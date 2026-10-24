import gc
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from .adapters import DTYPES, build_adapter, seed_everything
from .data import discover_datasets, synthetic_input
from .quality import evaluate_quality
from .reporting import make_manifest, write_results


def validate_config(settings):
    if settings.mode not in ("quality", "performance", "both"):
        raise ValueError("eval.mode must be quality, performance, or both")
    if settings.dtype not in DTYPES:
        raise ValueError(f"eval.dtype must be one of {list(DTYPES)}")
    if not settings.models or len(settings.models) != len(set(settings.models)):
        raise ValueError("eval.models must be nonempty and unique")
    if settings.mode in ("quality", "both") and not settings.metrics:
        raise ValueError("quality evaluation requires at least one metric")
    if not settings.quality_factors or any(
        type(q) is not int or not 1 <= q <= 100 for q in settings.quality_factors
    ):
        raise ValueError("eval.quality_factors must contain integer QFs in [1, 100]")
    if len(set(settings.quality_factors)) != len(settings.quality_factors):
        raise ValueError("quality factors must be unique")
    if settings.max_images is not None and (
        type(settings.max_images) is not int or settings.max_images < 1
    ):
        raise ValueError("eval.max_images must be positive")
    perf = settings.performance
    if (
        type(perf.warmup) is not int
        or type(perf.iterations) is not int
        or perf.warmup < 0
        or perf.iterations < 1
    ):
        raise ValueError("warmup must be nonnegative and iterations positive")
    if not perf.batch_sizes or any(
        type(b) is not int or b < 1 for b in perf.batch_sizes
    ):
        raise ValueError("performance batch sizes must be positive integers")
    if not perf.sizes or any(
        len(size) != 2 or any(type(x) is not int or x < 1 for x in size)
        for size in perf.sizes
    ):
        raise ValueError(
            "performance sizes must be [height, width] pairs of positive integers"
        )
    if not 1 <= perf.quality_factor <= 100:
        raise ValueError("performance JPEG quality must be in [1, 100]")


def resolve_specs(cfg):
    specs = {}
    for name in cfg.eval.models:
        if name not in cfg.eval.model_specs:
            raise KeyError(
                f"unknown model {name!r}; available: {list(cfg.eval.model_specs)}"
            )
        if any(char in name for char in "/\\") or name in (".", ".."):
            raise ValueError("model names must be safe directory names")
        spec = OmegaConf.to_container(cfg.eval.model_specs[name], resolve=True)
        for key in ("checkpoint", "config", "repo", "python"):
            if spec.get(key) and (key != "python" or "/" in spec[key]):
                spec[key] = to_absolute_path(spec[key])
        steps = spec.get("steps")
        if steps is not None and (type(steps) is not int or steps < 1):
            raise ValueError(f"{name}: steps must be a positive integer")
        specs[name] = spec
    return specs


def evaluate(cfg):
    settings = cfg.eval
    if settings.get("cpu_threads"):
        torch.set_num_threads(int(settings.cpu_threads))
    if "allow_tf32" in settings:
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(settings.allow_tf32)
    # Older training configurations remain valid entry points.
    if "models" not in settings:
        defaults = OmegaConf.load(
            Path(__file__).resolve().parents[2] / "config" / "evaluation.yaml"
        ).eval
        legacy = OmegaConf.to_container(settings, resolve=True)
        settings = OmegaConf.merge(defaults, legacy)
        settings.models = ["configured"] if settings.checkpoint else ["jpeg"]
        settings.model_specs.configured = {
            "kind": "native",
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "checkpoint": settings.checkpoint,
        }
        cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg, resolve=False), {"eval": settings}
        )
        settings = cfg.eval
    validate_config(settings)
    device = torch.device(
        settings.device
        if settings.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        if settings.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("selected CUDA device does not support bfloat16")
    if device.type == "cpu" and settings.dtype == "float16":
        raise ValueError("use float32 or bfloat16 for CPU evaluation")
    specs = resolve_specs(cfg)
    datasets = {}
    if settings.mode in ("quality", "both"):
        datasets = discover_datasets(
            cfg.dataset, settings.datasets, settings.max_images
        )
    output = Path(to_absolute_path(settings.output_dir))
    if output.exists() and any(output.iterdir()) and not settings.overwrite:
        raise FileExistsError(
            f"{output} is not empty; select another eval.output_dir or eval.overwrite=true"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(cfg, device, specs, datasets)
    quality_rows, image_rows, performance_rows, errors = [], [], [], []
    metrics = None

    def persist():
        write_results(
            output, manifest, quality_rows, image_rows, performance_rows, errors
        )

    persist()
    try:
        for name, spec in specs.items():
            adapter = None
            try:
                print(
                    f"evaluating {name} ({spec['kind']}, {device}, {settings.dtype})",
                    flush=True,
                )
                seed_everything(settings.seed)
                adapter = build_adapter(spec, device, settings.dtype)
                identity = manifest["models"][name]
                identity.update(
                    parameters=adapter.parameter_count(),
                    nfe=adapter.nfe,
                    scope=adapter.scope,
                    timing_scope=adapter.timing_scope,
                    weight_dtypes=adapter.weight_dtypes(),
                    trained=getattr(adapter, "trained", True),
                )
                if hasattr(adapter, "resolved_model_config"):
                    identity["resolved_model_config"] = adapter.resolved_model_config
                if hasattr(adapter, "worker_metadata"):
                    identity["worker_environment"] = adapter.worker_metadata
                if settings.mode in ("performance", "both"):
                    perf = settings.performance
                    for height, width in perf.sizes:
                        for batch in perf.batch_sizes:
                            row = {
                                "model": name,
                                "scope": adapter.scope,
                                "height": height,
                                "width": width,
                                "timing_scope": adapter.timing_scope,
                                "batch_size": batch,
                                "dtype": settings.dtype,
                                "device": str(device),
                                "parameters_m": adapter.parameter_count() / 1e6,
                                "trained": getattr(adapter, "trained", True),
                                "warmup": perf.warmup,
                                "iterations": perf.iterations,
                            }
                            try:
                                item = synthetic_input(
                                    height,
                                    width,
                                    batch,
                                    perf.quality_factor,
                                    settings.seed,
                                )
                                row.update(
                                    adapter.benchmark(
                                        item,
                                        perf.warmup,
                                        perf.iterations,
                                        settings.seed,
                                    ),
                                    status="ok",
                                )
                            except Exception as error:
                                row.update(status="error", error=str(error))
                                errors.append({**row, "pipeline": "performance"})
                                if not settings.continue_on_error:
                                    performance_rows.append(row)
                                    raise
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                            performance_rows.append(row)
                            print(
                                f"  {height}x{width}, batch {batch}: {row['status']}",
                                flush=True,
                            )
                if datasets:
                    if not getattr(adapter, "trained", True):
                        raise ValueError(
                            f"{name}: quality evaluation requires trained weights; set checkpoint"
                        )
                    from metrics import MetricSuite

                    metric_device = torch.device(settings.metric_device or str(device))
                    metrics = (
                        MetricSuite(
                            settings.metrics,
                            OmegaConf.to_container(
                                settings.metric_options, resolve=True
                            ),
                            metric_device,
                        )
                        .to(metric_device)
                        .eval()
                    )
                    manifest["metric_directions"] = {
                        key: "higher" if metric.higher_is_better else "lower"
                        for key, metric in metrics.metrics.items()
                    }
                    quality_rows.extend(
                        evaluate_quality(
                            adapter,
                            name,
                            datasets,
                            settings,
                            metrics,
                            metric_device,
                            image_rows,
                            errors,
                            output,
                        )
                    )
                    metrics = None
                identity["nfe"] = adapter.nfe
            except Exception as error:
                errors.append({"model": name, "pipeline": "model", "error": str(error)})
                print(f"  error: {error}", flush=True)
                if not settings.continue_on_error:
                    raise
            finally:
                metrics = None
                if adapter is not None:
                    adapter.close()
                adapter = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                persist()
    finally:
        persist()
    for title, rows in (("Quality", quality_rows), ("Performance", performance_rows)):
        if rows:
            print(
                f"\n{title}\n{pd.DataFrame(rows).drop(columns=['latencies_ms'], errors='ignore').to_string(index=False)}"
            )
    print(f"\nResults: {output}")
    if errors:
        raise RuntimeError(
            f"evaluation recorded {len(errors)} error(s); see {output / 'errors.csv'}"
        )
    return {
        "quality": quality_rows,
        "performance": performance_rows,
        "output_dir": output,
    }
