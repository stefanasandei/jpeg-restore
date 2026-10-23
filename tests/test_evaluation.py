import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from PIL import Image

from evaluation.adapters import JPEGAdapter, NativeAdapter, validate_output
from evaluation.data import EvaluationInput, case_seed, compress, tensor_from_image
from evaluation.external import JDECAdapter, PromptCIRAdapter
from evaluation.performance import measure
from evaluation.process import ProcessAdapter
from evaluation.runner import evaluate, validate_config
from metrics import MetricSuite

ROOT = Path(__file__).resolve().parents[1]


def config(name="evaluation", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name=name, overrides=list(overrides))


def test_paper_presets_and_validation():
    cfg = config("evaluation_paper")
    assert {"fbcnn", "jdec", "promptcir", "diffbir", "supir"} <= set(cfg.eval.models)
    assert not any("sodiff" in name.lower() for name in cfg.eval.model_specs)
    validate_config(cfg.eval)
    cfg.eval.performance.iterations = 0
    with pytest.raises(ValueError, match="iterations"):
        validate_config(cfg.eval)


def test_codec_and_case_seeds():
    from utils import jpeg_compress

    image = Image.fromarray(
        np.random.default_rng(1).integers(0, 256, (35, 47, 3), dtype=np.uint8)
    )
    item = compress(image, 10)
    assert item.jpeg.startswith(b"\xff\xd8")
    assert torch.equal(item.rgb, tensor_from_image(jpeg_compress(image, 10)))
    assert case_seed(0, "live1", "x", 5) == case_seed(0, "live1", "x", 5)
    assert case_seed(0, "live1", "x", 5) != case_seed(0, "live1", "x", 10)


def test_metric_rows_are_independent():
    suite = MetricSuite(["psnr", "ssim"], {}, "cpu")
    reference = torch.ones(1, 3, 33, 35)
    first = suite(reference * 0.8, reference)
    second = suite(reference * 0.2, reference)
    fresh = MetricSuite(["psnr", "ssim"], {}, "cpu")(reference * 0.2, reference)
    assert second == pytest.approx(fresh)
    assert first["psnr"] > second["psnr"]
    with pytest.raises(ValueError, match="unique"):
        MetricSuite(["clipqa", "clipiqa"], {}, "cpu")


def test_output_and_timing():
    adapter = JPEGAdapter({}, "cpu", "float32")
    item = EvaluationInput(torch.rand(2, 3, 16, 24))
    result = measure(adapter, item, warmup=1, iterations=3, seed=0)
    assert len(result["latencies_ms"]) == 3
    assert result["throughput_img_s"] == pytest.approx(2000 / result["latency_ms"])
    assert result["peak_allocated_gib"] is None
    with pytest.raises(ValueError, match="RGB output"):
        validate_output(torch.zeros(1), item.rgb.shape)
    with pytest.raises(FloatingPointError):
        validate_output(torch.full_like(item.rgb, float("nan")), item.rgb.shape)


@pytest.mark.parametrize(
    "architecture,steps,solver,nfe",
    [
        ("rela_dit_meanflow", 1, None, 1),
        ("rela_dit_meanflow", 2, None, 2),
        ("rela_dit", 2, "euler", 2),
        ("rela_dit", 2, "heun", 4),
    ],
)
def test_native_flow_adapter_with_actual_tiny_backbone(
    tmp_path, architecture, steps, solver, nfe
):
    spec = {
        "kind": "native",
        "architecture": architecture,
        "steps": steps,
        "overrides": {
            "model": {
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 2,
                "linear_head_dim": 16,
                "quality_split": 1,
            }
        },
    }
    if solver:
        spec["solver"] = solver
    adapter = NativeAdapter(spec, "cpu", "float32")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": adapter.model.state_dict()}, checkpoint)
    adapter.close()
    if architecture == "rela_dit_meanflow":
        # Published checkpoint exports contain the inner model's state only.
        state = torch.load(checkpoint, weights_only=True)["model"]
        torch.save(
            {key.removeprefix("model."): value for key, value in state.items()},
            checkpoint,
        )
    spec["checkpoint"] = str(checkpoint)
    adapter = NativeAdapter(spec, "cpu", "float32")
    item = EvaluationInput(torch.rand(1, 3, 17, 23))
    with torch.inference_mode():
        first = adapter.restore(adapter.prepare(item), 7)
        second = adapter.restore(adapter.prepare(item), 7)
    assert adapter.nfe == nfe
    assert adapter.trained
    assert torch.equal(first, second)
    validate_output(first, item.rgb.shape)
    adapter.close()


def test_quality_and_performance_outputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i, shape in enumerate(((33, 47, 3), (39, 51, 3))):
        array = np.random.default_rng(i).integers(0, 256, shape, dtype=np.uint8)
        Image.fromarray(array).save(images / f"{i}.png")
    cfg = config(
        overrides=[
            "eval.mode=both",
            "eval.datasets=[live1]",
            "eval.metrics=[psnr,ssim]",
            f"dataset.live1=[{images}]",
            f"eval.output_dir={tmp_path / 'results'}",
            "eval.save_images=true",
            "eval.performance.sizes=[[16,24]]",
            "eval.performance.warmup=0",
            "eval.performance.iterations=2",
        ]
    )
    evaluate(cfg)
    output = tmp_path / "results"
    results = json.loads((output / "results.json").read_text())
    assert len(results["quality"]) == 3
    assert all(row["images"] == 2 for row in results["quality"])
    assert len(results["performance"]) == 1
    assert not results["errors"]
    assert len(list((output / "images").rglob("*.png"))) == 6
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metric_directions"] == {"psnr": "higher", "ssim": "higher"}
    rows = [
        json.loads(line)
        for line in (output / "per_image.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    for summary in results["quality"]:
        scores = [row["psnr"] for row in rows if row["qf"] == summary["qf"]]
        assert summary["psnr"] == pytest.approx(np.mean(scores))
    with pytest.raises(FileExistsError):
        evaluate(cfg)


def test_missing_weights_failure_is_saved(tmp_path):
    cfg = config(
        overrides=[
            "eval.mode=performance",
            "eval.models=[fbcnn]",
            f"eval.output_dir={tmp_path / 'results'}",
        ]
    )
    with pytest.raises(RuntimeError, match="error"):
        evaluate(cfg)
    result = json.loads((tmp_path / "results/results.json").read_text())
    assert result["errors"][0]["model"] == "fbcnn"
    assert not result["performance"]


def test_untrained_quality_is_rejected(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "x.png")
    cfg = config(
        overrides=[
            "eval.models=[tiny]",
            "eval.datasets=[live1]",
            "eval.metrics=[psnr]",
            f"dataset.live1=[{images}]",
            f"eval.output_dir={tmp_path / 'results'}",
        ]
    )
    with open_dict(cfg.eval.model_specs):
        cfg.eval.model_specs.tiny = {
            "kind": "native",
            "model": {
                "_target_": "torch.nn.Conv2d",
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 1,
            },
        }
    with pytest.raises(RuntimeError):
        evaluate(cfg)
    errors = json.loads((tmp_path / "results/results.json").read_text())["errors"]
    assert "requires trained weights" in errors[0]["error"]


def test_external_worker_isolation_and_persistence(tmp_path):
    repo = tmp_path / "official"
    (repo / "models").mkdir(parents=True)
    (repo / "models/__init__.py").write_text("")
    (repo / "utils.py").write_text("raise RuntimeError('wrong utils imported')\n")
    (repo / "models/network_fbcnn.py").write_text("""
import torch
print("Official import log must not corrupt the worker protocol")
class FBCNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        return x + self.bias, None
""")
    weights = tmp_path / "official.pt"
    torch.save({"bias": torch.tensor(0.25)}, weights)
    adapter = ProcessAdapter(
        {
            "kind": "fbcnn",
            "repo": str(repo),
            "checkpoint": str(weights),
            "python": sys.executable,
            "timeout": 60,
        },
        "cpu",
        "float32",
    )
    try:
        pid = adapter.process.pid
        item = EvaluationInput(torch.zeros(1, 3, 17, 23))
        for _ in range(2):
            output = adapter.restore(adapter.prepare(item), seed=7)
            assert torch.all(output == 0.25)
        assert adapter.process.pid == pid
        assert adapter.parameter_count() == 1
        result = adapter.benchmark(item, 1, 2, 7)
        assert result["nfe"] == 1
        assert len(result["latencies_ms"]) == 2
    finally:
        adapter.close()
    assert adapter.process.poll() is not None


def test_promptcir_full_image_path(monkeypatch):
    class MyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, image):
            assert image.shape[-2] % 32 == image.shape[-1] % 32 == 0
            return image

    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(MyModel=MyModel))
    monkeypatch.setattr("evaluation.external.checkpoint_path", lambda _: "unused")
    monkeypatch.setattr("evaluation.external.load_weights", lambda _: {})
    adapter = PromptCIRAdapter({}, "cpu", "float32")
    image = torch.rand(1, 3, 33, 47)
    assert torch.equal(adapter.restore(image, 0), image)


def test_jdec_passes_coefficients_and_quantization(tmp_path):
    adapter = JDECAdapter.__new__(JDECAdapter)
    adapter.device = torch.device("cpu")
    adapter.jpeg_path = tmp_path / "input.jpg"
    coefficients = (
        None,
        torch.full((2, 8, 8), 2),
        torch.ones(1, 2, 2, 8, 8),
        torch.ones(2, 1, 1, 8, 8),
    )
    adapter.dm = types.SimpleNamespace(read_coefficients=lambda _: coefficients)

    def model(y, c, quantization):
        assert y.shape == (1, 1, 2, 2, 8, 8)
        assert c.shape == (1, 2, 1, 1, 8, 8)
        assert quantization.shape == (1, 2, 8, 8)
        assert torch.allclose(y, torch.full_like(y, (2 + 1024) / 2040 * 2 - 1))
        return torch.zeros(1, 3, 16, 16)

    adapter.model = model
    item = compress(Image.new("RGB", (13, 11)), 10)
    output = adapter.restore(adapter.prepare(item), 0)
    assert output.shape == (1, 3, 11, 13)
    assert torch.all(output == 0.5)
    with pytest.raises(ValueError, match="batch size"):
        adapter.prepare(EvaluationInput(item.rgb.repeat(2, 1, 1, 1), item.jpeg))


def test_supir_full_image_path_and_nfe(monkeypatch, tmp_path):
    from evaluation.external import SUPIRAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()

        def to(self, *args, **kwargs):
            return self

        def batchify_sample(self, image, prompts, **kwargs):
            assert image.shape[-2:] == (64, 64)
            assert prompts == [""]
            assert kwargs["num_steps"] == 50 and kwargs["seed"] == 7
            return self.model(image)

    monkeypatch.setitem(sys.modules, "SUPIR", types.ModuleType("SUPIR"))
    monkeypatch.setitem(
        sys.modules,
        "SUPIR.util",
        types.SimpleNamespace(create_SUPIR_model=lambda *a, **kw: Model()),
    )
    cfg_path = tmp_path / "supir.yaml"
    cfg_path.write_text("model: test\n")
    adapter = SUPIRAdapter({"config": str(cfg_path)}, "cuda:0", "float32")
    image = torch.rand(1, 3, 33, 47)
    assert torch.allclose(adapter.restore(image, 7), image)
    assert adapter.nfe == 1


def test_diffbir_official_pipeline_and_nfe(monkeypatch, tmp_path):
    from evaluation.external import DiffBIRAdapter

    arguments = {
        "strength": 1,
        "cleaner_tile_size": 512,
        "cleaner_tile_stride": 256,
        "vae_encoder_tile_size": 256,
        "vae_decoder_tile_size": 256,
        "cldm_tile_size": 512,
        "cldm_tile_stride": 256,
        "neg_prompt": "",
        "cfg_scale": 4,
        "start_point_type": "noise",
        "noise_aug": 0,
        "rescale_cfg": False,
        "s_churn": 0,
        "s_tmin": 0,
        "s_tmax": 300,
        "s_noise": 1,
        "eta": 1,
        "order": 1,
    }
    (tmp_path / "inference.py").write_text(
        "from argparse import Namespace\ndef parse_args():\n    return Namespace(**"
        + repr(arguments)
        + ")\n"
    )
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text("model: test\n")
    weights = tmp_path / "weights.pt"
    weights.touch()

    class Loop:
        def __init__(self, args):
            assert args.upscale == 1 and args.precision == "fp32"
            self.cleaner = torch.nn.Identity()
            self.cldm = torch.nn.Identity()
            self.pipeline = types.SimpleNamespace(run=self.run)

        def run(self, image, **kwargs):
            assert kwargs["steps"] == 50
            assert not any(
                kwargs[key]
                for key in (
                    "cleaner_tiled",
                    "cldm_tiled",
                    "vae_encoder_tiled",
                    "vae_decoder_tiled",
                )
            )
            self.cldm(torch.from_numpy(image))
            self.cldm(torch.from_numpy(image))
            return image

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "diffbir", types.ModuleType("diffbir"))
    monkeypatch.setitem(
        sys.modules,
        "diffbir.inference",
        types.SimpleNamespace(CustomInferenceLoop=Loop),
    )
    adapter = DiffBIRAdapter(
        {"kind": "diffbir", "config": str(cfg_path), "checkpoint": str(weights)},
        "cuda:0",
        "float32",
    )
    adapter.device = torch.device("cpu")
    item = EvaluationInput(torch.full((1, 3, 33, 47), 128 / 255))
    assert torch.equal(adapter.restore(adapter.prepare(item), 7), item.rgb)
    assert adapter.nfe == 2


def test_exact_match_psnr_is_valid_json(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32), (0, 0, 0)).save(images / "x.png")
    cfg = config(
        overrides=[
            "eval.datasets=[live1]",
            "eval.metrics=[psnr]",
            "eval.quality_factors=[20]",
            f"dataset.live1=[{images}]",
            f"eval.output_dir={tmp_path / 'results'}",
        ]
    )
    evaluate(cfg)
    result = json.loads((tmp_path / "results/results.json").read_text())
    row = result["quality"][0]
    assert row["psnr"] is None and row["psnr_std"] is None and row["psnr_infinite"]
