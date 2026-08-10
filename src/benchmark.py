import argparse
from contextlib import nullcontext
from pathlib import Path
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pandas as pd
import torch

from models import HaarWaveletRestoration
from utils import unpack_model_output


MODEL_NAMES = ("restormer", "adm_unet", "linear_dit")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "model"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
CHANNEL_ARGUMENTS = {
    "fbcnn": ("in_nc", "out_nc"),
    "restormer": ("inp_channels", "out_channels"),
    "adm_unet": ("in_channels", "out_channels"),
    "linear_dit": ("in_channels", "out_channels"),
}


def load_model(name, representation, device, dtype):
    cfg = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
    if representation == "haar":
        input_name, output_name = CHANNEL_ARGUMENTS[name]
        cfg.model[input_name] = 12
        cfg.model[output_name] = 12
        if name == "fbcnn":
            cfg.model.clamp_output = False

    model = instantiate(cfg.model)
    if representation == "haar":
        model = HaarWaveletRestoration(model)
    return model.eval().to(device=device, dtype=dtype)


def benchmark_batch(
    model, batch_size, height, width, warmup, iterations, device, dtype
):
    image = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
    autocast = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype != torch.float32
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        for _ in range(warmup):
            unpack_model_output(model(image))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(iterations):
            restored, _ = unpack_model_output(model(image))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if restored.shape != image.shape:
        raise RuntimeError(
            f"expected output {tuple(image.shape)}, got {tuple(restored.shape)}"
        )
    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    latency = elapsed / iterations
    return latency, batch_size / latency, peak_memory


def error_row(name, representation, batch_size, parameters, status):
    return {
        "Architecture": name,
        "Representation": representation,
        "Batch": batch_size,
        "Params (M)": parameters,
        "Latency (ms)": None,
        "Throughput (img/s)": None,
        "Peak memory (GiB)": None,
        "Status": status,
    }


def benchmark_model(name, representation, args, device, dtype):
    print(f"benchmarking {name} ({representation})...")
    try:
        model = load_model(name, representation, device, dtype)
        parameters = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    except (RuntimeError, ValueError) as error:
        return [
            error_row(
                name,
                representation,
                batch_size,
                None,
                str(error).splitlines()[0],
            )
            for batch_size in args.batch_sizes
        ]

    rows = []
    for batch_size in args.batch_sizes:
        try:
            latency, throughput, peak_memory = benchmark_batch(
                model,
                batch_size,
                args.height,
                args.width,
                args.warmup,
                args.iterations,
                device,
                dtype,
            )
            row = {
                "Architecture": name,
                "Representation": representation,
                "Batch": batch_size,
                "Params (M)": parameters,
                "Latency (ms)": latency * 1000,
                "Throughput (img/s)": throughput,
                "Peak memory (GiB)": peak_memory,
                "Status": "ok",
            }
        except torch.cuda.OutOfMemoryError:
            row = error_row(
                name, representation, batch_size, parameters, "out of memory"
            )
            torch.cuda.empty_cache()
        except RuntimeError as error:
            row = error_row(
                name,
                representation,
                batch_size,
                parameters,
                str(error).splitlines()[0],
            )
        rows.append(row)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def benchmark(args):
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    representations = (
        ["pixel", "haar"]
        if args.representation == "both"
        else [args.representation]
    )
    rows = []
    for representation in representations:
        for name in args.models:
            rows.extend(benchmark_model(name, representation, args, device, dtype))

    results = pd.DataFrame(rows)
    numeric_columns = [
        "Params (M)",
        "Latency (ms)",
        "Throughput (img/s)",
        "Peak memory (GiB)",
    ]
    results[numeric_columns] = results[numeric_columns].round(2)
    print(f"\nThroughput at {args.width}x{args.height} ({device}, {args.dtype})")
    print(results.to_string(index=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark restoration model throughput"
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--representation",
        choices=["pixel", "haar", "both"],
        default="haar",
        help="input representation; haar is the JPEG-restoration default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
