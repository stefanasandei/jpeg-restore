import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from .adapters import seed_everything, validate_output
from .data import EvaluationInput


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(adapter, item, warmup, iterations, seed):
    """Measure the declared boundary, with the same callable for warmup/timing."""
    with TemporaryDirectory(prefix="jpeg-benchmark-input-") as directory:
        if adapter.spec.get("benchmark_input") == "cached_jpeg_file":
            if item.jpeg is None or item.rgb.shape[0] != 1:
                raise ValueError("cached JPEG benchmarking requires one encoded image")
            source = Path(directory) / "input.jpg"
            source.write_bytes(item.jpeg)
            if adapter.spec.get("kind") == "jdec":
                # prepare stages identical JPEG bytes; restore reads coefficients
                # from its cached file inside the timed call.
                prepared = adapter.prepare(item)
                restore = lambda seed: adapter.restore_protocol(prepared, seed)
            else:
                def restore(seed):
                    rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
                    tensor = tensor.float().div_(255).unsqueeze(0)
                    prepared = adapter.prepare(EvaluationInput(tensor, item.jpeg))
                    return adapter.restore_protocol(prepared, seed)
        else:
            prepared = adapter.prepare(item)
            restore = lambda seed: adapter.restore_protocol(prepared, seed)
        return _measure_calls(adapter, item, restore, warmup, iterations, seed)


def _measure_calls(adapter, item, restore, warmup, iterations, seed):
    seed_everything(seed)
    device = adapter.device
    samples = []
    with torch.inference_mode(), adapter.precision_context():
        for index in range(warmup):
            output = restore(seed + index)
            validate_output(output, item.rgb.shape)
            del output
        synchronize(device)
        baseline = None
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        for index in range(iterations):
            synchronize(device)
            start = time.perf_counter()
            output = restore(seed + index)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
            if index != iterations - 1:
                del output
        # Capture memory before allocating validation temporaries.
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        )
        validate_output(output, item.rgb.shape)
        del output
    mean = float(np.mean(samples))
    cuda = device.type == "cuda"
    return {
        "latency_ms": mean,
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_std_ms": float(np.std(samples)),
        "throughput_img_s": item.rgb.shape[0] * 1000 / mean,
        "peak_allocated_gib": peak_allocated / 1024**3 if cuda else None,
        "peak_reserved_gib": peak_reserved / 1024**3 if cuda else None,
        "baseline_allocated_gib": baseline / 1024**3 if cuda else None,
        "latencies_ms": samples,
        "nfe": adapter.nfe,
    }
