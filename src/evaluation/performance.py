import time

import numpy as np
import torch

from .adapters import seed_everything, validate_output


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(adapter, item, warmup, iterations, seed):
    """Synchronized wall time includes encoding/sampling/decoding, excludes IPC/I/O."""
    prepared = adapter.prepare(item)
    seed_everything(seed)
    device = adapter.device
    samples = []
    with torch.inference_mode(), adapter.precision_context():
        for index in range(warmup):
            output = adapter.restore(prepared, seed + index)
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
            output = adapter.restore(prepared, seed + index)
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
