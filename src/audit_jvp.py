"""Compare the actual monolithic and segmented MeanFlow paths, including gradients.

Each method/seed runs in a fresh process. CUDA peaks include resident model/input
storage and loss forward/backward, but exclude optimizer state and optimizer.step.
The optional LPIPS term is excluded; MeanFlow and severity losses are included.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from evaluation.adapters import NativeAdapter, seed_everything
from evaluation.performance import synchronize
from evaluation.reporting import environment_metadata, file_identity


def compare(reference, candidate, rtol, atol):
    """Aggregate all tensors without flattening the entire model into one copy."""
    squared_error = reference_norm = candidate_norm = dot = maximum = 0.0
    passed = True
    count = nonzero = 0
    for name, ref in reference.items():
        other = candidate.get(name)
        if ref is None or other is None:
            passed &= ref is None and other is None
            continue
        if ref.shape != other.shape:
            raise ValueError(f"shape mismatch for {name}")
        passed &= bool(torch.allclose(ref, other, rtol=rtol, atol=atol))
        ref, other = ref.double(), other.double()
        difference = other - ref
        squared_error += difference.square().sum().item()
        reference_norm += ref.square().sum().item()
        candidate_norm += other.square().sum().item()
        dot += (ref * other).sum().item()
        maximum = max(maximum, difference.abs().max().item())
        count += ref.numel()
        nonzero += int(torch.count_nonzero(ref))
    if set(reference) != set(candidate):
        passed = False
    return {
        "passed": bool(passed),
        "relative_l2": (squared_error / reference_norm) ** 0.5
        if reference_norm else (0.0 if squared_error == 0 else None),
        "cosine": dot / (reference_norm * candidate_norm) ** 0.5
        if reference_norm and candidate_norm else None,
        "max_absolute_error": maximum,
        "reference_l2": reference_norm ** 0.5,
        "candidate_l2": candidate_norm ** 0.5,
        "absolute_l2_error": squared_error ** 0.5,
        "elements": count,
        "nonzero_reference_elements": nonzero,
    }


def worker(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    spec = {"kind": "native", "architecture": "rela_dit_meanflow"}
    if args.checkpoint:
        spec["checkpoint"] = args.checkpoint
    if args.smoke:
        spec["overrides"] = {"model": {
            "hidden_size": 32, "depth": 4, "num_heads": 2,
            "linear_head_dim": 16, "quality_split": 1,
        }}
    adapter = NativeAdapter(spec, args.device, args.dtype)
    model = adapter.model.train()
    # A random initialization has zero severity-head output projections. Exercise
    # those branches in smoke runs rather than accepting a degenerate zero test.
    if args.smoke and not args.checkpoint:
        with torch.no_grad():
            for module in (model.model.network.quality_head,
                           model.model.network.quality_embedder):
                torch.nn.init.normal_(module[2].weight, std=0.01)
    device, dtype = adapter.device, adapter.dtype
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    shape = (args.batch, 3, args.crop, args.crop)
    clean = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    degraded = clean.mul(0.8).add(0.1)
    condition = model.encode(model.model.pad(degraded))
    target = model.target(model.model.pad(clean), model.model.pad(degraded))
    noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
    r = torch.full((args.batch,), 0.2, device=device, dtype=dtype)
    t = torch.full_like(r, 0.7)
    quality = torch.full((args.batch, 1), 0.9, device=device, dtype=dtype)
    z = model.interpolate(target, noise, t)
    with torch.no_grad():
        velocity = model.u(z, t, t, condition)
    if args.method == "jvp":
        primal, tangent = torch.func.jvp(
            lambda z, r, t: model.u(z, r, t, condition),
            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)),
        )
    else:
        primal, tangent, _ = model._rcm_jvp(z, r, t, velocity, condition)
    values = {"primal": primal.detach().cpu(), "tangent": tangent.detach().cpu()}
    del primal, tangent, condition, target, z, velocity
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device) / 1024**3
    else:
        baseline = None
    synchronize(device)
    start = time.perf_counter()
    model.derivative_method = args.method
    loss = model.compute_loss(
        degraded, clean, quality, r=r, t=t, noise=noise,
        include_perceptual=False,
    )["loss"]
    loss.backward()
    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000
    measurement = {
        "elapsed_forward_backward_ms": elapsed,
        "baseline_allocated_gib": baseline,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda" else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    values["loss"] = loss.detach().cpu()
    values["gradients"] = {
        name: parameter.grad.detach().cpu() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    torch.save(values, args.worker_output)
    Path(args.worker_output + ".json").write_text(json.dumps(measurement))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="outputs/jvp-audit.json")
    parser.add_argument("--method", choices=["jvp", "rcm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if min(args.crop, args.batch, args.threads) < 1:
        parser.error("crop, batch, and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no GPU measurements were made")
    if args.worker_output:
        worker(args)
        return
    torch.set_num_threads(args.threads)
    report = {
        "environment": environment_metadata(torch.device(args.device)),
        "settings": vars(args),
        "audit_script_sha256": file_identity(__file__, True)["sha256"],
        "checkpoint": file_identity(args.checkpoint, True) if args.checkpoint else None,
        "scope": {
            "inputs": "fixed synthetic RGB, identical noise and r=0.2/t=0.7 per method",
            "loss": "adaptive MeanFlow plus severity regression; LPIPS excluded",
            "gradients": "all named parameters, including None/non-None pattern",
            "memory": "total process CUDA allocated/reserved peaks after reset; includes resident model/inputs and loss forward/backward; no optimizer constructed or stepped",
            "timing": "one synchronized loss forward/backward; diagnostic, not throughput benchmark",
            "claim": "finite test cases only; not proof or reconstruction of historical Table IV memory values",
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="jvp-audit-") as directory:
        for seed in args.seeds:
            records = {}
            for method in ("jvp", "rcm"):
                path = str(Path(directory) / f"{method}.pt")
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--method", method, "--seed", str(seed),
                           "--worker-output", path]
                for key in ("device", "dtype", "crop", "batch", "threads"):
                    command.extend([f"--{key}", str(getattr(args, key))])
                if args.smoke:
                    command.append("--smoke")
                if args.checkpoint:
                    command.extend(["--checkpoint", args.checkpoint])
                subprocess.run(command, check=True)
                records[method] = torch.load(path, map_location="cpu", weights_only=True)
                records[method]["measurement"] = json.loads(Path(path + ".json").read_text())
            reference, candidate = records["jvp"], records["rcm"]
            case = {"seed": seed, "measurements": {
                method: record["measurement"] for method, record in records.items()
            }}
            for name in ("primal", "tangent", "loss", "gradients"):
                a, b = reference[name], candidate[name]
                case[name] = compare(a if isinstance(a, dict) else {name: a},
                                     b if isinstance(b, dict) else {name: b},
                                     args.rtol, args.atol)
            case["loss"]["reference_value"] = float(reference["loss"])
            case["loss"]["candidate_value"] = float(candidate["loss"])
            report["cases"].append(case)
            print(json.dumps({"seed": seed, "gradients": case["gradients"]}), flush=True)
            del reference, candidate, records
    report["passed"] = all(case[name]["passed"] for case in report["cases"]
                           for name in ("primal", "tangent", "loss", "gradients"))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
