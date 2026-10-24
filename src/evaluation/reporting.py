import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from .external import SOURCES

ROOT = Path(__file__).resolve().parents[2]


def git_revision(directory):
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(device):
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "torchmetrics",
        "pyiqa",
        "transformers",
        "numpy",
        "opencv-python",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0] if device.type == "cuda" else None
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "nvidia_driver": driver,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "torch_cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "all_package_versions": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata and distribution.metadata.get("Name")
        )),
        "git_revision": git_revision(ROOT),
    }


def file_identity(path, hash_weights):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    identity = {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
    if hash_weights:
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def make_manifest(cfg, device, specs, datasets):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(device),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src").rglob("*.py"))
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "datasets": {
            name: [str(Path(p).resolve()) for p in paths]
            for name, paths in datasets.items()
        },
        "models": {
            name: {
                "spec": spec,
                "upstream_source": SOURCES.get(spec["kind"]),
                "repo_revision": git_revision(spec["repo"])
                if spec.get("repo")
                else None,
                "files": {
                    key: file_identity(spec[key], cfg.eval.get("hash_weights", True))
                    for key in ("checkpoint", "config")
                    if spec.get(key)
                },
            }
            for name, spec in specs.items()
        },
        "metric_directions": {},
    }


def write_results(directory, manifest, quality, images, performance, errors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "manifest.json": manifest,
        "results.json": {
            "quality": quality,
            "performance": performance,
            "errors": errors,
        },
    }
    for name, value in payloads.items():
        temporary = directory / (name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / name)
    for name, rows in (
        ("quality.csv", quality),
        ("per_image.csv", images),
        ("performance.csv", performance),
        ("errors.csv", errors),
    ):
        columns = {
            "quality.csv": [
                "model",
                "dataset",
                "qf",
                "images",
                "expected_images",
                "status",
            ],
            "per_image.csv": ["model", "dataset", "qf", "image", "seed", "status"],
            "performance.csv": [
                "model",
                "scope",
                "height",
                "width",
                "batch_size",
                "status",
            ],
            "errors.csv": ["model", "pipeline", "error"],
        }
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns[name])
        if "latencies_ms" in frame:
            frame = frame.drop(columns=["latencies_ms"])
        frame.to_csv(directory / name, index=False)
    with (directory / "per_image.jsonl").open("w") as stream:
        for row in images:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
