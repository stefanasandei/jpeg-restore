import math
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .adapters import seed_everything, validate_output
from .data import case_seed, compress, image_key, tensor_from_image


def evaluate_quality(
    adapter, model_name, datasets, cfg, metrics, device, image_rows, errors, output_dir
):
    summaries = []
    fingerprints = {
        path: sha256(Path(path).read_bytes()).hexdigest()
        for paths in datasets.values() for path in paths
    }
    for dataset_name, paths in datasets.items():
        for quality in cfg.quality_factors:
            rows = []
            for path in tqdm(paths, desc=f"{model_name}/{dataset_name}/QF{quality}"):
                seed = case_seed(
                    cfg.seed, dataset_name, fingerprints[path], quality
                )
                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "qf": quality,
                    "image": str(Path(path).resolve()),
                    "seed": seed,
                    "image_sha256": fingerprints[path],
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    with Image.open(path) as source:
                        clean = source.convert("RGB")
                    item = compress(clean, quality, cfg.get("input_pad_multiple", 1))
                    row["jpeg_sha256"] = sha256(item.jpeg).hexdigest()
                    row["encoded_height"], row["encoded_width"] = item.rgb.shape[-2:]
                    seed_everything(seed)
                    prepared = adapter.prepare(item)
                    with torch.inference_mode(), adapter.precision_context():
                        restored = adapter.restore_protocol(prepared, seed)
                    restored = validate_output(restored, item.rgb.shape).to(device)
                    restored = restored[..., :clean.height, :clean.width]
                    row["nfe"] = adapter.nfe
                    reference = tensor_from_image(clean).to(device)
                    with torch.inference_mode():
                        values = metrics(restored, reference)
                    if not all(
                        math.isfinite(value) or (name == "psnr" and value == math.inf)
                        for name, value in values.items()
                    ):
                        raise FloatingPointError(f"non-finite metric values: {values}")
                    # JSON has no Infinity literal. Exact-match PSNR is represented
                    # by null and an explicit flag, never an invalid JSON number.
                    if values.get("psnr") == math.inf:
                        values["psnr"] = None
                        row["psnr_infinite"] = True
                    row.update(
                        values, status="ok", height=clean.height, width=clean.width
                    )
                    if cfg.save_images:
                        destination = (
                            Path(output_dir)
                            / "images"
                            / model_name
                            / dataset_name
                            / f"qf{quality}"
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        pixels = (
                            restored[0]
                            .cpu()
                            .permute(1, 2, 0)
                            .mul(255)
                            .round()
                            .byte()
                            .numpy()
                        )
                        Image.fromarray(pixels).save(
                            destination / f"{image_key(path)}.png"
                        )
                except Exception as error:
                    row.update(status="error", error=str(error))
                    errors.append({**row, "pipeline": "quality"})
                    if not cfg.continue_on_error:
                        image_rows.append(row)
                        raise
                image_rows.append(row)
                row["finished_utc"] = datetime.now(timezone.utc).isoformat()
                with (Path(output_dir) / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
            successful = [row for row in rows if row["status"] == "ok"]
            summary = {
                "model": model_name,
                "dataset": dataset_name,
                "qf": quality,
                "images": len(successful),
                "expected_images": len(paths),
                "status": "ok" if len(successful) == len(paths) else "partial",
            }
            for name in metrics.metrics:
                values = [row[name] for row in successful if row.get(name) is not None]
                if name == "psnr" and any(
                    row.get("psnr_infinite") for row in successful
                ):
                    summary[name] = None
                    summary["psnr_infinite"] = True
                else:
                    summary[name] = float(np.mean(values)) if values else None
                summary[f"{name}_std"] = (
                    float(np.std(values))
                    if values and not (name == "psnr" and summary.get("psnr_infinite"))
                    else None
                )
            summaries.append(summary)
    return summaries
