"""Validate matched input identities, complete cohorts, summaries, and timings."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


LEGACY_MODELS = {"jpeg", "meanflow_1", "meanflow_2", "fbcnn", "promptcir", "jdec"}
MODELS = LEGACY_MODELS | {"codiff"}
WEIGHTS = {
    "codiff": "24607778dce971037cd4a013bbcba8475b02a29d5204a210a27e4996af19692b",
    "meanflow_1": "fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa",
    "meanflow_2": "fbb573706c64ae90da85e47d1ff3377734264142460770c54ebadc50a4ba4caa",
    "fbcnn": "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
    "jdec": "ab71edd4351d69cb08628a8b1310ddb8ec36f5cb052695230bbdddae1441015b",
    "promptcir": "728a2f3b9876b010c2c13386999baeb1a34023a3b2d6ded9c3419dc893ef6c47",
}
REVISIONS = {
    "codiff": "86e2f0499456298613472fa164417d1cce0a0cb7",
    "fbcnn": "54d1831927506b3247e2d4d245abb4f4dab1a1cd",
    "jdec": "e9fe8354754dd062dbf1defebb220e290507edd2",
    "promptcir": "3506b1aa0e6588fdf4e0b8345d90679adf1e2220",
}
METRICS = ("psnr", "ssim", "lpips", "dists", "musiq", "maniqa", "clipiqa")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tile_count(height, width):
    def count(length):
        starts = list(range(0, length - min(length, 224) + 1, 192))
        return len(starts) + (starts[-1] != length - min(length, 224))
    return count(height) * count(width)


def expected_nfe(model, height, width, spec):
    if model == "jpeg":
        return 0
    if model == "codiff":
        # Released CODiff native latent-grid algorithm; count UNet calls only.
        h, w = height // 8, width // 8
        if h * w <= 96 * 96:
            return 1
        tile = min(96, h, w)
        return (math.ceil((h - tile) / (tile - 32)) + 1) * (math.ceil((w - tile) / (tile - 32)) + 1)
    calls = tile_count(height, width) if spec.get("common_tiling") else 1
    return calls * (2 if model == "meanflow_2" else 1)


def check_csv(path, expected, excluded=()):
    with path.open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(expected), f"{path.name}: CSV row count differs")
    for csv_row, json_row in zip(actual, expected):
        for key, value in json_row.items():
            if key in excluded:
                continue
            require(key in csv_row, f"{path.name}: missing column {key}")
            cell = csv_row[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                require(math.isclose(float(cell), value, rel_tol=1e-9, abs_tol=1e-10), f"{path.name}: value differs in {key}")
            else:
                require(cell == ("" if value is None else str(value)), f"{path.name}: value differs in {key}")


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    config = manifest["config"]["eval"]
    models = config["models"]
    require(set(models) in (LEGACY_MODELS, MODELS) and len(models) == len(set(models)), "publication model set differs")
    current = "codiff" in models
    require(not results["errors"], "run contains errors")
    require(config["max_images"] is None, "subset runs cannot supply paper tables")
    require(config["dtype"] == "float32" and config["allow_tf32"], "precision policy differs")
    require(config["input_pad_multiple"] == 32, "input padding differs")
    require(config["performance"]["sizes"] == [[512, 512], [1024, 1024]], "timing sizes differ")
    require(config["quality_factors"] == [5, 10, 20], "quality factors differ")
    require(set(config["metrics"]) == set(METRICS), "metric suite differs")
    require({k: len(v) for k, v in manifest["datasets"].items()} ==
            {"live1": 29, "urban100": 100, "div2k_val": 100}, "dataset counts differ")
    for model in models:
        entry = manifest["models"][model]
        require(entry.get("trained"), f"{model}: trained checkpoint missing")
        require(entry["spec"]["benchmark_input"] == "cached_jpeg_file", "timing boundary differs")
        expected_tiling = None if current and model in ("meanflow_1", "meanflow_2", "codiff") else {"tile": 224, "overlap": 32, "clip": True}
        require(entry["spec"].get("common_tiling") == expected_tiling, f"{model}: inference policy differs")
        if model != "jpeg":
            require(entry["weight_dtypes"] == ["torch.float32"], f"{model}: weight dtype differs")
            require(entry["files"]["checkpoint"]["sha256"] == WEIGHTS[model], f"{model}: official checkpoint hash differs")
        if model == "codiff":
            require(entry["files"]["cave_checkpoint"]["sha256"] == "d49ef67568ffc9c22123eea934d37f1ceccb460f276455ec379bb6942408840b", "CaVE weights differ")
            assets = json.loads((directory.parent / "codiff-assets.json").read_text())
            backbone = assets[2]
            for asset in backbone["files"]:
                relative = asset["file"].split("/sd21/", 1)[1]
                require(entry["files"]["pretrained_model/" + relative]["sha256"] == asset["sha256"], "SD2.1 component differs")
            require(entry["upstream_diff_sha256"] == hashlib.sha256((directory.parent / "codiff-fp32.patch").read_bytes()).hexdigest(), "CODiff source patch differs")
        if model in REVISIONS:
            require(entry["repo_revision"] == REVISIONS[model], f"{model}: upstream revision differs")
        worker = entry.get("worker_environment", manifest["environment"])
        require(worker["tf32_matmul"] and worker["tf32_cudnn"], "worker TF32 policy differs")
        require(worker["torch_cpu_threads"] == 4, "CPU thread policy differs")
    rows = [json.loads(line) for line in (directory / "per_image.jsonl").read_text().splitlines()]
    require(all(row["model"] in MODELS for row in rows), "unexpected model rows")
    cohorts, groups = {}, {}
    expected_keys = {(dataset, Path(path).name, qf)
                     for dataset, paths in manifest["datasets"].items()
                     for path in paths for qf in config["quality_factors"]}
    for model in models:
        keyed = {}
        for row in (r for r in rows if r["model"] == model):
            key = (row["dataset"], Path(row["image"]).name, row["qf"])
            require(key not in keyed, f"duplicate input: {model}/{key}")
            require(row["status"] == "ok", f"failed input: {model}/{key}")
            require(all(isinstance(row.get(m), (int, float)) and math.isfinite(row[m])
                        for m in METRICS), f"missing/non-finite metric: {model}/{key}")
            require(len(row["image_sha256"]) == len(row["jpeg_sha256"]) == 64, "invalid input hash")
            expected_seed = int.from_bytes(hashlib.sha256(
                f'{config["seed"]}/{row["dataset"]}/{row["image_sha256"]}/{row["qf"]}'.encode()
            ).digest()[:4], "little")
            require(row["seed"] == expected_seed, "seed does not follow content-based rule")
            h, w = row["encoded_height"], row["encoded_width"]
            require(h % 32 == w % 32 == 0, "encoded image not block-aligned")
            require(row["nfe"] == expected_nfe(model, h, w, manifest["models"][model]["spec"]), "quality NFE differs from declared inference policy")
            require(datetime.fromisoformat(row["finished_utc"]) >=
                    datetime.fromisoformat(row["started_utc"]), "invalid case timestamps")
            keyed[key] = (row["image_sha256"], row["jpeg_sha256"], row["seed"], h, w)
            groups.setdefault((model, row["dataset"], row["qf"]), []).append(row)
        require(set(keyed) == expected_keys, f"{model}: image cohort differs from manifest")
        cohorts[model] = keyed
    reference = cohorts[models[0]]
    require(all(cohort == reference for cohort in cohorts.values()), "input bytes/seeds differ across models")
    summary_keys = set()
    for row in results["quality"]:
        key = (row["model"], row["dataset"], row["qf"])
        require(key not in summary_keys, "duplicate quality summary")
        summary_keys.add(key)
        samples = groups[key]
        require(row["status"] == "ok" and row["images"] == row["expected_images"] == len(samples),
                "incomplete quality summary")
        for metric in METRICS:
            values = [sample[metric] for sample in samples]
            require(math.isclose(row[metric], statistics.mean(values), rel_tol=1e-9, abs_tol=1e-10),
                    f"summary mean differs: {key}/{metric}")
            require(math.isclose(row[metric + "_std"], statistics.pstdev(values), rel_tol=1e-9, abs_tol=1e-10),
                    f"summary standard deviation differs: {key}/{metric}")
    require(summary_keys == set(groups), "missing quality summaries")
    timing_keys = set()
    for row in results["performance"]:
        key = (row["model"], row["height"], row["width"])
        require(key not in timing_keys, "duplicate timing configuration")
        timing_keys.add(key)
        require(row["status"] == "ok" and row["timing_scope"] == "cached_jpeg_file_to_rgb", "invalid timing boundary/status")
        require(row["dtype"] == "float32" and row["batch_size"] == 1, "timing precision/batch differs")
        require(row["warmup"] == 5 and row["iterations"] == 20, "timing repetitions differ")
        samples = row["latencies_ms"]
        require(len(samples) == 20 and all(math.isfinite(x) and x > 0 for x in samples), "invalid timing samples")
        require(math.isclose(row["latency_ms"], statistics.mean(samples), rel_tol=1e-9), "timing mean differs")
        require(math.isclose(row["throughput_img_s"], 1000 / statistics.mean(samples), rel_tol=1e-9), "throughput differs")
        require(row["nfe"] == expected_nfe(*key, manifest["models"][row["model"]]["spec"]), "timing NFE differs from declared inference policy")
    require(timing_keys == {(m, size[0], size[1]) for m in models
                            for size in config["performance"]["sizes"]}, "missing timing configurations")
    check_csv(directory / "quality.csv", results["quality"])
    check_csv(directory / "performance.csv", results["performance"], excluded=("latencies_ms",))
    check_csv(directory / "per_image.csv", rows)
    files = ("per_image.csv", "manifest.json", "results.json", "per_image.jsonl", "quality.csv", "performance.csv")
    return {"validated_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "models": models, "cases_per_model": len(expected_keys), "total_image_results": len(rows),
            "timing_rows": len(timing_keys), "input_hashes_and_seeds_identical": True,
            "summaries_recomputed_from_raw_records": True, "csv_json_agreement": True,
            "official_checkpoint_hashes_and_upstream_revisions_verified": True,
            "inference_policy": {m: manifest["models"][m]["spec"].get("common_tiling") or ("native_codiff" if m == "codiff" else "full_image") for m in models}, "dtype": config["dtype"],
            "tf32": True, "timing_boundary": "cached_jpeg_file_to_rgb",
            "datasets": {k: len(v) for k, v in manifest["datasets"].items()},
            "validated_file_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    result = validate(args.directory)
    (Path(args.directory) / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
