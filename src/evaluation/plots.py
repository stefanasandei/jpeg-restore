"""Render the paper's evaluation plots from measured CSV reports."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "meanflow_1": "MeanFlow (1 step)",
    "meanflow_2": "MeanFlow (2 steps)",
    "fbcnn": "FBCNN",
    "jdec": "JDEC",
    "promptcir": "PromptCIR",
    "diffbir": "DiffBIR",
    "supir": "SUPIR",
    "restormer_backbone": "Restormer",
    "sana_backbone": "SANA Linear DiT",
    "rela_backbone": "RELA DiT",
    "dit_backbone": "DiT",
}


def read_report(directory, filename):
    path = Path(directory) / filename
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"no measurements in {path}")
    if "status" not in frame or (frame["status"] != "ok").any():
        raise ValueError(f"{path} contains failed or partial measurements")
    return frame


def save_figure(figure, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_scaling(directory, output, batch_size=1):
    frame = read_report(directory, "performance.csv")
    frame = frame[(frame.scope == "backbone") & (frame.batch_size == batch_size)]
    if frame.empty or (frame.height != frame.width).any():
        raise ValueError(
            "scaling plots require square backbone measurements at the selected batch size"
        )
    if frame.duplicated(["model", "height"]).any():
        raise ValueError("scaling plot has duplicate model/resolution measurements")
    columns = {
        "throughput_img_s": ("Throughput (images/s)", "throughput_vs_resolution"),
        "peak_allocated_gib": ("Peak allocated memory (GiB)", "memory_vs_resolution"),
    }
    # CPU runs have no CUDA memory readings; do not create an empty memory plot.
    if frame.peak_allocated_gib.isna().any():
        raise ValueError(
            "memory scaling requires CUDA measurements with peak_allocated_gib"
        )
    for column, (label, filename) in columns.items():
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model, rows in frame.groupby("model", sort=False):
            rows = rows.sort_values("height")
            axis.plot(
                rows.height, rows[column], marker="o", label=LABELS.get(model, model)
            )
        axis.set(
            xlabel="Square resolution (pixels)",
            ylabel=label,
            title=f"Backbone scaling, batch size {batch_size}",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        save_figure(figure, output, filename)


def plot_pareto(
    quality_directory,
    performance_directory,
    output,
    dataset="live1",
    quality_factor=10,
    size=1024,
):
    quality = read_report(quality_directory, "quality.csv")
    performance = read_report(performance_directory, "performance.csv")
    quality = quality[(quality.dataset == dataset) & (quality.qf == quality_factor)]
    performance = performance[
        (performance.scope == "system")
        & (performance.batch_size == 1)
        & (performance.height == size)
        & (performance.width == size)
    ]
    if quality.duplicated("model").any() or performance.duplicated("model").any():
        raise ValueError(
            "Pareto plot requires one quality and performance row per model"
        )
    if "lpips" not in quality:
        raise ValueError("quality reports must include LPIPS")
    paired = quality[["model", "lpips"]].merge(
        performance[["model", "throughput_img_s"]], on="model", validate="one_to_one"
    )
    if (
        paired.empty
        or paired[["lpips", "throughput_img_s"]].isna().any().any()
        or (paired.throughput_img_s <= 0).any()
    ):
        raise ValueError("no valid matching LPIPS and system-throughput measurements")
    # Mark the measured non-dominated points, without interpolating new results.
    frontier = paired.loc[
        [
            not (
                (paired.lpips <= row.lpips)
                & (paired.throughput_img_s >= row.throughput_img_s)
                & (
                    (paired.lpips < row.lpips)
                    | (paired.throughput_img_s > row.throughput_img_s)
                )
            ).any()
            for row in paired.itertuples()
        ]
    ].sort_values("throughput_img_s")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        frontier.throughput_img_s,
        frontier.lpips,
        color="0.65",
        linestyle="--",
        zorder=0,
    )
    for row in paired.itertuples():
        axis.scatter(row.throughput_img_s, row.lpips, s=45)
        axis.annotate(
            LABELS.get(row.model, row.model),
            (row.throughput_img_s, row.lpips),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xscale="log",
        xlabel=f"Throughput at {size}×{size} (images/s)",
        ylabel="LPIPS (lower is better)",
        title=f"{dataset}, QF {quality_factor}",
    )
    axis.invert_yaxis()
    axis.grid(alpha=0.25)
    save_figure(figure, output, "pareto_quality_vs_throughput")
    Path(output).mkdir(parents=True, exist_ok=True)
    paired.to_csv(Path(output) / "pareto_points.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("scaling", "pareto"))
    parser.add_argument(
        "--performance", type=Path, required=True, help="evaluation output directory"
    )
    parser.add_argument(
        "--quality", type=Path, help="quality output directory, required for pareto"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/paper/figures"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="live1")
    parser.add_argument("--qf", type=int, default=10)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    if args.kind == "scaling":
        plot_scaling(args.performance, args.output, args.batch_size)
    else:
        if args.quality is None:
            parser.error("pareto requires --quality")
        plot_pareto(
            args.quality,
            args.performance,
            args.output,
            args.dataset,
            args.qf,
            args.size,
        )


if __name__ == "__main__":
    main()
