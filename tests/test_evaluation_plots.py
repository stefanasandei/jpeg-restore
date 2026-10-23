import pandas as pd
import pytest

from evaluation.plots import plot_pareto, plot_scaling


def report(directory, filename, rows):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / filename, index=False)


def test_scaling_figures_and_missing_memory(tmp_path):
    rows = [
        {
            "model": model,
            "scope": "backbone",
            "batch_size": 1,
            "height": size,
            "width": size,
            "throughput_img_s": throughput,
            "peak_allocated_gib": memory,
            "status": "ok",
        }
        for model in ("sana_backbone", "rela_backbone")
        for size, throughput, memory in ((64, 100, 0.2), (128, 50, 0.3))
    ]
    directory = tmp_path / "run"
    output = tmp_path / "figures"
    report(directory, "performance.csv", rows)
    plot_scaling(directory, output)
    for name in ("throughput_vs_resolution", "memory_vs_resolution"):
        for extension in ("png", "pdf"):
            assert (output / f"{name}.{extension}").stat().st_size > 0
    for row in rows:
        row["peak_allocated_gib"] = None
    report(directory, "performance.csv", rows)
    with pytest.raises(ValueError, match="CUDA"):
        plot_scaling(directory, output)


def test_pareto_pairs_reports_and_excludes_jpeg(tmp_path):
    quality = tmp_path / "quality"
    performance = tmp_path / "performance"
    output = tmp_path / "figures"
    report(
        quality,
        "quality.csv",
        [
            {
                "model": name,
                "dataset": "live1",
                "qf": 10,
                "lpips": lpips,
                "status": "ok",
            }
            for name, lpips in (
                ("meanflow_1", 0.20),
                ("meanflow_2", 0.16),
                ("jpeg", 0.30),
            )
        ],
    )
    report(
        performance,
        "performance.csv",
        [
            {
                "model": name,
                "scope": scope,
                "height": 1024,
                "width": 1024,
                "batch_size": 1,
                "throughput_img_s": throughput,
                "status": "ok",
            }
            for name, scope, throughput in (
                ("meanflow_1", "system", 16),
                ("meanflow_2", "system", 8),
                ("jpeg", "identity", 1000),
            )
        ],
    )
    plot_pareto(quality, performance, output)
    points = pd.read_csv(output / "pareto_points.csv")
    assert set(points.model) == {"meanflow_1", "meanflow_2"}
    assert (output / "pareto_quality_vs_throughput.pdf").stat().st_size > 0
    assert (output / "pareto_quality_vs_throughput.png").stat().st_size > 0
    with pytest.raises(ValueError, match="no valid matching"):
        plot_pareto(quality, performance, output, quality_factor=5)


def test_plot_rejects_partial_reports(tmp_path):
    report(tmp_path, "performance.csv", [{"status": "error"}])
    with pytest.raises(ValueError, match="failed or partial"):
        plot_scaling(tmp_path, tmp_path / "figures")
