from pathlib import Path
import random

import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from checkpoint import load_checkpoint
from dataset import HRDataset
from train import train_step
import utils


def configured_paths(cfg, names):
    return [path for name in names for path in cfg.dataset[name]]


def fixed_objective(model, batch, device, seed):
    compressed, clean, quality = (tensor.to(device) for tensor in batch)
    generator = torch.Generator(device=device).manual_seed(seed)
    t = torch.sigmoid(
        model.time_logit_mean
        + model.time_logit_std
        * torch.randn(
            clean.shape[0], device=device, dtype=clean.dtype, generator=generator
        )
    )
    noise = torch.randn(
        clean.shape, device=device, dtype=clean.dtype, generator=generator
    )
    return (compressed, clean, quality), {"t": t, "noise": noise}


def objective_probes(model, batch, device, seed, count):
    """Create held-out flow states for a repeatable one-batch loss check."""
    if count < 1:
        raise ValueError("smoke.probe_count must be positive")
    fixed_batch = None
    probes = []
    for offset in range(count):
        fixed_batch, objective = fixed_objective(
            model, batch, device, seed + offset
        )
        probes.append(objective)
    return fixed_batch, probes


def measure_loss(model, batch, probes):
    model.eval()
    totals = {}
    with torch.no_grad():
        for objective in probes:
            losses = utils.compute_loss(model, *batch, **objective)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item()
    return {name: value / len(probes) for name, value in totals.items()}


def jpeg_boundary_error(image, clean):
    """Measure discontinuities in the prediction error across the JPEG grid."""
    error = image - clean
    vertical = (error[..., :, 8::8] - error[..., :, 7:-1:8]).abs().mean()
    horizontal = (error[..., 8::8, :] - error[..., 7:-1:8, :]).abs().mean()
    return ((vertical + horizontal) / 2).item()


def restoration_metrics(image, clean, lpips):
    return {
        "mae": F.l1_loss(image, clean).item(),
        "psnr": peak_signal_noise_ratio(image, clean, data_range=1.0).item(),
        "ssim": structural_similarity_index_measure(
            image, clean, data_range=1.0
        ).item(),
        "lpips": lpips(image, clean).item(),
        "boundary": jpeg_boundary_error(image, clean),
    }


def save_predictions(path, compressed, restored, clean, quality):
    rows = min(4, len(compressed))
    figure, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
    quality_factors = (100 * (1 - quality)).round().flatten().tolist()
    for row in range(rows):
        images = (compressed[row], restored[row], clean[row])
        titles = (
            f"JPEG input (QF {quality_factors[row]:.0f})",
            "Overfit prediction",
            "Clean target",
        )
        for axis, image, title in zip(axes[row], images, titles):
            axis.imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1),
                        interpolation="nearest")
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def print_prediction_metrics(compressed, restored, clean, quality, device):
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    with torch.inference_mode():
        input_metrics = restoration_metrics(compressed, clean, lpips)
        prediction_metrics = restoration_metrics(restored, clean, lpips)

    print(
        "quality_factors="
        + ",".join(
            f"{value:.0f}" for value in (100 * (1 - quality)).round().flatten()
        )
    )
    print("prediction_metrics (higher PSNR/SSIM; lower MAE/LPIPS/boundary)")
    print("source       MAE       PSNR      SSIM      LPIPS     boundary_error")
    for name, metrics in (
        ("jpeg_input", input_metrics),
        ("prediction", prediction_metrics),
    ):
        print(
            f"{name:10s} {metrics['mae']:.7f} {metrics['psnr']:9.4f} "
            f"{metrics['ssim']:.6f} {metrics['lpips']:.6f} "
            f"{metrics['boundary']:.7f}"
        )
    prediction_to_input = F.l1_loss(restored, compressed).item()
    target_correction = F.l1_loss(clean, compressed).item()
    correction_ratio = prediction_to_input / max(target_correction, 1e-12)
    print(
        f"prediction_to_input_mae={prediction_to_input:.7f} "
        f"target_correction_mae={target_correction:.7f} "
        f"correction_ratio={correction_ratio:.3f}"
    )
    return input_metrics, prediction_metrics, correction_ratio


@hydra.main(version_base=None, config_path="../config", config_name="smoke_overfit")
def main(cfg: DictConfig) -> None:
    smoke_cfg = cfg.smoke
    seed = smoke_cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = HRDataset(
        configured_paths(cfg, cfg.train.datasets),
        train=True,
        crop_size=cfg.train.get("crop_size", 128),
        quality_range=(smoke_cfg.quality_factor, smoke_cfg.quality_factor),
    )
    loader = DataLoader(
        dataset,
        batch_size=smoke_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = instantiate(cfg.model).to(device)
    batch, probes = objective_probes(
        model,
        next(iter(loader)),
        device,
        seed + 10_000,
        smoke_cfg.probe_count,
    )
    optimizer = optim.Adam(
        model.parameters(), cfg.train.lr, betas=(0.9, 0.95), weight_decay=0
    )
    scheduler_cfg = cfg.get("lr_scheduler", cfg.train.scheduler)
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
    checkpoint = cfg.train.get("start_checkpoint")
    if checkpoint:
        checkpoint = Path(hydra.utils.get_original_cwd()) / checkpoint
        start_epoch = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            device,
            resume_optimizer=cfg.train.get("resume_optimizer", True),
            resume_scheduler=cfg.train.get("resume_scheduler", True),
            scheduler_config=OmegaConf.to_container(scheduler_cfg, resolve=True),
        )
        print(
            f"checkpoint={checkpoint} checkpoint_epoch={start_epoch - 1} "
            f"lr={optimizer.param_groups[0]['lr']:.8g}"
        )

    initial = measure_loss(model, batch, probes)
    print(
        f"step=0 probe_loss={initial['loss']:.7f} "
        f"probe_states={len(probes)}"
    )
    model.train()
    training_generator = torch.Generator(device=device).manual_seed(seed + 1)
    for step in range(1, smoke_cfg.steps + 1):
        losses, grad_norm = train_step(
            model,
            batch,
            optimizer,
            device,
            # Keep the image pair fixed, but sample a fresh time and noise as
            # production training does. A single frozen state only tests
            # point memorization and says nothing about the sampled path.
            loss_kwargs={"generator": training_generator},
        )
        if step == 1 or step % smoke_cfg.log_every == 0:
            print(
                f"step={step} loss={losses['loss'].item():.7f} "
                f"grad_norm={grad_norm.item():.5f}"
            )

    final = measure_loss(model, batch, probes)
    compressed, clean, quality = batch
    model.eval()
    prediction_generator = torch.Generator(device=device).manual_seed(
        smoke_cfg.prediction_seed
    )
    with torch.inference_mode():
        restored, _ = utils.predict(model, compressed, prediction_generator)
    prediction_path = Path(smoke_cfg.prediction_path)
    if not prediction_path.is_absolute():
        prediction_path = Path(hydra.utils.get_original_cwd()) / prediction_path
    save_predictions(prediction_path, compressed, restored, clean, quality)
    input_metrics, prediction_metrics, correction_ratio = print_prediction_metrics(
        compressed, restored, clean, quality, device
    )

    reduction = 1.0 - final["loss"] / initial["loss"]
    print(
        f"initial_loss={initial['loss']:.7f} final_loss={final['loss']:.7f} "
        f"reduction={reduction:.1%}"
    )
    print(
        "final_components="
        + ", ".join(f"{name}={value:.7f}" for name, value in final.items())
    )
    print(f"predictions={prediction_path}")

    failures = []
    if reduction < smoke_cfg.min_loss_reduction:
        failures.append(
            f"required {smoke_cfg.min_loss_reduction:.0%} loss reduction, "
            f"observed {reduction:.1%}"
        )
    psnr_gain = prediction_metrics["psnr"] - input_metrics["psnr"]
    lpips_gain = input_metrics["lpips"] - prediction_metrics["lpips"]
    boundary_gain = input_metrics["boundary"] - prediction_metrics["boundary"]
    mae_ratio = prediction_metrics["mae"] / max(input_metrics["mae"], 1e-12)
    if psnr_gain <= smoke_cfg.min_psnr_gain:
        failures.append(
            f"PSNR gain {psnr_gain:+.4f} dB did not exceed "
            f"{smoke_cfg.min_psnr_gain:+.4f} dB"
        )
    if lpips_gain <= smoke_cfg.min_lpips_gain:
        failures.append(
            f"LPIPS reduction {lpips_gain:+.6f} did not exceed "
            f"{smoke_cfg.min_lpips_gain:+.6f}"
        )
    if boundary_gain <= smoke_cfg.min_boundary_gain:
        failures.append(
            f"boundary-error reduction {boundary_gain:+.7f} did not exceed "
            f"{smoke_cfg.min_boundary_gain:+.7f}"
        )
    if mae_ratio > smoke_cfg.max_mae_ratio:
        failures.append(
            f"prediction/input MAE ratio {mae_ratio:.3f} exceeded "
            f"{smoke_cfg.max_mae_ratio:.3f} (not a strong overfit)"
        )
    if correction_ratio < smoke_cfg.min_correction_ratio:
        failures.append(
            f"correction ratio {correction_ratio:.3f} was below "
            f"{smoke_cfg.min_correction_ratio:.3f} (too close to identity)"
        )
    if failures:
        raise RuntimeError("one-batch overfit failed:\n- " + "\n- ".join(failures))
    print(
        "PASS: sampled predictions overfit the batch and beat the JPEG input "
        f"(PSNR {psnr_gain:+.4f} dB, LPIPS {lpips_gain:+.6f}, "
        f"boundary error {boundary_gain:+.7f})"
    )


if __name__ == "__main__":
    main()
