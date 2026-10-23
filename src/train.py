import multiprocessing as mp
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from wandb.sdk.lib import runid

import utils
import wandb
from checkpoint import load_checkpoint, save_checkpoint
from dataset import HRDataset, configure_data_worker
from validation_metrics import RestorationMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dataset_paths(dataset_cfg, names):
    """Expand exact or glob-style dataset group names into image paths."""
    paths = []
    known_names = list(dataset_cfg.keys())
    for pattern in names:
        matched_names = [
            name for name in known_names if fnmatchcase(name, pattern)
        ]
        if not matched_names:
            raise KeyError(
                f"dataset pattern {pattern!r} matched no configured groups"
            )
        for name in matched_names:
            paths.extend(dataset_cfg[name])

    # Avoid sampling a directory twice when exact and wildcard groups overlap.
    return list(dict.fromkeys(paths))


def train_step(
    model,
    batch,
    device,
    exploration=1,
    perceptual_loss=None,
    loss_divisor=1,
):
    if exploration < 1:
        raise ValueError("exploration must be positive")
    if loss_divisor < 1:
        raise ValueError("loss_divisor must be positive")
    compressed, clean, quality = (
        tensor.to(device, non_blocking=True) for tensor in batch
    )
    loss_kwargs = (
        {"perceptual_loss": perceptual_loss}
        if perceptual_loss is not None
        else {}
    )

    if exploration > 1:
        if hasattr(model, "sample_times"):
            r, t = model.sample_times(clean)
            time_kwargs = {"r": r, "t": t}
        else:
            time_kwargs = {"t": model.sample_timestep(clean)}
        exploration_kwargs = dict(time_kwargs)
        if perceptual_loss is not None:
            exploration_kwargs["include_perceptual"] = False
        best_loss = best_noise = None
        with torch.no_grad():
            # Explorative Modeling: https://arxiv.org/abs/2607.27372
            for _ in range(exploration):
                noise = torch.randn_like(clean)
                loss = model.compute_loss(
                    compressed,
                    clean,
                    quality,
                    noise=noise,
                    reduction="none",
                    **exploration_kwargs,
                )["loss"]
                if best_loss is None:
                    best_loss, best_noise = loss, noise
                    continue
                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                improved = improved[:, None, None, None]
                best_noise = torch.where(improved, noise, best_noise)
        loss_kwargs.update(noise=best_noise, **time_kwargs)

    losses = utils.compute_loss(
        model, compressed, clean, quality, **loss_kwargs
    )
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")

    (loss / loss_divisor).backward()
    return losses


def optimizer_step(model, optimizer):
    """Clip gradients, rejecting an update if its norm is not finite."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    updated = torch.isfinite(grad_norm).item()
    if updated:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm, updated


def resolve_training_stages(train_cfg):
    """Return validated progressive-resolution stages."""
    configured = train_cfg.get("stages")
    if not configured:
        return [
            {
                "end_epoch": train_cfg.epochs,
                "crop_size": train_cfg.get("crop_size", 128),
                "batch_size": train_cfg.batch_size,
                "gradient_accumulation_steps": train_cfg.get(
                    "gradient_accumulation_steps", 1
                ),
            }
        ]

    stages = []
    previous_end = 0
    for stage in configured:
        resolved = {
            "end_epoch": stage.end_epoch,
            "crop_size": stage.crop_size,
            "batch_size": stage.batch_size,
            "gradient_accumulation_steps": stage.get(
                "gradient_accumulation_steps", 1
            ),
        }
        if resolved["end_epoch"] <= previous_end:
            raise ValueError("train.stages end epochs must be increasing")
        for name in (
            "crop_size",
            "batch_size",
            "gradient_accumulation_steps",
        ):
            if resolved[name] < 1:
                raise ValueError(f"train.stages.{name} must be positive")
        stages.append(resolved)
        previous_end = resolved["end_epoch"]

    if previous_end != train_cfg.epochs:
        raise ValueError("the final training stage must end at train.epochs")
    return stages


def run_training(cfg: DictConfig, run: wandb.Run) -> None:
    train_cfg = cfg.train
    loader_context = mp.get_context("spawn")
    read_concurrency = train_cfg.get("read_concurrency", 2)
    if read_concurrency < 1:
        raise ValueError("train.read_concurrency must be positive")
    read_semaphore = loader_context.Semaphore(read_concurrency)

    # 1. data
    quality_range = tuple(train_cfg.get("quality_range", (10, 95)))
    training_stages = resolve_training_stages(train_cfg)

    train_ds = HRDataset(
        resolve_dataset_paths(cfg.dataset, train_cfg.datasets),
        train=True,
        crop_size=training_stages[0]["crop_size"],
        quality_range=quality_range,
        read_semaphore=read_semaphore,
    )
    val_ds = HRDataset(
        resolve_dataset_paths(
            cfg.dataset, train_cfg.get("val_datasets", cfg.eval.datasets)
        ),
        train=False,
        crop_size=train_cfg.get("val_crop_size", train_cfg.get("crop_size", 128)),
        quality_range=tuple(train_cfg.get("val_quality_range", quality_range)),
        read_semaphore=read_semaphore,
    )
    val_max_images = train_cfg.get("val_max_images")
    if val_max_images is not None:
        if val_max_images < 1:
            raise ValueError("train.val_max_images must be positive")
        val_ds = Subset(
            val_ds, range(min(val_max_images, len(val_ds)))
        )
    print(f"training images={len(train_ds)}")

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": True,
        "multiprocessing_context": loader_context,
        "worker_init_fn": configure_data_worker,
    }
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get(
            "val_batch_size", training_stages[-1]["batch_size"]
        ),
        shuffle=False,
        num_workers=2,
        **loader_kwargs,
    )

    # 2. model
    model = instantiate(cfg.model).to(device)
    perceptual_loss = None
    if getattr(model, "lpips_weight", 0):
        perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            net_type=model.lpips_backbone,
            reduction="none",
            normalize=True,
        ).to(device)

    # 3. hyperparameters
    optimizer = optim.AdamW(
        model.parameters(), train_cfg.lr, betas=(0.9, 0.95), weight_decay=0
    )
    scheduler_cfg = cfg.get("lr_scheduler", train_cfg.scheduler)
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)

    start_epoch = 1
    if train_cfg.start_checkpoint:
        start_epoch = load_checkpoint(
            train_cfg.start_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
            resume_optimizer=train_cfg.get("resume_optimizer", True),
            resume_scheduler=train_cfg.get("resume_scheduler", True),
            scheduler_config=OmegaConf.to_container(scheduler_cfg, resolve=True),
        )
    utils.compile_model(model, train_cfg.get("compile"))

    checkpoint_name = train_cfg.get("checkpoint_name", f"{cfg.model_name}.pt")
    checkpoint_path = Path(hydra.utils.get_original_cwd()) / checkpoint_name
    val_metrics = RestorationMetrics(device)

    # 4. training!
    torch.set_float32_matmul_precision("high")
    sample_every = train_cfg.get("sample_every", train_cfg.epochs)
    checkpoint_every = train_cfg.get("checkpoint_every", sample_every)
    if sample_every < 1 or checkpoint_every < 1:
        raise ValueError("sample and checkpoint intervals must be positive")
    completed_epoch = start_epoch - 1
    last_saved_epoch = None
    optimizer_steps = 0
    active_stage = None
    train_loader = None

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        stage = next(
            stage for stage in training_stages if epoch <= stage["end_epoch"]
        )
        if stage is not active_stage:
            # Recreating the loader gives persistent workers the new transform.
            if train_loader is not None:
                del train_loader
            train_ds.set_crop_size(stage["crop_size"])
            train_loader = DataLoader(
                train_ds,
                batch_size=stage["batch_size"],
                shuffle=True,
                num_workers=8,
                prefetch_factor=4,
                in_order=False,
                drop_last=True,
                **loader_kwargs,
            )
            active_stage = stage
            print(
                f"training stage through epoch {stage['end_epoch']}: "
                f"crop={stage['crop_size']}, batch={stage['batch_size']}, "
                "effective_batch_size="
                f"{stage['batch_size'] * stage['gradient_accumulation_steps']}"
            )
        accumulation_steps = stage["gradient_accumulation_steps"]

        model.train()
        train_totals = defaultdict(float)
        completed_batches = 0
        accumulated_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader):
            try:
                losses = train_step(
                    model,
                    batch,
                    device,
                    train_cfg.get("exploration", 1),
                    perceptual_loss,
                    accumulation_steps,
                )
            except FloatingPointError:
                print("Unstable step detected. Skipping batch.")
                torch.save(batch[0], "bad_batch.pt")
                # Discard the whole incomplete accumulation window rather than
                # mixing a partially scaled update into the next one.
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                continue

            completed_batches += 1
            accumulated_batches += 1
            for name, value in losses.items():
                train_totals[name] += value.detach().item()

            metrics = {
                "train_loss": losses["loss"].item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, value in losses.items():
                if name != "loss":
                    metrics[f"train_{name}"] = value.item()

            if accumulated_batches == accumulation_steps:
                grad_norm, updated = optimizer_step(model, optimizer)
                accumulated_batches = 0
                optimizer_steps += int(updated)
                metrics["grad_norm"] = grad_norm.item()
                metrics["optimizer_step"] = optimizer_steps
                if not updated:
                    print("Non-finite gradient detected. Skipping update.")
                    metrics["skipped_update"] = 1
            run.log(metrics)

        if accumulated_batches:
            # Each microbatch was divided by a full accumulation window. Undo
            # that extra scaling for the shorter final window.
            scale = accumulation_steps / accumulated_batches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            grad_norm, updated = optimizer_step(model, optimizer)
            optimizer_steps += int(updated)
            metrics = {
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "optimizer_step": optimizer_steps,
            }
            if not updated:
                print("Non-finite gradient detected. Skipping update.")
                metrics["skipped_update"] = 1
            run.log(metrics)

        if not completed_batches:
            raise RuntimeError("no finite training batches were completed")
        train_losses = {
            name: total / completed_batches
            for name, total in train_totals.items()
        }

        # validation epoch
        model.eval()
        val_metrics.reset()
        vis_batch = None
        val_seed = train_cfg.get("val_seed", 0)
        generator = torch.Generator(device=device).manual_seed(val_seed)

        for i, batch in enumerate(val_loader):
            if i == 0:
                vis_batch = batch[:2]
            compressed, clean, quality = tuple(
                tensor.to(device, non_blocking=True) for tensor in batch
            )

            with torch.no_grad():
                restored, _ = utils.predict(model, compressed, generator)
            val_metrics.update(restored, clean, quality)

        metrics = val_metrics.compute()
        train_loss = train_losses["loss"]
        extra_losses = ", ".join(
            f"train_{name}={value:.6f}"
            for name, value in train_losses.items()
            if name != "loss"
        )
        if extra_losses:
            extra_losses = ", " + extra_losses

        print(
            f"epoch {epoch}: train_loss={train_loss:.7f}, "
            f"{val_metrics.summary(metrics)}{extra_losses}"
        )

        metrics.update(epoch=epoch, train_loss_epoch=train_loss)
        if epoch % sample_every == 0 or epoch == start_epoch:
            generator = torch.Generator(device=device).manual_seed(val_seed)
            metrics["predictions"] = utils.visualize(
                model, vis_batch, device, generator
            )

        run.log(metrics)
        scheduler.step()
        completed_epoch = epoch

        if epoch % checkpoint_every == 0 or epoch == start_epoch:
            save_checkpoint(
                checkpoint_path, cfg, epoch, model, optimizer, scheduler
            )
            last_saved_epoch = epoch

    if last_saved_epoch != completed_epoch:
        save_checkpoint(
            checkpoint_path, cfg, completed_epoch, model, optimizer, scheduler
        )


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    plt.close("all")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    name = f"{cfg.model_name}-{runid.generate_fast_id()}"
    with wandb.init(
        project="jpeg-artifact-removal", config=cfg_dict, name=name
    ) as run:
        run_training(cfg, run)


if __name__ == "__main__":
    main()
