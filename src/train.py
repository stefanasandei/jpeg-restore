from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
from wandb.sdk.lib import runid

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from checkpoint import load_checkpoint, save_checkpoint
from dataset import HRDataset, configure_data_worker
import utils
from validation_metrics import RestorationMetrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, batch, optimizer, device, exploration=1):
    if exploration < 1:
        raise ValueError("exploration must be positive")
    compressed, clean, quality = (
        tensor.to(device, non_blocking=True) for tensor in batch
    )
    loss_kwargs = {}

    if exploration > 1:
        timestep = model.sample_timestep(clean)
        best_loss = best_noise = None
        with torch.no_grad():
            # Explorative Modeling: https://arxiv.org/abs/2607.27372
            for _ in range(exploration):
                noise = torch.randn_like(clean)
                loss = model.compute_loss(
                    compressed,
                    clean,
                    quality,
                    t=timestep,
                    noise=noise,
                    reduction="none",
                )["loss"]
                if best_loss is None:
                    best_loss, best_noise = loss, noise
                    continue
                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                improved = improved[:, None, None, None]
                best_noise = torch.where(improved, noise, best_noise)
        loss_kwargs = {"t": timestep, "noise": best_noise}

    losses = utils.compute_loss(
        model, compressed, clean, quality, **loss_kwargs
    )
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return losses, grad_norm


def run_training(cfg: DictConfig, run: wandb.Run) -> None:
    train_cfg = cfg.train
    loader_context = mp.get_context("spawn")
    read_concurrency = train_cfg.get("read_concurrency", 2)
    if read_concurrency < 1:
        raise ValueError("train.read_concurrency must be positive")
    read_semaphore = loader_context.Semaphore(read_concurrency)

    # 1. data
    dataset_paths = lambda names: [
        path for name in names for path in cfg.dataset[name]
    ]
    quality_range = tuple(train_cfg.get("quality_range", (10, 95)))

    train_ds = HRDataset(
        dataset_paths(train_cfg.datasets),
        train=True,
        crop_size=train_cfg.get("crop_size", 128),
        quality_range=quality_range,
        read_semaphore=read_semaphore,
    )
    val_ds = HRDataset(
        dataset_paths(train_cfg.get("val_datasets", cfg.eval.datasets)),
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
    batches = len(train_ds) // train_cfg.batch_size
    print(f"training images={len(train_ds)}, batches={batches}")

    loader_kwargs = {
        "pin_memory": True,
        "persistent_workers": True,
        "multiprocessing_context": loader_context,
        "worker_init_fn": configure_data_worker,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        in_order=False,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        **loader_kwargs,
    )

    # 2. model
    model = instantiate(cfg.model).to(device)

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
    completed_epoch = start_epoch - 1
    last_saved_epoch = None

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        model.train()
        train_totals = defaultdict(float)
        completed_batches = 0

        for batch in tqdm(train_loader):
            try:
                losses, grad_norm = train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    train_cfg.get("exploration", 1),
                )
            except FloatingPointError:
                print("Unstable step detected. Skipping batch.")
                torch.save(batch[0], "bad_batch.pt")
                continue

            completed_batches += 1
            for name, value in losses.items():
                train_totals[name] += value.detach().item()

            metrics = {
                "train_loss": losses["loss"].item(),
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, value in losses.items():
                if name != "loss":
                    metrics[f"train_{name}"] = value.item()
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
            val_metrics.update(restored, compressed, clean, quality)

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

        if epoch % sample_every == 0 or epoch == 1:
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
