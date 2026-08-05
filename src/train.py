from collections import defaultdict
from pathlib import Path

import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
from wandb.sdk.lib import runid

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from checkpoint import load_checkpoint, save_checkpoint
from dataset import DF2KDataset
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(cfg: DictConfig, run: wandb.Run) -> None:
    df2k_cfg = cfg.dataset.df2k
    train_cfg = cfg.train

    # 1. data
    train_ds = DF2KDataset(df2k_cfg.train_dir, train=True)
    val_ds = DF2KDataset(df2k_cfg.val_dir, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 2. model
    model = instantiate(cfg.model).to(device)

    # 3. hyperparameters
    optimizer = optim.AdamW(model.parameters(), train_cfg.lr)
    scheduler_cfg = cfg.get("model_scheduler", train_cfg.scheduler)
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

    checkpoint_path = Path(hydra.utils.get_original_cwd()) / f"{cfg.model_name}.pt"
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 4. training!
    torch.set_float32_matmul_precision("high")
    sample_every = train_cfg.get("sample_every", train_cfg.epochs)
    completed_epoch = start_epoch - 1
    last_saved_epoch = None

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        model.train()
        train_totals = defaultdict(float)
        completed_batches = 0

        for compressed, clean, quality in tqdm(train_loader):
            compressed = compressed.to(device)
            clean = clean.to(device)
            quality = quality.to(device)

            losses = utils.compute_loss(model, compressed, clean, quality)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                print("Unstable step detected. Skipping batch.")
                torch.save(compressed, "bad_batch.pt")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            completed_batches += 1
            for name, value in losses.items():
                train_totals[name] += value.detach().item()
            reconstruction = losses.get("reconstruction", loss)
            run.log(
                {
                    "train_loss": reconstruction.detach().item(),
                    "train_objective": loss.detach().item(),
                    **{
                        f"train_{name}": value.detach().item()
                        for name, value in losses.items()
                        if name not in {"loss", "reconstruction"}
                    },
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm.item(),
                }
            )

        if not completed_batches:
            raise RuntimeError("no finite training batches were completed")
        train_losses = {
            name: total / completed_batches
            for name, total in train_totals.items()
        }

        # validation epoch
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        val_loss = 0.0
        vis_batch = None
        val_seed = train_cfg.get("val_seed", 0)
        generator = torch.Generator(device=device).manual_seed(val_seed)

        for i, (compressed, clean, _) in enumerate(val_loader):
            if i == 0:
                vis_batch = (compressed, clean)
            compressed = compressed.to(device)
            clean = clean.to(device)

            with torch.no_grad():
                restored, _ = utils.predict(model, compressed, generator)
                loss = F.l1_loss(restored, clean)

            val_loss += loss.item()
            psnr_metric.update(restored, clean)
            ssim_metric.update(restored, clean)

        val_loss /= len(val_loader)
        val_psnr = psnr_metric.compute().item()
        val_ssim = ssim_metric.compute().item()
        train_loss = train_losses.get("reconstruction", train_losses["loss"])
        extra_losses = ", ".join(
            f"train_{name}={value:.6f}"
            for name, value in train_losses.items()
            if name not in {"loss", "reconstruction", "quality"}
        )
        if extra_losses:
            extra_losses = ", " + extra_losses

        print(
            f"epoch {epoch}: train_loss={train_loss:.7f}, "
            f"val_loss={val_loss:.6f}, psnr={val_psnr:.4f}, "
            f"ssim={val_ssim:.6f}{extra_losses}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "val_loss": val_loss,
            "psnr": val_psnr,
            "ssim": val_ssim,
        }
        if epoch % sample_every == 0 or epoch == 1:
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
    mp.set_start_method("fork")
    plt.close("all")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    name = f"{cfg.model_name}-{runid.generate_fast_id()}"
    with wandb.init(
        project="jpeg-artifact-removal", config=cfg_dict, name=name
    ) as run:
        run_training(cfg, run)


if __name__ == "__main__":
    main()
