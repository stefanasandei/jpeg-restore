import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import random
from tqdm import tqdm
import wandb
from wandb.sdk.lib import runid

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import DF2KDataset
import utils

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def compile_model(model, compile_cfg):
    """Compile the network forward used by custom loss and sampling methods.

    Compiling the outer restoration wrapper would not accelerate calls to its
    ``compute_loss`` or ``sample`` methods. Replacing only the core forward also
    leaves state-dict keys unchanged, so checkpoints remain portable.
    """
    if not compile_cfg or not compile_cfg.get("enabled", False):
        return

    core = getattr(model, "model", model)
    kwargs = {
        "backend": compile_cfg.get("backend", "inductor"),
        "mode": compile_cfg.get("mode", "default"),
        "fullgraph": compile_cfg.get("fullgraph", False),
        "dynamic": compile_cfg.get("dynamic", False),
    }
    core.forward = torch.compile(core.forward, **kwargs)
    log.info("Enabled torch.compile for %s with %s", type(core).__name__, kwargs)


def save_checkpoint(path, cfg, epoch, model, optimizer, scheduler):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    resume_optimizer=True,
    resume_scheduler=True,
    scheduler_config=None,
):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model" not in checkpoint:
        incompatible = model.load_state_dict(checkpoint, strict=False)
        log.warning(
            "Loaded a weights-only checkpoint with %d missing and %d unexpected "
            "keys; optimizer and scheduler start fresh",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        return 1

    model.load_state_dict(checkpoint["model"])
    if resume_scheduler == "auto":
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_scheduler_config = checkpoint_config.get(
            "model_scheduler",
            checkpoint_config.get("train", {}).get("scheduler"),
        )
        resume_scheduler = checkpoint_scheduler_config == scheduler_config
        log.info(
            "%s scheduler from checkpoint (configuration %s)",
            "Resuming" if resume_scheduler else "Restarting",
            "matches" if resume_scheduler else "changed",
        )
    if resume_optimizer:
        # Constructing a scheduler can change the optimizer LR (for example, at
        # the first point of a warmup). Preserve that fresh LR when retaining
        # AdamW's moments but deliberately restarting the schedule.
        fresh_scheduler_lrs = scheduler.get_last_lr()
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not resume_scheduler:
            for group, lr in zip(optimizer.param_groups, fresh_scheduler_lrs):
                group["lr"] = lr
    if resume_scheduler:
        if not resume_optimizer:
            raise ValueError("resume_scheduler requires resume_optimizer")
        scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available():
        cuda_rng_state = [state.cpu() for state in checkpoint["cuda_rng_state"]]
        torch.cuda.set_rng_state_all(cuda_rng_state)

    start_epoch = checkpoint["epoch"] + 1
    log.info("Resuming training at epoch %d", start_epoch)
    return start_epoch


def run_training(cfg: DictConfig, run: wandb.Run) -> None:
    df2k_cfg = cfg.dataset.df2k
    train_cfg = cfg.train

    # 1. data
    train_ds = DF2KDataset(root_dir=df2k_cfg.train_dir, train=True)
    val_ds = DF2KDataset(root_dir=df2k_cfg.val_dir, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True
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
    epochs = train_cfg.epochs
    lr = train_cfg.lr

    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler_cfg = cfg.get("model_scheduler", train_cfg.scheduler)
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
    start_epoch = 1
    if train_cfg.start_checkpoint:
        start_epoch = load_checkpoint(
            train_cfg.start_checkpoint,
            model,
            optimizer,
            scheduler,
            resume_optimizer=train_cfg.get("resume_optimizer", True),
            resume_scheduler=train_cfg.get("resume_scheduler", True),
            scheduler_config=OmegaConf.to_container(
                scheduler_cfg, resolve=True
            ),
        )
    compile_model(model, train_cfg.get("compile"))
    checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.model_name}.pt"
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 4. training!
    torch.set_float32_matmul_precision('high')

    completed_epoch = start_epoch - 1
    last_saved_epoch = None
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_flow = 0.0
        train_dct = 0.0
        completed_batches = 0

        for compressed, clean, q_target in tqdm(train_loader):
            compressed, clean, q_target = (
                compressed.to(device),
                clean.to(device),
                q_target.to(device),
            )

            losses = model.compute_loss(compressed, clean, q_target)
            loss = losses["loss"]

            if not torch.isfinite(loss):
                print("Unstable step detected. Skipping batch.")
                torch.save(compressed, "bad_batch.pt")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            reconstruction = losses.get("reconstruction", loss).item()
            flow = losses.get("flow")
            dct = losses.get("dct")
            train_loss += reconstruction
            train_flow += losses.get("flow", loss.new_zeros(())).item()
            train_dct += losses.get("dct", loss.new_zeros(())).item()
            completed_batches += 1

            metrics = {
                "train_loss": reconstruction,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if flow is not None:
                metrics["train_flow"] = flow.item()
            if dct is not None:
                metrics["train_dct"] = dct.item()
            run.log(metrics)

        if completed_batches == 0:
            raise RuntimeError("no finite training batches were completed")

        # validation epoch
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        val_loss = 0.0
        vis_batch = None
        val_seed = train_cfg.get("val_seed", 0)
        sample_generator = torch.Generator(device=device).manual_seed(val_seed)
        for i, (compressed, clean, _) in enumerate(val_loader):
            if i == 0: vis_batch = (compressed, clean)
            compressed, clean = (
                compressed.to(device),
                clean.to(device),
            )

            with torch.no_grad():
                pred, _ = utils.predict(
                    model, compressed, sample_generator
                )
                loss = F.l1_loss(pred, clean)

            val_loss += loss.item()
            psnr_metric.update(pred, clean)
            ssim_metric.update(pred, clean)
            run.log({"val_loss": loss.item()})

        # stats
        train_loss /= completed_batches
        train_flow /= completed_batches
        train_dct /= completed_batches
        val_loss /= len(val_loader)
        val_psnr = psnr_metric.compute().item()
        val_ssim = ssim_metric.compute().item()

        flow_summary = (
            f", train_flow={train_flow:.6f}, train_dct={train_dct:.6f}"
            if getattr(model, "is_rectified_flow", False)
            else ""
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.7f}, "
            f"val_loss={val_loss:.6f}, psnr={val_psnr:.4f}, "
            f"ssim={val_ssim:.6f}"
            f"{flow_summary}"
        )

        metrics = {
            "psnr": val_psnr,
            "ssim": val_ssim,
        }
        if epoch % train_cfg.sample_every == 0 or epoch == 1:
            visualization_generator = torch.Generator(device=device).manual_seed(
                val_seed
            )
            metrics["predictions"] = utils.visualize(
                model, vis_batch, device, visualization_generator
            )

        run.log(metrics)
        scheduler.step()
        completed_epoch = epoch
        if epoch % train_cfg.sample_every == 0 or epoch == 1:
            save_checkpoint(checkpoint_path, cfg, epoch, model, optimizer, scheduler)
            last_saved_epoch = epoch

    if last_saved_epoch != completed_epoch:
        save_checkpoint(
            checkpoint_path, cfg, completed_epoch, model, optimizer, scheduler
        )


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    mp.set_start_method("fork")
    plt.close('all')

    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    with wandb.init(
        project="jpeg-artifact-removal",
        config=cfg_dict,
        name=f"{cfg.model_name}-{runid.generate_fast_id()}",
    ) as run:
        run_training(cfg, run)

if __name__ == "__main__":
    main()
