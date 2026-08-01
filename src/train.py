import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import logging
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


def run_training(cfg: DictConfig, run: wandb.Run) -> None:
    df2k_cfg = cfg.dataset.df2k
    train_cfg = cfg.train

    # 1. data
    train_ds = DF2KDataset(root_dir=df2k_cfg.train_dir, train=True)
    val_ds = DF2KDataset(root_dir=df2k_cfg.val_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. model
    model = instantiate(cfg.model).to(device)
    if cfg.train.start_checkpoint:
        model.load_state_dict(torch.load(cfg.train.start_checkpoint, weights_only=True))

    # 3. hyperparameters
    epochs = train_cfg.epochs
    lr = train_cfg.lr

    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler_cfg = cfg.get("model_scheduler", train_cfg.scheduler)
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    input_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    input_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 4. training!
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        for compressed, clean, q_target in tqdm(train_loader):
            compressed, clean, q_target = compressed.to(device), clean.to(device), q_target.to(device)

            losses = model.compute_loss(compressed, clean, q_target)
            loss = losses["loss"]

            if not torch.isfinite(loss):
                print("Unstable step detected. Skipping batch.")
                torch.save(compressed, "bad_batch.pt")
                continue

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += losses.get("reconstruction", loss).item()

            run.log({
                **{f"train_{name}": value.item() for name, value in losses.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": total_norm.item(),
            })

        # validation epoch
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        input_psnr_metric.reset()
        input_ssim_metric.reset()

        val_loss = 0.0
        val_flow = 0.0
        vis_batch = None
        val_seed = train_cfg.get("val_seed", 0)
        sample_generator = torch.Generator(device=device).manual_seed(val_seed)
        flow_generator = torch.Generator(device=device).manual_seed(val_seed + 1)
        for i, (compressed, clean, q_target) in enumerate(val_loader):
            if i == 0: vis_batch = (compressed, clean)
            compressed, clean, q_target = (
                compressed.to(device),
                clean.to(device),
                q_target.to(device),
            )

            with torch.no_grad():
                pred, _ = utils.predict(model, compressed, sample_generator)
                loss = F.l1_loss(pred, clean)
                if getattr(model, "is_rectified_flow", False):
                    flow = model.compute_loss(
                        compressed,
                        clean,
                        q_target,
                        generator=flow_generator,
                    )["flow"]
                    val_flow += flow.item()

            val_loss += loss.item()
            psnr_metric.update(pred, clean)
            ssim_metric.update(pred, clean)
            input_psnr_metric.update(compressed, clean)
            input_ssim_metric.update(compressed, clean)

        # stats
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        if getattr(model, "is_rectified_flow", False):
            val_flow /= len(val_loader)
        val_psnr = psnr_metric.compute().item()
        val_ssim = ssim_metric.compute().item()
        input_psnr = input_psnr_metric.compute().item()
        input_ssim = input_ssim_metric.compute().item()

        print(
            f"epoch {epoch}: train_loss={train_loss:.7f}, val_loss={val_loss:.6f}, "
            f"val_psnr={val_psnr:.4f} ({val_psnr - input_psnr:+.4f}), "
            f"val_ssim={val_ssim:.6f} ({val_ssim - input_ssim:+.6f})"
        )

        metrics = {
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "val_loss": val_loss,
            "psnr": val_psnr,
            "ssim": val_ssim,
            "input_psnr": input_psnr,
            "input_ssim": input_ssim,
            "psnr_gain": val_psnr - input_psnr,
            "ssim_gain": val_ssim - input_ssim,
        }
        if getattr(model, "is_rectified_flow", False):
            metrics["val_flow"] = val_flow
        if epoch % train_cfg.sample_every == 0 or epoch == 1:
            visualization_generator = torch.Generator(device=device).manual_seed(val_seed)
            metrics["predictions"] = utils.visualize(model, vis_batch, device, visualization_generator)

            torch.save(model.state_dict(), f"{hydra.utils.get_original_cwd()}/{cfg.model_name}.pt")

        run.log(metrics)
        scheduler.step()

    torch.save(model.state_dict(), f"{hydra.utils.get_original_cwd()}/{cfg.model_name}.pt")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    mp.set_start_method("fork")
    plt.close('all')

    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    with wandb.init(project="jpeg-artifact-removal", config=cfg_dict, name=f"{cfg.model_name}-{runid.generate_fast_id()}") as run:
        run_training(cfg, run)

if __name__ == "__main__":
    main()
