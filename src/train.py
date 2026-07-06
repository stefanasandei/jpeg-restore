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
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import DF2KDataset
from models import MODELS
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
    model = MODELS[cfg.model]().to(device)
    if cfg.train.start_checkpoint:
        model.load_state_dict(torch.load(cfg.train.start_checkpoint, weights_only=True))

    # 3. hyperparameters
    epochs = train_cfg.epochs
    lr = train_cfg.lr

    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler = instantiate(train_cfg.scheduler, optimizer=optimizer)
    criterion = nn.L1Loss()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 4. training!
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        for compressed, clean, q_target in tqdm(train_loader):
            compressed, clean, q_target = compressed.to(device), clean.to(device), q_target.to(device)

            pred, q_pred = model(compressed)
            loss_rec = criterion(pred, clean)
            loss_qf = criterion(q_pred, q_target)
            loss = loss_rec + 0.1 * loss_qf

            if torch.isnan(loss) or loss.item() > 1.0:
                print("Unstable step detected. Skipping batch.")
                torch.save(compressed, "bad_batch.pt")
                continue

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss_rec.item()

            run.log({"train_loss": loss_rec.item(), "lr": optimizer.param_groups[0]["lr"], "grad_norm": total_norm.item()})

        scheduler.step()

        # validation epoch
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        val_loss = 0.0
        vis_batch = None
        for i, (compressed, clean, _) in enumerate(val_loader):
            if i == 0: vis_batch = (compressed, clean)
            compressed, clean = compressed.to(device), clean.to(device)

            with torch.no_grad():
                pred, _ = model(compressed)
                loss = criterion(pred, clean)

            val_loss += loss.item()
            psnr_metric.update(pred, clean)
            ssim_metric.update(pred, clean)

            run.log({"val_loss": loss.item()})

        # stats
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_psnr = psnr_metric.compute().item()
        val_ssim = ssim_metric.compute().item()

        print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_psnr={val_psnr:.2f}, val_ssim={val_ssim:.4f}")

        img = utils.visualize(model, vis_batch, device)
        run.log({"psnr": val_psnr, "ssim": val_ssim, "predictions": img})

    torch.save(model.state_dict(), f"{hydra.utils.get_original_cwd()}/{cfg.model}.pt")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    mp.set_start_method("fork")
    plt.close('all')

    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    with wandb.init(project="jpeg-artifact-removal", config=cfg_dict, name=f"{cfg.model}-{runid.generate_fast_id()}") as run:
        run_training(cfg, run)

if __name__ == "__main__":
    main()
