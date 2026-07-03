import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import torch
from torch.utils.data import DataLoader

from dataset import DF2KDataset

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    df2k_cfg = cfg.dataset.df2k

    train_ds = DF2KDataset(root_dir=df2k_cfg.train_dir, train=True)
    val_ds = DF2KDataset(root_dir=df2k_cfg.val_dir, train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    run_training(cfg)

if __name__ == "__main__":
    main()
