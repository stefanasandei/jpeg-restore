"""Configurable quality and performance evaluation entry point."""

import hydra
from omegaconf import DictConfig

from evaluation.runner import evaluate


@hydra.main(version_base=None, config_path="../config", config_name="evaluation")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
