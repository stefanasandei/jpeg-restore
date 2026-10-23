"""Performance entry point sharing the evaluation adapters and measurement code."""

import hydra
from omegaconf import DictConfig

from evaluation.runner import evaluate


@hydra.main(
    version_base=None, config_path="../config", config_name="evaluation_backbones"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
