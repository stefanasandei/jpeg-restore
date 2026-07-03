import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from dataset import df2k_path

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    log.info("Info level message")
    log.debug("Debug level message")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
