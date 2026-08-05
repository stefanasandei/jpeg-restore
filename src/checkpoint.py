import logging
from pathlib import Path
import random

import torch
from omegaconf import OmegaConf


log = logging.getLogger(__name__)


def model_state(checkpoint):
    """Read model weights from either a training or weights-only checkpoint."""
    return checkpoint.get("model", checkpoint)


def save_checkpoint(path, cfg, epoch, model, optimizer, scheduler):
    state = {
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
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
    resume_optimizer=True,
    resume_scheduler=True,
    scheduler_config=None,
):
    state = torch.load(path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(model_state(state), strict=False)

    if "model" not in state:
        log.warning(
            "Loaded weights only (%d missing, %d unexpected keys)",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        return 1

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "training checkpoint does not match the model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    if resume_scheduler == "auto":
        old_cfg = state.get("config", {})
        old_scheduler = old_cfg.get(
            "lr_scheduler", old_cfg.get("train", {}).get("scheduler")
        )
        resume_scheduler = old_scheduler == scheduler_config
        log.info(
            "%s scheduler because its configuration %s",
            "Resuming" if resume_scheduler else "Restarting",
            "matches" if resume_scheduler else "changed",
        )

    if resume_optimizer:
        fresh_lrs = scheduler.get_last_lr()
        optimizer.load_state_dict(state["optimizer"])
        if not resume_scheduler:
            for group, lr in zip(optimizer.param_groups, fresh_lrs):
                group["lr"] = lr

    if resume_scheduler:
        if not resume_optimizer:
            raise ValueError("resume_scheduler requires resume_optimizer")
        scheduler.load_state_dict(state["scheduler"])

    if "random_state" in state:
        random.setstate(state["random_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state"]])

    start_epoch = state["epoch"] + 1
    log.info("Resuming training at epoch %d", start_epoch)
    return start_epoch
