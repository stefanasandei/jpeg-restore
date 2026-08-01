import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Linear warmup followed by cosine decay, stepped once per epoch."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs)")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def lr_multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        decay_epochs = total_epochs - warmup_epochs
        progress = min(max(epoch - warmup_epochs, 0), decay_epochs) / decay_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)
