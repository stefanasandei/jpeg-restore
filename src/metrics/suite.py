from torch import nn

from .base import scalar
from .clipiqa import CLIPIQA
from .dists import DISTS
from .lpips import LPIPS
from .maniqa import MANIQA
from .musiq import MUSIQ
from .psnr import PSNR
from .ssim import SSIM


METRICS = {
    metric.name: metric
    for metric in (PSNR, SSIM, LPIPS, DISTS, MUSIQ, MANIQA, CLIPIQA)
}
ALIASES = {"clipqa": "clipiqa"}


class MetricSuite(nn.Module):

    def __init__(self, names, options, device):
        super().__init__()
        names = [ALIASES.get(name.lower(), name.lower()) for name in names]
        unknown = sorted(set(names) - METRICS.keys())
        if unknown:
            available = ", ".join(METRICS)
            raise ValueError(
                f"unknown evaluation metrics: {', '.join(unknown)}; "
                f"available metrics: {available}"
            )
        if len(names) != len(set(names)):
            raise ValueError("evaluation metrics must be unique")

        self.metrics = nn.ModuleDict({
            name: METRICS[name](device=device, **options.get(name, {}))
            for name in names
        })

    @property
    def labels(self):
        return {name: metric.label for name, metric in self.metrics.items()}

    def forward(self, restored, reference):
        scores = {}
        for name, metric in self.metrics.items():
            inputs = (
                (restored, reference)
                if metric.requires_reference
                else (restored,)
            )
            scores[name] = scalar(metric(*inputs))
        return scores
