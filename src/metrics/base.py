from abc import ABC, abstractmethod

import torch
from torch import nn


class ImageMetric(nn.Module, ABC):
    name: str
    label: str
    requires_reference: bool
    higher_is_better: bool

    def __init__(self, device=None):
        super().__init__()

    @abstractmethod
    def forward(self, restored, reference=None):
        pass


class PyiqaMetric(ImageMetric):
    requires_reference = False
    higher_is_better = True

    def __init__(self, model_name, device, **kwargs):
        super().__init__()
        try:
            import pyiqa
        except ImportError as error:
            raise ImportError(
                f"{self.label} requires pyiqa; install requirements-eval.txt"
            ) from error

        self.metric = pyiqa.create_metric(
            model_name, device=device, **kwargs
        ).eval()

    def forward(self, restored, reference=None):
        return self.metric(restored).mean()


def scalar(value):
    if not isinstance(value, torch.Tensor):
        return float(value)
    return value.detach().mean().item()
