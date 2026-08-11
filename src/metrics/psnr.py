from torchmetrics.functional.image import peak_signal_noise_ratio

from .base import ImageMetric


class PSNR(ImageMetric):
    name = "psnr"
    label = "PSNR"
    requires_reference = True
    higher_is_better = True

    def forward(self, restored, reference=None):
        return peak_signal_noise_ratio(restored, reference, data_range=1.0)
