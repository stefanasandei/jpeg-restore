from torchmetrics.functional.image import structural_similarity_index_measure

from .base import ImageMetric


class SSIM(ImageMetric):
    name = "ssim"
    label = "SSIM"
    requires_reference = True
    higher_is_better = True

    def forward(self, restored, reference=None):
        return structural_similarity_index_measure(
            restored, reference, data_range=1.0
        )
