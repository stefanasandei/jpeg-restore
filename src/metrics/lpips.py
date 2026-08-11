from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .base import ImageMetric


class LPIPS(ImageMetric):
    name = "lpips"
    label = "LPIPS"
    requires_reference = True
    higher_is_better = False

    def __init__(self, device, net_type="alex"):
        super().__init__()
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, normalize=True
        ).to(device).eval()

    def forward(self, restored, reference=None):
        return self.metric(restored, reference)
