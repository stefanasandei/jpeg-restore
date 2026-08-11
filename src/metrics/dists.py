from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity

from .base import ImageMetric


class DISTS(ImageMetric):
    name = "dists"
    label = "DISTS"
    requires_reference = True
    higher_is_better = False

    def __init__(self, device):
        super().__init__()
        self.metric = DeepImageStructureAndTextureSimilarity().to(device).eval()

    def forward(self, restored, reference=None):
        return self.metric(restored, reference)
