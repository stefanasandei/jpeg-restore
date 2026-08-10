import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


QUALITY_BANDS = ((5, 9), (10, 29), (30, 49))


class RestorationMetrics:

    def __init__(self, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

        self.quality_psnr = {
            band: PeakSignalNoiseRatio(data_range=1.0).to(device)
            for band in QUALITY_BANDS
        }

        self.reset()

    def reset(self):
        for metric in (
            self.psnr,
            self.ssim,
            self.lpips,
            *self.quality_psnr.values(),
        ):
            metric.reset()
        self.loss = 0.0
        self.quality_counts = {band: 0 for band in QUALITY_BANDS}
        self.batches = 0

    def update(self, restored, clean, quality):
        self.loss += F.l1_loss(restored, clean).item()
        self.batches += 1

        self.psnr.update(restored, clean)
        self.ssim.update(restored, clean)
        self.lpips.update(restored, clean)

        quality_factors = (100 * (1 - quality)).round().flatten()
        for band in QUALITY_BANDS:
            low, high = band
            selected = (quality_factors >= low) & (quality_factors <= high)
            if selected.any():
                self.quality_counts[band] += selected.sum().item()
                self.quality_psnr[band].update(restored[selected], clean[selected])

    def compute(self):
        loss = self.loss / self.batches
        psnr = self.psnr.compute().item()
        ssim = self.ssim.compute().item()

        metrics = {
            "val_loss": loss,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": self.lpips.compute().item(),
        }
        for band, metric in self.quality_psnr.items():
            if not self.quality_counts[band]:
                continue
            name = f"qf_{band[0]}_{band[1]}"
            metrics[f"psnr_{name}"] = metric.compute().item()
        return metrics

    @staticmethod
    def summary(metrics):
        return (
            f"val_loss={metrics['val_loss']:.6f}, "
            f"psnr={metrics['psnr']:.4f}, "
            f"ssim={metrics['ssim']:.6f}, "
            f"lpips={metrics['lpips']:.6f}"
        )
