import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


QUALITY_BANDS = ((5, 9), (10, 29), (30, 49))


class RestorationMetrics:

    def __init__(self, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.baseline_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.baseline_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.quality_psnr = {
            band: PeakSignalNoiseRatio(data_range=1.0).to(device)
            for band in QUALITY_BANDS
        }
        self.baseline_quality_psnr = {
            band: PeakSignalNoiseRatio(data_range=1.0).to(device)
            for band in QUALITY_BANDS
        }

        self.reset()

    def reset(self):
        for metric in (
            self.psnr,
            self.ssim,
            self.baseline_psnr,
            self.baseline_ssim,
            *self.quality_psnr.values(),
            *self.baseline_quality_psnr.values(),
        ):
            metric.reset()
        self.loss = 0.0
        self.baseline_loss = 0.0
        self.quality_counts = {band: 0 for band in QUALITY_BANDS}
        self.batches = 0

    def update(self, restored, compressed, clean, quality):
        self.loss += F.l1_loss(restored, clean).item()
        self.baseline_loss += F.l1_loss(compressed, clean).item()
        self.batches += 1

        self.psnr.update(restored, clean)
        self.ssim.update(restored, clean)
        self.baseline_psnr.update(compressed, clean)
        self.baseline_ssim.update(compressed, clean)

        quality_factors = (100 * (1 - quality)).round().flatten()
        for band in QUALITY_BANDS:
            low, high = band
            selected = (quality_factors >= low) & (quality_factors <= high)
            if selected.any():
                self.quality_counts[band] += selected.sum().item()
                self.quality_psnr[band].update(restored[selected], clean[selected])
                self.baseline_quality_psnr[band].update(
                    compressed[selected], clean[selected]
                )

    def compute(self):
        loss = self.loss / self.batches
        baseline_loss = self.baseline_loss / self.batches
        psnr = self.psnr.compute().item()
        ssim = self.ssim.compute().item()

        metrics = {
            "val_loss": loss,
            "val_loss_gain": baseline_loss - loss,
            "psnr": psnr,
            "psnr_gain": psnr - self.baseline_psnr.compute().item(),
            "ssim": ssim,
            "ssim_gain": ssim - self.baseline_ssim.compute().item(),
        }
        for band, metric in self.quality_psnr.items():
            if not self.quality_counts[band]:
                continue
            name = f"qf_{band[0]}_{band[1]}"
            quality_psnr = metric.compute().item()
            baseline_psnr = self.baseline_quality_psnr[band].compute().item()
            metrics[f"psnr_{name}"] = quality_psnr
            metrics[f"psnr_gain_{name}"] = quality_psnr - baseline_psnr
        return metrics

    @staticmethod
    def summary(metrics):
        return (
            f"val_loss={metrics['val_loss']:.6f} "
            f"({metrics['val_loss_gain']:+.6f}), "
            f"psnr={metrics['psnr']:.4f} ({metrics['psnr_gain']:+.4f}), "
            f"ssim={metrics['ssim']:.6f} ({metrics['ssim_gain']:+.6f})"
        )
