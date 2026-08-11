import glob
import os
import random
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

from utils import jpeg_compress


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def configure_data_worker(_worker_id):
    # each loader process single-threaded inside OpenCV.
    cv2.setNumThreads(1)


def image_paths(paths):
    def discover(path):
        if any(ch in path for ch in "*?["):
            yield from glob.glob(path, recursive=True)
        elif os.path.isdir(path):
            yield from (
                os.path.join(path, filename)
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
            )

    return sorted(
        path
        for base in paths
        for path in discover(base)
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    )


class JPEGCompression:
    def __init__(self, quality_range=(10, 95), random_quality=True):
        self.low, self.high = quality_range
        self.random_quality = random_quality

    def __call__(self, image, index=None):
        if self.random_quality:
            quality = random.randint(self.low, self.high)
        else:
            if index is None:
                raise ValueError("index is required for deterministic JPEG quality")
            count = self.high - self.low + 1
            # 37 is coprime with the 86 qualities in [10, 95].
            quality = self.low + (index * 37) % count
        return jpeg_compress(image, quality), 1.0 - quality / 100.0


class HRDataset(Dataset):
    def __init__(
        self,
        paths,
        train: bool = True,
        *,
        crop_size: int = 128,
        quality_range=(10, 95),
        read_semaphore=None,
    ):
        super().__init__()

        self.images = image_paths(paths)
        if not self.images:
            raise ValueError("no dataset images found")
        self.read_semaphore = read_semaphore

        if train:
            self.preprocess = v2.Compose([
                v2.RandomCrop(crop_size),
                v2.RandomVerticalFlip(),
                v2.RandomHorizontalFlip(),
            ])
        else:
            self.preprocess = v2.CenterCrop(crop_size)

        # Validation covers the same range with reproducible quality factors.
        self.compress = JPEGCompression(quality_range, random_quality=train)
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __getitem__(self, idx):
        if self.read_semaphore is None:
            source_context = Image.open(self.images[idx])
        else:
            # Keep SATA queue depth low, then release the disk slot before the
            # comparatively CPU-heavy PNG decode and JPEG augmentation.
            with self.read_semaphore:
                encoded = Path(self.images[idx]).read_bytes()
            source_context = Image.open(BytesIO(encoded))
        with source_context as source:
            image = source.convert("RGB")
        image = self.preprocess(image)
        compressed, quality = self.compress(image, idx)

        return (
            self.normalize(compressed),
            self.normalize(image),
            torch.tensor([quality], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    torch.random.manual_seed(42)

    cfg = OmegaConf.load("config/base.yaml")
    train_ds = HRDataset(cfg.dataset.df2k_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

    # 1. smoke test
    batch = next(iter(train_loader))
    print([b.shape for b in batch])

    # 2. visualization
    sample_idx = 3
    compressed, gt = batch[0][sample_idx], batch[1][sample_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(F.to_pil_image(compressed))
    ax1.set_title("Compressed")
    ax1.axis("off")
    ax2.imshow(F.to_pil_image(gt))
    ax2.set_title("Ground Truth")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("assets/sample_comparison.png", dpi=150)
