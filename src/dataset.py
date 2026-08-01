import os
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torch

from utils import jpeg_compress


class RandomJPEG:
    def __init__(self, quality_range=(10, 95)):
        self.low, self.high = quality_range

    def __call__(self, img):
        q = random.randint(self.low, self.high)
        return jpeg_compress(img, q), 1.0 - q / 100.0


class DF2KDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True):
        super().__init__()

        self.root_dir = root_dir
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".png")]

        if train:
            self.preprocess = v2.Compose([
                v2.RandomCrop(128),
                v2.RandomVerticalFlip(),
                v2.RandomHorizontalFlip(),
            ])
        else:
            self.preprocess = v2.CenterCrop(128)

        self.compress = RandomJPEG((10, 95) if train else (30, 30))
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __getitem__(self, idx):
        path = f"{self.root_dir}/{self.images[idx]}"

        img = Image.open(path).convert("RGB")
        img = self.preprocess(img)
        compressed, q_target = self.compress(img)

        return (
            self.normalize(compressed),
            self.normalize(img),
            torch.tensor([q_target], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from omegaconf import OmegaConf

    torch.random.manual_seed(42)

    cfg = OmegaConf.load("config/base.yaml")
    train_ds = DF2KDataset(root_dir=cfg.dataset.df2k.train_dir)
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
