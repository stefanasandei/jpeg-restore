import io
from PIL import Image

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import torch


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    compressed = Image.open(buf)
    compressed.load()
    return compressed


def visualize(model, batch, device):
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    compressed_batch = batch[0].to(device)
    gt_batch = batch[1].to(device)

    with torch.no_grad():
        pred_batch = model(compressed_batch)[0]

    for i in range(4):
        compressed = compressed_batch[i]
        pred = pred_batch[i]
        gt = gt_batch[i]

        axes[i, 0].imshow(F.to_pil_image(compressed))
        axes[i, 0].set_title("Compressed")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(F.to_pil_image(pred))
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(F.to_pil_image(gt))
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.tight_layout()
    return fig
