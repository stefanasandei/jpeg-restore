from PIL import Image

import cv2
import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import torch


def numpy_jpeg_compress(img_rgb, quality):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(encimg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def jpeg_compress(img, quality):
    return Image.fromarray(numpy_jpeg_compress(np.array(img.convert("RGB")), quality))


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
