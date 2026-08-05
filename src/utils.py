import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch
import torch.nn.functional as torch_F


def unpack_model_output(output):
    if isinstance(output, torch.Tensor):
        return output, None
    return output


def predict(model, image, generator=None):
    """Use deterministic noise when a model exposes a stochastic sampler."""
    if generator is not None and hasattr(model, "sample"):
        return unpack_model_output(model.sample(image, generator=generator))
    return unpack_model_output(model(image))


def compute_loss(model, degraded, clean, quality):
    """Adapt regression and objective-specific models to one training API."""
    if hasattr(model, "compute_loss"):
        return model.compute_loss(degraded, clean, quality)

    restored, _ = unpack_model_output(model(degraded))
    reconstruction = torch_F.l1_loss(restored, clean)
    return {"loss": reconstruction, "reconstruction": reconstruction}


def compile_model(model, cfg):
    if not cfg or not cfg.get("enabled", False):
        return

    core = getattr(model, "model", model)
    core.forward = torch.compile(
        core.forward,
        backend=cfg.get("backend", "inductor"),
        mode=cfg.get("mode", "default"),
        fullgraph=cfg.get("fullgraph", False),
        dynamic=cfg.get("dynamic", False),
    )


def numpy_jpeg_compress(img_rgb, quality):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(
        ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def jpeg_compress(img, quality):
    return Image.fromarray(numpy_jpeg_compress(np.array(img.convert("RGB")), quality))


def visualize(model, batch, device, generator=None):
    compressed_batch = batch[0].to(device)
    gt_batch = batch[1].to(device)

    with torch.no_grad():
        pred_batch, _ = predict(model, compressed_batch, generator)

    rows = min(4, len(compressed_batch))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
    columns = ("Compressed", "Prediction", "Ground Truth")

    for row, images in enumerate(zip(compressed_batch, pred_batch, gt_batch)):
        if row == rows:
            break
        for axis, title, image in zip(axes[row], columns, images):
            axis.imshow(F.to_pil_image(image.clamp(0, 1)))
            axis.set_title(title)
            axis.axis("off")

    plt.tight_layout()
    return fig
