from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class EvaluationInput:
    rgb: torch.Tensor
    jpeg: bytes | None = None


def tensor_from_image(image):
    array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).unsqueeze(0)


def compress(image, quality):
    """Encode once; RGB and coefficient-based models share the same bitstream."""
    rgb = np.asarray(image.convert("RGB"))
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    payload = encoded.tobytes()
    # Match the training codec, including its decoder and chroma subsampling.
    decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return EvaluationInput(tensor_from_image(Image.fromarray(decoded)), payload)


def case_seed(seed, dataset, image_name, quality):
    """Stable across model selection, dataset ordering, and process restarts."""
    key = f"{seed}/{dataset}/{image_name}/{quality}".encode()
    return int.from_bytes(sha256(key).digest()[:4], "little")


def discover_datasets(dataset_cfg, names, limit=None):
    from dataset import image_paths

    datasets = {}
    for name in names:
        if name not in dataset_cfg:
            raise KeyError(f"unknown dataset {name!r}; configure dataset.{name}")
        paths = sorted(set(image_paths(dataset_cfg[name])))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"no evaluation images found for {name!r}")
        datasets[name] = paths
    return datasets


def image_key(path):
    path = Path(path)
    return f"{path.stem}-{sha256(str(path).encode()).hexdigest()[:10]}"


def synthetic_input(height, width, batch_size, quality, seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    item = compress(image, quality)
    item.rgb = item.rgb.repeat(batch_size, 1, 1, 1)
    return item


def jpeg_size(payload):
    with Image.open(BytesIO(payload)) as image:
        return image.height, image.width
