from pathlib import Path
import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torchvision.transforms.v2 as v2
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

from models import MODELS
from utils import jpeg_compress

device = "cuda" if torch.cuda.is_available() else "cpu"
QUALITY_FACTORS = [10, 20, 30, 40]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    ds_name = cfg.eval.dataset
    ds_cfg = cfg.dataset[ds_name]

    checkpoint = cfg.eval.checkpoint
    val_dir = ds_cfg.val_dir

    if checkpoint:
        print(f"checkpoint: {checkpoint}")
        model = MODELS[cfg.model]().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
    else:
        print("checkpoint: none (baseline JPEG)")
        model = None

    images = sorted(p for p in Path(val_dir).iterdir() if p.suffix in IMAGE_EXTS)

    normalize = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    results = {qf: {"psnr": [], "ssim": []} for qf in QUALITY_FACTORS}

    for img_path in tqdm(images, desc="eval"):
        clean = Image.open(img_path)
        clean_tensor = normalize(clean).to(device)

        for qf in QUALITY_FACTORS:
            compressed = jpeg_compress(clean, qf)
            compressed_tensor = normalize(compressed).unsqueeze(0).to(device)

            if model is not None:
                with torch.no_grad():
                    pred, _ = model(compressed_tensor)
            else:
                pred = compressed_tensor

            psnr = peak_signal_noise_ratio(pred, clean_tensor.unsqueeze(0), data_range=1.0).item()
            ssim = structural_similarity_index_measure(pred, clean_tensor.unsqueeze(0), data_range=1.0).item()

            results[qf]["psnr"].append(psnr)
            results[qf]["ssim"].append(ssim)

    df = pd.DataFrame({
        "QF": QUALITY_FACTORS,
        "PSNR": [np.mean(results[qf]["psnr"]) for qf in QUALITY_FACTORS],
        "SSIM": [np.mean(results[qf]["ssim"]) for qf in QUALITY_FACTORS],
    })
    print(f"\n{ds_name}")
    print(df)


if __name__ == "__main__":
    main()
