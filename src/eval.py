import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.v2 as v2

from checkpoint import model_state
from dataset import image_paths
from utils import jpeg_compress, predict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUALITY_FACTORS = [5, 10, 20]


def load_model(cfg, checkpoint):
    model = instantiate(cfg.model).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(model_state(state))
    return model.eval()


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    dataset_names = cfg.eval.datasets
    checkpoint = cfg.eval.checkpoint

    if checkpoint:
        print(f"checkpoint: {checkpoint}")
        model = load_model(cfg, checkpoint)
    else:
        print("checkpoint: none (baseline JPEG)")
        model = None

    paths = [path for name in dataset_names for path in cfg.dataset[name]]
    images = image_paths(paths)
    if not images:
        raise ValueError("no evaluation images found")

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(cfg.eval.get("seed", 0))
    results = {
        quality: {"psnr": [], "ssim": [], "lpips": []}
        for quality in QUALITY_FACTORS
    }

    with torch.inference_mode():
        for image_path in tqdm(images, desc="eval"):
            clean = Image.open(image_path).convert("RGB")
            clean_tensor = to_tensor(clean).unsqueeze(0).to(device)

            for quality in QUALITY_FACTORS:
                compressed = jpeg_compress(clean, quality)
                compressed = to_tensor(compressed).unsqueeze(0).to(device)
                restored = compressed
                if model is not None:
                    restored, _ = predict(model, compressed, generator)

                results[quality]["psnr"].append(
                    peak_signal_noise_ratio(
                        restored, clean_tensor, data_range=1.0
                    ).item()
                )
                results[quality]["ssim"].append(
                    structural_similarity_index_measure(
                        restored, clean_tensor, data_range=1.0
                    ).item()
                )
                results[quality]["lpips"].append(
                    lpips(restored, clean_tensor).item()
                )

    table = pd.DataFrame({
        "QF": QUALITY_FACTORS,
        "PSNR": [np.mean(results[q]["psnr"]) for q in QUALITY_FACTORS],
        "SSIM": [np.mean(results[q]["ssim"]) for q in QUALITY_FACTORS],
        "LPIPS": [np.mean(results[q]["lpips"]) for q in QUALITY_FACTORS],
    })
    print(f"\n{', '.join(dataset_names)}")
    print(table)


if __name__ == "__main__":
    main()
