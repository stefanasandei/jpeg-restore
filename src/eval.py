import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.v2 as v2

from checkpoint import model_state
from dataset import image_paths
from metrics import MetricSuite
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
    metric_options = OmegaConf.to_container(
        cfg.eval.get("metric_options", {}), resolve=True
    )
    metrics = MetricSuite(cfg.eval.metrics, metric_options, device).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(cfg.eval.get("seed", 0))
    results = {
        quality: {name: [] for name in metrics.metrics}
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

                for name, value in metrics(restored, clean_tensor).items():
                    results[quality][name].append(value)

    table_data = {"QF": QUALITY_FACTORS}
    table_data.update({
        label: [np.mean(results[q][name]) for q in QUALITY_FACTORS]
        for name, label in metrics.labels.items()
    })
    table = pd.DataFrame(table_data)
    print(f"\n{', '.join(dataset_names)}")
    print(table)


if __name__ == "__main__":
    main()
