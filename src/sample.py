import argparse
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as v2

from checkpoint import model_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "/mnt/projects/models/best-mf/rela_dit_meanflow_qf5_20.pt"


def load_model(checkpoint, device):
    cfg = OmegaConf.load(
        PROJECT_ROOT / "config" / "model" / "rela_dit_meanflow.yaml"
    )
    model = instantiate(cfg.model).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(model_state(state))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description="Restore a JPEG image with the MeanFlow checkpoint")
    parser.add_argument("input", type=Path, help="path to the input JPEG image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output image path (default: input_stem_restored.png)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        choices=(1, 2),
        help="number of MeanFlow sampling steps (default: 2)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    degraded = to_tensor(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        restored, _ = model.sample(degraded, sample_steps=args.steps)

    output = args.output or args.input.with_name(
        f"{args.input.stem}_restored.png"
    )
    F.to_pil_image(restored[0].cpu()).save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
