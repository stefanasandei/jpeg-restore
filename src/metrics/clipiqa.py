import torch
import torch.nn.functional as F

from .base import ImageMetric


PROMPTS = (
    "Good image",
    "bad image",
    "Sharp image",
    "blurry image",
    "sharp edges",
    "blurry edges",
    "High resolution image",
    "low resolution image",
    "Noise-free image",
    "noisy image",
)


class CLIPIQA(ImageMetric):
    name = "clipiqa"
    label = "CLIPIQA"
    requires_reference = False
    higher_is_better = True

    def __init__(
        self,
        device,
        model_name="openai/clip-vit-base-patch16",
        local_files_only=True,
    ):
        super().__init__()
        try:
            from transformers import AutoProcessor, CLIPModel
        except ImportError as error:
            raise ImportError("CLIPIQA requires transformers") from error

        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.device = device

        inputs = self.processor(text=PROMPTS, return_tensors="pt", padding=True)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            anchors = self.model.get_text_features(**inputs).pooler_output
        self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))

    def forward(self, restored, reference=None):
        inputs = self.processor(
            images=[image.cpu() for image in restored],
            return_tensors="pt",
            do_rescale=False,
        )
        features = self.model.get_image_features(
            pixel_values=inputs.pixel_values.to(self.device)
        ).pooler_output
        features = F.normalize(features.float(), dim=-1)
        logits = self.model.logit_scale.exp() * features @ self.anchors.T
        prompt_pairs = logits.reshape(logits.shape[0], -1, 2)
        return prompt_pairs.softmax(dim=-1)[..., 0].mean()
