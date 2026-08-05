import math

import torch
from torch import nn


class JPEGDCTLoss(nn.Module):
    """L1 distance between AC coefficients of 8x8 YCbCr blocks."""

    def __init__(self):
        super().__init__()
        index = torch.arange(8, dtype=torch.float32)
        basis = torch.cos(math.pi / 8 * index[:, None] * (index[None] + 0.5))
        basis[0] /= math.sqrt(2)

        self.register_buffer("basis", basis * 0.5, persistent=False)
        self.register_buffer(
            "rgb_to_ycbcr",
            torch.tensor([
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ]),
            persistent=False,
        )

    def transform(self, image):
        matrix = self.rgb_to_ycbcr.to(image.dtype)
        image = torch.einsum("bchw,dc->bdhw", image, matrix)
        blocks = image.unfold(2, 8, 8).unfold(3, 8, 8)
        basis = self.basis.to(image.dtype)
        return basis @ blocks @ basis.t()

    def forward(self, prediction, target):
        if prediction.shape[-2] % 8 or prediction.shape[-1] % 8:
            raise ValueError("DCT loss requires dimensions divisible by 8")
        error = (self.transform(prediction) - self.transform(target)).abs()
        return error.flatten(-2)[..., 1:].mean()
