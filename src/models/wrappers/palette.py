import torch
from torch import nn

from models.linear_dit import SanaLinearDiT
from models.utils import pad_to_multiple
from models.wavelet import haar_decode, haar_encode


class Palette(nn.Module):
    """Condition a backbone on a degraded image in the same representation."""

    backbone = None

    def __init__(self, wavelet_levels=0, **kwargs):
        super().__init__()
        if wavelet_levels < 0:
            raise ValueError("wavelet_levels must be non-negative")
        self.wavelet_levels = wavelet_levels
        image_channels = kwargs.pop("out_channels", 3)
        latent_channels = image_channels * 4**wavelet_levels
        self.network = self.backbone(
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            **kwargs,
        )

    @property
    def pixel_multiple(self):
        return 2**self.wavelet_levels * self.network.patch_size

    def pad(self, image):
        return pad_to_multiple(image, self.pixel_multiple)

    def encode(self, image):
        for _ in range(self.wavelet_levels):
            image = haar_encode(image)
        return image

    def decode(self, state):
        for _ in range(self.wavelet_levels):
            state = haar_decode(state)
        return state

    def forward(self, state, timestep, condition):
        return self.network(torch.cat((state, condition), dim=1), timestep)


class LinearDiTPalette(Palette):
    backbone = SanaLinearDiT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nn.init.normal_(self.network.final_layer.linear.weight, std=1e-3)
        nn.init.zeros_(self.network.final_layer.linear.bias)
