import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import unpack_model_output


def haar_encode(image):
    """Apply an orthonormal one-level 2D Haar transform."""
    top_left = image[..., 0::2, 0::2]
    top_right = image[..., 0::2, 1::2]
    bottom_left = image[..., 1::2, 0::2]
    bottom_right = image[..., 1::2, 1::2]

    low = (top_left + top_right + bottom_left + bottom_right) * 0.5
    # Match PyWavelets' (LL, LH, HL, HH) channel convention, which is also the
    # convention used by the official DiMSUM implementation.  The distinction
    # matters here because Mamba is sensitive to the order of the subbands.
    vertical = (top_left + top_right - bottom_left - bottom_right) * 0.5
    horizontal = (top_left - top_right + bottom_left - bottom_right) * 0.5
    diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return torch.cat((low, vertical, horizontal, diagonal), dim=1)


def haar_decode(coefficients):
    """Invert coefficients produced by :func:`haar_encode`."""
    low, vertical, horizontal, diagonal = coefficients.chunk(4, dim=1)
    top_left = (low + horizontal + vertical + diagonal) * 0.5
    top_right = (low - horizontal + vertical - diagonal) * 0.5
    bottom_left = (low + horizontal - vertical - diagonal) * 0.5
    bottom_right = (low - horizontal - vertical + diagonal) * 0.5

    batch, channels, height, width = low.shape
    image = coefficients.new_empty(batch, channels, height * 2, width * 2)
    image[..., 0::2, 0::2] = top_left
    image[..., 0::2, 1::2] = top_right
    image[..., 1::2, 0::2] = bottom_left
    image[..., 1::2, 1::2] = bottom_right
    return image


class HaarWaveletRestoration(nn.Module):
    """Run a restoration model in a one-level Haar wavelet domain."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        height, width = image.shape[-2:]
        pad_height = height % 2
        pad_width = width % 2
        padded = image
        if pad_height or pad_width:
            padded = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
        restored, auxiliary = unpack_model_output(self.model(haar_encode(padded)))
        return haar_decode(restored)[..., :height, :width], auxiliary
