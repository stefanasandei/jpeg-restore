import torch
from torch import nn
import torch.nn.functional as F

from models.dimsum import DiMSUM, pad_to_multiple
from models.losses import JPEGDCTLoss


class DiMSUMPalette(nn.Module):
    """Predict velocity from the current state and compressed-image condition."""

    def __init__(self, **kwargs):
        super().__init__()
        self.network = DiMSUM(in_channels=6, **kwargs)

    def forward(self, x, t, y):
        height, width = x.shape[-2:]
        multiple = self.network.patch_size * 4
        x = pad_to_multiple(x, multiple)
        y = pad_to_multiple(y, multiple)
        return self.network(torch.cat((x, y), dim=1), t)[..., :height, :width]


class DiMSUMRectifiedFlow(nn.Module):
    """Conditional rectified flow from noise x_0 to clean image x_1"""

    def __init__(
        self,
        sample_steps=20,
        timestep_scale=1000.0,
        endpoint_loss_weight=1.0,
        dct_loss_weight=0.1,
        **kwargs,
    ):
        super().__init__()
        if sample_steps < 1 or timestep_scale <= 0:
            raise ValueError("sample_steps and timestep_scale must be positive")
        if endpoint_loss_weight < 0 or dct_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")

        self.sample_steps = sample_steps
        self.timestep_scale = timestep_scale
        self.endpoint_loss_weight = endpoint_loss_weight
        self.dct_loss_weight = dct_loss_weight
        self.model = DiMSUMPalette(**kwargs)
        self.dct_loss = JPEGDCTLoss()

    def velocity(self, x, t, y):
        return self.model(x, t * self.timestep_scale, y)

    def compute_loss(self, y, x_1, quality=None, *, generator=None, t=None, noise=None):
        del quality
        if t is None:
            t = torch.rand(y.shape[0], device=y.device, dtype=y.dtype, generator=generator)
        else:
            t = t.to(y)
        if t.shape != (y.shape[0],):
            raise ValueError(f"expected t shape {(y.shape[0],)}, got {t.shape}")

        if noise is None:
            noise = torch.randn_like(y, generator=generator)
        else:
            noise = noise.to(y)
        if noise.shape != y.shape:
            raise ValueError("noise must have the same shape as y")

        x_0 = noise
        t_image = t[:, None, None, None]
        x_t = (1 - t_image) * x_0 + t_image * x_1
        dx = x_1 - x_0

        v = self.velocity(x_t, t, y)
        flow = F.mse_loss(v, dx)

        x_1_hat = x_t + (1 - t_image) * v
        endpoint = F.l1_loss(x_1_hat, x_1)
        dct = self.dct_loss(x_1_hat, x_1)
        loss = flow + self.endpoint_loss_weight * endpoint + self.dct_loss_weight * dct
        
        return {
            "loss": loss,
            "flow": flow,
            "reconstruction": endpoint.detach(),
            "dct": dct.detach(),
        }

    def sample(self, y, *, generator=None, noise=None, sample_steps=None):
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        if noise is None:
            noise = torch.randn_like(y, generator=generator)
        else:
            noise = noise.to(y)
        if noise.shape != y.shape:
            raise ValueError("noise must have the same shape as y")

        x = noise
        dt = 1 / steps
        for step in range(steps):
            t = y.new_full((y.shape[0],), step * dt)
            v = self.velocity(x, t, y)
            dx = v * dt
            x = x + dx

        return x.clamp(0, 1), None

    def forward(self, y):
        return self.sample(y)
