import torch
from torch import nn
import torch.nn.functional as F

from models.dimsum import DiMSUM, DiMSUMRestoration


class DiMSUMPalette(nn.Module):
    """Palette-style source conditioning through channel concatenation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.network = DiMSUM(in_channels=6, **kwargs)

    def forward(self, x, t, compressed):
        height, width = x.shape[-2:]
        multiple = self.network.patch_size * 4
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple

        x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        compressed = F.pad(
            compressed, (0, pad_width, 0, pad_height), mode="replicate"
        )

        output = self.network(torch.cat((x, compressed), dim=1), t)
        return output[..., :height, :width]


class DiMSUMRectifiedFlow(nn.Module):
    """Conditional rectified flow in normalized JPEG-residual space.

    The transported variable is ``delta / residual_scale``, where
    ``delta = clean - compressed``.  Sampling therefore starts from Gaussian
    noise in residual space and ends at a residual which is added to the JPEG
    input.  This is equivalent to transporting ``compressed + noise`` to the
    clean image, but keeps the model focused on the uncertain restoration
    component.
    """

    is_rectified_flow = True

    def __init__(
        self,
        sample_steps=20,
        residual_scale=0.1,
        noise_scale=1.0,
        timestep_scale=1000.0,
        **kwargs,
    ):
        super().__init__()
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if noise_scale <= 0:
            raise ValueError("noise_scale must be positive for generative flow")
        if timestep_scale <= 0:
            raise ValueError("timestep_scale must be positive")

        self.sample_steps = sample_steps
        self.residual_scale = residual_scale
        self.noise_scale = noise_scale
        self.timestep_scale = timestep_scale
        self.model = DiMSUMPalette(**kwargs)

    def velocity(self, x_t, t, compressed):
        # DiMSUM's sinusoidal embedding was designed for diffusion timesteps
        # spanning roughly [0, 1000], rather than continuous values in [0, 1].
        return self.model(x_t, t * self.timestep_scale, compressed)

    @staticmethod
    def _randn_like(reference, generator=None):
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def compute_loss(
        self,
        compressed,
        clean,
        quality=None,
        *,
        generator=None,
        t=None,
        noise=None,
    ):
        batch_size = compressed.shape[0]
        if t is None:
            t = torch.rand(
                batch_size,
                device=compressed.device,
                dtype=compressed.dtype,
                generator=generator,
            )
        else:
            t = t.to(device=compressed.device, dtype=compressed.dtype)
            if t.shape != (batch_size,):
                raise ValueError(f"expected t shape {(batch_size,)}, got {tuple(t.shape)}")

        if noise is None:
            noise = self._randn_like(compressed, generator)
        elif noise.shape != compressed.shape:
            raise ValueError("noise must have the same shape as compressed")
        noise = noise.to(device=compressed.device, dtype=compressed.dtype)

        # Straight conditional flow: Gaussian source -> normalized clean delta.
        x_0 = self.noise_scale * noise
        x_1 = (clean - compressed) / self.residual_scale
        t_broadcast = t[:, None, None, None]
        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        v_t = x_1 - x_0

        predicted_v_t = self.velocity(x_t, t, compressed)
        flow = F.mse_loss(predicted_v_t, v_t)
        return {"loss": flow, "flow": flow}

    def sample(self, compressed, *, generator=None, noise=None, sample_steps=None):
        """Generate a restoration; supplied ``noise`` is unscaled N(0, I)."""
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        if noise is None:
            noise = self._randn_like(compressed, generator)
        elif noise.shape != compressed.shape:
            raise ValueError("noise must have the same shape as compressed")
        x_t = self.noise_scale * noise.to(
            device=compressed.device, dtype=compressed.dtype
        )
        dt = 1.0 / steps

        for step in range(steps):
            t = compressed.new_full((compressed.shape[0],), step * dt)
            v = self.velocity(x_t, t, compressed)
            x_t = x_t + dt * v

        restored = compressed + self.residual_scale * x_t
        return restored.clamp(0, 1), None

    def forward(self, compressed):
        return self.sample(compressed)


def dimsum_model(
    objective="regression",
    sample_steps=20,
    residual_scale=0.1,
    noise_scale=1.0,
    timestep_scale=1000.0,
    **kwargs,
):
    if objective == "regression":
        return DiMSUMRestoration(**kwargs)
    if objective == "rectified_flow":
        return DiMSUMRectifiedFlow(
            sample_steps=sample_steps,
            residual_scale=residual_scale,
            noise_scale=noise_scale,
            timestep_scale=timestep_scale,
            **kwargs,
        )
    raise ValueError(f"unknown objective: {objective}")
