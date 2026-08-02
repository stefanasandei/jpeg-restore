import math

import torch
from torch import nn
import torch.nn.functional as F

from models.dimsum import DiMSUM, DiMSUMRestoration


class DiMSUMPalette(nn.Module):
    """Palette-style source conditioning through channel concatenation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.network = DiMSUM(in_channels=6, **kwargs)

    def forward(
        self,
        restoration,
        time,
        compressed_image,
    ):
        height, width = restoration.shape[-2:]
        multiple = self.network.patch_size * 4
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple

        restoration = F.pad(
            restoration, (0, pad_width, 0, pad_height), mode="replicate"
        )
        compressed_image = F.pad(
            compressed_image,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )

        velocity = self.network(
            torch.cat((restoration, compressed_image), dim=1),
            time,
        )
        return velocity[..., :height, :width]


class DiMSUMRectifiedFlow(nn.Module):
    """Conditional rectified flow from a noisy JPEG image to a clean image.

    The flow state is always an RGB restoration image. Sampling starts from
    ``compressed_image + N(0, I)`` and integrates the predicted image velocity
    to the clean-image endpoint. All states, velocities, and losses use raw
    image-space units; the model learns the required correction scale.
    """

    is_rectified_flow = True

    def __init__(
        self,
        sample_steps=20,
        timestep_scale=1000.0,
        endpoint_loss_weight=1.0,
        dct_loss_weight=0.1,
        **kwargs,
    ):
        super().__init__()
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if timestep_scale <= 0:
            raise ValueError("timestep_scale must be positive")
        if min(
            endpoint_loss_weight,
            dct_loss_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")

        self.sample_steps = sample_steps
        self.timestep_scale = timestep_scale
        self.endpoint_loss_weight = endpoint_loss_weight
        self.dct_loss_weight = dct_loss_weight
        self.model = DiMSUMPalette(**kwargs)

        indices = torch.arange(8, dtype=torch.float32)
        frequencies = indices[:, None]
        basis = torch.cos(math.pi / 8 * (indices[None] + 0.5) * frequencies)
        basis[0] /= math.sqrt(2)
        self.register_buffer("dct_basis", basis * 0.5, persistent=False)
        self.register_buffer(
            "rgb_to_ycbcr",
            torch.tensor(
                [
                    [0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312],
                ]
            ),
            persistent=False,
        )

    def predict_velocity(self, restoration, time, compressed_image):
        """Predict raw RGB velocity for the current restoration image."""
        # DiMSUM's sinusoidal embedding was designed for diffusion timesteps
        # spanning roughly [0, 1000], rather than continuous values in [0, 1].
        return self.model(
            restoration,
            time * self.timestep_scale,
            compressed_image,
        )

    def _dct_loss(self, prediction, target):
        if prediction.shape[-2] % 8 or prediction.shape[-1] % 8:
            raise ValueError("JPEG DCT loss requires image dimensions divisible by 8")

        def block_dct(image):
            matrix = self.rgb_to_ycbcr.to(dtype=image.dtype)
            image = torch.einsum("bchw,dc->bdhw", image, matrix)
            blocks = image.unfold(2, 8, 8).unfold(3, 8, 8)
            basis = self.dct_basis.to(dtype=image.dtype)
            return torch.matmul(torch.matmul(basis, blocks), basis.t())

        error = (block_dct(prediction) - block_dct(target)).abs()
        return error.flatten(-2)[..., 1:].mean()

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
        compressed_image,
        clean_image,
        quality=None,
        *,
        generator=None,
        t=None,
        noise=None,
    ):
        # Kept for the common training API; blind DiMSUM flow does not consume
        # JPEG quality labels.
        del quality
        batch_size = compressed_image.shape[0]
        if t is None:
            t = torch.rand(
                batch_size,
                device=compressed_image.device,
                dtype=compressed_image.dtype,
                generator=generator,
            )
        else:
            t = t.to(
                device=compressed_image.device, dtype=compressed_image.dtype
            )
            if t.shape != (batch_size,):
                raise ValueError(
                    f"expected t shape {(batch_size,)}, got {tuple(t.shape)}"
                )

        if noise is None:
            noise = self._randn_like(compressed_image, generator)
        elif noise.shape != compressed_image.shape:
            raise ValueError(
                "noise must have the same shape as compressed_image"
            )
        noise = noise.to(
            device=compressed_image.device, dtype=compressed_image.dtype
        )

        noisy_source_image = compressed_image + noise
        time = t[:, None, None, None]
        restoration = (1 - time) * noisy_source_image + time * clean_image
        target_velocity = clean_image - noisy_source_image
        predicted_velocity = self.predict_velocity(
            restoration, t, compressed_image
        )

        flow = F.mse_loss(predicted_velocity, target_velocity)
        predicted_clean_image = restoration + (1 - time) * predicted_velocity

        endpoint = F.l1_loss(predicted_clean_image, clean_image)
        dct = self._dct_loss(predicted_clean_image, clean_image)
        loss = flow + self.endpoint_loss_weight * endpoint
        loss = loss + self.dct_loss_weight * dct

        return {
            "loss": loss,
            "flow": flow,
            "reconstruction": endpoint.detach(),
            "dct": dct.detach(),
        }

    def sample(
        self,
        compressed_image,
        *,
        generator=None,
        noise=None,
        sample_steps=None,
    ):
        """Restore a JPEG image; supplied ``noise`` is unscaled N(0, I)."""
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        if noise is None:
            noise = self._randn_like(compressed_image, generator)
        elif noise.shape != compressed_image.shape:
            raise ValueError(
                "noise must have the same shape as compressed_image"
            )
        source_noise = noise.to(
            device=compressed_image.device, dtype=compressed_image.dtype
        )
        restoration = compressed_image + source_noise
        dt = 1.0 / steps

        for step in range(steps):
            time = compressed_image.new_full(
                (compressed_image.shape[0],), step * dt
            )
            predicted_velocity = self.predict_velocity(
                restoration, time, compressed_image
            )
            restoration = restoration + dt * predicted_velocity

        return restoration.clamp(0, 1), None

    def forward(self, compressed_image):
        return self.sample(compressed_image)


def dimsum_model(
    objective="regression",
    sample_steps=20,
    timestep_scale=1000.0,
    **kwargs,
):
    if objective == "regression":
        return DiMSUMRestoration(**kwargs)
    if objective == "rectified_flow":
        return DiMSUMRectifiedFlow(
            sample_steps=sample_steps,
            timestep_scale=timestep_scale,
            **kwargs,
        )
    raise ValueError(f"unknown objective: {objective}")
