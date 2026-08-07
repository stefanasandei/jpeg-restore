import torch
from torch import nn
import torch.nn.functional as F

from models.dimsum import DiMSUM, pad_to_multiple
from models.losses import JPEGDCTLoss
from models.wavelet import haar_decode, haar_encode


class DiMSUMPalette(nn.Module):
    """Predict a clean restoration residual in an invertible Haar domain."""

    def __init__(self, wavelet_levels=0, **kwargs):
        super().__init__()
        if wavelet_levels < 0:
            raise ValueError("wavelet_levels must be non-negative")
        self.wavelet_levels = wavelet_levels
        image_channels = kwargs.pop("out_channels", 3)
        latent_channels = image_channels * 4**wavelet_levels
        self.network = DiMSUM(
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            **kwargs,
        )

    @property
    def pixel_multiple(self):
        return 2**self.wavelet_levels * self.network.patch_size * 4

    def encode(self, image):
        for _ in range(self.wavelet_levels):
            image = haar_encode(image)
        return image

    def decode(self, coefficients):
        for _ in range(self.wavelet_levels):
            coefficients = haar_decode(coefficients)
        return coefficients

    def forward(self, x, t, y):
        return self.network(torch.cat((x, y), dim=1), t)


class DiMSUMRectifiedFlow(nn.Module):
    """Conditional flow over clean restoration residuals in a Haar domain.

    Following JiT, the network directly predicts the on-manifold clean
    residual while training and sampling use its derived velocity.  The state
    is a normalized two-level Haar representation of ``clean - degraded``;
    sampling starts from Gaussian noise and exact inverse Haar reconstruction
    adds the generated correction to the degraded image.
    """

    def __init__(
        self,
        sample_steps=20,
        timestep_scale=1000.0,
        min_velocity_denom=0.05,
        time_logit_mean=-0.8,
        time_logit_std=0.8,
        sampling_method="heun",
        residual_scale=16.0,
        endpoint_loss_weight=1.0,
        dct_loss_weight=0.1,
        quality_loss_weight=0.1,
        **kwargs,
    ):
        super().__init__()
        if (
            sample_steps < 1
            or timestep_scale <= 0
            or min_velocity_denom <= 0
            or residual_scale <= 0
        ):
            raise ValueError(
                "sample_steps, timestep_scale, min_velocity_denom, and "
                "residual_scale must be positive"
            )
        if time_logit_std <= 0:
            raise ValueError("time_logit_std must be positive")
        if sampling_method not in ("euler", "heun"):
            raise ValueError("sampling_method must be 'euler' or 'heun'")
        if (
            endpoint_loss_weight < 0
            or dct_loss_weight < 0
            or quality_loss_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")

        self.sample_steps = sample_steps
        self.timestep_scale = timestep_scale
        self.min_velocity_denom = min_velocity_denom
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.sampling_method = sampling_method
        self.residual_scale = residual_scale
        self.endpoint_loss_weight = endpoint_loss_weight
        self.dct_loss_weight = dct_loss_weight
        self.quality_loss_weight = quality_loss_weight
        self.model = DiMSUMPalette(**kwargs)
        self.dct_loss = JPEGDCTLoss()

    def predict(self, x, t, y):
        """Predict the clean Haar residual and JPEG quality factor."""
        return self.model(x, t * self.timestep_scale, y)

    def encode(self, image):
        return self.model.encode(image)

    def decode(self, coefficients):
        return self.model.decode(coefficients)

    def target_residual(self, clean, degraded):
        return (self.encode(clean) - self.encode(degraded)) * self.residual_scale

    def restore(self, degraded_latent, residual):
        return self.decode(degraded_latent + residual / self.residual_scale)

    def interpolate(self, x_0, x_1, t):
        shape = (-1, 1, 1, 1)
        t = t.view(shape)
        return t * x_1 + (1 - t) * x_0

    def velocity(self, x_t, t, x_1_hat):
        denominator = (1 - t).clamp_min(self.min_velocity_denom)
        return (x_1_hat - x_t) / denominator[:, None, None, None]

    def endpoint(self, x_t, t, x_1_hat):
        """The network directly predicts the clean endpoint."""
        del x_t, t
        return x_1_hat

    def sample_timestep(self, reference, generator=None):
        logits = torch.randn(
            reference.shape[0],
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        return torch.sigmoid(
            self.time_logit_mean + self.time_logit_std * logits
        )

    def compute_loss(self, y, x_1, quality=None, *, generator=None, t=None, noise=None):
        if t is None:
            t = self.sample_timestep(y, generator)
        else:
            t = t.to(y)
        if t.shape != (y.shape[0],):
            raise ValueError(f"expected t shape {(y.shape[0],)}, got {t.shape}")

        y_latent = self.encode(y)
        x_1_latent = self.target_residual(x_1, y)
        if noise is None:
            x_0 = torch.randn_like(x_1_latent, generator=generator)
        else:
            noise = noise.to(x_1_latent)
            if noise.shape == y.shape:
                x_0 = self.encode(noise)
            elif noise.shape == x_1_latent.shape:
                x_0 = noise
            else:
                raise ValueError("noise must match the image or wavelet-state shape")
        x_t = self.interpolate(x_0, x_1_latent, t)

        x_1_hat_latent, q_pred = self.predict(x_t, t, y_latent)
        x_1_hat = self.restore(y_latent, x_1_hat_latent)

        velocity_hat = self.velocity(x_t, t, x_1_hat_latent)
        velocity = x_1_latent - x_0
        flow = F.mse_loss(velocity_hat, velocity)
        endpoint = F.l1_loss(x_1_hat, x_1)
        dct = self.dct_loss(x_1_hat, x_1) if self.dct_loss_weight else flow.new_zeros(())
        quality_loss = (
            F.l1_loss(q_pred, quality.reshape_as(q_pred))
            if quality is not None
            else flow.new_zeros(())
        )
        loss = (
            flow
            + self.endpoint_loss_weight * endpoint
            + self.dct_loss_weight * dct
            + self.quality_loss_weight * quality_loss
        )

        return {
            "loss": loss,
            "flow": flow.detach(),
            "reconstruction": endpoint.detach(),
            "dct": dct.detach(),
            "quality": quality_loss.detach(),
        }

    def sample(self, y, *, generator=None, noise=None, sample_steps=None):
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        height, width = y.shape[-2:]
        y = pad_to_multiple(y, self.model.pixel_multiple)
        y_latent = self.encode(y)
        if noise is None:
            noise = torch.randn_like(y_latent, generator=generator)
        else:
            noise = noise.to(y_latent)
            if noise.shape == y.shape:
                noise = self.encode(noise)
        if noise.shape != y_latent.shape:
            raise ValueError("noise must match the image or wavelet-state shape")

        x = noise
        dt = 1 / steps
        for step in range(steps):
            t = y_latent.new_full((y_latent.shape[0],), step * dt)
            x_1_hat, _ = self.predict(x, t, y_latent)
            velocity = self.velocity(x, t, x_1_hat)
            if self.sampling_method == "euler":
                x = x + velocity * dt
                continue

            t_next = t + dt
            proposal = x + velocity * dt
            x_1_next, _ = self.predict(proposal, t_next, y_latent)
            velocity_next = self.velocity(proposal, t_next, x_1_next)
            x = x + (velocity + velocity_next) * (dt / 2)

        return self.restore(y_latent, x)[..., :height, :width].clamp(0, 1), None

    def forward(self, y):
        return self.sample(y)
