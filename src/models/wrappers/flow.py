import torch
from torch import nn


class ConditionalFlow(nn.Module):
    """Shared representation adapter for conditional flow objectives."""

    def __init__(self, model, timestep_scale=1000.0, residual_scale=16.0):
        super().__init__()
        if timestep_scale <= 0 or residual_scale <= 0:
            raise ValueError("flow scales must be positive")
        self.model = model
        self.timestep_scale = timestep_scale
        self.residual_scale = residual_scale

    def predict(self, state, timestep, condition):
        output = self.model(
            state, timestep * self.timestep_scale, condition
        )
        return output if isinstance(output, tuple) else (output, None)

    def encode(self, image):
        return self.model.encode(image)

    def decode(self, state):
        return self.model.decode(state)

    def target(self, clean, degraded):
        return (self.encode(clean) - self.encode(degraded)) * self.residual_scale

    def restore(self, condition, residual):
        return self.decode(condition + residual / self.residual_scale)

    def prepare_noise(self, noise, reference, image, image_size, generator):
        if noise is None:
            return torch.randn_like(reference, generator=generator)

        noise = noise.to(reference)
        if noise.shape[-2:] == image_size:
            noise = self.model.pad(noise)
        if noise.shape == image.shape:
            noise = self.encode(noise)
        if noise.shape != reference.shape:
            raise ValueError("noise must match the image or encoded state")
        return noise
