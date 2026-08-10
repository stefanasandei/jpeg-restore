import torch

from .flow import ConditionalFlow


class RectifiedFlow(ConditionalFlow):
    """Conditional rectified flow with clean endpoint prediction."""

    def __init__(
        self,
        model,
        sample_steps=20,
        timestep_scale=1000.0,
        min_velocity_denom=0.05,
        time_logit_mean=-0.8,
        time_logit_std=0.8,
        sampling_method="heun",
        residual_scale=16.0,
    ):
        super().__init__(model, timestep_scale, residual_scale)
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if min_velocity_denom <= 0:
            raise ValueError("min_velocity_denom must be positive")
        if time_logit_std <= 0:
            raise ValueError("time_logit_std must be positive")
        if sampling_method not in ("euler", "heun"):
            raise ValueError("sampling_method must be 'euler' or 'heun'")

        self.sample_steps = sample_steps
        self.min_velocity_denom = min_velocity_denom
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.sampling_method = sampling_method

    @staticmethod
    def interpolate(noise, target, timestep):
        timestep = timestep.view((-1,) + (1,) * (target.ndim - 1))
        return timestep * target + (1 - timestep) * noise

    def velocity(self, state, timestep, endpoint):
        denominator = (1 - timestep).clamp_min(self.min_velocity_denom)
        denominator = denominator.view((-1,) + (1,) * (state.ndim - 1))
        return (endpoint - state) / denominator

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

    def compute_loss(
        self,
        degraded,
        clean,
        quality=None,
        *,
        generator=None,
        t=None,
        noise=None,
        reduction="mean",
    ):
        del quality
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")

        timestep = (
            self.sample_timestep(degraded, generator)
            if t is None
            else t.to(degraded)
        )
        if timestep.shape != (degraded.shape[0],):
            raise ValueError("timestep must have one value per image")

        image_size = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        clean = self.model.pad(clean)
        condition = self.encode(degraded)
        target = self.target(clean, degraded)
        noise = self.prepare_noise(
            noise, target, degraded, image_size, generator
        )

        state = self.interpolate(noise, target, timestep)
        endpoint, _ = self.predict(state, timestep, condition)
        predicted_velocity = self.velocity(state, timestep, endpoint)
        loss = (predicted_velocity - (target - noise)).square().flatten(1).mean(1)
        if reduction == "mean":
            loss = loss.mean()
        return {"loss": loss, "flow": loss.detach()}

    def sample(self, degraded, *, generator=None, noise=None, sample_steps=None):
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        height, width = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        condition = self.encode(degraded)
        noise = self.prepare_noise(
            noise, condition, degraded, (height, width), generator
        )

        state = noise
        dt = 1 / steps
        for step in range(steps):
            timestep = condition.new_full((condition.shape[0],), step * dt)
            endpoint, _ = self.predict(state, timestep, condition)
            velocity = self.velocity(state, timestep, endpoint)
            if self.sampling_method == "euler":
                state = state + velocity * dt
                continue

            proposal = state + velocity * dt
            next_timestep = timestep + dt
            endpoint, _ = self.predict(proposal, next_timestep, condition)
            next_velocity = self.velocity(proposal, next_timestep, endpoint)
            state = state + (velocity + next_velocity) * (dt / 2)

        restored = self.restore(condition, state)[..., :height, :width]
        return restored.clamp(0, 1), None

    def forward(self, degraded):
        return self.sample(degraded)
