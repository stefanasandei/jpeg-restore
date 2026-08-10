import torch
import torch.nn.functional as F

from .flow import ConditionalFlow


def expand(timestep, reference):
    return timestep.view((-1,) + (1,) * (reference.ndim - 1))


class MeanFlow(ConditionalFlow):
    """Pixel MeanFlow with the improved velocity-space objective."""

    def __init__(
        self,
        model,
        sample_steps=1,
        timestep_scale=1000.0,
        min_time=0.05,
        time_logit_mean=0.8,
        time_logit_std=0.8,
        instantaneous_ratio=0.5,
        adaptive_power=1.0,
        adaptive_eps=0.01,
        lpips_weight=0.0,
        lpips_max_time=0.8,
        lpips_backbone="vgg",
        quality_loss_weight=0.0,
        residual_scale=16.0,
        derivative_method="jvp",
        finite_difference_eps=1e-3,
    ):
        super().__init__(model, timestep_scale, residual_scale)
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if min_time <= 0 or time_logit_std <= 0 or adaptive_eps <= 0:
            raise ValueError("mean flow scales must be positive")
        if not 0 <= instantaneous_ratio <= 1:
            raise ValueError("instantaneous_ratio must be in [0, 1]")
        if adaptive_power < 0:
            raise ValueError("adaptive_power must be non-negative")
        if lpips_weight < 0 or not 0 < lpips_max_time <= 1:
            raise ValueError("invalid LPIPS configuration")
        if lpips_backbone not in ("alex", "vgg", "squeeze"):
            raise ValueError("invalid LPIPS backbone")
        if quality_loss_weight < 0:
            raise ValueError("quality_loss_weight must be non-negative")
        if derivative_method not in ("jvp", "rcm", "forward_difference"):
            raise ValueError(
                "derivative_method must be 'jvp', 'rcm', or "
                "'forward_difference'"
            )
        if finite_difference_eps <= 0:
            raise ValueError("finite_difference_eps must be positive")

        self.sample_steps = sample_steps
        self.min_time = min_time
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.instantaneous_ratio = instantaneous_ratio
        self.adaptive_power = adaptive_power
        self.adaptive_eps = adaptive_eps
        self.lpips_weight = lpips_weight
        self.lpips_max_time = lpips_max_time
        self.lpips_backbone = lpips_backbone
        self.quality_loss_weight = quality_loss_weight
        self.derivative_method = derivative_method
        self.finite_difference_eps = finite_difference_eps

    def _adaptive_loss(self, loss):
        weight = (loss.detach() + self.adaptive_eps).pow(
            -self.adaptive_power
        )
        return weight * loss

    def sample_times(self, reference, generator=None):
        logits = torch.randn(
            reference.shape[0],
            2,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        times = torch.sigmoid(
            self.time_logit_mean + self.time_logit_std * logits
        ).sort(dim=1).values
        r, t = times.unbind(1)
        instantaneous = torch.rand(
            r.shape, device=r.device, generator=generator
        ) < self.instantaneous_ratio
        r = torch.where(instantaneous, t, r)
        return r, t

    @staticmethod
    def interpolate(x, e, t):
        t = expand(t, x)
        return (1 - t) * x + t * e

    def u(self, z, r, t, condition, return_quality=False):
        x, quality = self.predict(z, t - r, condition)
        average = (z - x) / expand(t.clamp_min(self.min_time), z)
        return (average, quality) if return_quality else average

    @staticmethod
    def _jvp_segment(function, primals, tangents):
        """Run one exact JVP segment and discard its tangent graph."""
        output, tangent = torch.func.jvp(function, primals, tangents)
        if isinstance(tangent, tuple):
            tangent = tuple(value.detach() for value in tangent)
        else:
            tangent = tangent.detach()
        return output, tangent

    def _rcm_jvp(self, z, r, t, v, condition):
        """Exact, memory-efficient JVP segmented like the rCM network."""
        network = getattr(self.model, "network", None)
        required = (
            "x_embedder",
            "t_embedder",
            "t_block",
            "blocks",
            "final_layer",
            "unpatchify",
            "positional_embedding",
        )
        if network is None or any(
            not hasattr(network, name) for name in required
        ):
            raise TypeError(
                "the rcm JVP currently requires a Palette wrapper around "
                "SanaLinearDiT"
            )

        image = torch.cat((z, condition), dim=1)
        image_tangent = torch.cat((v, torch.zeros_like(condition)), dim=1)
        timestep = (t - r) * self.timestep_scale
        timestep_tangent = torch.ones_like(t) * self.timestep_scale

        def embed_fn(image, timestep):
            x = network.x_embedder(image)
            height, width = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)
            x = x + network.positional_embedding(
                height,
                width,
                network.hidden_size,
                x.device,
                x.dtype,
                network.positional_reference_size,
            )
            embedded_timestep = network.t_embedder(timestep)
            block_timestep = network.t_block(embedded_timestep)
            return x, embedded_timestep, block_timestep

        (x, embedded_timestep, block_timestep), (
            x_tangent,
            embedded_timestep_tangent,
            block_timestep_tangent,
        ) = self._jvp_segment(
            embed_fn,
            (image, timestep),
            (image_tangent, timestep_tangent),
        )

        block_condition = None
        if network.condition_embedder is not None:
            block_condition = network.condition_embedder(
                network.condition.expand(z.shape[0], -1, -1)
            )

        height = image.shape[-2] // network.patch_size
        width = image.shape[-1] // network.patch_size
        split = network.quality_split or len(network.blocks)

        def run_block(block, x, x_tangent):
            def block_fn(x, block_timestep):
                return block(
                    x,
                    block_condition,
                    block_timestep,
                    height,
                    width,
                )

            return self._jvp_segment(
                block_fn,
                (x, block_timestep),
                (x_tangent, block_timestep_tangent),
            )

        for block in network.blocks[:split]:
            x, x_tangent = run_block(block, x, x_tangent)

        quality = None
        if network.quality_head is not None:

            def quality_fn(x, embedded_timestep):
                quality = network.quality_head(x.mean(dim=1))
                embedded_timestep = (
                    embedded_timestep
                    + network.quality_embedder(quality)
                )
                return (
                    embedded_timestep,
                    network.t_block(embedded_timestep),
                    quality,
                )

            (embedded_timestep, block_timestep, quality), (
                embedded_timestep_tangent,
                block_timestep_tangent,
                _,
            ) = self._jvp_segment(
                quality_fn,
                (x, embedded_timestep),
                (x_tangent, embedded_timestep_tangent),
            )

        for block in network.blocks[split:]:
            x, x_tangent = run_block(block, x, x_tangent)

        def output_fn(x, embedded_timestep, z, t):
            prediction = network.unpatchify(
                network.final_layer(x, embedded_timestep),
                height,
                width,
            )
            return (z - prediction) / expand(
                t.clamp_min(self.min_time), z
            )

        u, du_dt = self._jvp_segment(
            output_fn,
            (x, embedded_timestep, z, t),
            (x_tangent, embedded_timestep_tangent, v, torch.ones_like(t)),
        )
        return u, du_dt, quality

    def compute_loss(
        self,
        degraded,
        clean,
        quality=None,
        *,
        generator=None,
        r=None,
        t=None,
        noise=None,
        perceptual_loss=None,
        include_perceptual=True,
        reduction="mean",
    ):
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        if (r is None) != (t is None):
            raise ValueError("r and t must be provided together")

        image_size = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        clean = self.model.pad(clean)
        condition = self.encode(degraded)
        x = self.target(clean, degraded)
        e = self.prepare_noise(noise, x, degraded, image_size, generator)

        if r is None:
            r, t = self.sample_times(x, generator)
        else:
            r, t = r.to(x), t.to(x)
        if r.shape != (x.shape[0],) or t.shape != r.shape:
            raise ValueError("r and t must have one value per image")
        if torch.any((r < 0) | (r > t) | (t > 1)):
            raise ValueError("times must satisfy 0 <= r <= t <= 1")

        z = self.interpolate(x, e, t)

        def u_fn(z, r, t):
            return self.u(z, r, t, condition)

        with torch.no_grad():
            v = u_fn(z, t, t)
        tangents = (v, torch.zeros_like(r), torch.ones_like(t))
        predicted_quality = None
        if self.derivative_method == "jvp":
            if self.quality_loss_weight:

                def u_with_quality(z, r, t):
                    return self.u(
                        z, r, t, condition, return_quality=True
                    )

                u, du_dt, predicted_quality = torch.func.jvp(
                    u_with_quality, (z, r, t), tangents, has_aux=True
                )
            else:
                u, du_dt = torch.func.jvp(
                    u_fn, (z, r, t), tangents
                )
        elif self.derivative_method == "rcm":
            u, du_dt, predicted_quality = self._rcm_jvp(
                z, r, t, v, condition
            )
        else:
            if self.quality_loss_weight:
                u, predicted_quality = self.u(
                    z, r, t, condition, return_quality=True
                )
            else:
                u = u_fn(z, r, t)
            epsilon = self.finite_difference_eps
            with torch.no_grad():
                shifted = u_fn(z + epsilon * v, r, t + epsilon)
                du_dt = (shifted - u.detach()) / epsilon
        V = u + expand(t - r, u) * du_dt.detach()
        dx = V - (e - x)

        mean_flow = dx.square().flatten(1).mean(1)
        loss = self._adaptive_loss(mean_flow)

        quality_loss = None
        if self.quality_loss_weight:
            if quality is None or predicted_quality is None:
                raise ValueError("quality prediction is required")
            quality_loss = F.l1_loss(
                predicted_quality,
                quality.to(predicted_quality),
                reduction="none",
            ).flatten(1).mean(1)
            loss = loss + self.quality_loss_weight * quality_loss

        lpips = mean_flow.new_zeros(mean_flow.shape)
        if self.lpips_weight and include_perceptual:
            if perceptual_loss is None:
                raise ValueError(
                    "perceptual_loss is required when LPIPS is enabled"
                )
            selected = t < self.lpips_max_time
            if selected.any():
                predicted_x = z - expand(t.clamp_min(self.min_time), z) * u
                predicted = self.restore(condition, predicted_x)
                height, width = image_size
                scores = perceptual_loss(
                    predicted[selected, :, :height, :width].clamp(0, 1),
                    clean[selected, :, :height, :width],
                ).reshape(-1)
                perceptual_loss.reset()
                indices = selected.nonzero(as_tuple=False).flatten()
                lpips = lpips.index_copy(0, indices, scores)
            loss = loss + self.lpips_weight * self._adaptive_loss(lpips)

        if reduction == "mean":
            loss = loss.mean()
            mean_flow = mean_flow.mean()
            if quality_loss is not None:
                quality_loss = quality_loss.mean()
            selected_count = (t < self.lpips_max_time).sum().clamp_min(1)
            lpips = lpips.sum() / selected_count
        losses = {"loss": loss, "mean_flow": mean_flow.detach()}
        if quality_loss is not None:
            losses["quality"] = quality_loss.detach()
        if self.lpips_weight and include_perceptual:
            losses["lpips"] = lpips.detach()
        return losses

    def sample(self, degraded, *, generator=None, noise=None, sample_steps=None):
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        height, width = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        condition = self.encode(degraded)
        z = self.prepare_noise(
            noise, condition, degraded, (height, width), generator
        )

        batch = degraded.shape[0]
        for step in range(steps, 0, -1):
            t = condition.new_full((batch,), step / steps)
            r = t - 1 / steps
            u = self.u(z, r, t, condition)
            z = z - expand(t - r, z) * u

        restored = self.restore(condition, z)[..., :height, :width]
        return restored.clamp(0, 1), None

    def forward(self, degraded):
        return self.sample(degraded)
