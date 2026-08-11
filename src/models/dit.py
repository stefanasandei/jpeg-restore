"""Canonical Diffusion Transformer (DiT) for image restoration.

This is a dependency-free implementation of the architecture introduced in
"Scalable Diffusion Models with Transformers".  It uses global softmax
self-attention, a GELU MLP, adaLN-Zero blocks, and a patchwise output head.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps with sinusoidal features and an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(next(self.parameters()).dtype))


class Attention(nn.Module):
    """Canonical global multi-head softmax self-attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .unbind(2)
        )
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, channels))


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float):
        super().__init__()
        mlp_size = int(hidden_size * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """Canonical DiT backbone accepting an image and one timestep per sample."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int = 2,
        hidden_size: int = 480,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if patch_size < 1 or depth < 1:
            raise ValueError("patch_size and depth must be positive")
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.x_embedder = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    @staticmethod
    def positional_embedding(
        height: int,
        width: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four")
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        omega = torch.arange(
            hidden_size // 4,
            device=device,
            dtype=torch.float32,
        ) / (hidden_size // 4)
        omega = 1 / 10_000**omega
        y = y.flatten()[:, None] * omega[None]
        x = x.flatten()[:, None] * omega[None]
        embedding = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return embedding.unsqueeze(0).to(dtype)

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.xavier_uniform_(
            self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1)
        )
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        patch = self.patch_size
        tokens = tokens.reshape(
            batch,
            height,
            width,
            patch,
            patch,
            self.out_channels,
        )
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(
            batch,
            self.out_channels,
            height * patch,
            width * patch,
        )

    def forward(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.shape != (image.shape[0],):
            raise ValueError("timestep must have one value per image")
        x = self.x_embedder(image)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_embedding(
            height,
            width,
            self.hidden_size,
            x.device,
            x.dtype,
        )
        condition = self.t_embedder(timestep)
        for block in self.blocks:
            x = block(x, condition)
        return self.unpatchify(
            self.final_layer(x, condition),
            height,
            width,
        )


class DiTRestoration(DiT):
    """Expose canonical DiT as a direct, single-NFE restoration model."""

    def __init__(self, timestep: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.timestep = timestep

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
        timestep = image.new_full((image.shape[0],), self.timestep)
        restored = super().forward(image, timestep)
        return restored[..., :height, :width]
