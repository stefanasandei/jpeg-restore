# Copyright (c) 2024, VinAI. All rights reserved.
# Architecture portions are adapted under the BSD 3-Clause License.

"""DiMSUM spatial-frequency diffusion backbone adapted for restoration.

The block topology follows the official implementation:
https://github.com/VinAIResearch/DiMSUM/blob/main/dimsum/models_dim.py

Each block splits spatial and Haar wavelet-packet streams, applies a Mamba
mixer to both, fuses them with cross-attention, and follows with a gated MLP.
A single globally shared Transformer block is reused every few layers.
"""

import math

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

from models.wavelet import haar_decode, haar_encode


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def pad_to_multiple(image, multiple):
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(next(self.parameters()).dtype))


class GatedMLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.w12 = nn.Linear(hidden_size, hidden_features * 2)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x):
        value, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.gelu(value, approximate="tanh") * gate)


class MambaMixer(nn.Module):
    def __init__(
        self,
        hidden_size,
        state_size=16,
        convolution_size=4,
        expansion=2,
        condition_size=None,
    ):
        super().__init__()
        self.mixer = Mamba(
            d_model=hidden_size,
            d_state=state_size,
            d_conv=convolution_size,
            expand=expansion,
        )
        self.condition_projection = nn.Linear(condition_size, hidden_size)

    def forward(self, x, condition):
        return self.mixer(x + self.condition_projection(condition)[:, None])


def scan_tokens(x, height, width, reverse=False, transpose=False):
    x = x.view(x.shape[0], height, width, x.shape[-1])
    if transpose:
        x = x.transpose(1, 2)
    x = x.reshape(x.shape[0], height * width, x.shape[-1])
    return x.flip(1) if reverse else x


def unscan_tokens(x, height, width, reverse=False, transpose=False):
    if reverse:
        x = x.flip(1)
    if transpose:
        x = x.view(x.shape[0], width, height, x.shape[-1]).transpose(1, 2)
    return x.reshape(x.shape[0], height * width, x.shape[-1])


def wavelet_window_scan(x, height, width, column_first=False):
    """Group a two-level wavelet packet into its 4x4 subband windows.

    This is the rectangular-grid equivalent of the official implementation's
    ``local_scan(..., w=grid_size // 4)``.  Row-first and column-first scans are
    alternated by successive DiMSUM blocks.
    """
    if height % 4 or width % 4:
        raise ValueError("two-level wavelet scanning requires a grid divisible by four")
    batch, tokens, channels = x.shape
    if tokens != height * width:
        raise ValueError("token count does not match the supplied wavelet grid")
    window_height, window_width = height // 4, width // 4
    x = x.view(batch, 4, window_height, 4, window_width, channels)
    if column_first:
        return x.permute(0, 3, 1, 4, 2, 5).reshape(batch, tokens, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, tokens, channels)


def wavelet_window_unscan(x, height, width, column_first=False):
    """Invert :func:`wavelet_window_scan`."""
    batch, tokens, channels = x.shape
    window_height, window_width = height // 4, width // 4
    if column_first:
        x = x.view(batch, 4, 4, window_width, window_height, channels)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(batch, tokens, channels)
    x = x.view(batch, 4, 4, window_height, window_width, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, tokens, channels)


def wavelet_packet_encode(x, height, width):
    x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
    x = haar_encode(haar_encode(x)) * 0.25
    subbands = x.chunk(16, dim=1)
    indices = [index % 4 * 4 + index // 4 for index in range(16)]
    x = torch.cat([subbands[index] for index in indices], dim=1)
    return rearrange(x, "b (c p q) h w -> b (h p w q) c", p=4, q=4)


def wavelet_packet_decode(x, height, width):
    x = rearrange(
        x * 4,
        "b (h p w q) c -> b (c p q) h w",
        p=4,
        q=4,
        h=height // 4,
        w=width // 4,
    )
    subbands = x.chunk(16, dim=1)
    indices = [index % 4 * 4 + index // 4 for index in range(16)]
    x = torch.cat([subbands[index] for index in indices], dim=1)
    return rearrange(haar_decode(haar_decode(x)), "b c h w -> b (h w) c")


class ConditionedMambaBranch(nn.Module):
    def __init__(
        self,
        hidden_size,
        condition_size,
        state_size,
        convolution_size,
        expansion,
        wavelet=False,
        reverse=False,
        transpose=False,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.reverse = reverse
        self.transpose = transpose
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_size, hidden_size * 3)
        )
        self.mixer = MambaMixer(
            hidden_size,
            state_size,
            convolution_size,
            expansion,
            condition_size,
        )

    def forward(self, x, condition, height, width):
        if self.wavelet:
            x = wavelet_packet_encode(x, height, width)
            x = wavelet_window_scan(x, height, width, self.transpose)
            if self.reverse:
                x = x.flip(1)
        else:
            x = scan_tokens(x, height, width, self.reverse, self.transpose)
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        x = x + gate[:, None] * self.mixer(modulate(x, shift, scale), condition)
        if self.wavelet:
            if self.reverse:
                x = x.flip(1)
            x = wavelet_window_unscan(x, height, width, self.transpose)
            x = wavelet_packet_decode(x, height, width)
        else:
            x = unscan_tokens(x, height, width, self.reverse, self.transpose)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        branch_size = hidden_size // 2
        if branch_size % num_heads:
            raise ValueError("half of hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = branch_size // num_heads
        self.qkv_spatial = nn.Linear(branch_size, branch_size * 3)
        self.qkv_frequency = nn.Linear(branch_size, branch_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def _qkv(self, projection, x):
        batch, tokens, channels = x.shape
        q, k, v = projection(x).view(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).unbind(2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, spatial, frequency):
        spatial_q, spatial_k, spatial_v = self._qkv(self.qkv_spatial, spatial)
        frequency_q, frequency_k, frequency_v = self._qkv(
            self.qkv_frequency, frequency
        )
        spatial = F.scaled_dot_product_attention(
            spatial_q, frequency_k, frequency_v, dropout_p=0.0
        )
        frequency = F.scaled_dot_product_attention(
            frequency_q, spatial_k, spatial_v, dropout_p=0.0
        )
        spatial = spatial.transpose(1, 2).flatten(2)
        frequency = frequency.transpose(1, 2).flatten(2)
        return self.proj(torch.cat((spatial, frequency), dim=-1))


class DiMSUMBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        layer_index,
        state_size,
        convolution_size,
        expansion,
        mlp_ratio,
    ):
        super().__init__()
        branch_size = hidden_size // 2
        reverse = layer_index % 2 == 1
        transpose = layer_index % 4 >= 2
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.spatial = ConditionedMambaBranch(
            branch_size,
            hidden_size,
            state_size,
            convolution_size,
            expansion,
            reverse=reverse,
            transpose=transpose,
        )
        self.frequency = ConditionedMambaBranch(
            branch_size,
            hidden_size,
            state_size,
            convolution_size,
            expansion,
            wavelet=True,
            transpose=reverse,
        )
        self.fusion = CrossAttentionFusion(hidden_size, num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = GatedMLP(hidden_size, mlp_ratio)
        self.mlp_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 3)
        )

    def forward(self, x, residual, condition, height, width):
        residual = x if residual is None else residual + x
        x = self.norm(residual)
        spatial, frequency = x.chunk(2, dim=-1)
        spatial = self.spatial(spatial, condition, height, width)
        frequency = self.frequency(frequency, condition, height, width)
        x = x + self.fusion(spatial, frequency)
        shift, scale, gate = self.mlp_modulation(condition).chunk(3, dim=-1)
        x = x + gate[:, None] * self.mlp(modulate(self.norm_mlp(x), shift, scale))
        return x, residual


class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_attention = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm_mlp = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.mlp = GatedMLP(hidden_size, mlp_ratio)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6)
        )

    def forward(self, x, condition):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        attention_input = modulate(self.norm_attention(x), shift_attn, scale_attn)
        batch, tokens, channels = attention_input.shape
        q, k, v = self.qkv(attention_input).view(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).unbind(2)
        attention = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, tokens, channels)
        attention = self.proj(attention)
        x = x + gate_attn[:, None] * attention
        x = x + gate_mlp[:, None] * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2)
        )

    def forward(self, x, condition):
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class HierarchicalPatch8(nn.Module):
    """Embed pixels through aligned 4x4 cells and 8x8 JPEG-block tokens."""

    def __init__(self, in_channels: int, hidden_dim: int, token_dim: int) -> None:
        super().__init__()
        # Merge four local 4x4 embeddings into each 8x8 JPEG-block token.
        self.local_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.merge = nn.Conv2d(hidden_dim, token_dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(self.local_embed(x))


class HierarchicalUnpatch8(nn.Module):
    """Decode stride-8 tokens through 4x4 cells back to pixels."""

    def __init__(self, token_dim: int, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.to_blocks = nn.Sequential(
            nn.Conv2d(token_dim, hidden_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.to_coefficients = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels * 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError("token count does not match the supplied grid")
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.to_coefficients(self.to_blocks(x))


class DiMSUM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        patch_size=8,
        patch_hidden_size=96,
        hidden_size=384,
        depth=16,
        num_heads=8,
        global_num_heads=16,
        state_size=16,
        convolution_size=4,
        expansion=2,
        mlp_ratio=4.0,
        global_attention_interval=4,
    ):
        super().__init__()
        if hidden_size % 2 or hidden_size // 2 % num_heads:
            raise ValueError("hidden_size/2 must be divisible by num_heads")
        if hidden_size % global_num_heads:
            raise ValueError("hidden_size must be divisible by global_num_heads")
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four")
        if global_attention_interval < 1:
            raise ValueError("global_attention_interval must be positive")
        if patch_size != 8:
            raise ValueError("the hierarchical patch encoder requires patch_size=8")
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.global_attention_interval = global_attention_interval
        self.x_embedder = HierarchicalPatch8(
            in_channels, patch_hidden_size, hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiMSUMBlock(
                    hidden_size,
                    num_heads,
                    index,
                    state_size,
                    convolution_size,
                    expansion,
                    mlp_ratio,
                )
                for index in range(depth)
            ]
        )
        self.shared_transformer = SharedTransformerBlock(
            hidden_size, global_num_heads, mlp_ratio
        )
        self.final_layer = FinalLayer(hidden_size)
        self.output_decoder = HierarchicalUnpatch8(
            hidden_size, patch_hidden_size, out_channels
        )
        self.initialize_weights()

    @staticmethod
    def positional_embedding(height, width, dim, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        omega = 1.0 / 10000 ** (
            torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)
        )
        y = y.flatten()[:, None] * omega[None]
        x = x.flatten()[:, None] * omega[None]
        embedding = torch.cat(
            (x.sin(), x.cos(), y.sin(), y.cos()), dim=-1
        )
        return embedding[None].to(dtype)

    def initialize_weights(self):
        # Convolution layers keep PyTorch's defaults. The conditioning paths
        # and final projection start at zero, as in diffusion transformers.
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            for modulation in (
                block.spatial.modulation,
                block.frequency.modulation,
                block.mlp_modulation,
            ):
                nn.init.zeros_(modulation[-1].weight)
                nn.init.zeros_(modulation[-1].bias)
        nn.init.zeros_(self.shared_transformer.modulation[-1].weight)
        nn.init.zeros_(self.shared_transformer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        output = self.output_decoder.to_coefficients[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, image, timestep):
        x = self.x_embedder(image)
        height, width = x.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError("DiMSUM's token grid must be divisible by four")
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_embedding(
            height, width, self.hidden_size, x.device, x.dtype
        )
        condition = self.t_embedder(timestep)
        residual = None
        for index, block in enumerate(self.blocks):
            x, residual = block(x, residual, condition, height, width)
            if (index + 1) % self.global_attention_interval == 0:
                x = self.shared_transformer(x, condition)
        # DiMSUM blocks use an Add -> LayerNorm -> Mixer residual stream. The
        # last mixer output must be merged just like it is between blocks.
        if residual is not None:
            x = x + residual
        x = self.final_layer(x, condition)
        return self.output_decoder(x, height, width)


class DiMSUMRestoration(DiMSUM):
    """Expose DiMSUM through the repository's restoration interface."""

    def __init__(self, timestep=0, **kwargs):
        super().__init__(**kwargs)
        self.timestep = timestep

    def forward(self, image):
        height, width = image.shape[-2:]
        image = pad_to_multiple(image, self.patch_size * 4)
        timestep = torch.full(
            (image.shape[0],),
            self.timestep,
            device=image.device,
            dtype=image.dtype,
        )
        restored = super().forward(image, timestep)[..., :height, :width]
        return restored, None
