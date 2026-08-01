# Copyright (c) Meta Platforms, Inc. and affiliates.
# Architecture portions are used under the upstream BSD-style license.

"""ADM U-Net from TorchMultimodal, adapted to the local model interface.

The architecture follows Meta's official implementation:
https://github.com/facebookresearch/multimodal/tree/main/torchmultimodal/diffusion_labs/models/adm_unet
"""

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x):
        output = F.group_norm(
            x.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timestep):
        half_dim = self.embed_dim // 2
        scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timestep.device) * -scale)
        embeddings = timestep.unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.embed_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ADMResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim_cond,
        use_upsample=False,
        use_downsample=False,
        skip_conv=None,
        rescale_skip_connection=False,
        scale_shift_conditional=True,
        pre_outconv_dropout=0.1,
        norm_groups=32,
        norm_eps=1e-5,
    ):
        super().__init__()
        if skip_conv is None and in_channels != out_channels:
            raise ValueError("A skip convolution is required when channel dimensions change")
        if use_downsample and use_upsample:
            raise ValueError("Cannot use both upsample and downsample in one residual block")
        if in_channels % norm_groups != 0 or out_channels % norm_groups != 0:
            raise ValueError("Channel dimensions must be divisible by 32")

        if use_downsample:
            hidden_resize = nn.AvgPool2d(2, 2)
            skip_resize = nn.AvgPool2d(2, 2)
        elif use_upsample:
            hidden_resize = nn.Upsample(scale_factor=2, mode="nearest")
            skip_resize = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            hidden_resize = nn.Identity()
            skip_resize = nn.Identity()

        cond_channels = out_channels * 2 if scale_shift_conditional else out_channels
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, cond_channels))
        self.in_block = nn.Sequential(
            Fp32GroupNorm(norm_groups, in_channels, eps=norm_eps),
            nn.SiLU(),
            hidden_resize,
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.out_group_norm = Fp32GroupNorm(norm_groups, out_channels, eps=norm_eps)
        self.out_block = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(pre_outconv_dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_block = nn.Sequential(skip_resize, skip_conv or nn.Identity())
        self.scale_shift_conditional = scale_shift_conditional
        self.rescale_skip_connection = rescale_skip_connection

    def forward(self, x, conditional_embedding):
        condition = self.cond_proj(conditional_embedding).unsqueeze(-1).unsqueeze(-1)
        skip = self.skip_block(x)
        h = self.in_block(x)
        if self.scale_shift_conditional:
            scale, shift = torch.chunk(condition, 2, dim=1)
            h = self.out_block(self.out_group_norm(h) * (1 + scale) + shift)
        else:
            h = self.out_block(self.out_group_norm(h + condition))
        if self.rescale_skip_connection:
            return (skip + h) / 1.414
        return skip + h


def adm_res_block(in_channels, out_channels, dim_cond):
    skip_conv = None
    if in_channels != out_channels:
        skip_conv = nn.Conv2d(in_channels, out_channels, 1)
    return ADMResBlock(in_channels, out_channels, dim_cond, skip_conv=skip_conv)


def adm_res_downsample_block(num_channels, dim_cond):
    return ADMResBlock(num_channels, num_channels, dim_cond, use_downsample=True)


def adm_res_upsample_block(num_channels, dim_cond):
    return ADMResBlock(num_channels, num_channels, dim_cond, use_upsample=True)


class ADMAttentionBlock(nn.Module):
    def __init__(self, num_channels, dim_cond=None, num_heads=1, norm_groups=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.norm = Fp32GroupNorm(norm_groups, num_channels)
        self.query = nn.Linear(num_channels, num_channels)
        self.key = nn.Linear(num_channels, num_channels)
        self.value = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, num_channels)
        self.cond_proj = nn.Linear(dim_cond, num_channels * 2) if dim_cond is not None else None

    def _heads(self, x):
        return x.unflatten(-1, (self.num_heads, -1)).movedim(-2, 1)

    def forward(self, x, conditional_embedding=None):
        batch, channels, height, width = x.shape
        h = self.norm(x).movedim(1, -1)
        q = self._heads(self.query(h)).flatten(2, -2)
        k = self._heads(self.key(h)).flatten(2, -2)
        v = self._heads(self.value(h)).flatten(2, -2)

        if conditional_embedding is not None and self.cond_proj is not None:
            condition = self._heads(self.cond_proj(conditional_embedding))
            cond_k, cond_v = condition.chunk(2, dim=-1)
            k = torch.cat([cond_k.flatten(2, -2), k], dim=2)
            v = torch.cat([cond_v.flatten(2, -2), v], dim=2)

        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.unflatten(2, (height, width)).movedim(1, -2).flatten(-2)
        h = self.output(h).movedim(-1, 1)
        assert h.shape == (batch, channels, height, width)
        return x + h


class ADMStackModuleType(Enum):
    RESIDUAL = 0
    ATTENTION = 1
    SIMPLE = 2


class ADMStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.block_types = []

    def append_residual_block(self, block):
        self.blocks.append(block)
        self.block_types.append(ADMStackModuleType.RESIDUAL)

    def append_attention_block(self, block):
        self.blocks.append(block)
        self.block_types.append(ADMStackModuleType.ATTENTION)

    def append_simple_block(self, block):
        self.blocks.append(block)
        self.block_types.append(ADMStackModuleType.SIMPLE)

    def forward(self, x, residual_condition, attention_condition):
        for block_type, block in zip(self.block_types, self.blocks):
            if block_type == ADMStackModuleType.RESIDUAL:
                x = block(x, residual_condition)
            elif block_type == ADMStackModuleType.ATTENTION:
                x = block(x, attention_condition)
            else:
                x = block(x)
        return x


def adm_stack_res(in_channels, out_channels, dim_cond):
    stack = ADMStack()
    stack.append_residual_block(adm_res_block(in_channels, out_channels, dim_cond))
    return stack


def adm_stack_res_attn(in_channels, out_channels, dim_res_cond, dim_attn_cond=None):
    stack = adm_stack_res(in_channels, out_channels, dim_res_cond)
    stack.append_attention_block(ADMAttentionBlock(out_channels, dim_attn_cond))
    return stack


def adm_stack_res_down(num_channels, dim_cond):
    stack = ADMStack()
    stack.append_residual_block(adm_res_downsample_block(num_channels, dim_cond))
    return stack


@dataclass
class DiffusionOutput:
    prediction: Tensor
    variance_value: Tensor | None


class ADMUNet(nn.Module):
    def __init__(
        self,
        *,
        channels_per_layer,
        num_resize,
        num_res_per_layer,
        use_attention_for_layer,
        dim_res_cond,
        dim_attn_cond=None,
        embed_dim=None,
        embed_name="context",
        in_channels=3,
        out_channels=3,
        time_embed_dim=512,
        predict_variance_value=False,
    ):
        super().__init__()
        if len(channels_per_layer) != len(use_attention_for_layer):
            raise ValueError("Attention must be specified for every layer")
        if len(channels_per_layer) < num_resize:
            raise ValueError("Not enough layers for the requested number of resizes")

        self.channels_per_layer = channels_per_layer
        self.num_resize = num_resize
        self.num_res_per_layer = num_res_per_layer
        self.use_attention_for_layer = use_attention_for_layer
        self.dim_res_cond = dim_res_cond
        self.dim_attn_cond = dim_attn_cond
        self.embed_name = embed_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.predict_variance_value = predict_variance_value
        self.timestep_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, dim_res_cond),
            nn.SiLU(),
            nn.Linear(dim_res_cond, dim_res_cond),
        )
        self.res_cond_proj = None
        self.attn_cond_proj = None
        if embed_dim is not None:
            self.res_cond_proj = nn.ModuleDict(
                {embed_name: nn.Linear(embed_dim, dim_res_cond)}
            )
            if dim_attn_cond is not None:
                self.attn_cond_proj = nn.ModuleDict(
                    {
                        embed_name: nn.Sequential(
                            nn.Linear(embed_dim, dim_attn_cond * 4),
                            nn.Unflatten(-1, (4, dim_attn_cond)),
                        )
                    }
                )

        self.down, down_channels = self._create_downsampling_encoder()
        self.bottleneck = self._create_bottleneck(down_channels[-1])
        self.up = self._create_upsampling_decoder(down_channels)

    def _create_downsampling_encoder(self):
        down_channels = [self.channels_per_layer[0]]
        initial = ADMStack()
        initial.append_simple_block(nn.Conv2d(self.in_channels, self.channels_per_layer[0], 3, padding=1))
        stacks = []
        in_channels = self.channels_per_layer[0]

        for layer, out_channels in enumerate(self.channels_per_layer):
            for _ in range(self.num_res_per_layer):
                if self.use_attention_for_layer[layer]:
                    stack = adm_stack_res_attn(
                        in_channels, out_channels, self.dim_res_cond, self.dim_attn_cond
                    )
                else:
                    stack = adm_stack_res(in_channels, out_channels, self.dim_res_cond)
                stacks.append(stack)
                down_channels.append(out_channels)
                in_channels = out_channels
            if layer < self.num_resize:
                stacks.append(adm_stack_res_down(out_channels, self.dim_res_cond))
                down_channels.append(out_channels)
        return nn.ModuleList([initial] + stacks), down_channels

    def _create_bottleneck(self, channels):
        stack = ADMStack()
        stack.append_residual_block(adm_res_block(channels, channels, self.dim_res_cond))
        stack.append_attention_block(ADMAttentionBlock(channels, self.dim_attn_cond))
        stack.append_residual_block(adm_res_block(channels, channels, self.dim_res_cond))
        return stack

    def _create_upsampling_decoder(self, down_channels):
        layer_channels = list(reversed(self.channels_per_layer))
        layer_attention = list(reversed(self.use_attention_for_layer))
        stacks = []
        in_channels = layer_channels[0]

        for layer, out_channels in enumerate(layer_channels):
            for _ in range(self.num_res_per_layer + 1):
                skip_channels = down_channels.pop() if down_channels else 0
                if layer_attention[layer]:
                    stack = adm_stack_res_attn(
                        in_channels + skip_channels,
                        out_channels,
                        self.dim_res_cond,
                        self.dim_attn_cond,
                    )
                else:
                    stack = adm_stack_res(
                        in_channels + skip_channels, out_channels, self.dim_res_cond
                    )
                stacks.append(stack)
                in_channels = out_channels
            if layer < self.num_resize:
                stacks[-1].append_residual_block(adm_res_upsample_block(out_channels, self.dim_res_cond))

        output = ADMStack()
        output.append_simple_block(
            nn.Sequential(
                Fp32GroupNorm(32, layer_channels[-1]),
                nn.SiLU(),
                nn.Conv2d(layer_channels[-1], self.out_channels, 3, padding=1),
            )
        )
        return nn.ModuleList(stacks + [output])

    def _get_conditional_projections(self, timestep, conditional_inputs):
        residual_conditions = [self.timestep_encoder(timestep)]
        attention_conditions = []
        for name, condition in (conditional_inputs or {}).items():
            if self.res_cond_proj is not None and name in self.res_cond_proj:
                residual_conditions.append(self.res_cond_proj[name](condition))
            if self.attn_cond_proj is not None and name in self.attn_cond_proj:
                attention_conditions.append(self.attn_cond_proj[name](condition))
        residual_condition = torch.stack(residual_conditions).sum(dim=0)
        attention_condition = (
            torch.cat(attention_conditions, dim=1) if attention_conditions else None
        )
        return residual_condition, attention_condition

    def forward(self, x, timestep, conditional_inputs=None):
        residual_condition, attention_condition = self._get_conditional_projections(
            timestep, conditional_inputs
        )
        hidden_states = []
        h = x
        for block in self.down:
            h = block(h, residual_condition, attention_condition)
            hidden_states.append(h)
        h = self.bottleneck(h, residual_condition, attention_condition)
        for block in self.up:
            if hidden_states:
                h = torch.cat([h, hidden_states.pop()], dim=1)
            h = block(h, residual_condition, attention_condition)

        if self.predict_variance_value:
            prediction, variance = torch.chunk(h, 2, dim=1)
            return DiffusionOutput(prediction, (variance + 1) / 2)
        return DiffusionOutput(h, None)


class ADMRestoration(nn.Module):
    """Unconditional ADM denoiser exposed as an image-restoration model."""

    def __init__(
        self,
        channels_per_layer=(64, 128, 192, 256),
        num_resize=3,
        num_res_per_layer=2,
        use_attention_for_layer=(False, False, False, True),
        dim_res_cond=256,
        time_embed_dim=256,
        timestep=0,
        in_channels=3,
        out_channels=3,
    ):
        super().__init__()
        self.timestep = timestep
        self.model = ADMUNet(
            channels_per_layer=list(channels_per_layer),
            num_resize=num_resize,
            num_res_per_layer=num_res_per_layer,
            use_attention_for_layer=list(use_attention_for_layer),
            dim_res_cond=dim_res_cond,
            in_channels=in_channels,
            out_channels=out_channels,
            time_embed_dim=time_embed_dim,
            predict_variance_value=False,
        )

    def forward(self, image):
        h, w = image.shape[-2:]
        divisor = 2**self.model.num_resize
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        timestep = torch.full(
            (image.shape[0],), self.timestep, device=image.device, dtype=image.dtype
        )
        restored = self.model(image, timestep).prediction[..., :h, :w]
        return restored, None
