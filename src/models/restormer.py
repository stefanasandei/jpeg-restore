"""Restormer architecture for image restoration.

Architecture code follows the official Restormer implementation:
https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
"""

import numbers

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        batch, heads, channels, pixels = q.shape
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn.view(batch, heads, channels, channels)
        attn = (attn * self.temperature).softmax(dim=-1).flatten(0, 1)
        out = torch.bmm(attn, v).view(batch, heads, channels, pixels)
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        dual_pixel_task=False,
    ):
        super().__init__()
        block = lambda features, head: TransformerBlock(
            features, head, ffn_expansion_factor, bias, layer_norm_type
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[block(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[block(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[block(dim * 2, heads[0]) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[block(dim * 2, heads[0]) for _ in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

    def forward(self, image):
        enc1 = self.encoder_level1(self.patch_embed(image))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))

        dec3 = self.reduce_chan_level3(torch.cat([self.up4_3(latent), enc3], dim=1))
        dec3 = self.decoder_level3(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([self.up3_2(dec3), enc2], dim=1))
        dec2 = self.decoder_level2(dec2)
        dec1 = self.decoder_level1(torch.cat([self.up2_1(dec2), enc1], dim=1))
        dec1 = self.refinement(dec1)

        if self.dual_pixel_task:
            return self.output(dec1 + self.skip_conv(enc1))
        return self.output(dec1) + image


class RestormerRestoration(Restormer):
    """Restormer with shape padding and the repository's output contract."""

    def forward(self, image):
        h, w = image.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        restored = super().forward(padded)[..., :h, :w]
        return restored, None
