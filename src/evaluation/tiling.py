"""A single grid, equal-weight blend, and seed schedule for matched inference."""

import torch


def tile_starts(length, tile, overlap):
    size = min(length, tile)
    starts = list(range(0, length - size + 1, tile - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def stitch(height, width, device, run_tile, tile=512, overlap=64, clip=True):
    if tile < 1 or not 0 <= overlap < tile:
        raise ValueError("tiling requires 0 <= overlap < tile")
    output = torch.zeros(1, 3, height, width, device=device, dtype=torch.float32)
    weights = torch.zeros(1, 1, height, width, device=device, dtype=torch.float32)
    count = 0
    for top in tile_starts(height, tile, overlap):
        for left in tile_starts(width, tile, overlap):
            h, w = min(tile, height), min(tile, width)
            restored = run_tile(top, left, h, w, count)
            if tuple(restored.shape) != (1, 3, h, w):
                raise ValueError("tile restoration returned an unexpected shape")
            if not torch.isfinite(restored).all():
                raise FloatingPointError("tile restoration returned non-finite RGB")
            restored = restored.float()
            if clip:
                restored = restored.clamp(0, 1)
            output[..., top:top+h, left:left+w].add_(restored)
            weights[..., top:top+h, left:left+w].add_(1)
            count += 1
    return output.div_(weights), count
