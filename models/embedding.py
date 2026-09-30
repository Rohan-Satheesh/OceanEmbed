"""Satellite embedding engine (PRD Section 12): a mask-aware convolutional masked autoencoder.

The encoder turns the stacked surface fields (values + explicit observed-mask channels, PRD 12.3)
into a per-grid-cell embedding. It is pretrained self-supervised — hide random patches of the input
and reconstruct them — so it needs no GLORYS labels and learns to cope with missing data (e.g. gappy
SSS) rather than being corrupted by it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.unet import UNetTrunk


class SurfaceEncoder(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 64, base: int = 32, dropout: float = 0.0):
        super().__init__()
        self.trunk = UNetTrunk(in_ch, base, dropout)
        self.proj = nn.Conv2d(base, embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(x))                            # (B, E, H, W)


def hide_patches(x: torch.Tensor, n_fields: int, window: int, ratio: float = 0.5, patch: int = 8,
                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly hide square patches across ALL fields and days (like a real coverage gap).

    `x` is (B, window*C*2, H, W): [values (window*C), masks (window*C)]. Hidden pixels get value 0 and
    mask 0 — indistinguishable from genuinely missing data. Returns (x_hidden, hidden) where `hidden`
    is (B, 1, H, W), 1 where a patch was hidden."""
    b, _, h, w = x.shape
    ph, pw = -(-h // patch), -(-w // patch)
    tiles = (torch.rand(b, 1, ph, pw, generator=generator, device=x.device) < ratio).float()
    hidden = torch.nn.functional.interpolate(tiles, scale_factor=patch, mode="nearest")[..., :h, :w]
    n = window * n_fields
    out = x.clone()
    out[:, :n] = out[:, :n] * (1 - hidden)
    out[:, n:] = out[:, n:] * (1 - hidden)
    return out, hidden


class MaskedAutoencoder(nn.Module):
    """Encoder + light decoder that reconstructs the last day's surface fields."""

    def __init__(self, in_ch: int, n_fields: int, window: int, embed_dim: int = 64, base: int = 32):
        super().__init__()
        self.n_fields, self.window = n_fields, window
        self.encoder = SurfaceEncoder(in_ch, embed_dim, base)
        self.decoder = nn.Sequential(nn.Conv2d(embed_dim, base, 3, padding=1), nn.GELU(), nn.Conv2d(base, n_fields, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x: torch.Tensor, ratio: float = 0.5, patch: int = 8,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """MSE on pixels that were genuinely observed but hidden from the encoder."""
        hidden_x, hidden = hide_patches(x, self.n_fields, self.window, ratio, patch, generator)
        recon = self(hidden_x)
        n = self.window * self.n_fields
        last = slice(n - self.n_fields, n)                         # last day's values
        target, observed = x[:, last], x[:, n:][:, self.n_fields * (self.window - 1):]
        w = hidden * observed
        return (((recon - target) ** 2) * w).sum() / w.sum().clamp(min=1.0)
