"""U-Net trunk + the plain U-Net baseline (PRD 13.5, 15 baseline 4: 'no embedding stage').

GroupNorm (not BatchNorm) is used throughout so train/eval behave identically at small batch sizes
and MC-dropout inference doesn't disturb normalization statistics. Inputs of any H, W are
reflect-padded up to a multiple of 8 internally and cropped back.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetTrunk(nn.Module):
    """3-level encoder/decoder returning `base` feature channels at full resolution."""

    def __init__(self, in_ch: int, base: int = 32, dropout: float = 0.0):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(in_ch, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.bott = ConvBlock(base * 4, base * 8, dropout)
        self.d3 = ConvBlock(base * 8 + base * 4, base * 4, dropout)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2, dropout)
        self.d1 = ConvBlock(base * 2 + base, base, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d3 = self.d3(torch.cat([up(b, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return d1[..., :h, :w]


class UNet(nn.Module):
    """Plain U-Net: surface fields (+masks) -> all `n_depths` levels directly, no embedding stage."""

    def __init__(self, in_ch: int, n_depths: int = 15, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.trunk = UNetTrunk(in_ch, base, dropout)
        self.head = nn.Conv2d(base, n_depths, 1)

    def forward(self, x: torch.Tensor, *_unused) -> torch.Tensor:
        return self.head(self.trunk(x))
