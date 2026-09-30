"""Depth-conditioned reconstruction model, OceanEmbed (PRD Section 13).

  surface fields (+masks) --encoder--> per-cell embedding
  [embedding | raw surface skip | day-of-year | lat/lon] --fusion convs--> features
  queried depth --sinusoidal encoding--> MLP --> FiLM (gamma, beta)   (PRD 13.7 FiLM candidate)
  features modulated per depth --1x1 head--> temperature at that depth  (all depths in one pass)

Dropout is kept in the fusion block so MC dropout at inference gives a per-pixel, per-depth
mean/std (PRD Section 19).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from models.embedding import SurfaceEncoder
from models.unet import _norm


def depth_encoding(depths_m: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """Sinusoidal encoding of log-scaled depth (dense near the surface where profiles change fastest)."""
    d = torch.log1p(depths_m) / math.log1p(1000.0)                  # ~[0, 1]
    freqs = (2.0 ** torch.arange(n_freq, dtype=d.dtype, device=d.device)) * math.pi
    ang = d[:, None] * freqs[None]
    return torch.cat([d[:, None], torch.sin(ang), torch.cos(ang)], dim=1)   # (Z, 1 + 2*n_freq)


class OceanEmbedModel(nn.Module):
    def __init__(self, in_ch: int, depths_m: list[float], lat: np.ndarray, lon: np.ndarray, embed_dim: int = 64,
                 base: int = 32, film_ch: int = 24, dropout: float = 0.1, use_encoder: bool = True):
        super().__init__()
        self.use_encoder = use_encoder
        self.encoder = SurfaceEncoder(in_ch, embed_dim, base) if use_encoder else None
        fuse_in = (embed_dim if use_encoder else 0) + in_ch + 2 + 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in, 64, 3, padding=1), _norm(64), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, film_ch, 3, padding=1), _norm(film_ch), nn.GELU(), nn.Dropout2d(dropout),
        )
        self.register_buffer("depths", torch.as_tensor(depths_m, dtype=torch.float32))
        self.film = nn.Sequential(nn.Linear(1 + 2 * 8, 64), nn.GELU(), nn.Linear(64, 2 * film_ch))
        self.head_w = nn.Parameter(torch.randn(film_ch) / math.sqrt(film_ch))
        self.head_b = nn.Parameter(torch.zeros(len(depths_m)))          # per-depth offset
        latn = (torch.as_tensor(lat, dtype=torch.float32) - 17.5) / 12.5
        lonn = (torch.as_tensor(lon, dtype=torch.float32) - 75.0) / 30.0
        self.register_buffer("latlon", torch.stack(torch.meshgrid(latn, lonn, indexing="ij"))[None])   # (1,2,H,W)
        self.film_ch = film_ch

    def forward(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        parts = [x, doy[:, :, None, None].expand(b, 2, h, w), self.latlon.expand(b, -1, -1, -1)]
        if self.use_encoder:
            parts.insert(0, self.encoder(x))
        f = self.fusion(torch.cat(parts, dim=1))                    # (B, F, H, W)
        gamma, beta = self.film(depth_encoding(self.depths)).chunk(2, dim=1)    # (Z, F) each
        m = f[:, None] * (1 + gamma[None, :, :, None, None]) + beta[None, :, :, None, None]
        out = torch.einsum("bzfhw,f->bzhw", torch.nn.functional.gelu(m), self.head_w)
        return out + self.head_b[None, :, None, None]

    @torch.no_grad()
    def predict_mc(self, x: torch.Tensor, doy: torch.Tensor, n: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-dropout mean and std (normalized units). Only dropout layers are switched to train mode."""
        was_training = self.training
        self.eval()
        for mod in self.modules():
            if isinstance(mod, nn.Dropout2d):
                mod.train()
        draws = torch.stack([self(x, doy) for _ in range(n)])
        self.train(was_training)
        return draws.mean(0), draws.std(0)
