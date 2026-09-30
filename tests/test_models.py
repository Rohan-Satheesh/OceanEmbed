import numpy as np
import pytest
import torch

from models.embedding import MaskedAutoencoder, SurfaceEncoder, hide_patches
from models.reconstruction import OceanEmbedModel, depth_encoding
from models.unet import UNet, UNetTrunk

C, WIN = 7, 2
IN_CH = WIN * C * 2
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
H, W = 24, 40                                    # small grid for speed; padding path tested separately


def _batch(b=2, h=H, w=W, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(b, WIN * C, h, w, generator=g)
    masks = (torch.rand(b, WIN * C, h, w, generator=g) > 0.2).float()
    return torch.cat([vals * masks, masks], 1), torch.randn(b, 2, generator=g)


def _model(dropout=0.1, use_encoder=True, h=H, w=W):
    return OceanEmbedModel(IN_CH, DEPTHS, np.linspace(5, 30, h), np.linspace(45, 105, w), embed_dim=16, base=8,
                           film_ch=8, dropout=dropout, use_encoder=use_encoder)


def test_unet_handles_real_grid_size_via_padding():
    x, _ = _batch(1, 101, 241)                                   # 101x241 is not a multiple of 8
    y = UNet(IN_CH, 15, base=8)(x)
    assert y.shape == (1, 15, 101, 241) and torch.isfinite(y).all()
    assert UNetTrunk(IN_CH, 8)(x).shape == (1, 8, 101, 241)


def test_encoder_embedding_shape():
    x, _ = _batch()
    assert SurfaceEncoder(IN_CH, embed_dim=16, base=8)(x).shape == (2, 16, H, W)


def test_hide_patches_zeroes_values_and_masks_together():
    x, _ = _batch(4, 32, 32)
    xh, hidden = hide_patches(x, C, WIN, ratio=0.5, patch=8, generator=torch.Generator().manual_seed(1))
    n = WIN * C
    h = hidden.bool().expand_as(xh[:, :n])
    assert (xh[:, :n][h] == 0).all() and (xh[:, n:][h] == 0).all()             # hidden == missing
    assert torch.equal(xh[:, :n][~h], x[:, :n][~h])                            # visible pixels untouched
    assert 0.2 < hidden.mean().item() < 0.8


def test_mae_pretraining_reduces_loss():
    torch.manual_seed(0)
    x, _ = _batch(4, 32, 32)
    x[:, :WIN * C] = x[:, :WIN * C].mean(dim=(-1, -2), keepdim=True) + 0 * x[:, :WIN * C]   # smooth, learnable fields
    x[:, :WIN * C] *= x[:, WIN * C:]
    mae = MaskedAutoencoder(IN_CH, C, WIN, embed_dim=16, base=8)
    opt = torch.optim.AdamW(mae.parameters(), 3e-3)
    first = mae.loss(x, generator=torch.Generator().manual_seed(0)).item()
    for i in range(60):
        opt.zero_grad()
        l = mae.loss(x, generator=torch.Generator().manual_seed(i))
        l.backward()
        opt.step()
    assert l.item() < 0.5 * first and np.isfinite(l.item())


def test_reconstruction_shape_depth_conditioning_and_gradients():
    m = _model()
    x, doy = _batch()
    y = m(x, doy)
    assert y.shape == (2, 15, H, W) and torch.isfinite(y).all()
    assert (y[:, 0] - y[:, -1]).abs().mean() > 1e-4                            # output depends on queried depth
    y.square().mean().backward()
    assert m.encoder.proj.weight.grad is not None and m.film[0].weight.grad is not None


def test_depth_encoding_distinct_per_depth():
    e = depth_encoding(torch.tensor(DEPTHS, dtype=torch.float32))
    assert e.shape == (15, 17) and torch.unique(e.round(decimals=4), dim=0).shape[0] == 15


def test_mc_dropout_gives_spread_only_when_dropout_on():
    x, doy = _batch()
    mean, std = _model(dropout=0.3).predict_mc(x, doy, n=6)
    assert mean.shape == (2, 15, H, W) and std.mean() > 0
    _, std0 = _model(dropout=0.0).predict_mc(x, doy, n=4)
    assert std0.max() < 1e-6
    m = _model(); m.train()
    m.predict_mc(x, doy, n=2)
    assert m.training                                                         # mode restored


def test_without_encoder_variant_and_all_missing_input_is_finite():
    x, doy = _batch()
    x[:, :] = 0                                                               # every field missing, all masks 0
    assert torch.isfinite(_model(use_encoder=False)(x, doy)).all()
    assert torch.isfinite(_model()(x, doy)).all()
