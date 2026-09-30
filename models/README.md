# /models

Architectures (PRD Sections 12-13, Chunks 5-6). All use GroupNorm so train/eval behave identically at small batch
sizes and MC-dropout inference does not disturb normalization.

- `unet.py` — `UNetTrunk` (3-level, any H x W via reflect padding) and `UNet`, the plain "no embedding" baseline
  that predicts all 15 depths directly.
- `embedding.py` — `SurfaceEncoder` (per-cell embedding), `hide_patches` (hides square patches across all fields and
  days by zeroing values *and* masks, i.e. indistinguishable from a real coverage gap) and `MaskedAutoencoder`
  (self-supervised pretraining: reconstruct hidden-but-observed pixels, so no GLORYS labels are needed).
- `reconstruction.py` — `OceanEmbedModel`: encoder embedding + raw surface skip + day-of-year + lat/lon -> fusion
  convs -> FiLM modulation from a sinusoidal depth encoding -> temperature at every queried depth in one pass.
  `predict_mc` gives MC-dropout mean/std (PRD Section 19).

Tests: `tests/test_models.py`.
