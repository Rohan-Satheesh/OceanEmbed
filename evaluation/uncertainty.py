"""Uncertainty calibration checks (PRD Section 19): does the MC-dropout spread track the real error,
and does a stated prediction interval cover the truth about as often as claimed?

Scored against GLORYS only (see evaluation/metrics.py for the reference caveat).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

Z90, Z95 = 1.6449, 1.96


def calibration(mean: np.ndarray, std: np.ndarray, true: np.ndarray, n_bins: int = 10,
                max_samples: int = 400_000, seed: int = 0) -> dict:
    """Calibration summary over all valid cells of (T, Z, H, W) arrays.

    - coverage_90 / coverage_95: fraction of truths inside mean +/- z*std (ideal 0.90 / 0.95)
    - spread_error_spearman: rank correlation of predicted std with |error| (>0 = spread is informative)
    - reliability: predicted-std quantile bins with the observed RMSE in each (ideal: rmse ~ mean_std)
    - sharpness: mean predicted std (a model can be 'calibrated' but useless if this is huge)
    """
    ok = ~np.isnan(true) & ~np.isnan(mean) & ~np.isnan(std)
    m, s, t = mean[ok].astype("float64"), std[ok].astype("float64"), true[ok].astype("float64")
    if len(m) > max_samples:
        pick = np.random.default_rng(seed).choice(len(m), max_samples, replace=False)
        m, s, t = m[pick], s[pick], t[pick]
    err = np.abs(m - t)
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    reliability = [
        {"mean_std": float(s[which == b].mean()), "rmse": float(np.sqrt(np.mean(err[which == b] ** 2))),
         "n": int((which == b).sum())}
        for b in range(n_bins) if (which == b).any()
    ]
    rho = spearmanr(s, err).statistic if s.std() > 0 else float("nan")
    return {
        "n": int(len(m)), "coverage_90": float(np.mean(err <= Z90 * s)), "coverage_95": float(np.mean(err <= Z95 * s)),
        "spread_error_spearman": float(rho), "sharpness": float(s.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "reliability": reliability,
    }
