"""Masked evaluation metrics (PRD Section 18): RMSE / MAE / bias / correlation / R2 per depth,
stratified by region (Bay of Bengal vs Arabian Sea) and season (monsoon vs non-monsoon), plus a
skill score against the climatology baseline.

All functions take predictions and truth as (T, Z, H, W) arrays in physical units (deg C) and score
only cells where the truth is valid (NaN in `true` = land / below bathymetry / not observed).

Caveat carried with every result (PRD 9.3, 18.3): GLORYS is both the training target and the
evaluation reference here, and GLORYS assimilates ARGO. Scores are against GLORYS only.
"""

from __future__ import annotations

import numpy as np

REGIONS = {                       # (lat_min, lat_max, lon_min, lon_max)
    "bay_of_bengal": (5.0, 23.0, 80.0, 100.0),
    "arabian_sea": (5.0, 25.0, 45.0, 77.0),
}
MONSOON_MONTHS = (6, 7, 8, 9)     # JJAS


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    ok = ~np.isnan(true) & ~np.isnan(pred)
    n = int(ok.sum())
    if n == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan, r2=np.nan)
    p, t = pred[ok].astype("float64"), true[ok].astype("float64")
    err = p - t
    sst = float(np.sum((t - t.mean()) ** 2))
    return dict(
        n=n, rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)),
        corr=float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else np.nan,
        r2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
    )


def per_depth(pred: np.ndarray, true: np.ndarray, depths: np.ndarray) -> list[dict]:
    """One metrics row per standard depth (each of the 15 reported separately — PRD 18.2)."""
    return [{"depth_m": float(z), **_metrics(pred[:, k], true[:, k])} for k, z in enumerate(depths)]


def region_mask(lat: np.ndarray, lon: np.ndarray, region: str) -> np.ndarray:
    la0, la1, lo0, lo1 = REGIONS[region]
    return ((lat[:, None] >= la0) & (lat[:, None] <= la1) & (lon[None, :] >= lo0) & (lon[None, :] <= lo1))


def season_mask(time: np.ndarray, season: str) -> np.ndarray:
    months = time.astype("datetime64[M]").astype(int) % 12 + 1
    monsoon = np.isin(months, MONSOON_MONTHS)
    return monsoon if season == "monsoon" else ~monsoon


def stratified(pred: np.ndarray, true: np.ndarray, time: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               depths: np.ndarray) -> dict[str, list[dict]]:
    """Per-depth metrics for every region x season combination (PRD 18.4 a, b)."""
    out: dict[str, list[dict]] = {}
    for region in REGIONS:
        rm = region_mask(lat, lon, region)
        for season in ("monsoon", "non_monsoon"):
            sm = season_mask(time, season)
            if not sm.any():
                continue
            p, t = pred[sm].copy(), true[sm].copy()
            t[:, :, ~rm] = np.nan
            out[f"{region}/{season}"] = per_depth(p, t, depths)
    return out


def skill_vs_climatology(rmse_model: np.ndarray, rmse_clim: np.ndarray) -> np.ndarray:
    """MSE-based skill score per depth: 1 = perfect, 0 = no better than climatology, <0 = worse."""
    return 1.0 - (np.asarray(rmse_model) ** 2) / (np.asarray(rmse_clim) ** 2)


def summarize(rows: list[dict]) -> dict[str, float]:
    """Depth-averaged RMSE / bias / correlation for a one-line comparison table."""
    rm = np.array([r["rmse"] for r in rows], dtype=float)
    return dict(mean_rmse=float(np.nanmean(rm)), surface_rmse=float(rm[0]), rmse_300m_plus=float(np.nanmean(rm[-5:])),
                mean_bias=float(np.nanmean([r["bias"] for r in rows])),
                mean_corr=float(np.nanmean([r["corr"] for r in rows])))
