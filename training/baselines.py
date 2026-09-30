"""Classical baselines (PRD Section 15): climatology, persistence, random forest.

All baselines and (later) the deep models are scored on the SAME split, evaluation days, masks and
metrics (PRD 15 'Comparison Requirements'). Everything is fit on the training period only.

    python -m training.baselines --config configs/demo.yaml

Caveats: with a one-year demo training period the "climatology" is that single year smoothed, so it
is a weak floor; and all scores are against GLORYS only (see evaluation/metrics.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestRegressor

from data.catalog import BaseConfig, load_base_config
from evaluation import metrics as M
from training.data import (REPO_ROOT, Normalizer, StoreArrays, eval_positions, load_arrays,
                           split_indices)
from training.run_logger import RunLogger

PERSISTENCE_LAG_DAYS = 7      # "most recent available estimate": the analysis from a week earlier


def _doy(time: np.ndarray) -> np.ndarray:
    days = time.astype("datetime64[D]")
    return (days - days.astype("datetime64[Y]")).astype(int) + 1


def fit_climatology(y: np.ndarray, time: np.ndarray, train_idx: np.ndarray, smooth_days: int = 15) -> np.ndarray:
    """Per-cell, per-depth day-of-year mean over the TRAINING days, circularly smoothed.
    Returns (366, Z, H, W); days of year absent from the training period borrow from the nearest
    day that has data (a one-season training set gives a constant-season floor, never NaN)."""
    yt, doy = y[train_idx], _doy(time[train_idx])
    sums = np.zeros((366,) + y.shape[1:], dtype="float64")
    cnts = np.zeros_like(sums)
    for d in np.unique(doy):
        block = yt[doy == d]
        sums[d - 1] = np.nansum(block, axis=0)
        cnts[d - 1] = np.sum(~np.isnan(block), axis=0)
    have = cnts.sum(axis=(1, 2, 3)) > 0                             # doys with any training data
    if not have.all():                                              # borrow from the nearest doy that has data
        avail = np.flatnonzero(have)
        for d in np.flatnonzero(~have):
            gap = np.abs(avail - d)
            nearest = avail[np.argmin(np.minimum(gap, 366 - gap))]
            sums[d], cnts[d] = sums[nearest], cnts[nearest]
    sums = uniform_filter1d(sums, smooth_days, axis=0, mode="wrap")
    cnts = uniform_filter1d(cnts, smooth_days, axis=0, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(cnts > 1e-9, sums / cnts, np.nan)
    return clim.astype("float32")


def predict_climatology(clim: np.ndarray, time: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return clim[_doy(time[positions]) - 1]


def predict_persistence(y: np.ndarray, positions: np.ndarray, lag: int = PERSISTENCE_LAG_DAYS) -> np.ndarray:
    return y[positions - lag]


def _rf_features(arrays: StoreArrays, norm: Normalizer, t: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Raw surface fields (normalized, NaN->0) + their observed masks + lat/lon + day-of-year — no embedding.
    Indexes single pixels directly (never materializes whole fields per sample)."""
    raw = arrays.x[t, :, rows, cols]                               # (n, C)
    z = (raw - norm.x_mean[None]) / norm.x_std[None]
    obs = ~np.isnan(z)
    ang = 2 * np.pi * (_doy(arrays.time[t]) - 1) / 365.25
    return np.column_stack([
        np.where(obs, z, 0.0), obs.astype("float32"),
        (arrays.lat[rows] - 17.5) / 12.5, (arrays.lon[cols] - 75.0) / 30.0, np.sin(ang), np.cos(ang),
    ]).astype("float32")


def fit_random_forest(arrays: StoreArrays, norm: Normalizer, train_idx: np.ndarray, n_samples: int = 60_000,
                      seed: int = 42) -> RandomForestRegressor:
    rng = np.random.default_rng(seed)
    tt, rr, cc = np.nonzero(~np.isnan(arrays.x[train_idx, 0]))     # SST-valid (ocean) pixels
    pick = rng.choice(len(tt), size=min(n_samples, len(tt)), replace=False)
    t, rows, cols = train_idx[tt[pick]], rr[pick], cc[pick]
    yz = (arrays.y[t, :, rows, cols] - norm.y_mean[None]) / norm.y_std[None]      # (n, Z)
    target = np.nan_to_num(yz, nan=0.0)                            # invalid depths -> 0 (= train mean)
    rf = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(_rf_features(arrays, norm, t, rows, cols), target)
    return rf


def predict_random_forest(rf: RandomForestRegressor, arrays: StoreArrays, norm: Normalizer,
                          positions: np.ndarray) -> np.ndarray:
    out = np.full((len(positions),) + arrays.y.shape[1:], np.nan, dtype="float32")
    for i, t in enumerate(positions):
        rr, cc = np.nonzero(~np.isnan(arrays.x[t, 0]))
        feats = _rf_features(arrays, norm, np.full(len(rr), t), rr, cc)
        z = rf.predict(feats)                                      # (n_pix, Z) normalized
        out[i][:, rr, cc] = (z * norm.y_std[None] + norm.y_mean[None]).T
    return out


def evaluate(cfg: BaseConfig, stride: int = 3, buffer_days: int = 3, rf_samples: int = 60_000,
             root: Path = REPO_ROOT, log: bool = True) -> dict:
    arrays = load_arrays(cfg, root)
    idx = split_indices(arrays.time, cfg.periods, buffer_days)
    norm = Normalizer.fit(arrays.x[idx["train"]], arrays.y[idx["train"]])       # train-only statistics
    clim = fit_climatology(arrays.y, arrays.time, idx["train"])
    rf = fit_random_forest(arrays, norm, idx["train"], rf_samples, cfg.seed)
    results: dict = {"data_source": cfg.data_source, "periods": cfg.periods, "depths_m": arrays.depth.tolist(),
                     "eval_stride_days": stride, "persistence_lag_days": PERSISTENCE_LAG_DAYS,
                     "reference": "GLORYS only (GLORYS assimilates ARGO; no independent validation)", "splits": {}}
    logger = RunLogger(f"baselines-{cfg.data_source}", {"data_source": cfg.data_source, "stride": stride}) if log else None
    for split in ("validation", "test"):
        pos = eval_positions(idx[split], stride)
        true = arrays.y[pos]
        preds = {
            "climatology": predict_climatology(clim, arrays.time, pos),
            "persistence": predict_persistence(arrays.y, pos),
            "random_forest": predict_random_forest(rf, arrays, norm, pos),
        }
        rows = {name: M.per_depth(p, true, arrays.depth) for name, p in preds.items()}
        clim_rmse = np.array([r["rmse"] for r in rows["climatology"]])
        results["splits"][split] = {"n_eval_days": int(len(pos)), "models": {}}
        for name, r in rows.items():
            skill = M.skill_vs_climatology([x["rmse"] for x in r], clim_rmse)
            summ = {**M.summarize(r), "mean_skill_vs_clim": float(np.nanmean(skill))}
            results["splits"][split]["models"][name] = {"summary": summ, "per_depth": r}
            if logger:
                logger.log({f"{split}/{name}/{k}": v for k, v in summ.items()})
    if logger:
        logger.finish()
    return results


def print_table(results: dict) -> None:
    print(f"data_source={results['data_source']}  reference: {results['reference']}")
    for split, blob in results["splits"].items():
        print(f"\n[{split}] {blob['n_eval_days']} eval days")
        print(f"{'model':15s} {'mean RMSE':>10s} {'surface':>8s} {'>=300m':>8s} {'bias':>7s} {'corr':>6s} {'skill/clim':>10s}")
        for name, m in blob["models"].items():
            s = m["summary"]
            print(f"{name:15s} {s['mean_rmse']:10.3f} {s['surface_rmse']:8.3f} {s['rmse_300m_plus']:8.3f} "
                  f"{s['mean_bias']:7.3f} {s['mean_corr']:6.3f} {s['mean_skill_vs_clim']:10.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit and score the classical baselines (PRD Section 15).")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "demo.yaml"), help="override YAML; 'none' = base only")
    ap.add_argument("--stride", type=int, default=3, help="score every Nth day of each split")
    ap.add_argument("--rf-samples", type=int, default=60_000)
    args = ap.parse_args(argv)
    cfg = load_base_config(override=None if args.config == "none" else args.config)
    results = evaluate(cfg, stride=args.stride, rf_samples=args.rf_samples)
    out = REPO_ROOT / cfg.paths["runs_dir"] / f"baselines_{cfg.data_source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print_table(results)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
