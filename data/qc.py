"""Data-quality checks (PRD Section 10.2, Chunk 2 deliverable): missing-day report,
spatial-coverage report, per-channel physical-range sanity checks.

Reports are written as JSON under data/processed/qc/<run_id>/ — machine-readable for later
chunks, human-scannable for the team.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from data.catalog import Channel

# Approximate physical bounds per variable, for sanity-checking regridded/harmonized values —
# not a scientific QC standard, just a sanity net to catch unit errors or bad fill values.
RANGE_BOUNDS: dict[str, tuple[float, float]] = {
    "analysed_sst": (-2.0, 40.0),   # degC
    "sos": (0.0, 45.0),              # psu
    "sla": (-3.0, 3.0),               # m
    "adt": (-3.0, 3.0),                # m
    "u": (-5.0, 5.0),                   # m/s (currents)
    "v": (-5.0, 5.0),
    "uwnd": (-60.0, 60.0),               # m/s (winds)
    "vwnd": (-60.0, 60.0),
    "thetao": (-2.0, 35.0),           # degC (GLORYS target, PRD Section 8.3)
    "TEMP": (-2.0, 40.0),               # degC (ARGO gridded VAM)
    "SAL": (0.0, 45.0),                  # psu (ARGO gridded VAM)
}


def mask_invalid_values(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Replace physically-impossible values (fill values, land-extended garbage, etc.) with
    NaN *before* regridding (PRD Section 10.2, which precedes 10.3 Spatial Harmonization).

    This matters beyond flagging: conservative regridding area-weight-averages every source
    pixel in a target cell's footprint, NaN-excluded — a raw fill/sentinel value that survives
    as a normal (non-NaN) number gets blended into real neighbouring ocean values and corrupts
    coastal cells. Masking out-of-bounds values first keeps them out of that average.
    """
    out = ds.copy()
    for var in variables:
        bounds = RANGE_BOUNDS.get(var)
        if bounds is None or var not in out:
            continue
        lo, hi = bounds
        out[var] = out[var].where((out[var] >= lo) & (out[var] <= hi))
    return out


def missing_day_report(ds: xr.Dataset, var: str, time_dim: str = "time") -> dict[str, Any]:
    """Days where every cell of `var` is NaN (i.e. no gap-fill could recover it)."""
    all_nan_per_day = ds[var].isnull().all(dim=[d for d in ds[var].dims if d != time_dim])
    missing_days = ds[time_dim].where(all_nan_per_day, drop=True).values
    return {
        "variable": var,
        "total_days": int(ds.sizes[time_dim]),
        "missing_days": int(len(missing_days)),
        "missing_day_list": [np.datetime_as_string(d, unit="D") for d in missing_days],
    }


def coverage_report(ds: xr.Dataset, var: str, time_dim: str = "time") -> dict[str, Any]:
    """Per-day fraction of non-NaN spatial cells for `var`."""
    spatial_dims = [d for d in ds[var].dims if d != time_dim]
    valid_fraction = 1.0 - ds[var].isnull().mean(dim=spatial_dims)
    return {
        "variable": var,
        "mean_coverage_fraction": float(valid_fraction.mean().item()),
        "min_coverage_fraction": float(valid_fraction.min().item()),
        "days_below_50pct_coverage": int((valid_fraction < 0.5).sum().item()),
    }


def range_check(ds: xr.Dataset, var: str) -> dict[str, Any]:
    """Fraction of non-NaN values for `var` falling outside its configured physical bounds."""
    bounds = RANGE_BOUNDS.get(var)
    if bounds is None:
        return {"variable": var, "checked": False, "reason": "no configured bounds"}
    lo, hi = bounds
    values = ds[var]
    valid = values.notnull()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {"variable": var, "checked": True, "bounds": bounds, "n_valid": 0, "n_out_of_range": 0}
    out_of_range = ((values < lo) | (values > hi)) & valid
    return {
        "variable": var,
        "checked": True,
        "bounds": bounds,
        "n_valid": n_valid,
        "n_out_of_range": int(out_of_range.sum().item()),
        "pass": bool(out_of_range.sum().item() == 0),
    }


EDGE_RATIO_BOUNDS = (0.85, 1.15)   # a healthy edge/inner-neighbour ratio is ~1.00; a partly-covered edge gives 0.25-0.67
EDGE_MIN_PAIRS = 5                 # ocean cells needed on a side before its ratio means anything


def edge_ratio_report(ds: xr.Dataset, var: str) -> dict[str, Any]:
    """Median ratio of each boundary row/column to the row/column just inside it (time-mean, surface level).

    Catches the partial-coverage regrid bug (edge cells holding a fraction of the true value) without needing a
    reference: real fields do not change by 15%+ across one 0.25 deg step along the whole edge. Only run on strictly
    positive fields (SST, SSS, ADT, temperature) where a ratio is meaningful."""
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=0)
    other = [d for d in da.dims if d not in ("lat", "lon")]
    field = (da.mean(other, skipna=True) if other else da).transpose("lat", "lon").values
    if not np.isfinite(field).any() or np.nanmin(field) <= 0:
        return {"variable": var, "checked": False, "reason": "field is not strictly positive"}
    sides = {"south": (field[0, :], field[1, :]), "north": (field[-1, :], field[-2, :]),
             "west": (field[:, 0], field[:, 1]), "east": (field[:, -1], field[:, -2])}
    ratios: dict[str, float | None] = {}
    for name, (edge, inner) in sides.items():
        ok = np.isfinite(edge) & np.isfinite(inner)
        ratios[name] = float(np.median(edge[ok] / inner[ok])) if ok.sum() >= EDGE_MIN_PAIRS else None
    lo, hi = EDGE_RATIO_BOUNDS
    suspect = [n for n, r in ratios.items() if r is not None and not lo <= r <= hi]
    return {"variable": var, "checked": True, "median_edge_over_inner": ratios, "suspect_sides": suspect,
            "pass": not suspect}


def run_channel_qc(ds: xr.Dataset, channel: Channel) -> dict[str, Any]:
    report: dict[str, Any] = {"channel": channel.name, "variables": {}}
    for var in channel.variables:
        if var not in ds:
            report["variables"][var] = {"error": "variable not present in harmonized dataset"}
            continue
        report["variables"][var] = {
            "missing_days": missing_day_report(ds, var),
            "coverage": coverage_report(ds, var),
            "range_check": range_check(ds, var),
            "edge_check": edge_ratio_report(ds, var),
        }
        edge = report["variables"][var]["edge_check"]
        if edge.get("checked") and not edge["pass"]:
            print(f"    [warn] {channel.name}.{var}: boundary cells differ from their inner neighbours on "
                  f"{edge['suspect_sides']} (ratios {edge['median_edge_over_inner']}) — likely partly-covered edge cells")
    return report


def write_qc_report(report: dict[str, Any], run_id: str, out_dir: Path) -> Path:
    qc_dir = out_dir / "qc" / run_id
    qc_dir.mkdir(parents=True, exist_ok=True)
    path = qc_dir / f"{report['channel']}_qc.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
