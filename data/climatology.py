"""Per-cell, per-depth climatology (PRD Sections 8.3, 8.4) to support anomaly-based training and
evaluation framing (Chunk 6). Computed only over the configured training period
(`configs/base.yaml`'s `periods.train`) — never validation/test — so no leakage of held-out
statistics into the anomaly baseline.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def _select_training_period(ds: xr.Dataset, start: str, end: str, time_dim: str = "time") -> xr.Dataset:
    return ds.sel({time_dim: slice(start, end)})


def compute_daily_climatology(
    ds: xr.Dataset, var: str, start: str, end: str, time_dim: str = "time"
) -> xr.DataArray:
    """Per-cell (and per-depth, if present) mean of `var` grouped by day-of-year, over
    [start, end] only."""
    training = _select_training_period(ds, start, end, time_dim)
    clim = training[var].groupby(f"{time_dim}.dayofyear").mean(dim=time_dim, skipna=True)
    clim.name = f"{var}_climatology_daily"
    return clim


def compute_monthly_climatology(
    ds: xr.Dataset, var: str, start: str, end: str, time_dim: str = "time"
) -> xr.DataArray:
    """Per-cell (and per-depth, if present) mean of `var` grouped by calendar month, over
    [start, end] only."""
    training = _select_training_period(ds, start, end, time_dim)
    clim = training[var].groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)
    clim.name = f"{var}_climatology_monthly"
    return clim


def write_climatology(clim: xr.DataArray, out_dir: Path) -> Path:
    clim_dir = out_dir / "climatology"
    clim_dir.mkdir(parents=True, exist_ok=True)
    path = clim_dir / f"{clim.name}.nc"
    clim.to_netcdf(path)
    return path
