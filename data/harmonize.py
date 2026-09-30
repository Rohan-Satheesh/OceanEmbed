"""Temporal harmonization + harmonized-store writer (PRD Sections 10.4, 10.6, 10.7, 10.8).

- Sub-daily products (CCMP 6-hourly) are aggregated to a daily mean.
- All channels are aligned to the same daily UTC 00:00 timestamp convention.
- Short gaps (<= 3 days) are forward/backward-filled; longer gaps stay missing.
- Missing/never-available data (e.g. pre-2015 SSS) is carried as an explicit `<var>_mask`
  variable rather than silently dropped, so downstream mask-aware training (Chunk 5) has it.
- Output: one NetCDF file per channel per day (PRD's `oceanembed_<variable>_<resolution>_<date>.nc`
  convention) plus an append to a single time-chunked Zarr store for efficient multi-variable
  training access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from data.catalog import Channel, Domain
from data.provenance import build_provenance

MAX_GAP_DAYS = 3


def daily_aggregate(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Aggregate sub-daily timestamps to a daily mean (PRD Section 10.4)."""
    if ds.sizes.get(time_dim, 1) <= 1:
        return ds
    return ds.resample({time_dim: "1D"}).mean(keep_attrs=True)


def align_daily_utc(ds: xr.Dataset, start: str, end: str, time_dim: str = "time") -> xr.Dataset:
    """Reindex onto a complete daily UTC-00:00 timestamp axis for [start, end], introducing NaNs
    for any day the source simply has no timestamp for (distinct from a masked/flagged bad value)."""
    full_index = xr.date_range(start, end, freq="1D", tz=None)
    return ds.reindex({time_dim: full_index})


def fill_short_gaps(ds: xr.Dataset, time_dim: str = "time", max_gap_days: int = MAX_GAP_DAYS) -> xr.Dataset:
    """Forward/backward-fill gaps of at most max_gap_days; longer gaps are left as NaN (PRD 10.4)."""
    return ds.ffill(time_dim, limit=max_gap_days).bfill(time_dim, limit=max_gap_days)


def add_mask_variable(ds: xr.Dataset, var_names: list[str]) -> xr.Dataset:
    """Attach `<var>_mask` (1 = observed, 0 = missing/gap-filled-beyond-window) per variable,
    computed from NaN status after gap-filling (PRD Sections 10.6, 10.8, 12.3)."""
    out = ds.copy()
    for var in var_names:
        if var in out:
            out[f"{var}_mask"] = (~out[var].isnull()).astype("int8")
    return out


def harmonize_channel(
    raw: xr.Dataset,
    channel: Channel,
    start: str,
    end: str,
) -> xr.Dataset:
    """Full temporal-harmonization sequence for one channel's regridded, masked dataset."""
    ds = daily_aggregate(raw)
    ds = align_daily_utc(ds, start, end)
    ds = fill_short_gaps(ds)
    ds = add_mask_variable(ds, channel.variables)

    provenance = build_provenance(
        source=channel.source,
        dataset_id=channel.dataset_id or channel.short_name or "unknown",
        processing_steps=channel.processing_steps + ["daily_aggregate", "align_daily_utc", "fill_short_gaps"],
    )
    ds.attrs.update(provenance)
    ds.attrs["units"] = channel.units
    return ds


def write_netcdf(ds: xr.Dataset, channel: Channel, domain: Domain, out_dir: Path) -> list[Path]:
    """Write one NetCDF file per day per PRD's oceanembed_<variable>_<resolution>_<date>.nc
    convention, under data/processed/netcdf/<variable>/<year>/."""
    resolution_tag = f"{domain.resolution_deg}deg"
    written = []
    for t in ds["time"].values:
        date_str = np.datetime_as_string(t, unit="D")
        year = date_str[:4]
        day_ds = ds.sel(time=[t])
        target_dir = out_dir / "netcdf" / channel.name / year
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"oceanembed_{channel.name}_{resolution_tag}_{date_str}.nc"
        day_ds.to_netcdf(path)
        written.append(path)
    return written


def append_zarr(ds: xr.Dataset, channel: Channel, out_dir: Path) -> Path:
    """Append this channel's harmonized daily data to the shared, time-chunked Zarr store.

    Each channel lives in its own Zarr group (named after the channel) within one store, so
    channels with different variable sets can each be appended along `time` independently
    without xarray's append-mode requiring identical variables across groups.
    """
    zarr_path = out_dir / "zarr" / "oceanembed.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    group_exists = zarr_path.exists() and (zarr_path / channel.name).exists()
    if group_exists:
        # Idempotent re-runs: skip days already stored rather than appending duplicate time steps.
        # (To overwrite existing days, delete the group first.)
        with xr.open_zarr(zarr_path, group=channel.name, consolidated=False) as existing:
            stored = existing["time"].values
        already = np.isin(ds["time"].values, stored)
        if already.any():
            print(f"    [skip] {int(already.sum())} day(s) already in zarr group '{channel.name}'")
            ds = ds.isel(time=~already)
        if ds.sizes["time"] == 0:
            return zarr_path
    chunked = ds.chunk({"time": 30})

    if group_exists:
        chunked.to_zarr(zarr_path, mode="a", append_dim="time", group=channel.name)
    else:
        chunked.to_zarr(zarr_path, mode="a", group=channel.name)
    return zarr_path
