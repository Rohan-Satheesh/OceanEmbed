"""Chunk 2 orchestrator: download -> regrid -> mask -> harmonize -> QC (PRD Section 35 Chunk 2).

Usage:
    python -m data.pipeline --start 2024-06-01 --end 2024-06-07 --channels sst,ssh
    python -m data.pipeline --start 2024-06-01 --end 2024-06-07  # all channels

Downloads land into data/raw/<channel>/, harmonized NetCDF + Zarr land into data/processed/,
QC reports land into data/processed/qc/<run_id>/. All three are git-ignored.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

from data.bathymetry import build_land_sea_mask, fetch_gebco
from data.catalog import Channel, Domain, load_base_config, load_catalog
from data.harmonize import append_zarr, harmonize_channel, write_netcdf
from data.qc import mask_invalid_values, run_channel_qc, write_qc_report
from data.regrid import regrid_dataset
from data.sources import copernicus, earthdata

UNIT_CONVERSIONS = {
    "kelvin_to_celsius": lambda da: da - 273.15,
}


def _normalize_time_dtype(ds: xr.Dataset) -> xr.Dataset:
    """Some PO.DAAC products (e.g. OSCAR, on a non-standard calendar) decode `time` to
    dtype=object — cftime.datetime instances — rather than numpy datetime64.
    harmonize.align_daily_utc() reindexes against an xr.date_range() (datetime64) index, which
    silently matches nothing (not an error, just 100% NaN) against a cftime/object time
    coordinate. Normalize eagerly here, before that mismatch can happen."""
    if "time" not in ds.coords or np.issubdtype(ds["time"].dtype, np.datetime64):
        return ds
    values = ds["time"].values
    if len(values) and isinstance(values[0], cftime.datetime):
        # unsafe=True: silences xarray's calendar-mismatch warning. Verified intentional here —
        # this preserves the same Y-M-D labels (see test_normalize_time_dtype_converts_cftime_datetimes),
        # which is what daily cross-source alignment needs; it is not doing elapsed-time arithmetic
        # across a range where Julian/Gregorian would actually diverge.
        return ds.assign_coords(
            time=xr.CFTimeIndex(values).to_datetimeindex(unsafe=True, time_unit="ns")
        )
    return ds.assign_coords(time=pd.to_datetime(values))


def download_channel(channel: Channel, start: str, end: str, domain: Domain, raw_dir: Path) -> xr.Dataset:
    out_dir = raw_dir / channel.name
    if channel.source == "copernicus":
        path = copernicus.download(channel, start, end, domain, out_dir)
        return _normalize_time_dtype(xr.open_dataset(path))
    if channel.source == "earthdata":
        paths = earthdata.download(channel, start, end, domain, out_dir)
        return _normalize_time_dtype(xr.open_mfdataset([str(p) for p in paths], combine="by_coords"))
    raise ValueError(f"unknown source '{channel.source}' for channel '{channel.name}'")


def apply_unit_conversion(ds: xr.Dataset, channel: Channel) -> xr.Dataset:
    if not channel.unit_conversion:
        return ds
    convert = UNIT_CONVERSIONS[channel.unit_conversion]
    out = ds.copy()
    for var in channel.variables:
        if var in out:
            out[var] = convert(out[var])
    return out


def apply_land_mask(ds: xr.Dataset, land_sea: xr.Dataset, channel: Channel) -> xr.Dataset:
    out = ds.copy()
    sea = land_sea["sea_mask"].reindex_like(out, method="nearest")
    # Mask every gridded data variable, not just channel.variables: derived fields (e.g. winds'
    # `ws`, `nobs`) would otherwise stay unmasked over land while uwnd/vwnd are NaN there.
    for var in list(out.data_vars):
        if {"lat", "lon"} <= set(out[var].dims) and not var.endswith("_mask"):
            out[var] = out[var].where(sea)
    return out


def process_channel(
    channel: Channel,
    start: str,
    end: str,
    domain: Domain,
    land_sea: xr.Dataset,
    raw_dir: Path,
    processed_dir: Path,
    weights_cache_dir: Path,
    run_id: str,
) -> None:
    print(f"\n=== {channel.name} ({channel.label}) ===")

    print("  downloading...")
    raw = download_channel(channel, start, end, domain, raw_dir)

    print("  unit conversion + range-based QC mask (PRD 10.2, before regridding)...")
    converted = apply_unit_conversion(raw, channel)
    cleaned = mask_invalid_values(converted, channel.variables)

    print("  regridding (conservative)...")
    regridded = regrid_dataset(cleaned, domain, weights_cache_dir)

    print("  applying land mask...")
    masked = apply_land_mask(regridded, land_sea, channel)

    print("  harmonizing (temporal alignment, gap-fill, mask variable)...")
    harmonized = harmonize_channel(masked, channel, start, end)
    # Earthdata channels come back as dask-backed (xr.open_mfdataset); materialize now so
    # downstream boolean masking in qc.py doesn't hit dask's disallowed boolean-array indexing.
    harmonized = harmonized.load()

    print("  writing NetCDF + Zarr...")
    netcdf_paths = write_netcdf(harmonized, channel, domain, processed_dir)
    zarr_path = append_zarr(harmonized, channel, processed_dir)
    print(f"    {len(netcdf_paths)} NetCDF file(s), zarr store at {zarr_path}")

    print("  running QC...")
    report = run_channel_qc(harmonized, channel)
    qc_path = write_qc_report(report, run_id, processed_dir)
    print(f"    QC report: {qc_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OceanEmbed Chunk 2 data acquisition + harmonization pipeline")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--channels", default=None, help="comma-separated channel names (default: all)")
    parser.add_argument("--out", default=None, help="override paths.processed_dir from configs/base.yaml")
    args = parser.parse_args(argv)

    base = load_base_config()
    catalog = load_catalog()
    domain = base.domain

    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / base.paths["raw_dir"]
    processed_dir = Path(args.out) if args.out else repo_root / base.paths["processed_dir"]
    weights_cache_dir = processed_dir / "regrid_weights"
    bathymetry_dir = processed_dir / "bathymetry"

    channel_names = args.channels.split(",") if args.channels else None
    channels = catalog.resolve(channel_names)

    run_id = uuid.uuid4().hex[:8]
    print(f"Run {run_id}: {args.start} -> {args.end}, channels: {[c.name for c in channels]}")

    print("\n=== bathymetry / land-sea mask ===")
    gebco = fetch_gebco(catalog.bathymetry, domain, bathymetry_dir)
    land_sea = build_land_sea_mask(gebco, catalog.bathymetry, domain)
    land_sea.to_netcdf(bathymetry_dir / "land_sea_mask.nc")
    print(f"  land-sea mask written to {bathymetry_dir / 'land_sea_mask.nc'}")

    for channel in channels:
        process_channel(
            channel, args.start, args.end, domain, land_sea,
            raw_dir, processed_dir, weights_cache_dir, run_id,
        )

    print(f"\nDone. Run {run_id} complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
