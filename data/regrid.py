"""Conservative, area-weighted regridding to the common 0.25° grid (PRD Section 10.3).

Never nearest-neighbour. Regridder weight files are cached per (source-grid-shape, target-grid)
pair so a run doesn't rebuild ESMF weights for every day of the same channel.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

from data.catalog import Domain


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Rename whatever this provider calls its spatial dims/coords to lat/lon.

    Different Copernicus Marine / PO.DAAC products name these differently — e.g. SST ships
    lat/lon dims directly, SSH/DUACS ships latitude/longitude dims, and OSCAR (PO.DAAC) ships
    dims literally named "latitude"/"longitude" with separate non-dimension "lat"/"lon"
    coordinate *variables* attached ("Dimensions without coordinates" case) — xESMF's
    conservative method needs a proper lat/lon *index* coordinate to find/build cell bounds,
    so this checks ds.dims (not just ds.coords) before every channel reaches build_regridder().
    Renaming the dim also relabels any coordinate variable that already carries that dim, which
    is what turns OSCAR's dangling "lat"/"lon" variables into real index coordinates here.
    """
    rename = {}
    if "lat" not in ds.dims and "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "lon" not in ds.dims and "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if not rename:
        return ds

    # xarray warns here that rename() alone won't create an index when the target name
    # already exists as a dangling (non-dimension) coordinate variable — e.g. OSCAR's case.
    # That's expected and handled by the set_index() call immediately below, so the warning
    # is suppressed rather than left to alarm callers about something already fixed.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '.*' to '.*' does not create an index anymore")
        ds = ds.rename(rename)
    to_index = [c for c in ("lat", "lon") if c in ds.coords and c not in ds.indexes]
    if to_index:
        ds = ds.set_index({c: c for c in to_index})
    return ds


def target_grid(domain: Domain) -> xr.Dataset:
    """The common PoC-domain 0.25° grid every channel and the target are regridded onto."""
    lat = np.arange(domain.lat_min, domain.lat_max + domain.resolution_deg / 2, domain.resolution_deg)
    lon = np.arange(domain.lon_min, domain.lon_max + domain.resolution_deg / 2, domain.resolution_deg)
    return xr.Dataset(coords={"lat": lat, "lon": lon})


def _weights_path(cache_dir: Path, src: xr.Dataset, domain: Domain) -> Path:
    if "lat" not in src.sizes or "lon" not in src.sizes:
        raise ValueError(
            "source dataset has no 'lat'/'lon' dims — call standardize_latlon() before regridding"
        )
    src_shape = f"{src.sizes['lat']}x{src.sizes['lon']}"
    # The source EXTENT is part of the key: two sources with the same shape but different coverage need different
    # weights, and reusing the wrong file silently mis-weights every cell.
    la, lo = src["lat"].values, src["lon"].values
    extent = f"{la[0]:.5f}_{la[-1]:.5f}_{lo[0]:.5f}_{lo[-1]:.5f}"
    key = f"{src_shape}_{extent}_{domain.lat_min}_{domain.lat_max}_{domain.lon_min}_{domain.lon_max}_{domain.resolution_deg}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"conservative_{digest}.nc"


def build_regridder(src: xr.Dataset, domain: Domain, cache_dir: Path) -> xe.Regridder:
    """Build (or load a cached) conservative regridder from src's grid to the domain's 0.25° grid."""
    weights_path = _weights_path(cache_dir, src, domain)
    tgt = target_grid(domain)
    reuse = weights_path.exists()
    return xe.Regridder(
        src,
        tgt,
        method="conservative",
        filename=str(weights_path),
        reuse_weights=reuse,
    )


COVERAGE_MIN = 0.999   # a target cell must be (almost) fully covered by the source grid to be trusted


def source_coverage(ds: xr.Dataset, regridder: xe.Regridder) -> xr.DataArray:
    """Fraction (0-1) of each target cell's area covered by the SOURCE grid, ignoring NaN values. Regridding a field
    of ones gives it: conservative weights normalise by the whole target cell, so a cell that sticks out past the
    source edge comes back as (covered fraction) x (true value) — silently wrong — unless it is masked."""
    ones = xr.DataArray(np.ones((ds.sizes["lat"], ds.sizes["lon"])), dims=("lat", "lon"),
                        coords={"lat": ds["lat"], "lon": ds["lon"]})
    return regridder(ones)


def regrid_dataset(ds: xr.Dataset, domain: Domain, cache_dir: Path) -> xr.Dataset:
    """Conservatively regrid every data variable in ds onto the common 0.25° domain grid.

    Target cells only partly covered by the source (the domain edge, when the source was cropped too tightly) are set
    to NaN rather than returned scaled down, and the count is recorded in `attrs['regrid_partial_cells_masked']`."""
    ds = standardize_latlon(ds)
    regridder = build_regridder(ds, domain, cache_dir)
    out = regridder(ds, keep_attrs=True)
    covered = source_coverage(ds, regridder) >= COVERAGE_MIN
    n_bad = int((~covered).sum())
    if n_bad:
        print(f"    [warn] {n_bad} target cell(s) are only partly covered by the source grid and were set to NaN "
              "(download the subset with a margin beyond the domain edge)")
        for name in out.data_vars:
            if {"lat", "lon"} <= set(out[name].dims):
                out[name] = out[name].where(covered)
    out.attrs.update(ds.attrs)
    out.attrs["regrid_method"] = "conservative"
    out.attrs["regrid_partial_cells_masked"] = n_bad
    return out
