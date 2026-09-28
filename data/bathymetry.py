"""ETOPO 2022 bathymetry: land/seafloor mask on the common 0.25° grid (PRD Sections 4.2, 7.3).

ETOPO's native grid (60 arc-second, ~0.0167°) is far finer than the 0.25° PoC grid, so the
land/sea decision per 0.25° cell is a coverage fraction, not a single point sample: cells with
more than `land_fraction_threshold` of their native ETOPO pixels above sea level are excluded.
Per-cell mean seafloor depth (over ocean pixels only) supports the below-bathymetry depth
masking used later in the reconstruction pipeline (Chunk 6).

See configs/channels.yaml for why ETOPO 2022 is used in place of GEBCO 2024 (no stable,
unauthenticated bbox-subset API for GEBCO at pipeline-build time).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

from data.catalog import BathymetryConfig, Domain
from data.retry import retry

CACHE_FILENAME = "bathymetry_subset.nc"


def _erddap_subset_url(bathy_cfg: BathymetryConfig, domain: Domain, buffer_deg: float) -> str:
    lat_lo, lat_hi = domain.lat_min - buffer_deg, domain.lat_max + buffer_deg
    lon_lo, lon_hi = domain.lon_min - buffer_deg, domain.lon_max + buffer_deg
    return (
        f"{bathy_cfg.opendap_url}.nc?"
        f"{bathy_cfg.variable}[({lat_lo}):({lat_hi})][({lon_lo}):({lon_hi})]"
    )


def fetch_gebco(bathy_cfg: BathymetryConfig, domain: Domain, cache_dir: Path, buffer_deg: float = 0.5) -> xr.Dataset:
    """Subset the configured bathymetry source to the domain (with a small buffer) via its
    ERDDAP griddap bbox query, caching the subset locally. Named fetch_gebco for continuity
    with the PRD's original GEBCO framing even though the configured source is ETOPO 2022."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILENAME

    if not cache_path.exists():
        url = _erddap_subset_url(bathy_cfg, domain, buffer_deg)

        def _download() -> bytes:
            # ERDDAP's griddap query syntax uses literal, unescaped [ ] ( ) : characters — the
            # `requests` library's URL normalization percent-encodes them (breaking the
            # request), so this uses urllib directly with the URL passed through as-is.
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()

        def _on_retry(attempt: int, exc: Exception) -> None:
            print(f"  [retry {attempt}] bathymetry fetch failed: {exc}")

        content = retry(_download, on_retry=_on_retry)
        cache_path.write_bytes(content)

    ds = xr.open_dataset(cache_path)
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    return ds.rename({lat_name: "lat", lon_name: "lon"})


def build_land_sea_mask(gebco: xr.Dataset, bathy_cfg: BathymetryConfig, domain: Domain) -> xr.Dataset:
    """Downsample native-resolution GEBCO elevation to per-0.25-degree-cell land fraction + seafloor depth."""
    elevation = gebco[bathy_cfg.variable]
    is_land = elevation >= 0

    lat_edges = np.arange(
        domain.lat_min - domain.resolution_deg / 2,
        domain.lat_max + domain.resolution_deg,
        domain.resolution_deg,
    )
    lon_edges = np.arange(
        domain.lon_min - domain.resolution_deg / 2,
        domain.lon_max + domain.resolution_deg,
        domain.resolution_deg,
    )
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    land_fraction = (
        is_land.astype("float32")
        .groupby_bins("lat", lat_edges, labels=lat_centers)
        .mean()
        .groupby_bins("lon", lon_edges, labels=lon_centers)
        .mean()
        .rename({"lat_bins": "lat", "lon_bins": "lon"})
    )

    ocean_depth = (-elevation).where(~is_land)
    seafloor_depth_m = (
        ocean_depth.groupby_bins("lat", lat_edges, labels=lat_centers)
        .mean()
        .groupby_bins("lon", lon_edges, labels=lon_centers)
        .mean()
        .rename({"lat_bins": "lat", "lon_bins": "lon"})
    )

    sea_mask = land_fraction < bathy_cfg.land_fraction_threshold

    out = xr.Dataset(
        {
            "land_fraction": land_fraction,
            "sea_mask": sea_mask,
            "seafloor_depth_m": seafloor_depth_m,
        }
    )
    out.attrs["land_fraction_threshold"] = bathy_cfg.land_fraction_threshold
    out.attrs["source"] = bathy_cfg.label
    return out


def below_bathymetry_mask(land_sea: xr.Dataset, depths_m: list[int]) -> xr.DataArray:
    """True where a given standard depth is deeper than the cell's seafloor (PRD Section 4.4)."""
    depth_da = xr.DataArray(depths_m, dims="depth", coords={"depth": depths_m})
    return depth_da > land_sea["seafloor_depth_m"]
