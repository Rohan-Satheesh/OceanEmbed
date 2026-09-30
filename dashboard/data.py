"""Data access + calculations for the dashboard (PRD 17.4). Pure functions, no Streamlit imports, so they
are unit-testable. Reads the Zarr written by `python -m evaluation.predict`.

Error views compare the reconstruction with GLORYS only (GLORYS assimilates ARGO; there is no independent
in-situ check in this build).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data.catalog import load_base_config
from evaluation.metrics import REGIONS, region_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "demo.yaml"

MAP_VARIABLES = {                      # label -> (kind, colormap, unit)
    "Temperature": ("temp", "turbo", "deg C"),
    "Uncertainty (MC-dropout std)": ("temp_std", "magma", "deg C"),
    "Difference vs GLORYS (reconstruction - GLORYS)": ("error", "RdBu_r", "deg C"),
    "GLORYS (reference)": ("glorys", "turbo", "deg C"),
    "Mixed layer depth": ("mld", "viridis", "m"),
    "20 C isotherm depth (D20)": ("d20", "viridis", "m"),
    "Tropical cyclone heat potential": ("tchp", "YlOrRd", "kJ/cm2"),
}
DEPTH_FREE = {"mld", "d20", "tchp"}    # 2-D products: the depth slider does not apply


def _cfg():
    return load_base_config(override=os.environ.get("OCEANEMBED_CONFIG", str(DEFAULT_CONFIG)))


def predictions_path() -> Path:
    """OCEANEMBED_PREDICTIONS overrides; otherwise <processed_dir>/outputs/oceanembed_predictions.zarr from the config."""
    env = os.environ.get("OCEANEMBED_PREDICTIONS")
    return Path(env) if env else REPO_ROOT / _cfg().paths["processed_dir"] / "outputs" / "oceanembed_predictions.zarr"


def runs_path() -> Path:
    env = os.environ.get("OCEANEMBED_RUNS")
    return Path(env) if env else REPO_ROOT / _cfg().paths["runs_dir"]


def open_predictions(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def data_banner(ds: xr.Dataset) -> str | None:
    """Text for the always-visible warning when the data are synthetic."""
    return ds.attrs.get("synthetic_warning") if ds.attrs.get("data_source") == "synthetic" else None


def map_field(ds: xr.Dataset, var: str, time_idx: int, depth_idx: int = 0) -> xr.DataArray:
    """2-D (lat, lon) field for a map. `error` = reconstruction minus GLORYS (GLORYS only)."""
    if var in DEPTH_FREE:
        return ds[var].isel(time=time_idx).load()
    if var == "error":
        return (ds["temp"] - ds["glorys"]).isel(time=time_idx, depth=depth_idx).load()
    return ds[var].isel(time=time_idx, depth=depth_idx).load()


def nearest_cell(ds: xr.Dataset, lat: float, lon: float) -> tuple[float, float]:
    return (float(ds.lat.sel(lat=lat, method="nearest")), float(ds.lon.sel(lon=lon, method="nearest")))


def is_ocean(ds: xr.Dataset, lat: float, lon: float) -> bool:
    la, lo = nearest_cell(ds, lat, lon)
    return bool(ds["quality_flag"].sel(lat=la, lon=lo).isel(depth=0) == 0)


def profile(ds: xr.Dataset, time_idx: int, lat: float, lon: float) -> pd.DataFrame:
    la, lo = nearest_cell(ds, lat, lon)
    sel = ds.isel(time=time_idx).sel(lat=la, lon=lo)
    return pd.DataFrame({"depth_m": ds.depth.values, "temp": sel["temp"].values, "std": sel["temp_std"].values,
                         "glorys": sel["glorys"].values})


def timeseries(ds: xr.Dataset, depth_idx: int, lat: float, lon: float) -> pd.DataFrame:
    la, lo = nearest_cell(ds, lat, lon)
    sel = ds.isel(depth=depth_idx).sel(lat=la, lon=lo)
    return pd.DataFrame({"time": ds.time.values, "temp": sel["temp"].values, "std": sel["temp_std"].values,
                         "glorys": sel["glorys"].values}).set_index("time")


def regional_summary(ds: xr.Dataset, region: str, depth_idx: int) -> pd.DataFrame:
    """Regional-mean reconstruction vs GLORYS and RMSE over the region, per day (Bay of Bengal / Arabian Sea)."""
    m = xr.DataArray(region_mask(ds.lat.values, ds.lon.values, region), dims=("lat", "lon"), coords={"lat": ds.lat, "lon": ds.lon})
    pred, ref = ds["temp"].isel(depth=depth_idx).where(m), ds["glorys"].isel(depth=depth_idx).where(m)
    rmse = np.sqrt(((pred - ref) ** 2).mean(("lat", "lon"), skipna=True))
    return pd.DataFrame({"time": ds.time.values, "reconstruction": pred.mean(("lat", "lon"), skipna=True).values,
                         "glorys": ref.mean(("lat", "lon"), skipna=True).values, "rmse": rmse.values}).set_index("time")


def default_location(ds: xr.Dataset) -> tuple[float, float]:
    """A sensible open-ocean start point (central Bay of Bengal), falling back to any ocean cell."""
    for lat, lon in ((15.0, 88.0), (15.0, 65.0)):
        if is_ocean(ds, lat, lon):
            return lat, lon
    ocean = (ds["quality_flag"].isel(depth=0) == 0).values
    i, j = np.argwhere(ocean)[len(np.argwhere(ocean)) // 2]
    return float(ds.lat[i]), float(ds.lon[j])


REGION_LABELS = {"bay_of_bengal": "Bay of Bengal", "arabian_sea": "Arabian Sea"}
assert set(REGION_LABELS) == set(REGIONS)
