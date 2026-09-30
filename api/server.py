"""HTTP backend for the React frontend (`frontend/`). Thin read-only layer over `dashboard.data` (the same pure
functions the Streamlit dashboard uses), so both UIs show identical numbers.

    uvicorn api.server:app --port 8000

Serves the predictions Zarr written by `python -m evaluation.predict`. Path resolution: OCEANEMBED_PREDICTIONS if
set, else the real-data output (`data/processed/outputs/`) when it exists, else the dashboard default (demo config).
Endpoints return the grid metadata as JSON, whole map fields as raw float32 (NaN = land / no data), and small
per-point series as JSON (NaN -> null).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from dashboard import data as D
from evaluation.metrics import REGIONS

REAL_PATH = D.REPO_ROOT / "data" / "processed" / "outputs" / "oceanembed_predictions.zarr"

app = FastAPI(title="OceanEmbed API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def resolve_path() -> Path:
    env = os.environ.get("OCEANEMBED_PREDICTIONS")
    if env:
        return Path(env)
    return REAL_PATH if REAL_PATH.exists() else D.predictions_path()


@lru_cache(maxsize=1)
def dataset() -> xr.Dataset:
    path = resolve_path()
    if not path.exists():
        raise HTTPException(503, f"No predictions at {path}. Run `python -m evaluation.predict` first.")
    return D.open_predictions(path)


def _clean(values) -> list:
    """Array -> JSON-safe list (NaN -> None)."""
    arr = np.asarray(values, dtype="float64")
    return [None if not np.isfinite(v) else float(v) for v in arr.ravel()]


def _check_time(ds: xr.Dataset, time: int) -> None:
    if not 0 <= time < ds.sizes["time"]:
        raise HTTPException(400, f"time index {time} out of range 0..{ds.sizes['time'] - 1}")


def _check_depth(ds: xr.Dataset, depth: int) -> None:
    if not 0 <= depth < ds.sizes["depth"]:
        raise HTTPException(400, f"depth index {depth} out of range 0..{ds.sizes['depth'] - 1}")


@app.get("/api/meta")
def meta():
    ds = dataset()
    warning = D.data_banner(ds)
    return {
        "times": [str(t)[:10] for t in ds.time.values],
        "depths": [float(d) for d in ds.depth.values],
        "lats": [float(v) for v in ds.lat.values],
        "lons": [float(v) for v in ds.lon.values],
        "attrs": {
            "dataSource": ds.attrs.get("data_source", "real"),
            "modelVersion": ds.attrs.get("model_version", "unknown"),
            "syntheticWarning": warning,
            "note": D.data_note(ds),
        },
    }


@app.get("/api/land")
def land():
    """Surface quality flag as raw uint8, row-major (lat, lon): 0 = ocean, non-zero = land / masked. Time-invariant."""
    ds = dataset()
    flag = ds["quality_flag"].isel(depth=0).load().values.astype("uint8")
    return Response(flag.tobytes(), media_type="application/octet-stream")


@app.get("/api/field")
def field(var: str, time: int = Query(0, ge=0), depth: int = Query(0, ge=0)):
    """One (lat, lon) map as raw little-endian float32, row-major; NaN where there is no data."""
    ds = dataset()
    if var not in {v[0] for v in D.MAP_VARIABLES.values()}:
        raise HTTPException(400, f"unknown variable {var!r}")
    _check_time(ds, time)
    if var not in D.DEPTH_FREE:
        _check_depth(ds, depth)
    da = D.map_field(ds, var, time, depth)
    return Response(da.values.astype("<f4").tobytes(), media_type="application/octet-stream")


@app.get("/api/profile")
def profile(time: int, lat: float, lon: float):
    ds = dataset()
    _check_time(ds, time)
    df = D.profile(ds, time, lat, lon)
    return {"depthM": _clean(df["depth_m"]), "temp": _clean(df["temp"]), "std": _clean(df["std"]),
            "glorys": _clean(df["glorys"])}


@app.get("/api/timeseries")
def timeseries(depth: int, lat: float, lon: float):
    ds = dataset()
    _check_depth(ds, depth)
    df = D.timeseries(ds, depth, lat, lon)
    return {"time": [str(t)[:10] for t in df.index.values], "temp": _clean(df["temp"]), "std": _clean(df["std"]),
            "glorys": _clean(df["glorys"])}


@app.get("/api/regional")
def regional(region: str, depth: int):
    ds = dataset()
    if region not in REGIONS:
        raise HTTPException(400, f"unknown region {region!r}; choose from {sorted(REGIONS)}")
    _check_depth(ds, depth)
    df = D.regional_summary(ds, region, depth)
    return {"time": [str(t)[:10] for t in df.index.values], "reconstruction": _clean(df["reconstruction"]),
            "glorys": _clean(df["glorys"]), "rmse": _clean(df["rmse"])}
