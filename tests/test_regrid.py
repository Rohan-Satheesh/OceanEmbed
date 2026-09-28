from pathlib import Path

import numpy as np
import xarray as xr

from data.catalog import Domain
from data.regrid import regrid_dataset, standardize_latlon, target_grid


def _domain() -> Domain:
    return Domain(lat_min=5.0, lat_max=10.0, lon_min=45.0, lon_max=50.0, resolution_deg=0.25)


def test_target_grid_shape():
    domain = _domain()
    grid = target_grid(domain)
    assert grid.sizes["lat"] == 21  # 5.0..10.0 inclusive at 0.25 steps
    assert grid.sizes["lon"] == 21


def test_standardize_latlon_renames_latitude_longitude():
    ds = xr.Dataset(
        {"sla": (("latitude", "longitude"), np.zeros((3, 3)))},
        coords={"latitude": [1, 2, 3], "longitude": [4, 5, 6]},
    )
    out = standardize_latlon(ds)
    assert "lat" in out.coords and "lon" in out.coords
    assert "latitude" not in out.coords and "longitude" not in out.coords


def test_standardize_latlon_handles_dangling_lat_lon_coord_variables():
    # Mirrors OSCAR/PO.DAAC files: dims are "latitude"/"longitude", with separate 1-D
    # "lat"/"lon" coordinate *variables* attached to those dims but not indexing them
    # ("Dimensions without coordinates" in xarray's repr).
    ds = xr.Dataset(
        {"u": (("time", "longitude", "latitude"), np.zeros((1, 3, 2)))},
        coords={
            "lat": (("latitude",), [1.0, 2.0]),
            "lon": (("longitude",), [3.0, 4.0, 5.0]),
            "time": [0],
        },
    )
    out = standardize_latlon(ds)
    assert "lat" in out.dims and "lon" in out.dims
    assert "latitude" not in out.dims and "longitude" not in out.dims
    assert list(out["lat"].values) == [1.0, 2.0]


def test_standardize_latlon_is_noop_when_already_lat_lon():
    ds = xr.Dataset({"sst": (("lat", "lon"), np.zeros((2, 2)))}, coords={"lat": [1, 2], "lon": [3, 4]})
    out = standardize_latlon(ds)
    assert list(out.coords) == list(ds.coords)


def test_conservative_regrid_produces_target_shape(tmp_path: Path):
    domain = _domain()
    src = xr.Dataset(
        {"sst": (("lat", "lon"), np.random.rand(10, 10))},
        coords={"lat": np.linspace(4, 11, 10), "lon": np.linspace(44, 51, 10)},
    )
    out = regrid_dataset(src, domain, cache_dir=tmp_path)
    assert out["sst"].sizes["lat"] == 21
    assert out["sst"].sizes["lon"] == 21
    assert out.attrs["regrid_method"] == "conservative"
