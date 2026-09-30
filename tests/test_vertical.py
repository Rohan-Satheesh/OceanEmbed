import numpy as np
import xarray as xr

from data.vertical import apply_below_bathymetry_mask, interpolate_to_standard_depths


def test_interpolate_to_standard_depths_linear_interior_value():
    # Native depths [0, 10, 20]; temperature drops linearly 20 -> 10 -> 0 degC.
    native_depths = [0, 10, 20]
    temp = xr.DataArray(
        [[20.0, 10.0, 0.0]], dims=("time", "depth"), coords={"time": [0], "depth": native_depths}
    )
    ds = xr.Dataset({"thetao": temp})

    out = interpolate_to_standard_depths(ds, depths_m=[5], depth_dim="depth")

    assert out["thetao"].sel(depth=5).item() == 15.0


def test_interpolate_to_standard_depths_holds_boundary_value_for_surface():
    # Native depths start at 0.5 (like GLORYS's real ~0.49m first level), not exactly 0 --
    # standard depth 0 must take the shallowest level's value (boundary hold), not NaN.
    native_depths = [0.5, 10, 20]
    temp = xr.DataArray(
        [[15.0, 10.0, 0.0]], dims=("time", "depth"), coords={"time": [0], "depth": native_depths}
    )
    ds = xr.Dataset({"thetao": temp})

    out = interpolate_to_standard_depths(ds, depths_m=[0, 10], depth_dim="depth")

    assert list(out["depth"].values) == [0, 10]
    assert out["thetao"].sel(depth=0).item() == 15.0
    assert out["thetao"].sel(depth=10).item() == 10.0


def test_interpolate_to_standard_depths_no_extrapolation_beyond_native_range():
    native_depths = [0, 10, 20]
    temp = xr.DataArray(
        [[20.0, 10.0, 0.0]], dims=("time", "depth"), coords={"time": [0], "depth": native_depths}
    )
    ds = xr.Dataset({"thetao": temp})

    # 1000m is far beyond the native [0, 20] range -> must come back NaN, not extrapolated.
    out = interpolate_to_standard_depths(ds, depths_m=[1000], depth_dim="depth")

    assert np.isnan(out["thetao"].sel(depth=1000).item())


def test_apply_below_bathymetry_mask_nans_out_deeper_than_seafloor():
    depths_m = [0, 100, 200]
    lat, lon = [10.0], [70.0]
    land_sea = xr.Dataset(
        {
            "sea_mask": (("lat", "lon"), [[True]]),
            "seafloor_depth_m": (("lat", "lon"), [[150.0]]),  # 200m is below the seafloor here
        },
        coords={"lat": lat, "lon": lon},
    )
    thetao = xr.DataArray(
        [[[[5.0]], [[5.0]], [[5.0]]]],
        dims=("time", "depth", "lat", "lon"),
        coords={"time": [0], "depth": depths_m, "lat": lat, "lon": lon},
    )
    ds = xr.Dataset({"thetao": thetao})

    out = apply_below_bathymetry_mask(ds, land_sea, depths_m, ["thetao"])

    assert out["thetao"].sel(depth=0).item() == 5.0
    assert out["thetao"].sel(depth=100).item() == 5.0
    assert np.isnan(out["thetao"].sel(depth=200).item())
