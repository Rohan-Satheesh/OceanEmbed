import numpy as np
import xarray as xr

from data.bathymetry import below_bathymetry_mask, build_land_sea_mask
from data.catalog import BathymetryConfig, Domain


def _bathy_cfg(threshold: float = 0.5) -> BathymetryConfig:
    return BathymetryConfig(
        label="test",
        source="gebco",
        url="",
        opendap_url="",
        variable="elevation",
        land_fraction_threshold=threshold,
    )


def _domain() -> Domain:
    # A single-point domain (lat_min == lat_max) so the whole synthetic native patch below
    # falls into exactly one 1deg target cell centered at (0.5, 0.5), covering [0,1)x[0,1).
    return Domain(lat_min=0.5, lat_max=0.5, lon_min=0.5, lon_max=0.5, resolution_deg=1.0)


def test_land_fraction_and_sea_mask_on_synthetic_grid():
    # 4x4 native grid over the single 1deg target cell [0,1]x[0,1]:
    # 12 ocean points (negative elevation), 4 land points (positive) -> 25% land fraction.
    lat = np.linspace(0.125, 0.875, 4)
    lon = np.linspace(0.125, 0.875, 4)
    elevation = np.full((4, 4), -100.0)
    elevation[0, 0] = 50.0
    elevation[0, 1] = 50.0
    elevation[1, 0] = 50.0
    elevation[1, 1] = 50.0
    gebco = xr.Dataset({"elevation": (("lat", "lon"), elevation)}, coords={"lat": lat, "lon": lon})

    land_sea = build_land_sea_mask(gebco, _bathy_cfg(threshold=0.5), _domain())

    assert abs(land_sea["land_fraction"].values.item() - 0.25) < 1e-6
    assert bool(land_sea["sea_mask"].values.item()) is True  # 25% < 50% threshold -> kept as sea


def test_land_fraction_threshold_excludes_mostly_land_cell():
    lat = np.linspace(0.125, 0.875, 4)
    lon = np.linspace(0.125, 0.875, 4)
    elevation = np.full((4, 4), 50.0)  # entirely land
    gebco = xr.Dataset({"elevation": (("lat", "lon"), elevation)}, coords={"lat": lat, "lon": lon})

    land_sea = build_land_sea_mask(gebco, _bathy_cfg(threshold=0.5), _domain())

    assert bool(land_sea["sea_mask"].values.item()) is False


def test_below_bathymetry_mask_flags_deeper_depths():
    land_sea = xr.Dataset(
        {"seafloor_depth_m": (("lat", "lon"), np.array([[150.0]]))},
        coords={"lat": [0.5], "lon": [0.5]},
    )
    mask = below_bathymetry_mask(land_sea, depths_m=[0, 100, 200])
    assert list(mask.sel(lat=0.5, lon=0.5).values) == [False, False, True]
