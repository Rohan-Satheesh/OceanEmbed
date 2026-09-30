import numpy as np
import xarray as xr

from data.catalog import ArgoConfig, Domain
from evaluation.argo import match_spatial, match_temporal, qc_argo

ARGO_CFG = ArgoConfig(
    label="test",
    source="incois_erddap",
    dataset_id="incois_argo_10d_VAM",
    erddap_base_url="https://erddap.incois.gov.in/erddap/griddap/incois_argo_10d_VAM",
    variables={"temp": "TEMP", "sal": "SAL", "temp_error": "TERR", "sal_error": "SERR"},
    native_resolution_deg=1.0,
    max_error_threshold=1.0,
)


def test_qc_argo_masks_out_of_range_values():
    lat, lon = [10.0], [70.0]
    ds = xr.Dataset(
        {
            "TEMP": (("time", "lat", "lon"), [[[999.0]]]),  # way out of physical range
            "SAL": (("time", "lat", "lon"), [[[35.0]]]),
            "TERR": (("time", "lat", "lon"), [[[0.1]]]),
            "SERR": (("time", "lat", "lon"), [[[0.1]]]),
        },
        coords={"time": [0], "lat": lat, "lon": lon},
    )
    out = qc_argo(ds, ARGO_CFG)
    assert np.isnan(out["TEMP"].values).all()
    assert float(out["SAL"].values[0, 0, 0]) == 35.0


def test_qc_argo_masks_high_error_cells():
    lat, lon = [10.0], [70.0]
    ds = xr.Dataset(
        {
            "TEMP": (("time", "lat", "lon"), [[[28.0]]]),
            "SAL": (("time", "lat", "lon"), [[[35.0]]]),
            "TERR": (("time", "lat", "lon"), [[[5.0]]]),  # above max_error_threshold=1.0
            "SERR": (("time", "lat", "lon"), [[[0.1]]]),
        },
        coords={"time": [0], "lat": lat, "lon": lon},
    )
    out = qc_argo(ds, ARGO_CFG)
    assert np.isnan(out["TEMP"].values).all()
    assert float(out["SAL"].values[0, 0, 0]) == 35.0  # SAL's own error (SERR) is fine


def test_match_spatial_nearest_neighbor_onto_finer_grid():
    domain = Domain(lat_min=9.0, lat_max=11.0, lon_min=69.0, lon_max=71.0, resolution_deg=0.5)
    ds = xr.Dataset(
        {"TEMP": (("lat", "lon"), [[28.0]])},
        coords={"lat": [10.0], "lon": [70.0]},
    )
    out = match_spatial(ds, domain)
    assert set(out["lat"].values) == {9.0, 9.5, 10.0, 10.5, 11.0}
    # Every finer-grid cell should pick up the single coarse source value via nearest-neighbor.
    assert (out["TEMP"].values == 28.0).all()


def test_match_temporal_nearest_composite_window():
    times = np.array(["2024-06-01", "2024-06-11"], dtype="datetime64[ns]")
    ds = xr.Dataset({"TEMP": (("time",), [28.0, 29.0])}, coords={"time": times})

    target_dates = xr.date_range("2024-06-01", "2024-06-05", freq="1D")
    out = match_temporal(ds, target_dates)

    # Every target day is closer to 2024-06-01's composite than to 2024-06-11's.
    assert (out["TEMP"].values == 28.0).all()
