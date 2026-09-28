import numpy as np
import xarray as xr

from data.qc import coverage_report, mask_invalid_values, missing_day_report, range_check


def test_missing_day_report_flags_fully_nan_days():
    time = xr.date_range("2024-06-01", periods=4, freq="1D")
    values = np.ones((4, 2, 2))
    values[1] = np.nan  # entire day 1 missing
    ds = xr.Dataset({"sst": (("time", "lat", "lon"), values)}, coords={"time": time, "lat": [0, 1], "lon": [0, 1]})
    report = missing_day_report(ds, "sst")
    assert report["missing_days"] == 1
    assert "2024-06-02" in report["missing_day_list"]


def test_coverage_report_computes_fraction():
    time = xr.date_range("2024-06-01", periods=1, freq="1D")
    values = np.array([[[1.0, np.nan], [1.0, 1.0]]])  # 3 of 4 cells valid
    ds = xr.Dataset({"sst": (("time", "lat", "lon"), values)}, coords={"time": time, "lat": [0, 1], "lon": [0, 1]})
    report = coverage_report(ds, "sst")
    assert abs(report["mean_coverage_fraction"] - 0.75) < 1e-9


def test_range_check_detects_out_of_bounds_values():
    time = xr.date_range("2024-06-01", periods=1, freq="1D")
    values = np.array([[[100.0, 20.0]]])  # 100 degC is absurd for SST
    ds = xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": [0], "lon": [0, 1]},
    )
    report = range_check(ds, "analysed_sst")
    assert report["checked"] is True
    assert report["n_out_of_range"] == 1
    assert report["pass"] is False


def test_mask_invalid_values_nans_out_of_bounds_before_regrid():
    # simulates a garbage land-extension value (-197) surviving in a raw source file, the
    # exact kind of value that corrupted coastal cells when it reached conservative regridding
    time = xr.date_range("2024-06-01", periods=1, freq="1D")
    values = np.array([[[20.0, -197.0]]])
    ds = xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": [0], "lon": [0, 1]},
    )
    out = mask_invalid_values(ds, ["analysed_sst"])
    assert out["analysed_sst"].isel(lon=0).item() == 20.0
    assert np.isnan(out["analysed_sst"].isel(lon=1).item())


def test_mask_invalid_values_leaves_unconfigured_variable_untouched():
    ds = xr.Dataset({"mystery_var": (("time",), [999.0])}, coords={"time": xr.date_range("2024-06-01", periods=1)})
    out = mask_invalid_values(ds, ["mystery_var"])
    assert out["mystery_var"].item() == 999.0


def test_range_check_skips_unconfigured_variable():
    ds = xr.Dataset({"mystery_var": (("time",), [1.0])}, coords={"time": xr.date_range("2024-06-01", periods=1)})
    report = range_check(ds, "mystery_var")
    assert report["checked"] is False
