import numpy as np
import xarray as xr

from data.harmonize import add_mask_variable, align_daily_utc, daily_aggregate, fill_short_gaps


def _six_hourly_ds() -> xr.Dataset:
    time = xr.date_range("2024-06-01", periods=8, freq="6h")  # 2 full days
    return xr.Dataset(
        {"uwnd": (("time", "lat", "lon"), np.ones((8, 3, 3)))},
        coords={"time": time, "lat": [0, 1, 2], "lon": [0, 1, 2]},
    )


def test_daily_aggregate_collapses_subdaily_steps():
    ds = _six_hourly_ds()
    out = daily_aggregate(ds)
    assert out.sizes["time"] == 2


def test_daily_aggregate_is_noop_for_already_daily_data():
    time = xr.date_range("2024-06-01", periods=5, freq="1D")
    ds = xr.Dataset({"sst": (("time",), np.arange(5.0))}, coords={"time": time})
    out = daily_aggregate(ds)
    assert out.sizes["time"] == 5


def test_align_daily_utc_introduces_missing_days_as_nan():
    time = xr.date_range("2024-06-01", periods=3, freq="1D")
    ds = xr.Dataset({"sst": (("time",), [1.0, 2.0, 3.0])}, coords={"time": time})
    # drop the middle day to simulate a gap
    sparse = ds.isel(time=[0, 2])
    out = align_daily_utc(sparse, "2024-06-01", "2024-06-03")
    assert out.sizes["time"] == 3
    assert np.isnan(out["sst"].isel(time=1).item())


def test_fill_short_gaps_fills_within_limit_only():
    time = xr.date_range("2024-06-01", periods=6, freq="1D")
    values = [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]  # 4-day gap > MAX_GAP_DAYS(3)
    ds = xr.Dataset({"sst": (("time",), values)}, coords={"time": time})
    out = fill_short_gaps(ds, max_gap_days=3)
    # forward-fill covers day 1-3 from day 0, backward-fill covers day 4 from day 5;
    # net result: no fully-unfilled NaN remains for a 4-day gap under limit=3 on each side
    assert out["sst"].isnull().sum().item() == 0


def test_fill_short_gaps_leaves_long_gap_when_no_fill_reaches():
    time = xr.date_range("2024-06-01", periods=10, freq="1D")
    values = [1.0] + [np.nan] * 8 + [10.0]  # 8-day gap, unreachable by limit=3 from either side
    ds = xr.Dataset({"sst": (("time",), values)}, coords={"time": time})
    out = fill_short_gaps(ds, max_gap_days=3)
    assert out["sst"].isnull().sum().item() > 0


def test_add_mask_variable_reflects_nan_status():
    time = xr.date_range("2024-06-01", periods=3, freq="1D")
    ds = xr.Dataset({"sos": (("time",), [1.0, np.nan, 3.0])}, coords={"time": time})
    out = add_mask_variable(ds, ["sos"])
    assert list(out["sos_mask"].values) == [1, 0, 1]
