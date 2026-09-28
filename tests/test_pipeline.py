import cftime
import numpy as np
import xarray as xr

from data.pipeline import _normalize_time_dtype


def test_normalize_time_dtype_converts_cftime_datetimes():
    # Reproduces the actual bug: OSCAR's raw files decode `time` to cftime.DatetimeJulian
    # (dtype=object), which then silently fails to align against a datetime64 reindex target
    # in harmonize.align_daily_utc() — not an error, just every day comes back all-NaN.
    times = np.array(
        [cftime.DatetimeJulian(2024, 6, 1), cftime.DatetimeJulian(2024, 6, 2)], dtype=object
    )
    ds = xr.Dataset({"u": (("time",), [1.0, 2.0])}, coords={"time": times})
    assert ds["time"].dtype == object

    out = _normalize_time_dtype(ds)
    assert np.issubdtype(out["time"].dtype, np.datetime64)
    assert list(out["time"].dt.strftime("%Y-%m-%d").values) == ["2024-06-01", "2024-06-02"]


def test_normalize_time_dtype_converts_generic_object_dtype_datetimes():
    # Plain python datetime.datetime objects get auto-upgraded to datetime64 by xarray on
    # construction, so date *strings* are used here to reliably keep dtype=object as input
    # and exercise the pd.to_datetime() fallback path (non-cftime object dtype).
    times = np.array(["2024-06-01", "2024-06-02"], dtype=object)
    ds = xr.Dataset({"u": (("time",), [1.0, 2.0])}, coords={"time": times})
    assert ds["time"].dtype == object

    out = _normalize_time_dtype(ds)
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_normalize_time_dtype_is_noop_for_datetime64():
    time = xr.date_range("2024-06-01", periods=2, freq="1D")
    ds = xr.Dataset({"sst": (("time",), [1.0, 2.0])}, coords={"time": time})
    out = _normalize_time_dtype(ds)
    assert np.issubdtype(out["time"].dtype, np.datetime64)
