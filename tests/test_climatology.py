import numpy as np
import xarray as xr

from data.climatology import compute_daily_climatology, compute_monthly_climatology


def _synthetic_ds():
    # Two years, Jan 1 and Feb 1 each year, one lat/lon cell.
    times = np.array(
        ["2015-01-01", "2015-02-01", "2016-01-01", "2016-02-01", "2020-01-01"],
        dtype="datetime64[ns]",
    )
    values = [10.0, 20.0, 30.0, 40.0, 999.0]  # 2020 value is outside the training window
    ds = xr.Dataset({"thetao": (("time",), values)}, coords={"time": times})
    return ds


def test_compute_daily_climatology_averages_within_training_period_only():
    ds = _synthetic_ds()
    clim = compute_daily_climatology(ds, "thetao", "2015-01-01", "2016-12-31")

    jan1_doy = 1
    feb1_doy = 32  # non-leap Feb 1
    assert float(clim.sel(dayofyear=jan1_doy).values) == 20.0  # mean(10, 30)
    assert float(clim.sel(dayofyear=feb1_doy).values) == 30.0  # mean(20, 40)
    # 2020's outlier value must not leak in since it's outside [2015-01-01, 2016-12-31].
    assert 999.0 not in clim.values


def test_compute_monthly_climatology_averages_within_training_period_only():
    ds = _synthetic_ds()
    clim = compute_monthly_climatology(ds, "thetao", "2015-01-01", "2016-12-31")

    assert float(clim.sel(month=1).values) == 20.0  # mean(10, 30)
    assert float(clim.sel(month=2).values) == 30.0  # mean(20, 40)
    assert 999.0 not in clim.values
