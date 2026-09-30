import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.catalog import Channel
from data.harmonize import append_zarr
from data.pipeline import apply_land_mask
from evaluation import argo


def _channel(variables=("uwnd",)):
    return Channel(name="t", label="t", source="earthdata", variables=list(variables),
                   native_resolution_deg=0.25, units="m s-1")


def _ds(days):
    t = pd.to_datetime(days)
    return xr.Dataset({"uwnd": (("time", "lat", "lon"), np.ones((len(t), 2, 2)))},
                      coords={"time": t, "lat": [0.0, 0.25], "lon": [0.0, 0.25]})


def test_append_zarr_skips_days_already_stored(tmp_path):
    ch = _channel()
    append_zarr(_ds(["2024-06-01", "2024-06-02"]), ch, tmp_path)
    append_zarr(_ds(["2024-06-02", "2024-06-03"]), ch, tmp_path)   # overlap + new
    append_zarr(_ds(["2024-06-01"]), ch, tmp_path)                 # fully duplicate
    out = xr.open_zarr(tmp_path / "zarr" / "oceanembed.zarr", group="t", consolidated=False)
    assert list(out.time.dt.strftime("%Y-%m-%d").values) == ["2024-06-01", "2024-06-02", "2024-06-03"]


def test_land_mask_covers_derived_variables():
    ds = _ds(["2024-06-01"])
    ds["ws"] = ds["uwnd"] * 2          # derived, not in channel.variables
    sea = xr.Dataset({"sea_mask": (("lat", "lon"), np.array([[True, False], [True, True]]))},
                     coords={"lat": ds.lat, "lon": ds.lon})
    out = apply_land_mask(ds, sea, _channel())
    assert np.isnan(out["ws"].values[0, 0, 1]) and np.isnan(out["uwnd"].values[0, 0, 1])


def test_tls_failure_raises_unless_opted_in(monkeypatch):
    import ssl
    def boom(*a, **k):
        raise ssl.SSLCertVerificationError("bad cert")
    monkeypatch.setattr(argo.urllib.request, "urlopen", boom)
    monkeypatch.delenv(argo.INSECURE_TLS_ENV, raising=False)
    with pytest.raises(RuntimeError, match=argo.INSECURE_TLS_ENV):
        argo._download_with_tls_fallback("https://example.invalid/x")
