from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data import synthetic as syn
from data.catalog import load_base_config

CFG = load_base_config(override=Path(__file__).resolve().parents[1] / "configs" / "demo.yaml")


def _gen(tmp_path, start="2022-06-01", end="2022-06-03", seed=42, chunk_days=30):
    # processed_dir=tmp_path has no bathymetry file -> deterministic crude-coastline fallback
    out = tmp_path / "syn"
    store = syn.generate(start, end, out, CFG.domain, CFG.depths_m, seed=seed, processed_dir=tmp_path,
                         chunk_days=chunk_days)
    return {ch: xr.open_zarr(store, group=ch, consolidated=False).load() for ch in syn.CHANNELS}


def test_demo_config_switches_to_synthetic():
    assert CFG.data_source == "synthetic"
    assert CFG.paths["processed_dir"] == "data/synthetic"
    assert CFG.periods["train"] == ["2022-01-01", "2022-12-31"]
    assert load_base_config().data_source == "real"          # base config untouched


def test_schema_matches_real_store(tmp_path):
    ds = _gen(tmp_path)
    lat, lon = syn.make_grid(CFG.domain)
    assert set(ds) == {"sst", "sss", "ssh", "currents", "winds", "glorys"}
    for ch, d in ds.items():
        assert np.array_equal(d.lat, lat) and np.array_equal(d.lon, lon) and d.sizes["time"] == 3
        assert d.attrs["data_source"] == "synthetic"
        for v in syn.CHANNELS[ch]:                            # mask: 1 = observed, 0 = missing
            assert np.array_equal(d[f"{v}_mask"].values.astype(bool), ~np.isnan(d[v].values))
    assert list(ds["glorys"].depth.values) == [float(z) for z in CFG.depths_m]
    assert ds["glorys"]["thetao"].dims == ("time", "depth", "lat", "lon")


def test_masks_and_bathymetry(tmp_path):
    ds = _gen(tmp_path)
    sea, floor_m = syn.load_geography(*syn.make_grid(CFG.domain), processed_dir=tmp_path)
    t = ds["glorys"]["thetao"].isel(time=0).values
    assert np.isnan(t[:, ~sea]).all()                                     # land
    below = floor_m[None] < np.array(CFG.depths_m, float)[:, None, None]
    assert np.isnan(t[below & sea[None]]).all()                           # below the seafloor
    assert np.isnan(ds["sst"]["analysed_sst"].values[:, ~sea]).all()
    sss_obs, sst_obs = ds["sss"]["sos_mask"].values[0].sum(), ds["sst"]["analysed_sst_mask"].values[0].sum()
    assert sss_obs < sst_obs                                              # SSS is the gappiest input


def test_fields_are_physical_and_learnable(tmp_path):
    ds = _gen(tmp_path, "2022-01-01", "2022-01-03")
    sst, thetao = ds["sst"]["analysed_sst"], ds["glorys"]["thetao"]
    assert 15 <= float(sst.min()) and float(sst.max()) <= 34
    assert 28 <= float(ds["sss"]["sos"].min()) and float(ds["sss"]["sos"].max()) <= 38
    assert np.allclose(thetao.isel(depth=0).values, sst.values, atol=0.5, equal_nan=True)   # surface ~ SST
    prof = thetao.mean(("time", "lat", "lon"), skipna=True).values
    assert (np.diff(prof) < 0).all()                                      # cools with depth on average
    # thermocline follows SSH: warm-core (high SLA) eddies -> warmer water at 100 m
    a, b = ds["ssh"]["sla"].values.ravel(), thetao.sel(depth=100).values.ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    assert np.corrcoef(a[ok], b[ok])[0, 1] > 0.4


def test_deterministic_and_chunk_independent(tmp_path):
    a = _gen(tmp_path / "a", chunk_days=30)
    b = _gen(tmp_path / "b", chunk_days=1)
    c = _gen(tmp_path / "c", seed=7)
    for ch in syn.CHANNELS:
        v = syn.CHANNELS[ch][0]
        assert np.array_equal(a[ch][v].values, b[ch][v].values, equal_nan=True)
    assert not np.array_equal(a["sst"]["analysed_sst"].values, c["sst"]["analysed_sst"].values, equal_nan=True)


def test_refuses_to_write_into_real_processed_dir():
    with pytest.raises(ValueError, match="real processed dir"):
        syn.generate("2022-01-01", "2022-01-01", syn.REPO_ROOT / "data" / "processed", CFG.domain, CFG.depths_m)
