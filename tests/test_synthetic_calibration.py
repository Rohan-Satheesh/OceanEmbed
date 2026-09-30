from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data import synthetic as syn
from data.catalog import load_base_config

REPO = Path(__file__).resolve().parents[1]
CFG = load_base_config(override=REPO / "configs" / "demo.yaml")
DEPTHS = np.array(CFG.depths_m, dtype=float)


def test_heave_identity_direction_and_clamping():
    tbg = np.broadcast_to((28.0 - 20.0 * (1 - np.exp(-DEPTHS / 300.0)))[:, None, None], (15, 3, 4)).copy()
    same = syn._heave(tbg, DEPTHS, np.zeros((3, 4)))
    assert np.allclose(same, tbg)                                        # zero displacement = the mean profile
    deeper = syn._heave(tbg, DEPTHS, np.full((3, 4), 60.0))
    assert (deeper >= tbg - 1e-9).all() and (deeper[3:10] > tbg[3:10]).all()     # thermocline pushed down -> warmer at fixed z
    shoaled = syn._heave(tbg, DEPTHS, np.full((3, 4), -60.0))
    assert (shoaled <= tbg + 1e-9).all() and (shoaled[3:10] < tbg[3:10]).all()
    extreme = syn._heave(tbg, DEPTHS, np.full((3, 4), 5000.0))
    assert np.isfinite(extreme).all() and extreme.min() >= tbg.min() - 1e-9 and extreme.max() <= tbg.max() + 1e-9


def test_fill_profiles_never_bleeds_shelf_values_into_deep_water():
    tbg = np.full((15, 20, 20), np.nan)
    tbg[:, :, 10:] = (7.0 + 20.0 * np.exp(-DEPTHS / 300.0))[:, None, None]      # deep ocean: real profile
    tbg[:9, :, :10] = 27.0                                                       # shelf: valid only in the top 9 levels
    out = syn._fill_profiles(tbg, sigma=1.0)
    assert np.isfinite(out).all()
    deep_truth = 7.0 + 20.0 * np.exp(-1000.0 / 300.0)
    assert abs(out[14, :, 12:].mean() - deep_truth) < 0.05                      # 1000 m ocean untouched by warm shelf water
    assert abs(out[14, :, :8].mean() - deep_truth) < 1.0                        # below-shelf cells borrow deep-water values


def test_slow_field_is_deterministic_smooth_in_time_and_unit_scale():
    a, b = syn._slow_field(100, (30, 40), 7), syn._slow_field(100, (30, 40), 7)
    assert np.array_equal(a, b) and not np.array_equal(a, syn._slow_field(100, (30, 40), 8))
    series = np.stack([syn._slow_field(d, (30, 40), 7) for d in range(200)])
    assert 0.6 < series.std() < 1.6
    c = np.corrcoef(series[:-1].ravel(), series[1:].ravel())[0, 1]
    assert c > 0.85                                                             # day-to-day persistence (not white noise)


def test_calibration_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cal = syn.Calibration(rng.normal(size=(4, 5)), rng.normal(size=(4, 5)), rng.normal(size=(4, 5)),
                          rng.normal(size=(15, 4, 5)), 1.6, 2.5, 1.2, 2.0, 170.0, 1.8, 0.99, "src")
    cal.save(tmp_path / "c.npz")
    back = syn.Calibration.load(tmp_path / "c.npz")
    for f in ("sst_map", "sss_map", "sla_map", "tbg"):
        assert np.array_equal(getattr(cal, f), getattr(back, f))
    assert (back.eddy_scale, back.wind_ne_scale, back.wind_noise_scale, back.eddy_density, back.heave_per_m,
            back.sst_sla_coef, back.sub_noise, back.source) == (1.6, 2.5, 1.2, 2.0, 170.0, 1.8, 0.99, "src")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A stand-in 'real' store (the uncalibrated generator, Q1 2022), a calibration fitted to it, and stores from both."""
    root = tmp_path_factory.mktemp("calib")
    syn.generate("2022-01-01", "2022-03-31", root / "fake_real", CFG.domain, CFG.depths_m, seed=3, processed_dir=root)
    cal = syn.fit_calibration(root / "fake_real", CFG.domain, seed=42)
    syn.generate("2022-02-01", "2022-02-20", root / "cal", CFG.domain, CFG.depths_m, seed=42, processed_dir=root,
                 calibration=cal)
    return root, cal


def _stats(root, store, group, var, sl):
    d = xr.open_zarr(root / store / "zarr" / "oceanembed.zarr", group=group, consolidated=False)[var].sel(time=sl)
    if "depth" in d.dims and group == "sss":
        d = d.isel(depth=0, drop=True)
    return d.load()


def test_fit_produces_sane_parameters(fitted):
    _, cal = fitted
    assert cal.eddy_density in (1.0, 2.0, 3.0, 4.0, 6.0) and 0.6 <= cal.eddy_scale <= 2.5
    assert 50.0 <= cal.heave_per_m <= 800.0 and 0.0 <= cal.sst_sla_coef <= 12.0 and cal.sub_noise > 0
    assert np.isfinite(cal.tbg).all() and np.isfinite(cal.sst_map).all() and cal.tbg.shape == (15, 101, 241)
    assert cal.source.startswith("real store")


def test_calibrated_output_reproduces_the_target_statistics(fitted):
    root, _ = fitted
    sl = slice("2022-02-01", "2022-02-20")
    for group, var, rel in (("sst", "analysed_sst", 0.03), ("sss", "sos", 0.03)):
        real, cal = _stats(root, "fake_real", group, var, sl), _stats(root, "cal", group, var, sl)
        assert abs(float(real.mean()) - float(cal.mean())) < rel * abs(float(real.mean())) + 0.3, var
    real_ws, cal_ws = _stats(root, "fake_real", "winds", "ws", sl), _stats(root, "cal", "winds", "ws", sl)
    assert abs(float(real_ws.mean()) - float(cal_ws.mean())) < 0.25 * float(real_ws.mean())
    rt, ct = _stats(root, "fake_real", "glorys", "thetao", sl), _stats(root, "cal", "glorys", "thetao", sl)
    for z in (100.0, 500.0):                                            # subsurface mean follows the target
        assert abs(float(rt.sel(depth=z).mean()) - float(ct.sel(depth=z).mean())) < 1.5
    assert 0.4 < float(ct.sel(depth=100.0).std()) / float(rt.sel(depth=100.0).std()) < 2.5      # spread at 100 m
    # the stand-in target is nearly uniform at 500 m (std ~0.04), so compare absolutely, not as a ratio
    assert abs(float(ct.sel(depth=500.0).std()) - float(rt.sel(depth=500.0).std())) < 0.3


def test_calibrated_generation_is_marked_deterministic_and_chunk_independent(fitted, tmp_path):
    root, cal = fitted
    ds = xr.open_zarr(root / "cal" / "zarr" / "oceanembed.zarr", group="glorys", consolidated=False)
    assert ds.attrs["data_source"] == "synthetic" and ds.attrs["calibrated_to"] == cal.source
    assert "calibrated_to=real store" in (root / "cal" / "SYNTHETIC_DATA.txt").read_text(encoding="utf-8")
    syn.generate("2022-02-01", "2022-02-20", tmp_path / "again", CFG.domain, CFG.depths_m, seed=42,
                 processed_dir=root, calibration=cal, chunk_days=7)
    a = _stats(root, "cal", "glorys", "thetao", slice(None, None))
    b = xr.open_zarr(tmp_path / "again" / "zarr" / "oceanembed.zarr", group="glorys", consolidated=False)["thetao"].load()
    assert np.array_equal(a.values, b.values, equal_nan=True)


def test_default_generation_is_not_calibrated(tmp_path):
    syn.generate("2022-02-01", "2022-02-02", tmp_path / "s", CFG.domain, CFG.depths_m, seed=1, processed_dir=tmp_path)
    ds = xr.open_zarr(tmp_path / "s" / "zarr" / "oceanembed.zarr", group="sst", consolidated=False)
    assert "calibrated_to" not in ds.attrs
    assert "calibrated_to=none" in (tmp_path / "s" / "SYNTHETIC_DATA.txt").read_text(encoding="utf-8")
