from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import data as D
from data import synthetic as syn
from data.catalog import load_base_config
from evaluation import predict as P
from training import train as T

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "dashboard" / "app.py")
CFG0 = load_base_config(override=REPO / "configs" / "demo.yaml")
PERIODS = {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-14"],
           "test": ["2022-02-15", "2022-02-28"]}
TINY = dict(epochs=2, pretrain_epochs=1, window=2, base=8, embed_dim=16, film_ch=8, batch_size=4, eval_stride=5,
            patience=3)


@pytest.fixture(scope="module")
def zarr_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashstore")
    syn.generate("2022-01-01", "2022-02-28", root / "syn", CFG0.domain, CFG0.depths_m, seed=11, processed_dir=root)
    cfg = replace(CFG0, periods=PERIODS, paths={**CFG0.paths, "processed_dir": "syn", "checkpoints_dir": "ckpt"})
    res = T.fit(cfg, T.TrainConfig(name="dash_t", **TINY), root=root, device="cpu", log=False)
    return P.run_inference(cfg, Path(res["checkpoint"]), "test", stride=4, mc=3, root=root, device="cpu")


@pytest.fixture
def ds(zarr_path):
    return D.open_predictions(zarr_path)


def test_paths_respect_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OCEANEMBED_PREDICTIONS", str(tmp_path / "x.zarr"))
    monkeypatch.setenv("OCEANEMBED_RUNS", str(tmp_path / "runs"))
    assert D.predictions_path() == tmp_path / "x.zarr" and D.runs_path() == tmp_path / "runs"
    monkeypatch.delenv("OCEANEMBED_PREDICTIONS")
    assert D.predictions_path() == REPO / "data" / "synthetic" / "outputs" / "oceanembed_predictions.zarr"   # demo config default


def test_banner_only_for_synthetic(ds):
    assert "SYNTHETIC DATA" in D.data_banner(ds)
    real = ds.copy()
    real.attrs = {**ds.attrs, "data_source": "real"}
    assert D.data_banner(real) is None


def test_map_fields(ds):
    assert D.map_field(ds, "temp", 0, 0).dims == ("lat", "lon")
    err = D.map_field(ds, "error", 1, 3)
    ref = (ds["temp"] - ds["glorys"]).isel(time=1, depth=3)
    assert np.allclose(err.values, ref.values, equal_nan=True)                 # error = reconstruction - GLORYS
    for var in D.DEPTH_FREE:
        assert D.map_field(ds, var, 0).dims == ("lat", "lon")
    assert {v[0] for v in D.MAP_VARIABLES.values()} >= {"temp", "temp_std", "error", "glorys", "mld", "d20", "tchp"}


def test_location_helpers_land_vs_ocean(ds):
    lat, lon = D.default_location(ds)
    assert D.is_ocean(ds, lat, lon)
    assert not D.is_ocean(ds, 25.0, 80.0)                                      # north India: land
    la, lo = D.nearest_cell(ds, 14.9, 88.1)
    assert (la, lo) == (15.0, 88.0)


def test_profile_timeseries_and_regions(ds):
    lat, lon = D.default_location(ds)
    prof = D.profile(ds, 0, lat, lon)
    assert list(prof.columns) == ["depth_m", "temp", "std", "glorys"] and len(prof) == 15 and prof["temp"].notna().any()
    assert prof["temp"].dropna().iloc[0] > prof["temp"].dropna().iloc[-1]      # warmer at the surface
    ts = D.timeseries(ds, 0, lat, lon)
    assert len(ts) == ds.sizes["time"] and ts["temp"].notna().all()
    for region in D.REGION_LABELS:
        s = D.regional_summary(ds, region, 0)
        assert len(s) == ds.sizes["time"] and np.isfinite(s["rmse"]).all() and (s["rmse"] >= 0).all()


def test_app_runs_shows_banner_and_every_map_variable(monkeypatch, zarr_path):
    monkeypatch.setenv("OCEANEMBED_PREDICTIONS", str(zarr_path))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("SYNTHETIC DATA" in e.value for e in at.error)                  # banner always visible
    assert len(at.tabs) == 6 and at.title[0].value.startswith("OceanEmbed")
    variable_box = next(s for s in at.selectbox if s.label == "Map variable")
    for label in variable_box.options:
        at = variable_box.select(label).run()
        assert not at.exception, f"app failed on map variable '{label}'"
        variable_box = next(s for s in at.selectbox if s.label == "Map variable")


def test_app_land_point_warns_instead_of_crashing(monkeypatch, zarr_path):
    monkeypatch.setenv("OCEANEMBED_PREDICTIONS", str(zarr_path))
    at = AppTest.from_file(APP, default_timeout=180).run()
    lat = next(n for n in at.number_input if n.label == "Latitude (N)")
    lon = next(n for n in at.number_input if n.label == "Longitude (E)")
    lat.set_value(25.0); lon.set_value(80.0)
    at = at.run()
    assert not at.exception and any("land" in w.value for w in at.sidebar.warning)


def test_app_without_predictions_explains_how_to_generate(monkeypatch, tmp_path):
    monkeypatch.setenv("OCEANEMBED_PREDICTIONS", str(tmp_path / "missing.zarr"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("evaluation.predict" in e.value for e in at.error)
