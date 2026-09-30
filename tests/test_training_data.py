from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data import synthetic as syn
from data.catalog import load_base_config
from evaluation import metrics as M
from training import baselines as B
from training import data as D

REPO = Path(__file__).resolve().parents[1]
CFG = load_base_config(override=REPO / "configs" / "demo.yaml")
PERIODS = {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-14"],
           "test": ["2022-02-15", "2022-02-28"]}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    syn.generate("2022-01-01", "2022-02-28", root / "syn", CFG.domain, CFG.depths_m, seed=1, processed_dir=root)
    cfg = replace(CFG, periods=PERIODS, paths={**CFG.paths, "processed_dir": "syn"})
    return cfg, root, D.load_arrays(cfg, root)


def test_split_is_chronological_with_buffer(small):
    _, _, arr = small
    idx = D.split_indices(arr.time, PERIODS, buffer_days=3)
    assert idx["train"].max() < idx["validation"].min() and idx["validation"].max() < idx["test"].min()
    assert len(idx["train"]) == 31 - 3 and len(idx["validation"]) == 14 - 3 and len(idx["test"]) == 14   # buffers
    assert set(idx["train"]).isdisjoint(idx["validation"])


def test_normalizer_uses_train_only_and_roundtrips(small):
    _, _, arr = small
    idx = D.split_indices(arr.time, PERIODS)
    poisoned_x, poisoned_y = arr.x.copy(), arr.y.copy()
    poisoned_x[idx["test"]] += 1000.0
    poisoned_y[idx["test"]] += 1000.0
    a = D.Normalizer.fit(arr.x[idx["train"]], arr.y[idx["train"]])
    b = D.Normalizer.fit(poisoned_x[idx["train"]], poisoned_y[idx["train"]])
    assert np.allclose(a.x_mean, b.x_mean) and np.allclose(a.y_std, b.y_std)      # test data can't leak in
    z, m = a.y(arr.y[:2])
    back = a.inv_y(z)
    assert np.allclose(back[m > 0], arr.y[:2][m > 0], atol=1e-3)
    assert (z[m == 0] == 0).all()


def test_window_dataset_shapes_masks_and_no_boundary_crossing(small):
    _, _, arr = small
    idx = D.split_indices(arr.time, PERIODS)
    norm = D.Normalizer.fit(arr.x[idx["train"]], arr.y[idx["train"]])
    ds = D.WindowDataset(arr, norm, idx["validation"], window=3)
    assert ds.targets.min() >= idx["validation"][0] + 2                    # first 2 days lack a full in-split window
    x, y, ym, doy, t = ds[0]
    C = len(D.INPUTS)
    assert x.shape == (3 * C * 2, 101, 241) and y.shape == (15, 101, 241) and doy.shape == (2,)
    vals, masks = x[: 3 * C], x[3 * C:]
    assert ((masks == 0) <= (vals == 0)).all()                             # missing -> value 0 with mask 0
    assert set(np.unique(masks.numpy())) <= {0.0, 1.0}


def test_eval_positions_identical_and_inside_split(small):
    _, _, arr = small
    idx = D.split_indices(arr.time, PERIODS)
    p = D.eval_positions(idx["test"], stride=3)
    assert set(p) <= set(idx["test"]) and p[0] == idx["test"][D.MIN_LEAD_DAYS]
    assert np.array_equal(p, D.eval_positions(idx["test"], stride=3))


def test_metrics_values_and_masking():
    true = np.zeros((2, 1, 4, 4)); pred = true + 2.0
    true[0, 0, 0, 0] = np.nan
    r = M.per_depth(pred, true, np.array([0.0]))[0]
    assert r["rmse"] == pytest.approx(2.0) and r["bias"] == pytest.approx(2.0) and r["n"] == 31
    assert M.skill_vs_climatology([1.0], [2.0])[0] == pytest.approx(0.75)


def test_region_and_season_masks():
    lat, lon = np.arange(5, 31, 0.25), np.arange(45, 105.1, 0.25)
    bob, arab = M.region_mask(lat, lon, "bay_of_bengal"), M.region_mask(lat, lon, "arabian_sea")
    assert bob.any() and arab.any() and not (bob & arab).any()
    t = np.array(["2022-07-15", "2022-01-15"], dtype="datetime64[ns]")
    assert M.season_mask(t, "monsoon").tolist() == [True, False]


def test_baselines_run_end_to_end_and_learn(small):
    cfg, root, _ = small
    res = B.evaluate(cfg, stride=4, rf_samples=4000, root=root, log=False)
    for split in ("validation", "test"):
        models = res["splits"][split]["models"]
        assert set(models) == {"climatology", "persistence", "random_forest"}
        assert {m["per_depth"][0]["n"] for m in models.values()} .__len__() >= 1
        for m in models.values():
            assert np.isfinite(m["summary"]["mean_rmse"])
        # the surface->subsurface signal exists, so the RF must beat a wrong-season climatology at the surface
        assert models["random_forest"]["summary"]["surface_rmse"] < models["climatology"]["summary"]["surface_rmse"]
    assert "GLORYS only" in res["reference"]
