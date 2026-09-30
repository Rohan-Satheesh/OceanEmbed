from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data import synthetic as syn
from data.catalog import load_base_config
from training import train as T

REPO = Path(__file__).resolve().parents[1]
CFG0 = load_base_config(override=REPO / "configs" / "demo.yaml")
PERIODS = {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-14"],
           "test": ["2022-02-15", "2022-02-28"]}
TINY = dict(epochs=2, pretrain_epochs=1, window=2, base=8, embed_dim=16, film_ch=8, batch_size=4, eval_stride=5,
            patience=3)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("trainstore")
    syn.generate("2022-01-01", "2022-02-28", root / "syn", CFG0.domain, CFG0.depths_m, seed=3, processed_dir=root)
    cfg = replace(CFG0, periods=PERIODS, paths={**CFG0.paths, "processed_dir": "syn",
                                                "checkpoints_dir": "ckpt", "runs_dir": "runs"})
    return cfg, root


def test_physics_loss_penalizes_inversions_not_stable_profiles():
    z = torch.tensor([0.0, 10.0, 20.0, 50.0])
    stable = (30.0 - 0.1 * z)[None, :, None, None].expand(1, 4, 3, 3).clone()      # linear cooling
    valid = torch.ones_like(stable)
    assert T.physics_loss(stable, valid, z, 1.0, 1.0).item() == pytest.approx(0.0, abs=1e-5)
    inverted = stable.clone()
    inverted[:, 2] += 3.0                                                            # warm layer below cold
    assert T.physics_loss(inverted, valid, z, 1.0, 0.0).item() > 0.1


def test_masked_huber_ignores_masked_pixels():
    pred, tgt = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    pred[0, 0, 0, 0] = 100.0
    mask = torch.ones_like(pred); mask[0, 0, 0, 0] = 0
    assert T.masked_huber(pred, tgt, mask).item() == 0.0


def test_config_hash_and_input_validation():
    a, b = T.TrainConfig(), T.TrainConfig(physics_strat=0.0)
    assert a.hash() == T.TrainConfig().hash() and a.hash() != b.hash()
    assert T.channel_indices(["sst", "ssh"]) == [0, 2]                               # PRD group names work
    assert T.channel_indices(["currents", "winds"]) == [3, 4, 5, 6] and T.channel_indices(["analysed_sst"]) == [0]
    with pytest.raises(ValueError, match="unknown input"):
        T.channel_indices(["salinity"])


@pytest.mark.parametrize("model", ["oceanembed", "unet"])
def test_fit_end_to_end_writes_checkpoint_and_scores(env, model):
    cfg, root = env
    res = T.fit(cfg, T.TrainConfig(model=model, name=f"t_{model}", **TINY), root=root, device="cpu", log=False)
    assert np.isfinite(res["best_val_loss"]) and res["epochs_run"] >= 1 and res["data_source"] == "synthetic"
    for split in ("validation", "test"):
        assert np.isfinite(res["splits"][split]["summary"]["mean_rmse"])
        assert len(res["splits"][split]["per_depth"]) == 15
    ck = torch.load(res["checkpoint"], weights_only=True)                            # plain types only
    assert ck["data_source"] == "synthetic" and ck["config"]["model"] == model and len(ck["depths"]) == 15
    assert "GLORYS only" in res["reference"]


def test_input_subset_ablation_and_window_guard(env):
    cfg, root = env
    res = T.fit(cfg, T.TrainConfig(name="t_sub", inputs=["sst", "ssh", "currents"], **TINY), root=root, device="cpu", log=False)
    assert np.isfinite(res["splits"]["test"]["summary"]["mean_rmse"])
    with pytest.raises(ValueError, match="shrink the eval set"):
        T.fit(cfg, T.TrainConfig(name="t_big", **{**TINY, "window": 9}), root=root, device="cpu", log=False)


def test_anomaly_framing_auto_resolves_from_train_length(env, capsys):
    cfg, root = env                                                     # ~1-month train -> auto OFF
    res = T.fit(cfg, T.TrainConfig(name="t_auto", **TINY), root=root, device="cpu", log=False)
    assert res["config"]["anomaly"] is False and "auto-OFF" in capsys.readouterr().out
    forced = T.fit(cfg, T.TrainConfig(name="t_forced", anomaly=True, **TINY), root=root, device="cpu", log=False)
    assert forced["config"]["anomaly"] is True                          # explicit setting always wins


def test_experiment_registry_is_valid():
    from training import experiments as E
    assert "oceanembed" in E.EXPERIMENTS and "unet" in E.EXPERIMENTS
    for name in E.EXPERIMENTS:
        tc = E.build(name)
        T.channel_indices(tc.inputs)                                   # inputs resolve
        assert tc.window <= 8 and tc.name == name
    assert len({E.build(n).hash() for n in E.EXPERIMENTS}) == len(E.EXPERIMENTS)   # every run is a distinct config
