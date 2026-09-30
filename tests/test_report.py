import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data import synthetic as syn
from data.catalog import load_base_config
from evaluation import report as R
from training import baselines as B
from training import train as T

REPO = Path(__file__).resolve().parents[1]
CFG0 = load_base_config(override=REPO / "configs" / "demo.yaml")
PERIODS = {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-14"],
           "test": ["2022-02-15", "2022-02-28"]}
TINY = dict(epochs=2, pretrain_epochs=1, window=2, base=8, embed_dim=16, film_ch=8, batch_size=4, eval_stride=5,
            patience=3)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("reportstore")
    syn.generate("2022-01-01", "2022-02-28", root / "syn", CFG0.domain, CFG0.depths_m, seed=9, processed_dir=root)
    cfg = replace(CFG0, periods=PERIODS, paths={**CFG0.paths, "processed_dir": "syn", "checkpoints_dir": "ckpt",
                                                "runs_dir": "runs"})
    runs = root / "runs"
    runs.mkdir()
    base = B.evaluate(cfg, stride=5, rf_samples=2000, root=root, log=False)
    (runs / "baselines_synthetic.json").write_text(json.dumps(base), encoding="utf-8")
    for name, model in (("oceanembed", "oceanembed"), ("unet", "unet")):
        res = T.fit(cfg, T.TrainConfig(model=model, name=name, **TINY), root=root, device="cpu", log=False)
        (runs / f"{name}_synthetic.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    return cfg, root, R.build_report(cfg, "oceanembed", "oceanembed", stride=5, mc=3, root=root)


def test_report_tables_and_ablation_reference(built):
    _, _, rep = built
    names = [r["name"] for r in rep["rows"]]
    assert names == ["climatology", "persistence", "random_forest", "oceanembed", "unet"]
    clim = next(r for r in rep["rows"] if r["name"] == "climatology")
    assert clim["test_skill_vs_clim"] == pytest.approx(0.0, abs=1e-9)          # skill of the floor vs itself
    abl = {a["name"]: a for a in rep["ablations"]}
    assert abl["oceanembed"]["delta_vs_reference"] == 0.0 and set(abl) == {"oceanembed", "unet"}


def test_report_analysis_and_calibration(built):
    _, _, rep = built
    a = rep["main_model"]
    assert a["stratified_mean_rmse"] and all(np.isfinite(v) for v in a["stratified_mean_rmse"].values())
    assert all(k.endswith("/non_monsoon") for k in a["stratified_mean_rmse"])      # test period is February only
    c = a["calibration"]
    assert 0.0 <= c["coverage_90"] <= c["coverage_95"] <= 1.0 and c["sharpness"] >= 0 and len(c["reliability"]) >= 3


def test_report_files_banner_and_caveat(built):
    cfg, root, rep = built
    runs = root / "runs"
    md = (runs / "report_synthetic.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in md and "GLORYS only" in md and "not independent validation" in md
    assert "| oceanembed |" in md and "![RMSE by depth]" in md
    assert set(rep["figures"]) == {"rmse_by_depth.png", "reliability.png"}
    assert all((runs / "figures" / f).stat().st_size > 1000 for f in rep["figures"])
    assert json.loads((runs / "report_synthetic.json").read_text(encoding="utf-8"))["data_source"] == "synthetic"
