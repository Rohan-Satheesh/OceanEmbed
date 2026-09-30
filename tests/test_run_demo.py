import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_demo", REPO / "scripts" / "run_demo.py")
run_demo = importlib.util.module_from_spec(spec)
sys.modules["run_demo"] = run_demo            # dataclasses resolve their module through sys.modules
spec.loader.exec_module(run_demo)


def test_plan_order_and_full_adds_ablations():
    names = [s.name for s in run_demo.plan(quick=False, full=False)]
    assert names == ["generate synthetic store", "classical baselines", "train OceanEmbed", "train U-Net baseline",
                     "predict (NetCDF + Zarr)", "evaluation report"]
    assert "ablation experiments" in [s.name for s in run_demo.plan(quick=False, full=True)]


def test_all_outputs_stay_in_demo_locations_not_real_processed_dir():
    for s in run_demo.plan(False, True):
        rel = s.output.relative_to(REPO).as_posix()
        assert not rel.startswith("data/processed") and not rel.startswith("data/raw"), rel


def test_dry_run_executes_nothing(monkeypatch, capsys):
    monkeypatch.setattr(run_demo, "plan", lambda *a, **k: [run_demo.Step("x", REPO / "nope.txt",
                                                                       lambda: (_ for _ in ()).throw(AssertionError("ran")))])
    assert run_demo.main(["--dry-run"]) == 0
    assert "[1/1] x: run" in capsys.readouterr().out


def test_quick_mode_passes_fewer_epochs(monkeypatch):
    seen = {}
    from training import train as T
    monkeypatch.setattr(T, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    step = next(s for s in run_demo.plan(True, False) if s.name == "train OceanEmbed")
    step.run()
    assert seen["argv"][seen["argv"].index("--epochs") + 1] == "6"
    assert "oceanembed" in seen["argv"]


def test_real_q1_plan_skips_synthetic_and_isolates_outputs():
    steps = run_demo.plan(False, False, REPO / "configs" / "demo_real_q1.yaml", stride=1, epochs_override=60)
    assert [s.name for s in steps][0] == "classical baselines"                     # no synthetic generation for real data
    assert "generate synthetic store" not in [s.name for s in steps]
    for s in steps:
        rel = s.output.relative_to(REPO).as_posix()
        assert (rel.startswith("data/processed/outputs") or "runs/real_q1" in rel) and not rel.startswith("data/synthetic"), rel


def test_real_q1_config_values():
    from data.catalog import load_base_config
    cfg = load_base_config(override=REPO / "configs" / "demo_real_q1.yaml")
    assert cfg.data_source == "real" and cfg.paths["processed_dir"] == "data/processed"
    assert cfg.paths["checkpoints_dir"] == "training/runs/real_q1/checkpoints"        # never the synthetic checkpoints
    assert cfg.periods == {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-28"],
                           "test": ["2022-03-01", "2022-03-31"]}


def test_stride_and_epochs_reach_the_underlying_commands(monkeypatch):
    from training import baselines, train as T
    seen = {}
    monkeypatch.setattr(T, "main", lambda argv: seen.setdefault("train", argv) and 0)
    monkeypatch.setattr(baselines, "main", lambda argv: seen.setdefault("base", argv) and 0)
    for s in run_demo.plan(False, False, REPO / "configs" / "demo_real_q1.yaml", stride=1, epochs_override=60):
        if s.name in ("train OceanEmbed", "classical baselines"):
            s.run()
    assert seen["train"][seen["train"].index("--epochs") + 1] == "60" and seen["train"][seen["train"].index("--eval-stride") + 1] == "1"
    assert seen["base"][seen["base"].index("--stride") + 1] == "1"
