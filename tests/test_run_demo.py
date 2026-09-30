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
    monkeypatch.setattr(run_demo, "plan", lambda q, f: [run_demo.Step("x", REPO / "nope.txt",
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
