"""Run the whole demo end to end (Chunks 4-8): on data/synthetic/ by default, or on real data via --config:

    python scripts/run_demo.py            # full reference run (~15 min on a laptop GPU)
    python scripts/run_demo.py --quick    # fewer epochs, for a fast smoke run
    python scripts/run_demo.py --full     # also run every ablation in training.experiments (~1 h)
    python scripts/run_demo.py --dry-run  # print the plan only
    python scripts/run_demo.py --config configs/demo_real_q1.yaml --stride 1 --epochs 60   # REAL Q1 2022 data

Steps (each skipped if its output already exists, unless --force):
  1. synthetic store   2. classical baselines   3. train OceanEmbed + U-Net
  4. predict (NetCDF + Zarr)   5. evaluation report   -> then launch the dashboard.

With the default config everything runs on SYNTHETIC data: scores demonstrate the pipeline, not real-ocean
skill. With --config configs/demo_real_q1.yaml it runs on the real Q1 2022 data (one training month: a
small-sample smoke test, not a skill assessment). Validation is against GLORYS only either way.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.catalog import load_base_config  # noqa: E402

DEMO_CONFIG = REPO_ROOT / "configs" / "demo.yaml"


@dataclass
class Step:
    name: str
    output: Path
    run: Callable[[], object]


def plan(quick: bool, full: bool, config: Path = DEMO_CONFIG, stride: int | None = None,
         epochs_override: int | None = None) -> list[Step]:
    config = Path(config)
    cfg = load_base_config(override=config)
    proc, runs = REPO_ROOT / cfg.paths["processed_dir"], REPO_ROOT / cfg.paths["runs_dir"]
    ds = cfg.data_source
    n_epochs = epochs_override if epochs_override is not None else (6 if quick else None)
    epochs = ["--epochs", str(n_epochs)] if n_epochs is not None else []
    stride_args = ["--stride", str(stride)] if stride is not None else []
    eval_stride = ["--eval-stride", str(stride)] if stride is not None else []

    def synth():
        from data import synthetic
        return synthetic.main([])

    def base():
        from training import baselines
        return baselines.main(["--config", str(config), *stride_args])

    def train(model: str):
        def go():
            from training import train as T
            return T.main(["--config", str(config), "--model", model, "--name", model, *epochs, *eval_stride])
        return go

    def experiments():
        from training import experiments as E
        return E.main(["--config", str(config)])

    def predict():
        from evaluation import predict as P
        return P.main(["--config", str(config), "--checkpoint", "oceanembed", *stride_args])

    def report():
        from evaluation import report as R
        return R.main(["--config", str(config), *stride_args])

    steps = [Step("generate synthetic store", proc / "zarr" / "oceanembed.zarr", synth)] if ds == "synthetic" else []
    steps += [
        Step("classical baselines", runs / f"baselines_{ds}.json", base),
        Step("train OceanEmbed", runs / f"oceanembed_{ds}.json", train("oceanembed")),
        Step("train U-Net baseline", runs / f"unet_{ds}.json", train("unet")),
    ]
    if full:
        steps.append(Step("ablation experiments", runs / f"experiments_{ds}.json", experiments))
    steps += [
        Step("predict (NetCDF + Zarr)", proc / "outputs" / "oceanembed_predictions.zarr", predict),
        Step("evaluation report", runs / f"report_{ds}.md", report),
    ]
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the synthetic-data demo end to end.")
    ap.add_argument("--config", default=str(DEMO_CONFIG), help="override YAML (default: the synthetic demo)")
    ap.add_argument("--stride", type=int, default=None, help="score every Nth eval day (default 3; use 1 for tiny splits)")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    ap.add_argument("--quick", action="store_true", help="6 epochs instead of 20")
    ap.add_argument("--full", action="store_true", help="also run the full ablation set")
    ap.add_argument("--force", action="store_true", help="re-run steps even if their output exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    steps = plan(args.quick, args.full, Path(args.config), args.stride, args.epochs)
    for i, s in enumerate(steps, 1):
        done = s.output.exists() and not args.force
        print(f"[{i}/{len(steps)}] {s.name}: {'skip (exists)' if done else 'run'}  -> {s.output.relative_to(REPO_ROOT)}")
        if args.dry_run or done:
            continue
        if s.run() not in (0, None):
            print(f"step '{s.name}' failed")
            return 1
    if not args.dry_run:
        cfg = load_base_config(override=Path(args.config))
        default = Path(args.config).resolve() == DEMO_CONFIG.resolve()
        env = "" if default else f"$env:OCEANEMBED_CONFIG='{args.config}'   # PowerShell; use `set` in cmd\n    "
        print(f"\nDone. Launch the dashboard with:\n    {env}streamlit run dashboard/app.py")
        if cfg.data_source == "synthetic":
            print("(all data shown is SYNTHETIC — see the banner in the app)")
        else:
            print("(real data, but check the Model performance tab for how much data the model was trained on)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
