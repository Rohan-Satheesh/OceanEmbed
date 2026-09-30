"""Run the model comparison + ablation set (PRD Sections 15-16) with identical budgets, resumably.

    python -m training.experiments                 # everything not yet done
    python -m training.experiments --only oceanembed,unet
    python -m training.experiments --list

Each run writes training/runs/<name>_<data_source>.json (skipped if present, so an interrupted batch
resumes). The combined table is training/runs/experiments_<data_source>.json. All runs use the same
splits, eval days, masks and metrics as the baselines, against GLORYS only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from data.catalog import load_base_config
from training.data import REPO_ROOT
from training.train import TrainConfig, fit

# name -> TrainConfig overrides. `oceanembed` is the reference configuration all ablations differ from.
EXPERIMENTS: dict[str, dict] = {
    "oceanembed": {},
    "unet": {"model": "unet"},                                        # no embedding stage (PRD 13.5 / 15)
    "oe_no_pretrain": {"pretrain_epochs": 0},                         # embedding without MAE pretraining
    "oe_no_physics": {"physics_strat": 0.0, "physics_smooth": 0.0},   # loss-term ablation (PRD 16)
    "oe_anomaly": {"anomaly": True},                                  # anomaly-from-climatology vs raw (PRD 16)
    "oe_window1": {"window": 1},                                      # temporal-window ablation (PRD 12.4, 16)
    "oe_window7": {"window": 7},
    "oe_in_sst": {"inputs": ["sst"]},                                 # input ablations (PRD 16)
    "oe_in_sst_sss": {"inputs": ["sst", "sss"]},
    "oe_in_sst_ssh": {"inputs": ["sst", "ssh"]},
    "oe_in_sst_ssh_currents": {"inputs": ["sst", "ssh", "currents"]},
    # Seed repeats of the two headline models (the reference runs above use seed 42): a gap between them is only
    # meaningful if it exceeds the seed-to-seed spread.
    "oceanembed_s1": {"seed": 1},
    "oceanembed_s2": {"seed": 2},
    "unet_s1": {"model": "unet", "seed": 1},
    "unet_s2": {"model": "unet", "seed": 2},
}


def build(name: str) -> TrainConfig:
    return replace(TrainConfig(name=name), **EXPERIMENTS[name])


def table(results: dict[str, dict]) -> str:
    lines = [f"{'run':26s} {'params':>9s} {'val RMSE':>9s} {'test RMSE':>10s} {'test surf':>10s} {'test >=300m':>12s} "
             f"{'test corr':>10s}"]
    for name, r in results.items():
        v, t = r["splits"]["validation"]["summary"], r["splits"]["test"]["summary"]
        lines.append(f"{name:26s} {r['n_params']:9,d} {v['mean_rmse']:9.3f} {t['mean_rmse']:10.3f} {t['surface_rmse']:10.3f} "
                     f"{t['rmse_300m_plus']:12.3f} {t['mean_corr']:10.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the model comparison / ablation set.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "demo.yaml"))
    ap.add_argument("--only", default=None, help="comma-separated subset of experiment names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    args = ap.parse_args(argv)
    if args.list:
        for n, o in EXPERIMENTS.items():
            print(f"{n:26s} {o}")
        return 0
    cfg = load_base_config(override=args.config)
    names = args.only.split(",") if args.only else list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; see --list")
    runs_dir = REPO_ROOT / cfg.paths["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for n in names:
        path = runs_dir / f"{n}_{cfg.data_source}.json"
        if path.exists() and not args.force:
            print(f"[skip] {n}: {path.name} exists")
        else:
            print(f"[run ] {n}: {EXPERIMENTS[n]}")
            res = fit(cfg, build(n))
            path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        results[n] = json.loads(path.read_text(encoding="utf-8"))
    (runs_dir / f"experiments_{cfg.data_source}.json").write_text(
        json.dumps({"data_source": cfg.data_source, "reference": "GLORYS only", "runs": results}, default=float),
        encoding="utf-8")
    print(f"\ndata_source={cfg.data_source}; reference: GLORYS only\n{table(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
