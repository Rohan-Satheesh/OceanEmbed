"""Evaluation report (PRD Section 18): model + ablation comparison on identical eval days, per-depth
skill, region x season breakdown and MC-dropout calibration for the main model, with figures.

    python -m evaluation.report --main oceanembed

Inputs (already produced): training/runs/baselines_<ds>.json, training/runs/<experiment>_<ds>.json.
Outputs: training/runs/report_<ds>.json, report_<ds>.md, figures/*.png.

Every table is scored against GLORYS only (GLORYS assimilates ARGO; no independent validation), and a
synthetic-data run says so at the top — those numbers show the pipeline works, not real-ocean skill.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np

from data.catalog import BaseConfig, load_base_config
from evaluation import metrics as M
from evaluation import uncertainty as U
from evaluation.predict import find_checkpoint, predict_split
from training.data import REPO_ROOT

SYNTHETIC_BANNER = ("SYNTHETIC DATA: these results come from a synthetic ocean and only demonstrate that the "
                    "pipeline runs end to end. They say nothing about skill on the real ocean.")
GLORYS_CAVEAT = "All scores are against GLORYS only; GLORYS assimilates ARGO, so this is not independent validation."


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_rows(baselines: dict, experiments: dict[str, dict]) -> list[dict]:
    """One row per model/ablation with validation + test summaries and depth-mean skill vs climatology."""
    clim_test = np.array([r["rmse"] for r in baselines["splits"]["test"]["models"]["climatology"]["per_depth"]])
    rows = []
    for name, blob in baselines["splits"]["test"]["models"].items():
        rows.append({"name": name, "kind": "baseline", "params": None,
                     "val": baselines["splits"]["validation"]["models"][name]["summary"], "test": blob["summary"],
                     "test_per_depth": [r["rmse"] for r in blob["per_depth"]]})
    for name, r in experiments.items():
        pd_ = [d["rmse"] for d in r["splits"]["test"]["per_depth"]]
        rows.append({"name": name, "kind": "model", "params": r["n_params"], "val": r["splits"]["validation"]["summary"],
                     "test": r["splits"]["test"]["summary"], "test_per_depth": pd_})
    for row in rows:
        row["test_skill_vs_clim"] = float(np.nanmean(M.skill_vs_climatology(row["test_per_depth"], clim_test)))
    return rows


SEED_RE = re.compile(r"^(?P<base>.+?)_s\d+$")          # e.g. oceanembed_s1 -> base "oceanembed"


def ablation_deltas(rows: list[dict], reference: str) -> list[dict]:
    """Deltas vs the reference run. Seed repeats are excluded here: they belong in seed_summary."""
    ref = next(r for r in rows if r["name"] == reference)["test"]["mean_rmse"]
    return [{"name": r["name"], "test_mean_rmse": r["test"]["mean_rmse"], "delta_vs_reference": r["test"]["mean_rmse"] - ref}
            for r in rows if r["kind"] == "model" and not SEED_RE.match(r["name"])]


def seed_summary(rows: list[dict]) -> list[dict]:
    """Mean / std / range of validation and test RMSE across seed repeats (`<name>` plus `<name>_s<k>`)."""
    bases = sorted({m["base"] for r in rows if (m := SEED_RE.match(r["name"]))})
    out = []
    for base in bases:
        runs = [r for r in rows if r["name"] == base or (SEED_RE.match(r["name"]) and SEED_RE.match(r["name"])["base"] == base)]
        if len(runs) < 2:
            continue
        test = np.array([r["test"]["mean_rmse"] for r in runs])
        val = np.array([r["val"]["mean_rmse"] for r in runs])
        out.append({"model": base, "n_seeds": len(runs), "test_mean": float(test.mean()), "test_std": float(test.std(ddof=1)),
                    "test_min": float(test.min()), "test_max": float(test.max()), "val_mean": float(val.mean()),
                    "val_std": float(val.std(ddof=1)), "runs": [r["name"] for r in runs]})
    return out


def main_model_analysis(cfg: BaseConfig, main: str, root: Path, stride: int, mc: int) -> dict:
    ck = find_checkpoint(cfg, main, root)
    r = predict_split(cfg, ck, "test", stride, mc, root)
    strat = M.stratified(r["mean"], r["truth"], r["time"], r["lat"], r["lon"], r["depth"])
    return {
        "checkpoint": ck.name,
        "stratified_mean_rmse": {k: float(np.nanmean([d["rmse"] for d in rows])) for k, rows in strat.items()},
        "stratified_per_depth": strat,
        "calibration": U.calibration(r["mean"], r["std"], r["truth"]),
    }


def make_figures(rows: list[dict], analysis: dict, depths: list[float], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        if row["name"] in ("climatology", "persistence", "random_forest", "unet", "oceanembed"):
            ax.plot(row["test_per_depth"], depths, marker="o", ms=3, label=row["name"],
                    lw=2.5 if row["name"] == "oceanembed" else 1.2)
    ax.invert_yaxis(); ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("test RMSE (deg C) vs GLORYS"); ax.set_ylabel("depth (m)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("RMSE by depth (test)")
    fig.tight_layout(); fig.savefig(out_dir / "rmse_by_depth.png", dpi=110); plt.close(fig); made.append("rmse_by_depth.png")

    rel = analysis["calibration"]["reliability"]
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(max(b["mean_std"] for b in rel), max(b["rmse"] for b in rel)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal")
    ax.plot([b["mean_std"] for b in rel], [b["rmse"] for b in rel], "o-", label="MC dropout")
    ax.set_xlabel("predicted std (deg C)"); ax.set_ylabel("observed RMSE (deg C)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Uncertainty reliability (test)")
    fig.tight_layout(); fig.savefig(out_dir / "reliability.png", dpi=110); plt.close(fig); made.append("reliability.png")
    return made


def to_markdown(report: dict) -> str:
    L = [f"# OceanEmbed evaluation report ({report['data_source']})", ""]
    if report["data_source"] == "synthetic":
        L += [f"> **{SYNTHETIC_BANNER}**", ""]
    days = (np.datetime64(report["periods"]["train"][1]) - np.datetime64(report["periods"]["train"][0])).astype(int) + 1
    if days < 365:
        L += [f"> **Training period is only {days} days (< 1 year).** Splits are chronological but seasonally uneven, so "
              "treat these numbers as a small-sample smoke test, not a skill assessment.", ""]
    L += [f"_{GLORYS_CAVEAT}_", "", f"Generated {report['generated_utc']}. Periods: {report['periods']}.", "",
          "## Model comparison (identical eval days, masks, metrics)", "",
          "| model | params | val RMSE | test RMSE | test surface | test >=300 m | test bias | test corr | skill vs clim |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        t, v = r["test"], r["val"]
        L.append(f"| {r['name']} | {r['params'] or '-'} | {v['mean_rmse']:.3f} | {t['mean_rmse']:.3f} | {t['surface_rmse']:.3f} | "
                 f"{t['rmse_300m_plus']:.3f} | {t['mean_bias']:.3f} | {t['mean_corr']:.3f} | {r['test_skill_vs_clim']:.3f} |")
    L += ["", f"## Ablations (reference: `{report['reference']}`; negative delta = better)", "",
          "| run | test RMSE | delta vs reference |", "|---|---:|---:|"]
    L += [f"| {a['name']} | {a['test_mean_rmse']:.3f} | {a['delta_vs_reference']:+.3f} |" for a in report["ablations"]]
    seeds = report.get("seeds", [])
    if seeds:
        L += ["", "## Seed variability (same configuration, different random seeds)", "",
              "| model | seeds | test RMSE mean +/- std | test range | val RMSE mean +/- std |", "|---|---:|---:|---:|---:|"]
        L += [f"| {s['model']} | {s['n_seeds']} | {s['test_mean']:.3f} +/- {s['test_std']:.3f} | "
              f"{s['test_min']:.3f}-{s['test_max']:.3f} | {s['val_mean']:.3f} +/- {s['val_std']:.3f} |" for s in seeds]
        L += ["", "A difference between two models (or ablations) smaller than these spreads is not meaningful."]
        if len(seeds) >= 2:
            (x, y) = sorted(seeds, key=lambda s: s["test_mean"])[:2]
            gap = y["test_mean"] - x["test_mean"]
            spread = max(x["test_std"], y["test_std"])
            L += [f"Best two: `{x['model']}` ({x['test_mean']:.3f}) vs `{y['model']}` ({y['test_mean']:.3f}); gap {gap:.3f} = "
                  f"{gap / spread:.1f}x the larger seed std." if spread > 0 else ""]
    a = report["main_model"]
    L += ["", f"## Region x season breakdown ({report['reference']}, test mean RMSE)", "", "| stratum | mean RMSE |", "|---|---:|"]
    L += [f"| {k} | {v:.3f} |" for k, v in a["stratified_mean_rmse"].items()]
    c = a["calibration"]
    L += ["", "## Uncertainty (MC dropout)", "",
          f"- 90% interval coverage: {c['coverage_90']:.3f} (ideal 0.90); 95%: {c['coverage_95']:.3f} (ideal 0.95)",
          f"- spread-error Spearman: {c['spread_error_spearman']:.3f}; sharpness (mean std): {c['sharpness']:.3f} deg C",
          "- Under-coverage means the stated interval is too narrow: treat the uncertainty as a relative confidence "
          "indicator, not a calibrated interval, unless coverage is near ideal.", "",
          "![RMSE by depth](figures/rmse_by_depth.png)", "", "![Reliability](figures/reliability.png)", ""]
    return "\n".join(L)


def build_report(cfg: BaseConfig, main: str = "oceanembed", reference: str = "oceanembed", stride: int = 3,
                 mc: int = 10, root: Path = REPO_ROOT) -> dict:
    runs = root / cfg.paths["runs_dir"]
    baselines = _read(runs / f"baselines_{cfg.data_source}.json")
    exps = {p.name[: -len(f"_{cfg.data_source}.json")]: _read(p) for p in sorted(runs.glob(f"*_{cfg.data_source}.json"))
            if not p.name.startswith(("baselines_", "experiments_", "report_"))}
    rows = comparison_rows(baselines, exps)
    analysis = main_model_analysis(cfg, main, root, stride, mc)
    report = {"data_source": cfg.data_source, "periods": cfg.periods, "reference": reference, "main": main,
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "caveat": GLORYS_CAVEAT, "rows": rows, "ablations": ablation_deltas(rows, reference), "seeds": seed_summary(rows),
              "main_model": analysis}
    figs = make_figures(rows, analysis, baselines["depths_m"], runs / "figures")
    (runs / f"report_{cfg.data_source}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (runs / f"report_{cfg.data_source}.md").write_text(to_markdown(report), encoding="utf-8")
    report["figures"] = figs
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the evaluation report from finished runs.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "demo.yaml"))
    ap.add_argument("--main", default="oceanembed", help="checkpoint/run name to analyse in depth")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--mc", type=int, default=10)
    args = ap.parse_args(argv)
    cfg = load_base_config(override=None if args.config == "none" else args.config)
    build_report(cfg, args.main, args.main, args.stride, args.mc)
    print((REPO_ROOT / cfg.paths["runs_dir"] / f"report_{cfg.data_source}.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
