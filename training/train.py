"""Training + scoring for OceanEmbed and the plain U-Net baseline (PRD Sections 13.6, 14, 16).

    python -m training.train --model oceanembed --name oe_full
    python -m training.train --model unet --name unet
    python -m training.train --model oceanembed --no-pretrain --name oe_nopretrain     # ablation
    python -m training.train --model oceanembed --physics 0 --name oe_nophysics        # ablation
    python -m training.train --model oceanembed --inputs sst,ssh --name oe_sst_ssh     # ablation

Loss: masked Huber on the normalized temperature (anomaly-from-climatology framing by default) plus
physics-aware terms (stratification stability, vertical smoothness) in physical units. Every run
logs through RunLogger with its data_source, and the checkpoint records the data source, config
hash and train-only normalization statistics. All models are scored on the same eval days and
masks as the baselines, against GLORYS only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.catalog import BaseConfig, load_base_config
from evaluation import metrics as M
from models.embedding import MaskedAutoencoder
from models.reconstruction import OceanEmbedModel
from models.unet import UNet
from training.baselines import fit_climatology
from training.data import (INPUT_NAMES, MIN_LEAD_DAYS, REPO_ROOT, Normalizer, StoreArrays, WindowDataset, eval_positions,
                           load_arrays, split_indices)
from training.run_logger import RunLogger


MIN_CLIM_YEARS = 3


@dataclass
class TrainConfig:
    model: str = "oceanembed"            # "oceanembed" | "unet"
    name: str = "run"
    window: int = 3                      # days of surface input (PRD 12.4: tune/ablate 1, 3, 7)
    epochs: int = 20
    pretrain_epochs: int = 5             # MAE embedding pretraining (oceanembed only); 0 = random-init encoder
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    base: int = 32
    embed_dim: int = 64
    film_ch: int = 24
    # Predict the anomaly relative to the train-period climatology (PRD 13.6). None = auto: only when the
    # training period spans >= MIN_CLIM_YEARS, because with fewer years the climatology is fit on the very days
    # it is subtracted from, so train anomalies collapse to ~0 and the model never sees real anomalies.
    anomaly: bool | None = None
    physics_strat: float = 0.1           # weight on unstable-stratification penalty (deg C units)
    physics_smooth: float = 1.0          # weight on vertical-gradient smoothness penalty
    patience: int = 6
    inputs: list[str] = field(default_factory=lambda: list(INPUT_NAMES))
    seed: int = 42
    buffer_days: int = 3
    eval_stride: int = 3

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------------------------- losses
def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=delta) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def physics_loss(pred_phys: torch.Tensor, valid: torch.Tensor, depths: torch.Tensor,
                 w_strat: float, w_smooth: float) -> torch.Tensor:
    """Physics-aware regularizers on the physical (deg C) profile (PRD 13.6 / 14.5).
    - stratification: penalize warmer-below-than-above (temperature increasing with depth)
    - smoothness: penalize abrupt changes of the vertical temperature gradient.
    NB real profiles do contain rare inversions (e.g. Bay of Bengal barrier layers), so this is a soft prior."""
    pair = valid[:, 1:] * valid[:, :-1]
    strat = (F.relu(pred_phys[:, 1:] - pred_phys[:, :-1]) * pair).sum() / pair.sum().clamp(min=1.0)
    dz = (depths[1:] - depths[:-1])[None, :, None, None]
    grad = (pred_phys[:, 1:] - pred_phys[:, :-1]) / dz
    tri = pair[:, 1:] * pair[:, :-1]
    smooth = ((grad[:, 1:] - grad[:, :-1]).abs() * tri).sum() / tri.sum().clamp(min=1.0)
    return w_strat * strat + w_smooth * smooth


# --------------------------------------------------------------------------------------------- building
INPUT_ALIASES = {"sst": ["analysed_sst"], "sss": ["sos"], "ssh": ["sla"], "currents": ["u", "v"],
                 "winds": ["uwnd", "vwnd"]}


def channel_indices(names: list[str]) -> list[int]:
    """Indices of the requested input variables. Accepts variable names or the PRD's group names
    (sst, sss, ssh, currents, winds), e.g. ['sst', 'ssh', 'currents'] for the SST+SSH+currents ablation."""
    expanded = [v for n in names for v in INPUT_ALIASES.get(n, [n])]
    missing = [n for n in expanded if n not in INPUT_NAMES]
    if missing:
        raise ValueError(f"unknown input variable(s) {missing}; choose from {INPUT_NAMES} or {list(INPUT_ALIASES)}")
    return [INPUT_NAMES.index(n) for n in expanded]


def build_model(tc: TrainConfig, arrays: StoreArrays) -> torch.nn.Module:
    in_ch = tc.window * len(channel_indices(tc.inputs)) * 2
    if tc.model == "unet":
        return UNet(in_ch, len(arrays.depth), tc.base, tc.dropout)
    if tc.model == "oceanembed":
        return OceanEmbedModel(in_ch, arrays.depth.tolist(), arrays.lat, arrays.lon, tc.embed_dim, tc.base,
                               tc.film_ch, tc.dropout)
    raise ValueError(f"unknown model '{tc.model}'")


def climatology_norm(clim: np.ndarray, norm: Normalizer) -> torch.Tensor:
    z = (clim - norm.y_mean[None, :, None, None]) / norm.y_std[None, :, None, None]
    return torch.from_numpy(np.nan_to_num(z, nan=0.0).astype("float32"))


def _doy_index(t_idx: torch.Tensor, time_axis: np.ndarray) -> torch.Tensor:
    days = time_axis[t_idx.cpu().numpy()].astype("datetime64[D]")
    return torch.from_numpy(((days - days.astype("datetime64[Y]")).astype(int)).astype("int64"))   # 0-based doy


def forward_norm(model: torch.nn.Module, x, doy, clim_n, t, time_axis, tc: TrainConfig, dev):
    """Model output in normalized target units (adds the climatology back when anomaly framing is on)."""
    out = model(x, doy)
    if tc.anomaly:
        out = out + clim_n[_doy_index(t, time_axis).to(dev)]
    return out


# --------------------------------------------------------------------------------------------- training
def pretrain_encoder(tc: TrainConfig, ds: WindowDataset, dev, log) -> dict | None:
    if tc.model != "oceanembed" or tc.pretrain_epochs <= 0:
        return None
    n_fields = len(channel_indices(tc.inputs))
    in_ch = tc.window * n_fields * 2
    mae = MaskedAutoencoder(in_ch, n_fields, tc.window, tc.embed_dim, tc.base).to(dev)
    opt = torch.optim.AdamW(mae.parameters(), tc.lr, weight_decay=tc.weight_decay)
    dl = DataLoader(ds, tc.batch_size, shuffle=True, drop_last=True)
    for ep in range(tc.pretrain_epochs):
        tot, n = 0.0, 0
        for x, *_ in dl:
            x = x.to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = mae.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        log({"pretrain/mae_loss": tot / max(n, 1)}, ep)
    return mae.encoder.state_dict()


def run_epoch(model, dl, opt, sched, clim_n, tc, arrays, norm_t, depths, dev, train: bool) -> float:
    model.train(train)
    tot, n = 0.0, 0
    for x, y, ym, doy, t in dl:
        x, y, ym, doy = x.to(dev), y.to(dev), ym.to(dev), doy.to(dev)
        with torch.set_grad_enabled(train), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = forward_norm(model, x, doy, clim_n, t, arrays.time, tc, dev).float()
            loss = masked_huber(pred, y, ym)
            if tc.physics_strat or tc.physics_smooth:
                phys = pred * norm_t[1][None, :, None, None] + norm_t[0][None, :, None, None]
                loss = loss + physics_loss(phys, ym, depths, tc.physics_strat, tc.physics_smooth)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        tot, n = tot + loss.item(), n + 1
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, ds: WindowDataset, clim_n, tc, arrays, norm: Normalizer, dev, mc: int = 0):
    """Physical-unit predictions (n, Z, H, W) for the dataset's eval positions; optionally MC-dropout std."""
    dl = DataLoader(ds, tc.batch_size, shuffle=False)
    preds, stds = [], []
    for x, _y, _ym, doy, t in dl:
        x, doy = x.to(dev), doy.to(dev)
        if mc:
            def fwd(x_, d_):
                return forward_norm(model, x_, d_, clim_n, t, arrays.time, tc, dev)
            model.eval()
            for mod in model.modules():
                if isinstance(mod, torch.nn.Dropout2d):
                    mod.train()
            draws = torch.stack([fwd(x, doy).float() for _ in range(mc)])
            mean, std = draws.mean(0), draws.std(0)
            stds.append(std.cpu().numpy() * norm.y_std[None, :, None, None])
        else:
            model.eval()
            mean = forward_norm(model, x, doy, clim_n, t, arrays.time, tc, dev).float()
        preds.append(norm.inv_y(mean.cpu().numpy()))
    return np.concatenate(preds), (np.concatenate(stds) if stds else None)


def fit(cfg: BaseConfig, tc: TrainConfig, root: Path = REPO_ROOT, device: str | None = None, log: bool = True) -> dict:
    if tc.window > MIN_LEAD_DAYS + 1:
        raise ValueError(f"window {tc.window} > {MIN_LEAD_DAYS + 1} days would shrink the eval set below the baselines'")
    if tc.anomaly is None:
        y0, y1 = (np.datetime64(d) for d in cfg.periods["train"])
        years = float((y1 - y0) / np.timedelta64(1, "D") + 1) / 365.25
        tc = replace(tc, anomaly=years >= MIN_CLIM_YEARS - 0.05)
        print(f"anomaly framing auto-{'ON' if tc.anomaly else 'OFF'} (train period = {years:.1f} yr, needs >= {MIN_CLIM_YEARS})")
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    arrays = load_arrays(cfg, root)
    idx = split_indices(arrays.time, cfg.periods, tc.buffer_days)
    norm = Normalizer.fit(arrays.x[idx["train"]], arrays.y[idx["train"]])            # train-only (PRD 11.6)
    clim_n = climatology_norm(fit_climatology(arrays.y, arrays.time, idx["train"]), norm).to(dev)
    chans = channel_indices(tc.inputs)
    mk = lambda split, pos=None: WindowDataset(arrays, norm, idx[split], tc.window, pos, chans)  # noqa: E731
    train_ds, val_ds = mk("train"), mk("validation")
    norm_t = (torch.from_numpy(norm.y_mean).float().to(dev), torch.from_numpy(norm.y_std).float().to(dev))
    depths = torch.from_numpy(arrays.depth).float().to(dev)

    logger = RunLogger(f"{tc.name}-{cfg.data_source}", {**asdict(tc), "data_source": cfg.data_source, "periods": cfg.periods}) if log else None
    lg = (lambda m, s=None: logger.log(m, s)) if logger else (lambda m, s=None: None)
    t0 = time.time()
    model = build_model(tc, arrays).to(dev)
    enc_state = pretrain_encoder(tc, train_ds, dev, lg)
    if enc_state is not None:
        model.encoder.load_state_dict(enc_state)

    enc_params = list(model.encoder.parameters()) if getattr(model, "encoder", None) is not None else []
    enc_ids = {id(p) for p in enc_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": tc.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": tc.lr * (0.3 if enc_state is not None else 1.0)})
    opt = torch.optim.AdamW(groups, weight_decay=tc.weight_decay)
    train_dl = DataLoader(train_ds, tc.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, tc.batch_size, shuffle=False)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=tc.epochs * len(train_dl),
                                                pct_start=0.1, anneal_strategy="cos")
    best, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(tc.epochs):
        tr = run_epoch(model, train_dl, opt, sched, clim_n, tc, arrays, norm_t, depths, dev, True)
        va = run_epoch(model, val_dl, None, None, clim_n, tc, arrays, norm_t, depths, dev, False)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        lg({"train_loss": tr, "val_loss": va}, ep)
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.patience:                                                   # early stopping on val loss
                break
    model.load_state_dict(best_state)

    results = {"name": tc.name, "config": asdict(tc), "config_hash": tc.hash(), "data_source": cfg.data_source,
               "reference": "GLORYS only (GLORYS assimilates ARGO; no independent validation)",
               "n_params": sum(p.numel() for p in model.parameters()), "best_val_loss": best,
               "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1), "history": history, "splits": {}}
    for split in ("validation", "test"):
        pos = eval_positions(idx[split], tc.eval_stride)
        ds = mk(split, pos)
        pred, _ = predict(model, ds, clim_n, tc, arrays, norm, dev)
        rows = M.per_depth(pred, arrays.y[ds.targets], arrays.depth)
        results["splits"][split] = {"n_eval_days": int(len(ds)), "summary": M.summarize(rows), "per_depth": rows}
        lg({f"{split}/{k}": v for k, v in results["splits"][split]["summary"].items()})
    ckpt_dir = root / cfg.paths["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{tc.name}-{tc.hash()}.pt"
    torch.save({"state": best_state, "config": asdict(tc), "data_source": cfg.data_source, "periods": cfg.periods,
                "norm": {k: v.tolist() for k, v in norm.__dict__.items()}, "lat": arrays.lat.tolist(), "lon": arrays.lon.tolist(),
                "depths": arrays.depth.tolist()}, ckpt)
    results["checkpoint"] = str(ckpt)
    if logger:
        logger.finish()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train + score OceanEmbed or the U-Net baseline.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "demo.yaml"), help="override YAML; 'none' = base only")
    ap.add_argument("--model", default="oceanembed", choices=["oceanembed", "unet"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-stride", type=int, default=None, help="score every Nth day of validation/test (default 3)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--no-pretrain", action="store_true", help="ablation: skip MAE embedding pretraining")
    ap.add_argument("--anomaly", dest="anomaly", action="store_true", default=None,
                    help="force climatology-anomaly framing (default: auto, needs >= 3 train years)")
    ap.add_argument("--no-anomaly", dest="anomaly", action="store_false", help="force raw-temperature target")
    ap.add_argument("--physics", type=float, default=None, help="scale both physics loss weights (0 = ablation)")
    ap.add_argument("--inputs", default=None, help="comma-separated subset: variables (" + ",".join(INPUT_NAMES) + ") or groups (sst,sss,ssh,currents,winds)")
    args = ap.parse_args(argv)
    tc = TrainConfig(model=args.model, name=args.name or args.model)
    if args.epochs is not None:
        tc.epochs = args.epochs
    if args.eval_stride is not None:
        tc.eval_stride = args.eval_stride
    if args.window is not None:
        tc.window = args.window
    if args.no_pretrain:
        tc.pretrain_epochs = 0
    if args.anomaly is not None:
        tc.anomaly = args.anomaly
    if args.physics is not None:
        tc.physics_strat *= args.physics
        tc.physics_smooth *= args.physics
    if args.inputs:
        tc.inputs = args.inputs.split(",")
    cfg = load_base_config(override=None if args.config == "none" else args.config)
    res = fit(cfg, tc)
    out = REPO_ROOT / cfg.paths["runs_dir"] / f"{tc.name}_{cfg.data_source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for split, blob in res["splits"].items():
        s = blob["summary"]
        print(f"[{split}] {tc.name}: mean RMSE {s['mean_rmse']:.3f}  surface {s['surface_rmse']:.3f}  "
              f">=300m {s['rmse_300m_plus']:.3f}  bias {s['mean_bias']:.3f}  corr {s['mean_corr']:.3f}")
    print(f"params={res['n_params']:,} epochs={res['epochs_run']} time={res['train_seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
