"""Inference pipeline (PRD Sections 17, 19, 21.4): checkpoint -> daily subsurface temperature fields with
MC-dropout uncertainty, quality flags and derived products (MLD, D20, TCHP).

    python -m evaluation.predict --checkpoint oceanembed --split test --stride 3

Writes, under <processed_dir>/outputs/:
    netcdf/oceanembed_temp_<YYYYMMDD>.nc      one CF-1.8 NetCDF4 file per day (PRD 17.2 naming)
    oceanembed_predictions.zarr               everything stacked in time, for the dashboard

It shares its preprocessing with training (same Normalizer statistics from the checkpoint, same window
dataset). The truth/GLORYS field is included in the Zarr only for side-by-side comparison; scores against
it are against GLORYS only (GLORYS assimilates ARGO). Synthetic-data runs stamp every file with a warning.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

from data.catalog import BaseConfig, load_base_config
from evaluation import derived
from training.baselines import fit_climatology
from training.data import REPO_ROOT, Normalizer, WindowDataset, load_arrays, split_indices
from training.train import TrainConfig, build_model, channel_indices, climatology_norm, predict

REFERENCE_NOTE = "Validated against GLORYS only; GLORYS assimilates ARGO, so agreement is not independent proof of skill."
SYNTHETIC_WARNING = "SYNTHETIC DATA - produced from synthetic inputs; NOT a real ocean reconstruction"


def find_checkpoint(cfg: BaseConfig, name_or_path: str, root: Path = REPO_ROOT) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    found = sorted((root / cfg.paths["checkpoints_dir"]).glob(f"{name_or_path}-*.pt"), key=lambda f: f.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no checkpoint matching '{name_or_path}' in {root / cfg.paths['checkpoints_dir']}")
    return found[-1]


def load_checkpoint(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    tc = TrainConfig(**ck["config"])
    meta = SimpleNamespace(lat=np.asarray(ck["lat"]), lon=np.asarray(ck["lon"]), depth=np.asarray(ck["depths"]))
    model = build_model(tc, meta)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    norm = Normalizer(**{k: np.asarray(v) for k, v in ck["norm"].items()})
    return model, tc, norm, ck


def static_geometry(y: np.ndarray) -> np.ndarray:
    """(Z, H, W) bool, True where a temperature exists: ocean and shallower than the local seafloor.
    Land/bathymetry are static, so a cell that is NaN on every day of the target record is masked."""
    return ~np.all(np.isnan(y), axis=0)


def quality_flag(valid: np.ndarray) -> np.ndarray:
    """0 = predicted, 1 = land (no ocean at the surface), 2 = below local bathymetry."""
    flag = np.zeros(valid.shape, dtype="int8")
    land = ~valid[0]
    flag[:, land] = 1
    flag[~valid & ~land[None]] = 2
    return flag


def derive(temp: np.ndarray, depths: np.ndarray) -> dict[str, np.ndarray]:
    return {"mld": derived.mixed_layer_depth(temp, depths), "d20": derived.isotherm_depth(temp, depths, 20.0),
            "tchp": derived.tchp(temp, depths, 26.0)}


def _day_dataset(t, temp, std, flag, der, lat, lon, depth, ck, model_version: str) -> xr.Dataset:
    ds = xr.Dataset(
        {"temp": (("time", "depth", "lat", "lon"), temp[None].astype("float32"),
                  {"units": "degree_Celsius", "long_name": "reconstructed sea water potential temperature",
                   "standard_name": "sea_water_potential_temperature"}),
         "temp_std": (("time", "depth", "lat", "lon"), std[None].astype("float32"),
                      {"units": "degree_Celsius", "long_name": "MC-dropout standard deviation of temperature"}),
         "quality_flag": (("depth", "lat", "lon"), flag, {"flag_values": np.array([0, 1, 2], dtype="int8"),
                                                          "flag_meanings": "predicted land below_bathymetry"}),
         "mld": (("time", "lat", "lon"), der["mld"][None].astype("float32"), {"units": "m", "long_name": "mixed layer depth (0.2C criterion)"}),
         "d20": (("time", "lat", "lon"), der["d20"][None].astype("float32"), {"units": "m", "long_name": "depth of the 20C isotherm"}),
         "tchp": (("time", "lat", "lon"), der["tchp"][None].astype("float32"), {"units": "kJ cm-2", "long_name": "tropical cyclone heat potential"})},
        coords={"time": [t], "depth": ("depth", depth, {"units": "m", "positive": "down", "axis": "Z"}),
                "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y"}),
                "lon": ("lon", lon, {"units": "degrees_east", "axis": "X"})},
    )
    ds.attrs.update(Conventions="CF-1.8", title="OceanEmbed subsurface temperature reconstruction",
                    model_version=model_version, data_source=ck["data_source"], reference_note=REFERENCE_NOTE,
                    generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    if ck["data_source"] == "synthetic":
        ds.attrs["synthetic_warning"] = SYNTHETIC_WARNING
    return ds


def predict_split(cfg: BaseConfig, checkpoint: Path, split: str = "test", stride: int = 3, mc: int = 10,
                  root: Path = REPO_ROOT, device: str | None = None) -> dict:
    """Run a checkpoint over a split's eval days -> physical-unit mean/std (masked to valid ocean cells),
    the GLORYS truth on the same days, and the metadata needed to write or score them."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tc, norm, ck = load_checkpoint(checkpoint, dev)
    start, end = cfg.periods[split]
    lead = np.datetime64(start) - np.timedelta64(tc.window - 1, "D")
    arrays = load_arrays(cfg, root, start=str(lead), end=end)
    idx_all = np.arange(len(arrays.time))
    pos = idx_all[(arrays.time >= np.datetime64(start))][::stride]
    ds_in = WindowDataset(arrays, norm, idx_all, tc.window, pos, channel_indices(tc.inputs))
    clim_n = None
    if tc.anomaly:
        train = load_arrays(cfg, root, *cfg.periods["train"])
        tr_idx = split_indices(train.time, {"train": cfg.periods["train"]}, 0)["train"]
        clim_n = climatology_norm(fit_climatology(train.y, train.time, tr_idx), norm).to(dev)
    mean, std = predict(model, ds_in, clim_n, tc, arrays, norm, dev, mc=mc)
    valid = static_geometry(arrays.y)
    return {"mean": np.where(valid[None], mean, np.nan), "std": np.where(valid[None], std, np.nan),
            "truth": arrays.y[ds_in.targets], "time": arrays.time[ds_in.targets], "lat": arrays.lat,
            "lon": arrays.lon, "depth": arrays.depth, "valid": valid, "tc": tc, "ck": ck}


def run_inference(cfg: BaseConfig, checkpoint: Path, split: str = "test", stride: int = 3, mc: int = 10,
                  root: Path = REPO_ROOT, device: str | None = None, write_netcdf: bool = True) -> Path:
    r = predict_split(cfg, checkpoint, split, stride, mc, root, device)
    mean, std, times, tc, ck = r["mean"], r["std"], r["time"], r["tc"], r["ck"]
    flag = quality_flag(r["valid"])
    version = f"{tc.name}-{tc.hash()}"
    out_dir = root / cfg.paths["processed_dir"] / "outputs"
    nc_dir = out_dir / "netcdf"
    nc_dir.mkdir(parents=True, exist_ok=True)
    days = []
    for i, t in enumerate(times):
        der = derive(mean[i], r["depth"])
        day = _day_dataset(t, mean[i], std[i], flag, der, r["lat"], r["lon"], r["depth"], ck, version)
        days.append(day)
        if write_netcdf:
            day.to_netcdf(nc_dir / f"oceanembed_temp_{str(t.astype('datetime64[D]')).replace('-', '')}.nc")
    stacked = xr.concat(days, dim="time", data_vars="minimal")      # quality_flag is static: stays (depth, lat, lon)
    stacked["glorys"] = (("time", "depth", "lat", "lon"), r["truth"].astype("float32"),
                         {"long_name": "GLORYS reanalysis (comparison reference, not independent)"})
    stacked.attrs.update(days[0].attrs)
    zarr_path = out_dir / "oceanembed_predictions.zarr"
    stacked.chunk({"time": 8}).to_zarr(zarr_path, mode="w")
    return zarr_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconstruct subsurface temperature with a trained checkpoint.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "demo.yaml"), help="override YAML; 'none' = base only")
    ap.add_argument("--checkpoint", default="oceanembed", help="checkpoint path or name prefix (latest match)")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--mc", type=int, default=10, help="MC-dropout samples for uncertainty")
    args = ap.parse_args(argv)
    cfg = load_base_config(override=None if args.config == "none" else args.config)
    ck = find_checkpoint(cfg, args.checkpoint)
    print(f"checkpoint {ck.name}; data_source={cfg.data_source}")
    print("wrote", run_inference(cfg, ck, args.split, args.stride, args.mc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
