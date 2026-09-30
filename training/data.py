"""Training data layer (PRD Sections 11, 13.2): load the harmonized store, split chronologically,
normalize with train-only statistics, and serve windowed samples.

Works unchanged on the real store and the synthetic demo store — the store location comes from
`paths.processed_dir` in the (possibly overridden) config.

Leakage rules enforced here (PRD 11.5/11.6): whole-period chronological splits, a buffer gap
dropped from the end of train and validation, normalization fit on train only, and windows that
never reach back across a split boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from data.catalog import BaseConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

# The six surface variables (SSH -> SLA; currents and winds are two components each) = 7 channels.
INPUTS: list[tuple[str, str]] = [
    ("sst", "analysed_sst"), ("sss", "sos"), ("ssh", "sla"),
    ("currents", "u"), ("currents", "v"), ("winds", "uwnd"), ("winds", "vwnd"),
]
INPUT_NAMES = [v for _, v in INPUTS]
TARGET = ("glorys", "thetao")
MIN_LEAD_DAYS = 7          # eval days skip the first 7 of a split so windows/persistence stay inside it


def store_path(cfg: BaseConfig, root: Path = REPO_ROOT) -> Path:
    return root / cfg.paths["processed_dir"] / "zarr" / "oceanembed.zarr"


@dataclass
class StoreArrays:
    time: np.ndarray            # datetime64[ns], (T,)
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    x: np.ndarray               # (T, C, H, W) float32, NaN = missing
    y: np.ndarray               # (T, Z, H, W) float32, NaN = land / below bathymetry


def load_arrays(cfg: BaseConfig, root: Path = REPO_ROOT, start: str | None = None,
                end: str | None = None) -> StoreArrays:
    path = store_path(cfg, root)
    if not path.exists():
        raise FileNotFoundError(f"no harmonized store at {path} (data_source={cfg.data_source})")
    sl = slice(start, end)
    chans = []
    for group, var in INPUTS:
        a = xr.open_zarr(path, group=group, consolidated=False)[var].sel(time=sl)
        for extra in [d for d in a.dims if d not in ("time", "lat", "lon")]:
            # The real SSS group carries a singleton `depth` (0 m) dimension; drop any such singleton.
            if a.sizes[extra] != 1:
                raise ValueError(f"{group}.{var} has a non-singleton extra dimension '{extra}'")
            a = a.isel({extra: 0}, drop=True)
        chans.append(a)
    tgt = xr.open_zarr(path, group=TARGET[0], consolidated=False)[TARGET[1]].sel(time=sl)
    ref = chans[0]
    for c in chans + [tgt]:
        if not np.array_equal(c.time.values, ref.time.values):
            raise ValueError("input/target groups have different time axes — re-pull so they line up")
    x = np.stack([c.transpose("time", "lat", "lon").values for c in chans], axis=1).astype("float32")
    return StoreArrays(ref.time.values, ref.lat.values, ref.lon.values, tgt.depth.values,
                       x, tgt.transpose("time", "depth", "lat", "lon").values.astype("float32"))


def split_indices(time: np.ndarray, periods: dict[str, list[str]], buffer_days: int = 3) -> dict[str, np.ndarray]:
    """Chronological index sets per period. The last `buffer_days` of train and validation are
    dropped so adjacent-day autocorrelation can't leak across the boundary (PRD 11.5)."""
    out: dict[str, np.ndarray] = {}
    for name, (s, e) in periods.items():
        idx = np.flatnonzero((time >= np.datetime64(s)) & (time <= np.datetime64(e)))
        if name in ("train", "validation") and buffer_days and len(idx) > buffer_days:
            idx = idx[:-buffer_days]
        out[name] = idx
    return out


def eval_positions(idx: np.ndarray, stride: int = 3, skip_first: int = MIN_LEAD_DAYS) -> np.ndarray:
    """Indices (into the store's time axis) used to score EVERY model — identical eval days for
    baselines and OceanEmbed (PRD 15 'Comparison Requirements')."""
    return idx[skip_first::stride]


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray) -> "Normalizer":
        """Statistics from the TRAINING period only (PRD 11.6), over valid pixels."""
        return cls(
            np.nanmean(x_train, axis=(0, 2, 3)), np.nanstd(x_train, axis=(0, 2, 3)) + 1e-6,
            np.nanmean(y_train, axis=(0, 2, 3)), np.nanstd(y_train, axis=(0, 2, 3)) + 1e-6,
        )

    def x(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalized inputs with NaN -> 0, plus the explicit observed-mask channels (1 = observed)."""
        z = (x - self.x_mean[None, :, None, None]) / self.x_std[None, :, None, None]
        m = ~np.isnan(z)
        return np.where(m, z, 0.0).astype("float32"), m.astype("float32")

    def y(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (y - self.y_mean[None, :, None, None]) / self.y_std[None, :, None, None]
        m = ~np.isnan(z)
        return np.where(m, z, 0.0).astype("float32"), m.astype("float32")

    def inv_y(self, z: np.ndarray) -> np.ndarray:
        return z * self.y_std[None, :, None, None] + self.y_mean[None, :, None, None]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: v.tolist() for k, v in self.__dict__.items()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls(**{k: np.asarray(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()})


class WindowDataset(Dataset):
    """Samples of `window` consecutive days of surface fields (+ their masks) -> the full
    15-depth temperature field at the last day. Windows stay inside the split (no reaching back
    across the boundary). Returns tensors:
        x   (window*C*2, H, W)   normalized values then masks, oldest day first
        y   (Z, H, W)            normalized target (0 where invalid)
        ym  (Z, H, W)            1 where the target is valid
        doy (2,)                 sin/cos day-of-year
        t   ()                   index into the store's time axis
    """

    def __init__(self, arrays: StoreArrays, norm: Normalizer, split_idx: np.ndarray, window: int = 1,
                 positions: np.ndarray | None = None, channels: list[int] | None = None):
        self.window = window
        self.idx = np.asarray(split_idx)
        lo, hi = int(self.idx[0]), int(self.idx[-1]) + 1
        self.offset = lo
        self.time = arrays.time[lo:hi]
        self.xv, self.xm = norm.x(arrays.x[lo:hi])
        if channels is not None:                                   # input ablations (PRD 16): keep a subset of variables
            self.xv, self.xm = self.xv[:, channels], self.xm[:, channels]
        self.yv, self.ym = norm.y(arrays.y[lo:hi])
        keep = self.idx if positions is None else np.asarray(positions)
        self.targets = np.array([t for t in keep if t - lo >= window - 1], dtype=int)
        days = arrays.time[self.targets].astype("datetime64[D]")
        doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
        ang = 2 * np.pi * (doy - 1) / 365.25
        self.doy = np.stack([np.sin(ang), np.cos(ang)], axis=1).astype("float32")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        t = int(self.targets[i]) - self.offset
        frames = slice(t - self.window + 1, t + 1)
        x = np.concatenate([self.xv[frames].reshape(-1, *self.xv.shape[2:]),
                            self.xm[frames].reshape(-1, *self.xm.shape[2:])], axis=0)
        return (torch.from_numpy(x), torch.from_numpy(self.yv[t]), torch.from_numpy(self.ym[t]),
                torch.from_numpy(self.doy[i]), torch.tensor(int(self.targets[i])))
