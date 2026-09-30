"""Derived products from a reconstructed temperature profile (PRD 17.3 / 17.3b): mixed layer depth
(MLD), 20 deg C isotherm depth (D20) and tropical cyclone heat potential (TCHP).

All functions take temperature as (Z, H, W) in deg C on the standard depth levels and return (H, W).
With only 15 levels (5-200 m spacing) these are coarse, linearly-interpolated estimates — good for
maps and relative comparison, not a substitute for a high-resolution profile analysis.
"""

from __future__ import annotations

import numpy as np

RHO_SEAWATER = 1026.0        # kg m-3
CP_SEAWATER = 3985.0         # J kg-1 K-1


def _cross_depth(temp: np.ndarray, depths: np.ndarray, level: float, start: int = 0) -> np.ndarray:
    """Depth where temperature first falls to `level` going down from level index `start` (linear
    interpolation between standard levels). NaN where the column never reaches `level` or is invalid;
    0-length columns (already at/below `level` at `start`) return depths[start]."""
    z = depths.astype("float64")
    out = np.full(temp.shape[1:], np.nan)
    done = np.zeros(temp.shape[1:], dtype=bool)
    start_below = temp[start] <= level
    out[start_below & ~np.isnan(temp[start])] = z[start]
    done |= start_below | np.isnan(temp[start])
    for k in range(start + 1, len(z)):
        t0, t1 = temp[k - 1], temp[k]
        hit = ~done & (t1 <= level) & ~np.isnan(t1) & ~np.isnan(t0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (t0 - level) / (t0 - t1)
        out[hit] = z[k - 1] + frac[hit] * (z[k] - z[k - 1])
        done |= hit | np.isnan(t1)
    return out


def isotherm_depth(temp: np.ndarray, depths: np.ndarray, level: float = 20.0) -> np.ndarray:
    """Depth of the `level` isotherm (default D20). 0 where the surface is already colder than `level`;
    NaN where the water stays warmer than `level` down to the deepest valid level."""
    return _cross_depth(temp, depths, level)


def mixed_layer_depth(temp: np.ndarray, depths: np.ndarray, ref_depth: float = 10.0, dt: float = 0.2) -> np.ndarray:
    """Temperature-threshold MLD: depth where T drops `dt` below the value at `ref_depth`
    (de Boyer Montegut et al. 2004 style). Columns that never cool by `dt` return their deepest valid depth."""
    ref = int(np.argmin(np.abs(depths - ref_depth)))
    t_ref = temp[ref]
    z = depths.astype("float64")
    mld = np.full(temp.shape[1:], np.nan)
    done = np.isnan(t_ref)
    thresh = t_ref - dt
    for k in range(ref + 1, len(z)):
        t0, t1 = temp[k - 1], temp[k]
        hit = ~done & (t1 <= thresh) & ~np.isnan(t1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (t0 - thresh) / (t0 - t1)
        mld[hit] = z[k - 1] + frac[hit] * (z[k] - z[k - 1])
        done |= hit
    deepest = np.nanmax(np.where(np.isnan(temp), np.nan, z[:, None, None]), axis=0)
    never = ~np.isnan(t_ref) & np.isnan(mld)
    mld[never] = deepest[never]
    return mld


def tchp(temp: np.ndarray, depths: np.ndarray, ref_temp: float = 26.0) -> np.ndarray:
    """Tropical cyclone heat potential, kJ cm-2: rho*cp * integral of (T - 26) dz over the connected warm surface
    layer down to the 26 deg C isotherm (trapezoid rule, last partial layer cut at the linearly-interpolated
    crossing). 0 where SST < 26; deeper warm layers beneath a cold one (inversions) are not counted."""
    z = depths.astype("float64")
    ok = temp >= ref_temp                                    # NaN compares False
    connected = np.logical_and.accumulate(ok, axis=0)        # True while every level so far is >= ref
    total = np.zeros(temp.shape[1:])
    for k in range(len(z) - 1):
        t0, t1, dz = temp[k], temp[k + 1], z[k + 1] - z[k]
        full = connected[k + 1]
        cut = connected[k] & ~ok[k + 1] & ~np.isnan(t1)      # crosses 26 inside this layer
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(cut, (t0 - ref_temp) / (t0 - t1), 0.0)
        total += np.nan_to_num(np.where(full, 0.5 * ((t0 - ref_temp) + (t1 - ref_temp)) * dz, 0.0))
        total += np.nan_to_num(np.where(cut, 0.5 * (t0 - ref_temp) * frac * dz, 0.0))
    return np.where(~np.isnan(temp[0]), RHO_SEAWATER * CP_SEAWATER * total * 1e-7, np.nan)   # J m-2 -> kJ cm-2
