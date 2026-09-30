"""Vertical interpolation onto the 15 standard reconstruction depths (PRD Sections 4.4, 8.3).

GLORYS ships on 50 native z-levels; ARGO's gridded VAM product ships on 24 native levels. Both
need linear interpolation onto the same standard depth axis so the target and independent
validation data align with each other and with the reconstruction model's output depths.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from data.bathymetry import below_bathymetry_mask


def interpolate_to_standard_depths(
    ds: xr.Dataset, depths_m: list[int], depth_dim: str = "depth"
) -> xr.Dataset:
    """Linearly interpolate every data variable along `depth_dim` onto `depths_m`.

    No extrapolation on the deep end: a standard depth deeper than the source profile's native
    range comes back NaN rather than a fabricated value (PRD Section 4.4 — "masked, never
    extrapolated") — that's a genuine no-observation case.

    The shallow end is different: most ocean products' shallowest native level isn't exactly 0m
    (e.g. GLORYS's first level is ~0.49m, ARGO's gridded product starts at 5m) — that's a
    boundary-representation quirk, not a missing observation, so any standard depth at or above
    the native minimum is boundary-held to that shallowest level's value rather than masked.
    Losing the standard depth-0 surface layer entirely (the single most-used depth) to a ~0.5m
    labeling difference would be a real, silent data-loss bug, not "no extrapolation" caution.
    """
    native_min = float(ds[depth_dim].min())
    query_depths = [max(d, native_min) if d <= native_min else d for d in depths_m]

    interped = ds.interp(
        {depth_dim: query_depths},
        method="linear",
        kwargs={"bounds_error": False, "fill_value": np.nan},
    )
    return interped.assign_coords({depth_dim: depths_m})


def apply_below_bathymetry_mask(
    ds: xr.Dataset, land_sea: xr.Dataset, depths_m: list[int], variables: list[str]
) -> xr.Dataset:
    """NaN out standard depths deeper than each cell's seafloor (PRD Section 4.4), reusing the
    already-tested `data.bathymetry.below_bathymetry_mask`."""
    too_deep = below_bathymetry_mask(land_sea, depths_m)
    out = ds.copy()
    for var in variables:
        if var in out:
            out[var] = out[var].where(~too_deep.reindex_like(out[var], method="nearest"))
    return out
