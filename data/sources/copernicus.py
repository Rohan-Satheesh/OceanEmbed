"""Copernicus Marine Service downloader (PRD Section 7.1: SST/OSTIA, SSS/SMAP-SMOS, SSH-SLA/DUACS).

Region/date subsetting happens at request time via copernicusmarine.subset(), so we never
pull more than the PoC domain needs (PRD Section 10.1).
"""

from __future__ import annotations

from pathlib import Path

import copernicusmarine

from data.catalog import Channel, Domain
from data.retry import retry


# The request extends this far beyond the domain on every side. Without it the subset ends exactly on the domain edge,
# and the destination cells centred on the edge (which reach half a cell outside it) are only partly covered by the
# source: the conservative regrid then returns a fraction of the true value there (SST/SSS/SSH x0.5, GLORYS x0.67,
# corners x0.25). One target cell (0.25 deg) is more than enough for every source resolution used here.
SUBSET_MARGIN_DEG = 0.5


def download(channel: Channel, start: str, end: str, domain: Domain, out_dir: Path,
             margin_deg: float = SUBSET_MARGIN_DEG) -> Path:
    """Subset+download one channel for [start, end] into out_dir. Returns the downloaded file path.

    The bounding box is the domain plus `margin_deg` on every side, and the margin is part of the cache filename so
    a file cropped exactly to the domain (the old, edge-corrupting behaviour) can never be picked up again."""
    if channel.source != "copernicus":
        raise ValueError(f"channel '{channel.name}' is not a copernicus channel")
    if not channel.dataset_id:
        raise ValueError(f"channel '{channel.name}' has no dataset_id configured")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{channel.name}_{start}_{end}_m{margin_deg:g}.nc"

    def _fetch():
        depth_kwargs = {}
        if channel.min_depth_m is not None:
            depth_kwargs["minimum_depth"] = channel.min_depth_m
        if channel.max_depth_m is not None:
            depth_kwargs["maximum_depth"] = channel.max_depth_m
        return copernicusmarine.subset(
            dataset_id=channel.dataset_id,
            variables=channel.variables,
            minimum_longitude=domain.lon_min - margin_deg,
            maximum_longitude=domain.lon_max + margin_deg,
            minimum_latitude=max(domain.lat_min - margin_deg, -90.0),
            maximum_latitude=min(domain.lat_max + margin_deg, 90.0),
            start_datetime=start,
            end_datetime=end,
            output_directory=str(out_dir),
            output_filename=out_filename,
            file_format="netcdf",
            skip_existing=True,
            disable_progress_bar=True,
            **depth_kwargs,
        )

    def _on_retry(attempt: int, exc: Exception) -> None:
        print(f"  [retry {attempt}] copernicus subset failed for {channel.name}: {exc}")

    retry(_fetch, on_retry=_on_retry)
    return out_dir / out_filename
