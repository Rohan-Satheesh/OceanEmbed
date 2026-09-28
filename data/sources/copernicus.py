"""Copernicus Marine Service downloader (PRD Section 7.1: SST/OSTIA, SSS/SMAP-SMOS, SSH-SLA/DUACS).

Region/date subsetting happens at request time via copernicusmarine.subset(), so we never
pull more than the PoC domain needs (PRD Section 10.1).
"""

from __future__ import annotations

from pathlib import Path

import copernicusmarine

from data.catalog import Channel, Domain
from data.retry import retry


def download(channel: Channel, start: str, end: str, domain: Domain, out_dir: Path) -> Path:
    """Subset+download one channel for [start, end] into out_dir. Returns the downloaded file path."""
    if channel.source != "copernicus":
        raise ValueError(f"channel '{channel.name}' is not a copernicus channel")
    if not channel.dataset_id:
        raise ValueError(f"channel '{channel.name}' has no dataset_id configured")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{channel.name}_{start}_{end}.nc"

    def _fetch():
        return copernicusmarine.subset(
            dataset_id=channel.dataset_id,
            variables=channel.variables,
            minimum_longitude=domain.lon_min,
            maximum_longitude=domain.lon_max,
            minimum_latitude=domain.lat_min,
            maximum_latitude=domain.lat_max,
            start_datetime=start,
            end_datetime=end,
            output_directory=str(out_dir),
            output_filename=out_filename,
            file_format="netcdf",
            skip_existing=True,
            disable_progress_bar=True,
        )

    def _on_retry(attempt: int, exc: Exception) -> None:
        print(f"  [retry {attempt}] copernicus subset failed for {channel.name}: {exc}")

    retry(_fetch, on_retry=_on_retry)
    return out_dir / out_filename
