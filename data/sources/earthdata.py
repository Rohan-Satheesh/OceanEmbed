"""NASA Earthdata / PO.DAAC downloader via `earthaccess` (PRD Section 7.1: OSCAR currents, CCMP winds).

earthaccess handles login via EARTHDATA_USERNAME/EARTHDATA_PASSWORD (strategy="environment",
already verified in scripts/auth_smoke_test.py).
"""

from __future__ import annotations

from pathlib import Path

import earthaccess
import netCDF4

from data.catalog import Channel, Domain
from data.retry import retry

_authenticated = False


def _ensure_login() -> None:
    global _authenticated
    if not _authenticated:
        auth = earthaccess.login(strategy="environment")
        if not auth.authenticated:
            raise RuntimeError("earthaccess login failed — check EARTHDATA_USERNAME/PASSWORD in .env")
        _authenticated = True


def download(channel: Channel, start: str, end: str, domain: Domain, out_dir: Path) -> list[Path]:
    """Search+download one channel's granules for [start, end] into out_dir. Returns downloaded file paths."""
    if channel.source != "earthdata":
        raise ValueError(f"channel '{channel.name}' is not an earthdata channel")
    if not channel.short_name:
        raise ValueError(f"channel '{channel.name}' has no short_name configured")

    _ensure_login()
    out_dir.mkdir(parents=True, exist_ok=True)

    bounding_box = (domain.lon_min, domain.lat_min, domain.lon_max, domain.lat_max)

    def _search():
        return earthaccess.search_data(
            short_name=channel.short_name,
            temporal=(start, end),
            bounding_box=bounding_box,
        )

    def _on_retry(attempt: int, exc: Exception) -> None:
        print(f"  [retry {attempt}] earthaccess search/download failed for {channel.name}: {exc}")

    granules = retry(_search, on_retry=_on_retry)
    if not granules:
        raise RuntimeError(f"no granules found for '{channel.name}' ({channel.short_name}) in [{start}, {end}]")

    return _download_all(granules, out_dir, channel.name, _on_retry)


def _is_valid_nc(path: Path) -> bool:
    """False for a truncated/corrupt file (e.g. left behind by an interrupted download)."""
    try:
        with netCDF4.Dataset(path) as ds:
            return len(ds.variables) > 0
    except OSError:
        return False


def _download_all(granules: list, out_dir: Path, name: str, on_retry, attempts: int = 4) -> list[Path]:
    """earthaccess.download() logs per-file failures and returns only the files that succeeded —
    it does not raise — so a dropped granule would silently become a gap-filled day downstream.
    Re-request until every granule is present (already-downloaded files are skipped by
    earthaccess), and raise if any are still missing."""
    downloaded: list[str] = []
    for attempt in range(1, attempts + 1):
        downloaded = retry(lambda: earthaccess.download(granules, local_path=str(out_dir)), on_retry=on_retry)
        if len(downloaded) >= len(granules):
            paths = [Path(p) for p in downloaded]
            corrupt = [p for p in paths if not _is_valid_nc(p)]
            if not corrupt:
                return paths
            # earthaccess skips files that already exist, so delete corrupt ones to force a re-fetch.
            for p in corrupt:
                p.unlink()
            on_retry(attempt, RuntimeError(f"{len(corrupt)} corrupt file(s) deleted for re-download"))
            continue
        on_retry(attempt, RuntimeError(f"{len(granules) - len(downloaded)} of {len(granules)} granules missing"))
    raise RuntimeError(
        f"'{name}': only {len(downloaded)} of {len(granules)} granules downloaded after {attempts} attempts"
    )
