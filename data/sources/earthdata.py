"""NASA Earthdata / PO.DAAC downloader via `earthaccess` (PRD Section 7.1: OSCAR currents, CCMP winds).

earthaccess handles login via EARTHDATA_USERNAME/EARTHDATA_PASSWORD (strategy="environment",
already verified in scripts/auth_smoke_test.py).
"""

from __future__ import annotations

from pathlib import Path

import earthaccess

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

    downloaded = retry(lambda: earthaccess.download(granules, local_path=str(out_dir)), on_retry=_on_retry)
    return [Path(p) for p in downloaded]
