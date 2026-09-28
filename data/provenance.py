"""Provenance metadata attached to every downloaded/processed file (PRD Section 7.4, 10.7)."""

from __future__ import annotations

import datetime as dt
from typing import Any

LICENSES = {
    "copernicus": "Copernicus Marine Service Product User Licence "
    "(https://marine.copernicus.eu/user-corner/service-commitments-and-licence)",
    "earthdata": "NASA Earth Science Data open-data policy "
    "(https://www.earthdata.nasa.gov/learn/use-data/data-use-policy)",
    "etopo": "NOAA NCEI ETOPO 2022 Global Relief Model, public domain (U.S. Government work) "
    "(https://www.ncei.noaa.gov/products/etopo-global-relief-model)",
}


def build_provenance(*, source: str, dataset_id: str, processing_steps: list[str]) -> dict[str, Any]:
    if source not in LICENSES:
        raise ValueError(f"unknown provenance source '{source}' — known: {sorted(LICENSES)}")
    return {
        "source_product": dataset_id,
        "source_provider": source,
        "license": LICENSES[source],
        "access_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "processing_steps": ",".join(processing_steps),
    }
