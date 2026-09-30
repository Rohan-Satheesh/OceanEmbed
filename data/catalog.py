"""Channel + bathymetry catalog loader (PRD Section 7.1, Chunk 2).

Reads configs/channels.yaml into typed, importable objects so the rest of the
pipeline never hardcodes a dataset id, variable name, or processing flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass(frozen=True)
class Channel:
    name: str
    label: str
    source: str  # "copernicus" | "earthdata"
    variables: list[str]
    native_resolution_deg: float
    units: str
    unit_conversion: str | None = None
    processing: str = ""
    dataset_id: str | None = None       # copernicusmarine dataset id
    short_name: str | None = None        # earthaccess CMR short_name
    available_from: str | None = None    # ISO date; earlier days are masked, not fetched
    native_temporal_resolution: str = "daily"
    min_depth_m: float | None = None     # copernicusmarine depth-subset bound (Chunk 3 target)
    max_depth_m: float | None = None

    @property
    def processing_steps(self) -> list[str]:
        return [s for s in self.processing.split(",") if s]


@dataclass(frozen=True)
class BathymetryConfig:
    label: str
    source: str
    url: str
    opendap_url: str
    variable: str
    land_fraction_threshold: float


@dataclass(frozen=True)
class Catalog:
    channels: dict[str, Channel]
    bathymetry: BathymetryConfig

    def get(self, name: str) -> Channel:
        try:
            return self.channels[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.channels))
            raise KeyError(f"unknown channel '{name}' — known channels: {known}") from exc

    def resolve(self, names: list[str] | None) -> list[Channel]:
        if not names:
            return list(self.channels.values())
        return [self.get(n) for n in names]


def load_catalog(path: Path | str | None = None) -> Catalog:
    path = Path(path) if path else CONFIG_DIR / "channels.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    channels = {
        name: Channel(name=name, **spec) for name, spec in raw["channels"].items()
    }
    bathymetry = BathymetryConfig(**raw["bathymetry"])
    return Catalog(channels=channels, bathymetry=bathymetry)


@dataclass(frozen=True)
class ArgoConfig:
    label: str
    source: str
    dataset_id: str
    erddap_base_url: str
    variables: dict[str, str]  # logical name -> ERDDAP variable name (TEMP, SAL, TERR, SERR)
    native_resolution_deg: float
    max_error_threshold: float


@dataclass(frozen=True)
class TargetCatalog:
    glorys: Channel
    argo: ArgoConfig


def load_target_catalog(path: Path | str | None = None) -> TargetCatalog:
    path = Path(path) if path else CONFIG_DIR / "target.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    glorys = Channel(name="glorys", **raw["glorys"])
    argo = ArgoConfig(**raw["argo"])
    return TargetCatalog(glorys=glorys, argo=argo)


@dataclass(frozen=True)
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float


@dataclass(frozen=True)
class BaseConfig:
    project: str
    domain: Domain
    depths_m: list[int]
    periods: dict[str, list[str]]
    paths: dict[str, str]
    seed: int


def load_base_config(path: Path | str | None = None) -> BaseConfig:
    path = Path(path) if path else CONFIG_DIR / "base.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BaseConfig(
        project=raw["project"],
        domain=Domain(**raw["domain"]),
        depths_m=raw["depths_m"],
        periods=raw["periods"],
        paths=raw["paths"],
        seed=raw["seed"],
    )
