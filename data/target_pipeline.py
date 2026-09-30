"""Chunk 3 target orchestrator: download GLORYS -> QC-mask -> horizontal regrid -> vertical
interpolate to the 15 standard depths -> bathymetry+land mask -> harmonize -> QC (PRD Section 35
Chunk 3, Section 8).

GLORYS is modeled as a `Channel` (data/catalog.py::load_target_catalog()) so it reuses the same
harmonize_channel/write_netcdf/append_zarr machinery Chunk 2 already built and tested — output
lands in the same `oceanembed.zarr` store (group "glorys") and NetCDF layout as every other
channel, just with an extra `depth` dimension.

Usage:
    python -m data.target_pipeline --start 2024-06-01 --end 2024-06-07
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv

load_dotenv()

from data.bathymetry import build_land_sea_mask, fetch_gebco
from data.catalog import Channel, Domain, load_base_config, load_catalog, load_target_catalog
from data.harmonize import append_zarr, harmonize_channel, write_netcdf
from data.pipeline import apply_land_mask, apply_unit_conversion, download_channel
from data.qc import mask_invalid_values, run_channel_qc, write_qc_report
from data.regrid import regrid_dataset
from data.vertical import apply_below_bathymetry_mask, interpolate_to_standard_depths

# CMEMS GLORYS ships its vertical coordinate as "depth" — standardized defensively in case a
# future dataset_id swap ships a different name (same defensiveness as regrid.py's
# standardize_latlon(), which exists precisely because provider dim-naming isn't stable).
_DEPTH_DIM_CANDIDATES = ("depth", "lev", "elevation")


def _standardize_depth_dim(ds: xr.Dataset) -> xr.Dataset:
    if "depth" in ds.dims:
        return ds
    for candidate in _DEPTH_DIM_CANDIDATES:
        if candidate in ds.dims:
            return ds.rename({candidate: "depth"})
    raise ValueError(f"no recognized depth dimension found in {list(ds.dims)} — extend _DEPTH_DIM_CANDIDATES")


def process_glorys(
    glorys: Channel,
    start: str,
    end: str,
    domain: Domain,
    depths_m: list[int],
    land_sea: xr.Dataset,
    raw_dir: Path,
    processed_dir: Path,
    weights_cache_dir: Path,
    run_id: str,
) -> None:
    print(f"\n=== {glorys.name} ({glorys.label}) ===")

    print("  downloading...")
    raw = download_channel(glorys, start, end, domain, raw_dir)
    raw = _standardize_depth_dim(raw)

    print("  unit conversion + range-based QC mask (PRD 10.2, before regridding)...")
    converted = apply_unit_conversion(raw, glorys)
    cleaned = mask_invalid_values(converted, glorys.variables)

    print("  regridding (conservative, horizontal)...")
    regridded = regrid_dataset(cleaned, domain, weights_cache_dir)

    print("  interpolating to the 15 standard depths (PRD Section 4.4)...")
    vertical = interpolate_to_standard_depths(regridded, depths_m)

    print("  applying below-bathymetry + land mask...")
    masked = apply_below_bathymetry_mask(vertical, land_sea, depths_m, glorys.variables)
    masked = apply_land_mask(masked, land_sea, glorys)

    print("  harmonizing (temporal alignment, gap-fill, mask variable)...")
    harmonized = harmonize_channel(masked, glorys, start, end)
    harmonized = harmonized.load()

    print("  writing NetCDF + Zarr...")
    netcdf_paths = write_netcdf(harmonized, glorys, domain, processed_dir)
    zarr_path = append_zarr(harmonized, glorys, processed_dir)
    print(f"    {len(netcdf_paths)} NetCDF file(s), zarr store at {zarr_path}")

    print("  running QC...")
    report = run_channel_qc(harmonized, glorys)
    qc_path = write_qc_report(report, run_id, processed_dir)
    print(f"    QC report: {qc_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OceanEmbed Chunk 3 GLORYS target pipeline")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="override paths.processed_dir from configs/base.yaml")
    args = parser.parse_args(argv)

    base = load_base_config()
    target_catalog = load_target_catalog()
    domain = base.domain

    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / base.paths["raw_dir"]
    processed_dir = Path(args.out) if args.out else repo_root / base.paths["processed_dir"]
    weights_cache_dir = processed_dir / "regrid_weights"
    bathymetry_dir = processed_dir / "bathymetry"

    run_id = uuid.uuid4().hex[:8]
    print(f"Run {run_id}: {args.start} -> {args.end}, target: glorys")

    print("\n=== bathymetry / land-sea mask ===")
    catalog = load_catalog()
    gebco = fetch_gebco(catalog.bathymetry, domain, bathymetry_dir)
    land_sea = build_land_sea_mask(gebco, catalog.bathymetry, domain)
    land_sea.to_netcdf(bathymetry_dir / "land_sea_mask.nc")
    print(f"  land-sea mask written to {bathymetry_dir / 'land_sea_mask.nc'}")

    process_glorys(
        target_catalog.glorys, args.start, args.end, domain, base.depths_m, land_sea,
        raw_dir, processed_dir, weights_cache_dir, run_id,
    )

    print(f"\nDone. Run {run_id} complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
