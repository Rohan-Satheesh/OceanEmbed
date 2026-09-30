"""Chunk 3 ARGO validation orchestrator: fetch -> QC -> spatial match -> temporal match ->
depth match -> independence caveat -> harmonize -> write (PRD Section 35 Chunk 3, Section 9).

ARGO is modeled as a `Channel` (name "argo_validation") purely to reuse Chunk 2's
harmonize_channel/write_netcdf/append_zarr machinery — this is independent *validation* data,
never touched by the training pipeline (PRD Section 9.3).

Usage:
    python -m evaluation.argo_pipeline --start 2024-06-01 --end 2024-06-07
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv

load_dotenv()

from data.catalog import ArgoConfig, Channel, Domain, load_base_config, load_target_catalog
from data.harmonize import append_zarr, harmonize_channel, write_netcdf
from data.qc import run_channel_qc, write_qc_report
from evaluation.argo import (
    attach_independence_caveat,
    fetch_argo,
    match_depth,
    match_spatial,
    match_temporal,
    qc_argo,
)


def build_argo_channel(argo_cfg: ArgoConfig) -> Channel:
    """ARGO doesn't have its own entry in configs/channels.yaml (it's validation, not a training
    input) — constructed here so the QC/harmonize/write steps below can reuse Chunk 2's
    Channel-shaped machinery unchanged."""
    return Channel(
        name="argo_validation",
        label=argo_cfg.label,
        source=argo_cfg.source,
        variables=[argo_cfg.variables["temp"], argo_cfg.variables["sal"]],
        native_resolution_deg=argo_cfg.native_resolution_deg,
        units="degC,PSU",
        dataset_id=argo_cfg.dataset_id,
        processing="qc_mask,spatial_match,temporal_match,depth_match",
    )


def process_argo(
    argo_cfg: ArgoConfig,
    start: str,
    end: str,
    domain: Domain,
    depths_m: list[int],
    cache_dir: Path,
    processed_dir: Path,
    run_id: str,
) -> None:
    channel = build_argo_channel(argo_cfg)
    print(f"\n=== {channel.name} ({channel.label}) ===")

    print("  fetching (INCOIS ERDDAP)...")
    raw = fetch_argo(argo_cfg, domain, start, end, cache_dir)

    print("  QC (range check + TERR/SERR error masking)...")
    cleaned = qc_argo(raw, argo_cfg)
    cleaned = cleaned[channel.variables]

    print("  spatial matching (nearest 0.25deg cell)...")
    spatial = match_spatial(cleaned, domain)

    print("  temporal matching (nearest 10-day composite window)...")
    target_dates = xr.date_range(start, end, freq="1D")
    temporal = match_temporal(spatial, target_dates)

    print("  depth matching (linear interp to the 15 standard depths)...")
    matched = match_depth(temporal, depths_m)
    matched = attach_independence_caveat(matched)

    print("  harmonizing (mask variable; PRD 10.4 daily alignment is a no-op here since "
          "temporal matching already produced one value per target day)...")
    harmonized = harmonize_channel(matched, channel, start, end)
    harmonized = harmonized.load()

    print("  writing NetCDF + Zarr...")
    netcdf_paths = write_netcdf(harmonized, channel, domain, processed_dir)
    zarr_path = append_zarr(harmonized, channel, processed_dir)
    print(f"    {len(netcdf_paths)} NetCDF file(s), zarr store at {zarr_path}")

    print("  running QC...")
    report = run_channel_qc(harmonized, channel)
    qc_path = write_qc_report(report, run_id, processed_dir)
    print(f"    QC report: {qc_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OceanEmbed Chunk 3 ARGO validation pipeline")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="override paths.processed_dir from configs/base.yaml")
    args = parser.parse_args(argv)

    base = load_base_config()
    target_catalog = load_target_catalog()
    domain = base.domain

    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = Path(args.out) if args.out else repo_root / base.paths["processed_dir"]
    cache_dir = repo_root / base.paths["raw_dir"] / "argo"

    run_id = uuid.uuid4().hex[:8]
    print(f"Run {run_id}: {args.start} -> {args.end}, validation: argo")

    process_argo(
        target_catalog.argo, args.start, args.end, domain, base.depths_m,
        cache_dir, processed_dir, run_id,
    )

    print(f"\nDone. Run {run_id} complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
