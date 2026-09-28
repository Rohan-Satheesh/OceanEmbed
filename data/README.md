# /data

Download and preprocessing scripts (PRD Section 10, Chunk 2).

## Layout

- `catalog.py` — loads `configs/channels.yaml` / `configs/base.yaml` into typed objects (`Channel`, `Domain`, `BathymetryConfig`). Nothing else in this package hardcodes a dataset id, variable name, or grid bound.
- `sources/copernicus.py`, `sources/earthdata.py` — per-provider downloaders (Copernicus Marine for SST/SSS/SSH-SLA, NASA Earthdata via `earthaccess` for OSCAR currents/CCMP winds), with retry/backoff (`retry.py`) and provenance metadata (`provenance.py`).
- `bathymetry.py` — land/seafloor mask on the common 0.25° grid, built from NOAA ETOPO 2022 (used in place of GEBCO — see the comment in `configs/channels.yaml` for why). Produces `land_fraction`, `sea_mask`, and `seafloor_depth_m` per cell.
- `regrid.py` — conservative, area-weighted regridding to the domain's 0.25° grid via xESMF, with weight-file caching.
- `harmonize.py` — temporal alignment (daily aggregation, gap-fill ≤3 days), mask-variable attachment, and writers for both the per-day NetCDF convention and the shared Zarr store.
- `qc.py` — missing-day, spatial-coverage, and physical-range-check reports, written as JSON.
- `pipeline.py` — CLI orchestrator tying the above together end to end.

## Running it

```bash
conda run -n oceanembed python -m data.pipeline --start 2024-06-01 --end 2024-06-07 --channels sst,ssh
```

`--channels` is a comma-separated subset of the names in `configs/channels.yaml` (`sst`, `sss`, `ssh`, `currents`, `winds`); omit it to run every channel. `--start`/`--end` are `YYYY-MM-DD`. Use a date range from at least a few weeks in the past — the Copernicus SST/SSS/SSH products used here are reprocessed ("REP"/"my") products with a reprocessing lag, so very recent dates may simply have no data yet.

- `raw/` — untouched downloads (git-ignored)
- `processed/` — harmonized NetCDF (`processed/netcdf/<channel>/<year>/`) + Zarr (`processed/zarr/oceanembed.zarr`, one group per channel) + bathymetry cache + QC reports (`processed/qc/<run_id>/`) + regrid weight cache (git-ignored)

## Tests

Unit tests for the pure preprocessing logic (regridding, harmonization, QC, bathymetry masking) live in `/tests` and run against synthetic data — no network access required:

```bash
pytest tests/
```
