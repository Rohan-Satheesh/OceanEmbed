# /data

Download and preprocessing scripts (PRD Sections 8, 10, Chunks 2-3).

## Layout

- `catalog.py` — loads `configs/channels.yaml` / `configs/target.yaml` / `configs/base.yaml` into typed objects (`Channel`, `Domain`, `BathymetryConfig`, `ArgoConfig`). Nothing else in this package hardcodes a dataset id, variable name, or grid bound.
- `sources/copernicus.py`, `sources/earthdata.py` — per-provider downloaders (Copernicus Marine for SST/SSS/SSH-SLA/GLORYS, NASA Earthdata via `earthaccess` for OSCAR currents/CCMP winds), with retry/backoff (`retry.py`) and provenance metadata (`provenance.py`).
- `bathymetry.py` — land/seafloor mask on the common 0.25° grid, built from NOAA ETOPO 2022 (used in place of GEBCO — see the comment in `configs/channels.yaml` for why). Produces `land_fraction`, `sea_mask`, `seafloor_depth_m`, and (`below_bathymetry_mask()`) a per-standard-depth "deeper than local seafloor" mask.
- `regrid.py` — conservative, area-weighted horizontal regridding to the domain's 0.25° grid via xESMF, with weight-file caching.
- `vertical.py` — linear interpolation onto the 15 standard reconstruction depths (`configs/base.yaml`'s `depths_m`), with a documented boundary-hold at the shallow end (most products' shallowest native level isn't exactly 0m) and genuine no-extrapolation masking at the deep end, combined with `bathymetry.below_bathymetry_mask()`.
- `climatology.py` — per-cell, per-depth daily/monthly climatology over the configured training period only (PRD Section 8.3/8.4, anomaly-based training support for Chunk 6).
- `harmonize.py` — temporal alignment (daily aggregation, gap-fill ≤3 days), mask-variable attachment, and writers for both the per-day NetCDF convention and the shared Zarr store.
- `qc.py` — missing-day, spatial-coverage, and physical-range-check reports, written as JSON.
- `pipeline.py` — Chunk 2 CLI orchestrator for the 5 surface input channels.
- `target_pipeline.py` — Chunk 3 CLI orchestrator for the GLORYS target: download → QC-mask → horizontal regrid → vertical interpolation to the 15 standard depths → below-bathymetry + land mask → harmonize → write → QC.

ARGO independent-validation ingestion/matching lives in `/evaluation` (see its README), not here — it's validation data, never touched by the training pipeline.

## Running it

```bash
# Chunk 2: surface input channels
conda run -n oceanembed python -m data.pipeline --start 2024-06-01 --end 2024-06-07 --channels sst,ssh

# Chunk 3: GLORYS target
conda run -n oceanembed python -m data.target_pipeline --start 2024-06-01 --end 2024-06-07
```

`--channels` is a comma-separated subset of the names in `configs/channels.yaml` (`sst`, `sss`, `ssh`, `currents`, `winds`); omit it to run every channel. `--start`/`--end` are `YYYY-MM-DD`. Use a date range from at least a few weeks in the past — the Copernicus SST/SSS/SSH/GLORYS products used here are reprocessed ("REP"/"my") products with a reprocessing lag, so very recent dates may simply have no data yet.

`target_pipeline.py` writes GLORYS into the same Zarr store as every other channel (group `glorys`, dims `time, depth, lat, lon`) and the same NetCDF convention (`processed/netcdf/glorys/<year>/`) — it's deliberately modeled as "just another `Channel`" (see `configs/target.yaml`) to reuse Chunk 2's tested harmonize/write machinery unchanged.

Climatology (`climatology.py`) is a separate step, run once the full training-period GLORYS pull exists — not part of the smoke-window pipeline run above, same treatment as the full historical input pull.

- `raw/` — untouched downloads (git-ignored)
- `processed/` — harmonized NetCDF (`processed/netcdf/<channel>/<year>/`) + Zarr (`processed/zarr/oceanembed.zarr`, one group per channel, including `glorys`) + bathymetry cache + climatology (`processed/climatology/`) + QC reports (`processed/qc/<run_id>/`) + regrid weight cache (git-ignored)

## Tests

Unit tests for the pure preprocessing logic (regridding, harmonization, QC, bathymetry masking, vertical interpolation, climatology) live in `/tests` and run against synthetic data — no network access required:

```bash
pytest tests/
```

## Synthetic demo store

`synthetic.py` (`python -m data.synthetic`) writes the same groups/variables/grid/depths/`*_mask` channels as the real
pipeline into `data/synthetic/zarr/oceanembed.zarr` — monsoon winds and SST, Arabian Sea upwelling, a fresh Bay of
Bengal plume, eddies that drive SSH/SST/currents/thermocline depth — on the real coastline and seafloor when
`data/processed/bathymetry/land_sea_mask.nc` exists. It refuses to write into `data/processed/`. The fields are
physically motivated, not calibrated; scores on them never measure real-ocean skill.
