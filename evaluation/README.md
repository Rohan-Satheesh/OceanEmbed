# /evaluation

Metrics, ARGO matching, and baseline comparisons (PRD Sections 9, 15-18, Chunks 3-4, 6).

## Layout (Chunk 3: ARGO independent validation)

- `argo.py` — fetch (public INCOIS ERDDAP griddap, no auth), QC (physical range check +
  TERR/SERR error-field masking, since this gridded VAM product doesn't carry per-profile QC
  flags), and matching (nearest-neighbor spatial match to the 0.25° grid, nearest 10-day
  composite window per target date, linear depth interpolation via `data.vertical`) for the
  INCOIS ARGO 10-day gridded Variational Analysis Methodology product (PRD Sections 9.1-9.2).
  Also attaches the GLORYS-assimilates-ARGO independence caveat (PRD Section 9.3) to every
  output dataset's attrs.
- `argo_pipeline.py` — CLI orchestrator: fetch → QC → spatial match → temporal match → depth
  match → caveat → write. Reuses `data.harmonize`'s write/mask-variable machinery by modeling
  ARGO as a `Channel` named `argo_validation` (it isn't a training input; this is purely to
  reuse Chunk 2's tested writer code).

ARGO is independent validation data — it is never touched by the training pipeline, and GLORYS
(the training target, prepared in `/data`) assimilates ARGO observations, so strong agreement
against this product is not by itself fully independent proof of skill (PRD Section 9.3).

## Running it

```bash
conda run -n oceanembed python -m evaluation.argo_pipeline --start 2024-06-01 --end 2024-06-07
```

`--start`/`--end` are `YYYY-MM-DD`. Output lands in the same `data/processed/zarr/oceanembed.zarr`
store as every other channel (group `argo_validation`) and the matching NetCDF convention
(`data/processed/netcdf/argo_validation/<year>/`).

## Tests

`tests/test_argo.py` covers QC masking and spatial/temporal matching against synthetic data — no
network access required. `tests/test_vertical.py` covers the depth-matching interpolation logic
shared with `/data`.
