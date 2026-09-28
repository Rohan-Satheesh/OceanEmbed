# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OceanEmbed: a satellite-embedding-based deep learning framework that reconstructs daily, 0.25°, 15-depth
subsurface ocean temperature fields over the North Indian Ocean (5–30°N, 45–105°E) from six daily surface
satellite variables (SST, SSS, SSH/SLA, surface currents, surface winds). Trained against GLORYS reanalysis,
validated independently against gridded ARGO. Built for a hackathon (SIH, Problem Statement 26066) entirely on
free-tier data access and free-tier compute (Kaggle/Colab).

The full spec, including data sources, model architecture candidates, loss design, evaluation methodology, risks,
and the 8-chunk work breakdown, lives in [`OceanEmbed_PRD.md`](OceanEmbed_PRD.md) — read the relevant section
before implementing anything non-trivial rather than guessing at requirements. Key sections: 7–9 (input/target/
validation data), 10 (preprocessing pipeline), 11 (splitting/leakage rules), 12–13 (embedding + reconstruction
model), 15–16 (baselines/ablations), 18–19 (evaluation, uncertainty), 35 (chunk-by-chunk work breakdown and
MVP-vs-Advanced staging).

**Current state:** only Chunk 1 (environment & data-access bootstrap) is implemented. `/data`, `/models`,
`/training`, `/evaluation`, `/dashboard` are placeholder directories (README only) awaiting Chunks 2–7.

## Work is organized into 8 sequential chunks (PRD Section 35)

1. Environment & data access setup (**done**)
2. Data acquisition & harmonization pipeline (regridding, masking, NetCDF/Zarr store)
3. Target & validation data prep (GLORYS regrid, climatology, ARGO matching)
4. Baseline models & dataset splitting (chronological split, climatology/persistence/RF/U-Net baselines)
5. Satellite embedding engine (masked-autoencoder / CNN-ViT, mask-aware training)
6. Reconstruction model, training & evaluation (depth-conditioned decoder, physics-aware loss, ablations)
7. Frontend / dashboard (Streamlit)
8. Deployment & PoC demo

Each chunk depends on the previous one finishing its deliverable — check the PRD table in Section 35 before
assuming a downstream component (e.g. the embedding engine) can be built ahead of its dependency (the harmonized
data store).

## Environment setup

Two parallel environment definitions are kept in sync — update both when adding a dependency:

- `environment.yml` — conda, **preferred**. xESMF/ESMPy (needed for conservative regridding) is far more
  reliable via conda-forge than pip, especially on Windows. PyTorch is still installed via pip inside the conda
  env (cu128 wheels — see the comment in the file for why: Blackwell GPU support while staying backward-compatible
  with Kaggle/Colab T4/P100).
- `requirements.txt` — pip-only fallback. If using this path on Windows, either install ESMF separately first or
  run preprocessing inside WSL2 / a conda env.

```bash
# conda (preferred)
conda env create -f environment.yml
conda activate oceanembed

# or venv + pip fallback
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Credentials: copy `.env.example` to `.env` and fill in Copernicus Marine Service, NASA Earthdata, and INCOIS LAS
credentials. `.env` is git-ignored; loaders read it via `python-dotenv`. Never commit real credentials.

## Commands

```bash
# Environment smoke test — package imports, GPU visibility, .env credential presence,
# and a minimal in-memory xESMF conservative regrid. No live network calls.
python scripts/smoke_test.py

# Authenticated smoke test — confirms real credentials work against live Copernicus
# Marine / NASA Earthdata services with a cheap request (catalogue query / login only,
# not a full download). Run inside the oceanembed env.
conda run -n oceanembed python scripts/auth_smoke_test.py

# Run tests
pytest

# Dummy experiment-tracking run (confirms W&B or the CSV/JSON fallback logs correctly)
python training/run_logger.py
```

There is no lint/build step configured yet.

## Architecture

- **Config-driven, not hardcoded.** `configs/base.yaml` holds the domain bounds (5–30°N, 45–105°E, 0.25°), the
  15 standard reconstruction depths, chronological train/val/test date ranges, path layout, and the global seed.
  Per-experiment configs are expected to override this via Hydra composition or a plain YAML merge — read
  `configs/base.yaml` before hardcoding any of these values elsewhere.
- **Pipeline shape** (PRD Section 21): raw multi-source downloads → harmonized daily 0.25° NetCDF/Zarr store →
  training/inference tensors → satellite embedding engine → depth-conditioned reconstruction decoder → validation
  (GLORYS test set + independent ARGO) → output store (NetCDF/Zarr) → Streamlit dashboard.
- **Training vs. inference pipelines share code paths.** Both use the same harmonization and embedding-engine
  code; only the training pipeline touches GLORYS labels and the loss/optimizer (PRD Section 21.4).
- **Experiment tracking fallback pattern** (`training/run_logger.py`): `RunLogger` uses W&B if `WANDB_API_KEY` is
  set in the environment, otherwise transparently falls back to appending JSONL records under `training/runs/`.
  Any future training code should go through this logger rather than calling `wandb` directly, so the no-signup
  fallback keeps working.
- **Mask-aware design is load-bearing, not optional.** SSS is only available from 2015 onward (SMAP era) and is
  the weakest/gappiest input in the Bay of Bengal (river plumes, RFI, coastal contamination). The embedding
  engine and preprocessing pipeline are both expected to degrade gracefully via explicit mask channels rather than
  dropping years or imputing silently (PRD Sections 10.6, 10.8, 12.3).
- **No shuffled splits, ever.** Splitting is strictly chronological, whole-year (train 2015–2021, val 2022, test
  2023–2024 per `configs/base.yaml`), with normalization statistics fit on the training period only and applied
  unchanged elsewhere. This is a hard leakage-prevention rule (PRD Section 11), not a default to optimize away.
- **GLORYS is the training target, ARGO is independent validation** — and GLORYS assimilates ARGO, so strong
  agreement against ARGO is not by itself proof of skill against a fully independent source. Any new evaluation
  code should carry this caveat rather than presenting ARGO agreement as ground truth (PRD Sections 9.3, 18.3).
- **Depths below local bathymetry are masked, never extrapolated.** Applies consistently across inputs, target,
  and model outputs.

## Data & artifacts (git-ignored, not committed)

- `data/raw/`, `data/processed/` — downloaded and harmonized datasets
- `training/runs/` — CSV/JSON experiment logs and checkpoints (the no-W&B fallback)
- `wandb/` — local W&B run directories
