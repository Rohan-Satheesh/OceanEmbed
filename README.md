# OceanEmbed

Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.

Full requirements: [`OceanEmbed_PRD.md`](OceanEmbed_PRD.md).

## Repository Structure

```
/data        - download and preprocessing scripts
/models      - embedding and reconstruction model architectures
/training    - training loops, run configs
/evaluation  - metrics, ARGO matching, baseline comparisons
/dashboard   - Streamlit app
/configs     - YAML run configurations
```

## Environment Setup (Chunk 1 — PRD Section 22.1b)

1. **Create a Python 3.10+ environment.**

   ```bash
   conda env create -f environment.yml
   conda activate oceanembed
   # or, with venv + pip:
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Register free accounts** and store credentials as environment variables — never commit them:
   - [Copernicus Marine Service](https://data.marine.copernicus.eu/) (satellite inputs + GLORYS target)
   - [NASA Earthdata](https://urs.earthdata.nasa.gov/) (OSTIA/SMAP/SMOS/OSCAR/CCMP/ASCAT-family products via PO.DAAC)
   - [INCOIS Live Access Server (LAS)](https://las.incois.gov.in/) (ARGO independent validation)

   Copy `.env.example` to `.env` and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

   `.env` is git-ignored. Loaders read it via `python-dotenv`.

3. **Provision free GPU compute.**
   - Kaggle Notebooks: enable the GPU accelerator (P100/T4, ~30 GPU-hrs/week)
   - Google Colab: open a notebook, Runtime → Change runtime type → GPU (T4, free tier)

   Confirm with the smoke test below.

4. **Run the smoke test** to confirm the environment, credentials, GPU, and a minimal regrid all work end to end:

   ```bash
   python scripts/smoke_test.py
   ```

5. **Experiment tracking.** Weights & Biases (free tier) is used if `WANDB_API_KEY` is set in `.env`; otherwise runs fall back to CSV/JSON logs under `training/runs/`. No setup is required for the fallback.

See PRD Section 22.1b for the full step-by-step bootstrap and Section 35 (Chunk 1) for scope.

## Demo on synthetic data (Chunks 4-8)

The real historical data pull is large and slow, so Chunks 4-8 are built and demoed on a **synthetic**
stand-in with the same groups, grid, depths and `*_mask` channels, kept separate in `data/synthetic/`.
Switching to real data is a config change (`paths.processed_dir` in `configs/demo.yaml`), not a code change.

```bash
conda activate oceanembed
python scripts/run_demo.py            # synthetic store -> baselines -> train -> predict -> report (~15 min on a laptop GPU)
python scripts/run_demo.py --quick    # fewer epochs
python scripts/run_demo.py --full     # + every ablation and the seed repeats (~1.5 h)
streamlit run dashboard/app.py        # explore the results
```

**What the numbers mean.** Everything under the demo config is synthetic. Scores show that the pipeline runs end
to end, not how well the model reconstructs the real ocean; validation is against GLORYS only (GLORYS assimilates
ARGO, so there is no independent check), and the demo trains on one year (2022; validation 2023, test 2024).
The dashboard and every output file carry a "SYNTHETIC DATA" marker.

## Pipeline stages and where they live

| Stage | Command / module |
|---|---|
| Real inputs + GLORYS target | `python -m data.pipeline`, `python -m data.target_pipeline` |
| Synthetic demo store | `python -m data.synthetic` |
| Baselines (climatology, persistence, random forest) | `python -m training.baselines` |
| OceanEmbed / U-Net training | `python -m training.train --model oceanembed` |
| Ablation set | `python -m training.experiments` |
| Inference (CF-1.8 NetCDF + Zarr, MC-dropout uncertainty, MLD/D20/TCHP) | `python -m evaluation.predict` |
| Evaluation report | `python -m evaluation.report` |
| Dashboard | `streamlit run dashboard/app.py` |

## Running the same pipeline on the real Q1 2022 data

`configs/demo_real_q1.yaml` points the identical pipeline at the real store in `data/processed/` (own run and
checkpoint folders, so nothing overwrites the synthetic results):

```bash
python scripts/run_demo.py --config configs/demo_real_q1.yaml --stride 1 --epochs 60    # ~3 min
$env:OCEANEMBED_CONFIG='configs/demo_real_q1.yaml'; streamlit run dashboard/app.py       # PowerShell
```

Only 90 days exist, so the split is by month (train January, validate February, test March) instead of whole
years. One training month plus a January-to-March seasonal shift make this a **smoke test of the real-data path**,
not a skill assessment: the learned models do worse than a 7-day-lag persistence baseline here, which is expected
(persistence uses GLORYS from a week earlier, which a satellite-only reconstruction would not have in operation).
The original synthetic ocean was also much smoother than the real one (SST spatial-temporal std 0.8 vs 2.6 deg C in
Jan-Mar 2022, near-zero spatial variation at 500-1000 m), so its scores were far too optimistic. The generator is now
calibrated to the real Q1 (see `data/README.md`), which closes most of that gap for January-March; the rest of the
year is still parametric. Meaningful results need the multi-year pull.

## Known issue: wrong values in the outermost ring of grid cells (real store)

In the real SST, SSS, SSH and GLORYS fields the outermost row/column of the 0.25 deg grid holds only a fraction of the
true value (SST/SSS/SSH x0.50, GLORYS x0.67, corners x0.25 — e.g. 13 C instead of ~27 C at 5N, 48.5E in February). The
Copernicus subsets are cut exactly at the domain bounds, and the conservative regrid scales partly-covered edge cells
down instead of flagging them. About 230 ocean cells per field are affected; currents and winds are not (their global
files are not cropped). Until the download is given a margin and the regrid gets a coverage guard, exclude the outer ring
when analysing real data, and fix this before the multi-year pull.
