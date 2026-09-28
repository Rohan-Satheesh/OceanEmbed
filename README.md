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
