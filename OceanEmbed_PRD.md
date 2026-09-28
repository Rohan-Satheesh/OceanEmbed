# OceanEmbed

## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

Product Requirements Document · Sep 28, 2026

---

# 1. Document Information

| Field | Value |
| --- | --- |
| Problem Statement ID | 26066 |
| PRD Version | 0.2 |
| Document Status | Draft |
| Team Members | Rohan Satheesh Ishita Sreedhar Govind Chandana Sahasra Rishi |
| Created Date | 28 Sep 2026 |
| Last Updated | 28 Sep 2026 |

---

# 2. Project Overview

### 2.1 Project Summary

OceanEmbed reconstructs a full three-dimensional ocean temperature field — 15 standard depths from 0 to 1000 m — over the North Indian Ocean at 0.25° daily resolution, using only six daily surface satellite observations (SST, SSS, SSH/SLA, surface currents, surface winds) as input. A self-supervised embedding engine compresses these surface fields into a compact latent “ocean-state” representation, which a depth-conditioned decoder maps to subsurface temperature. The system targets GLORYS reanalysis temperature as its training target and validates independently against gridded ARGO observations. A working Proof-of-Concept is demonstrated over the Bay of Bengal and Arabian Sea.

### 2.2 Background

Subsurface ocean temperature governs ocean circulation, upper-ocean heat content, stratification, air-sea interaction and marine ecosystem health, and is central to marine heatwave monitoring, fisheries management, cyclone intensity forecasting and ocean data assimilation. Direct subsurface measurement depends on ARGO floats, moored buoys, gliders and ship casts, whose spatial and temporal coverage is far too sparse to produce continuous, basin-scale subsurface fields. Satellite instruments, by contrast, observe the ocean surface daily and at relatively fine resolution; surface expressions of subsurface processes (thermocline displacement, mesoscale eddies, vertical mixing, ocean-atmosphere coupling) mean surface fields carry an indirect but learnable signature of what lies beneath. Deep learning and representation learning make it possible to learn this nonlinear surface-to-subsurface relationship more effectively than classical regression.

This surface-to-subsurface supervised learning framing is grounded in an established and growing body of satellite-to-subsurface deep learning literature — e.g., neural-network mappings from multi-channel surface satellite fields to subsurface temperature (Water, 13(3):388, 2021, doi:10.3390/w13030388), and long-record global deep-learning reconstructions validated against ARGO-type independent data (Remote Sensing, 14(13):3198, 2022, doi:10.3390/rs14133198). OceanEmbed is positioned as an incremental, PoC-scale contribution building on this line of work, not a claim of a wholly novel, unproven approach.

### 2.3 Problem Statement

There is a persistent gap between the near-global, daily coverage of satellite surface observations and the sparse, irregular coverage of direct subsurface measurements. Existing operational products either rely on data assimilation into physics-based ocean models (computationally expensive, requiring institutional infrastructure) or on coarse statistical relationships that do not exploit the full information content of multi-variable surface observations. OceanEmbed addresses this gap by learning a deep, multi-variable embedding of the surface ocean state and decoding it into a continuous, depth-resolved temperature field — using entirely free data and free compute.

---

# 3. Objectives

### 3.1 Primary Objective

Build and validate a satellite-embedding-based deep learning framework that reconstructs daily, 0.25°, 15-depth subsurface ocean temperature fields for the North Indian Ocean from six surface satellite variables, with skill that is quantifiably better than climatology and simple baselines, validated independently against ARGO.

### 3.2 Secondary Objectives

- Build an end-to-end, reproducible preprocessing and harmonization pipeline for five heterogeneous satellite/ocean data products plus the GLORYS training target.
- Develop a reusable, mask-aware satellite embedding engine that degrades gracefully when an input (notably SSS) is missing or low quality.
- Establish an honest, leakage-aware validation framework with explicit baselines and ablations, rather than reporting only headline skill numbers.
- Produce disaster-management-relevant derived products — mixed layer depth (MLD), 20°C isotherm depth (D20), and tropical cyclone heat potential (TCHP).
- Deliver everything on free-tier data access and free-tier compute (Kaggle/Colab), since the team has no dedicated funding.

### 3.3 Expected Outcome

A daily-updatable, 0.25° gridded 3D temperature product over the North Indian Ocean, with per-depth skill metrics (RMSE, correlation, bias) reported against both a held-out GLORYS test set and independent ARGO profiles, packaged with an interactive dashboard for depth-wise and time-series exploration, and demonstrated as a PoC over the Bay of Bengal and Arabian Sea.

**Priority labels (this revision):**

- 3.1 Primary objective (end-to-end reconstruction + PoC) — **Mandatory**
- 3.2a Preprocessing/harmonization pipeline — **Mandatory**
- 3.2b Mask-aware embedding engine — **Recommended**
- 3.2c Leakage-aware validation framework with baselines/ablations — **Mandatory**
- 3.2d MLD / D20 / TCHP derived products — **Recommended** (MLD, D20); TCHP — **Future Scope** if PoC time is constrained
- 3.2e Free-tier data/compute delivery — **Mandatory** (project constraint)
- 3.3 Expected outcome (daily-updated PoC field) — **Mandatory**

---

# 4. Scope

### 4.1 Geographic Scope

- **Region:** North Indian Ocean
- **Latitude:** 5°N – 30°N
- **Longitude:** 45°E – 105°E
- **PoC subregions:** Bay of Bengal and Arabian Sea

### 4.2 Spatial Scope

- **Target resolution:** 0.25° × 0.25°
- Full coverage of the domain above, excluding land
- Land and coastal masking applied consistently across all inputs, the target and all outputs; cells with a high land fraction (threshold fixed during preprocessing) are excluded
- Predictions below the local seafloor depth (from bathymetry) are masked rather than extrapolated

### 4.3 Temporal Scope

- **Target temporal resolution:** Daily
- **Training period:** indicative 2015–2021 (SMAP-era onward; to be confirmed after data audit)
- **Validation period:** indicative 2022
- **Test period:** indicative 2023–2024
- Near-real-time / operational inference is out of scope for this phase (see Section 33, Future Scope)

### 4.4 Depth Scope

Standard reconstruction depths:

| # | Depth (m) |
| --- | --- |
| 1 | 0 |
| 2 | 5 |
| 3 | 10 |
| 4 | 20 |
| 5 | 30 |
| 6 | 50 |
| 7 | 75 |
| 8 | 100 |
| 9 | 125 |
| 10 | 150 |
| 11 | 200 |
| 12 | 300 |
| 13 | 500 |
| 14 | 700 |
| 15 | 1000 |

Depths deeper than the local bathymetry, or where GLORYS/ARGO have no valid observation, are masked rather than filled with an extrapolated value.

### Non-Goals (Future Scope)

The following are explicitly out of scope for this hackathon PoC:

- Real-time operational forecasting or a production operational service
- Full-depth abyssal reconstruction below 2000 m
- Salinity reconstruction (temperature only for this PoC)
- Global-ocean deployment beyond the PoC region (Section 20)

---

# 5. Users and Stakeholders

### 5.1 Primary Users

- Ocean scientists studying circulation, heat content and stratification
- Researchers monitoring marine heatwaves and Indian Ocean climate variability
- Operational ocean forecasting and data assimilation teams
- Fisheries advisory teams
- Hackathon evaluators / technical reviewers

### 5.2 Stakeholders

- Ministry of Earth Sciences (MoES)
- Indian National Centre for Ocean Information Services (INCOIS), Ocean Valley
- Project mentors and domain experts
- Data providers: Copernicus Marine Service, NASA PO.DAAC/Earthdata
- &#91;Additional institutional stakeholders\]

### 5.3 User Needs

- Reliable, depth-wise temperature fields with a stated uncertainty, not a single unqualified number
- An accessible way to explore the fields (depth slider, time series, regional views) without writing code
- Standard, documented output formats (NetCDF/CF-conventions) that plug into existing ocean-science workflows
- A reproducible pipeline whose skill and limitations are clearly reported

---

# 6. Use Cases

### UC-01: Subsurface Temperature Reconstruction

- **Actor:** OceanEmbed pipeline
- **Input:** daily surface satellite observations (SST, SSS, SSH/SLA, currents, winds)
- **Processing:** harmonize inputs → embed via the satellite embedding engine → decode via the depth-conditioned reconstruction model
- **Output:** depth-wise temperature field (15 depths) plus uncertainty, at 0.25° daily resolution

### UC-02: Regional Ocean Analysis

- **Actor:** ocean scientist / forecaster
- **Behavior:** browse the dashboard by region, date and depth to inspect stratification, vertical temperature profiles, mixed layer depth and 20°C isotherm depth for a chosen location and time

### UC-03: Independent Validation

- **Actor:** evaluator / validation pipeline
- **Behavior:** match model output against gridded ARGO profiles by location, date and depth, and compute RMSE, correlation and bias, reported per depth, region and season

### UC-04: PoC Demonstration

- **Actor:** project team
- **Behavior:** run the full pipeline end to end over the Bay of Bengal and Arabian Sea for a representative period, including a historical cyclone case, and present the reconstructed fields, validation results and derived products

### Additional Use Cases

- **UC-05 Ablation Run:** a developer toggles the input set or embedding architecture and compares resulting skill metrics against the baseline configuration
- **UC-06 Disaster Product Query:** a forecaster retrieves cyclone heat potential, mixed layer depth or 20°C isotherm depth for a chosen date and region ahead of a developing storm

### UC-05: Thermocline / Mixed-Layer Depth Product for Disaster Management (Recommended)

- **Actor:** INCOIS forecaster / disaster-management analyst
- **Goal:** Access derived mixed-layer depth (MLD) and thermocline-depth fields alongside the primary temperature field, to support cyclone-intensification and heat-potential assessments
- **Value:** Adds decision-relevant derived products beyond raw temperature, consistent with Objective 3.2's MLD/D20/TCHP goals (Section 17.3b)

---

# 7. Input Data Requirements

### 7.1 Surface Variables

| Variable | Dataset/Product | Native Resolution | Required Resolution | Temporal Resolution | Source | Required Processing |
| --- | --- | --- | --- | --- | --- | --- |
| SST | OSTIA | 0.05° | 0.25° | Daily | Copernicus Marine Service ([moi-00168](https://doi.org/10.48670/moi-00168)) | Conservative (area-weighted) regrid, land mask |
| SSS | SMAP / SMOS | 0.125° | 0.25° | Daily | Copernicus Marine Service ([moi-00051](https://doi.org/10.48670/moi-00051)) | Regrid, RFI/coastal-contamination flagging, gap handling, mask-aware treatment of missing values |
| SSH / SLA | DUACS | 0.25° | 0.25° | Daily | Copernicus Marine Service ([moi-00145](https://doi.org/10.48670/moi-00145)) | Minimal regrid, quality check |
| Surface currents (U, V) | OSCAR L4 | 0.25° | 0.25° | Daily | NASA PO.DAAC ([OSCAR\_L4\_OC\_FINAL\_V2.0](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0)) | Regrid if needed, unit check |
| Surface winds (U, V) | CCMP 10 m L4 v3.1 (primary); ASCAT L2 Coastal (alternative) | 0.25° (CCMP, 6-hourly) | 0.25° | Daily | NASA PO.DAAC ([CCMP\_WINDS\_10M6HR\_L4\_V3.1](https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1), [ASCATC-L2-Coastal](https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal)) | Daily aggregation of 6-hourly CCMP; ASCAT is swath data and deprioritized |

### 7.2 Input Data Selection

- CCMP is preferred over ASCAT L2 for winds because it is already a gridded L4 product; ASCAT L2 is swath data and harder to regrid conservatively.
- SSS is the weakest input in the Bay of Bengal (river plumes, RFI, coastal contamination) and is only available from 2015 onward (SMAP era) — it is treated as an optional, maskable input, and its contribution is quantified via ablation (Section 16).
- SSH/SLA and currents are already close to the 0.25° target resolution and need minimal regridding.
- Final product selection, exact coverage windows and missing-data characteristics for each variable will be confirmed during the initial data audit (Section 31, Open Questions).

### 7.3 Additional Input Variables

- Static bathymetry / land–sea mask (e.g., GEBCO or ETOPO) for masking below-seafloor depths
- Day-of-year and/or climatological SST anomaly as an auxiliary feature
- A multi-day lag window (e.g., the past 5–10 days) of each surface variable, to capture the delay between a surface signature and its subsurface expression

### 7.4 Data Licensing & Access Notes (Recommended documentation)

Each of the 7 mandatory surface input channels is sourced from an open-access route: Copernicus Marine Service (SST, SSS, SSH/SLA) under the Copernicus open-data licence, and NASA PO.DAAC/Earthdata (surface currents, winds) under NASA's open-data policy. Recording the exact licence and access route per channel is **Recommended** practice for reproducibility. An input-channel ablation study (Section 16) will independently assess each channel's marginal contribution to reconstruction skill; this does not change the 7-channel input requirement, which remains **Mandatory**.

---

# 8. Target Dataset Requirements

### 8.1 Primary Target Dataset — GLORYS Global Ocean Reanalysis

- **Reference:** <https://doi.org/10.48670/moi-00021>
- **Required variable:** ocean potential temperature (thetao)
- **Spatial coverage:** subset to 5–30°N, 45–105°E
- **Temporal coverage:** matched to the input period (indicatively 2015 onward for training; a longer window may be pulled separately to build climatology)
- **Native resolution:** approximately 1/12° (\~0.083°)
- **Required preprocessing:** conservative regrid to 0.25°; vertical interpolation from native z-levels to the 15 standard depths (Section 4.4)

### 8.2 Target Variable

- **Variable:** subsurface ocean temperature
- **Depth levels:** the 15 standard depths specified in Section 4.4
- **Spatial resolution:** 0.25°
- **Temporal resolution:** Daily

### 8.3 Target Data Preparation

- Conservative area-weighted regridding from native GLORYS resolution to 0.25°
- Vertical interpolation to the 15 standard depths
- Quality control against physical bounds (e.g., −2 °C to 35 °C)
- Missing-value handling: interpolate short gaps, mask larger ones
- Land/sea and bathymetry-based masking consistent with the input pipeline
- Generation of daily and monthly climatology, to support anomaly-based training and evaluation (Sections 13.6, 18)

### 8.4 Reanalysis Caveat (Recommended context)

GLORYS is itself a data-assimilating ocean reanalysis, not a direct observation — it blends a numerical ocean model with available in-situ and satellite data, and inherits that model's biases, particularly below well-observed upper layers. Predicting a temperature **anomaly relative to climatology**, rather than the raw temperature field, reduces sensitivity to the reanalysis's mean-state bias, following the climatology-anomaly framing used in satellite-to-subsurface deep learning reconstructions (Remote Sensing, 14(13):3198, 2022, doi:10.3390/rs14133198). This is **Recommended** context; GLORYS remains the **Mandatory** training target.

---

# 9. Independent Validation Dataset

### 9.1 ARGO Data

- **Product:** Gridded ARGO — 10-day averaged composite, Variational Analysis Methodology (INCOIS LAS "Gridded Product based on Variational Analysis Methodology" → 10Days Averaged)
- **Source:** INCOIS Live Access Server (LAS)
- **Coverage:** North Indian Ocean domain
- **Temporal period:** matched to the validation and test periods (Section 4.3)
- **Note:** individual ARGO floats profile roughly every 10 days, so no daily gridded ARGO product exists; the 10-day composite is used as a pragmatic PoC-timeline choice over sourcing and matching raw, per-float profiles (e.g., via `argopy`/the ARGO GDAC), which would be the more rigorous but more time-consuming alternative (Section 25).

### 9.2 Validation Data Preparation

- Quality control per ARGO QC flags
- Spatial matching: nearest 0.25° grid cell to each ARGO profile location
- Temporal matching: nearest 10-day composite window to each model output day (not a daily match, since the gridded ARGO product is itself a 10-day average)
- Depth matching: linear interpolation of each ARGO profile onto the 15 standard depths

### 9.3 Independence Criteria

GLORYS assimilates ARGO observations, so validating model output against GLORYS-derived truth and separately against ARGO is not a fully independent check — a caveat this PRD states explicitly rather than treating strong ARGO agreement as unqualified proof of skill. Mitigations:

Additional QC context (**Recommended**): ARGO profiles used for independent validation should be filtered by ARGO QC flags (retaining good / probably-good flagged levels only), and delayed-mode versus real-time provenance should be tracked, since delayed-mode profiles carry additional calibration. Where GLORYS assimilation coverage information is available, validation results should be reported separately for ARGO floats used versus not used in GLORYS assimilation, sharpening the independence caveat stated above.

- Chronological, whole-year train/validation/test splits (Section 11) so that no test-period information leaks into training statistics
- Where feasible, a secondary and more independent check against RAMA moorings or ship CTD casts
- Reporting the GLORYS-assimilation caveat alongside every ARGO-based skill number, rather than presenting it as ground truth

---

# 10. Data Preprocessing and Harmonization Pipeline

### 10.1 Data Acquisition

- Scripted, automated downloads via the `copernicusmarine` Python client (Copernicus Marine Service) and `earthaccess` (NASA PO.DAAC/Earthdata)
- ARGO access via the INCOIS Live Access Server (LAS)
- Region/date subsetting at request time to limit download volume
- Retry/resume logic, given free-tier bandwidth and session limits

### 10.2 Quality Control

- Physical range checks per variable
- Masking of fill values and flagged bad-quality data
- Removal of SSS retrievals flagged for RFI or land contamination near the coast

### 10.3 Spatial Harmonization

- Common 0.25° × 0.25° grid over the full domain (5–30°N, 45–105°E)
- Conservative, area-weighted regridding via xESMF — never nearest-neighbour
- Consistent land mask applied across all variables and the target
- Coastal handling: mask cells above a configurable land-fraction threshold

### 10.4 Temporal Harmonization

- Daily-mean aggregation for sub-daily products (e.g., 6-hourly CCMP)
- All sources aligned to the same daily timestamp convention (UTC, 00:00 reference)
- Short gaps (≤ 3 days) forward/backward-filled; longer gaps marked missing rather than filled

### 10.5 Variable Harmonization

- Standard units (°C, PSU, m, m/s) and CF-convention variable names and coordinates
- Metadata (units, standard\_name) attached to every variable
- Normalization: per-variable z-score using training-period statistics only

### 10.6 Data Alignment

- Outer join on date, inner join on spatial grid across all sources
- Explicit handling of SSS's later start date (2015) via a variable-length input window or an availability flag channel, so the pipeline does not silently drop pre-2015 years for every variable

### 10.7 Preprocessed Dataset Format

- NetCDF4 (CF-conventions) as the primary interchange format; Zarr, chunked by time, for efficient training access
- Directory structure organized by variable and year
- Consistent file naming convention (e.g., `oceanembed_<variable>_<resolution>_<date>.nc`)
- Metadata recorded for every processed file (source product, processing steps applied, version)

### 10.8 Missing-Data Handling (Recommended)

- A gap-filling strategy for missing or cloud-obscured satellite inputs (notably SSS RFI/coastal gaps) should be documented explicitly once chosen — candidate approaches: short-window temporal interpolation, climatological gap-fill, or a masked-input / masked-autoencoder-style approach that trains the embedding engine to reconstruct masked regions from surrounding context, following the sparse/irregular-observation handling used in graph-attention subsurface reconstruction approaches (International Journal of Applied Earth Observation and Geoinformation, 149:105237, 2026, doi:10.1016/j.jag.2026.105237).
- The mask-aware embedding design (Section 12) already accounts for missing SSS at the architecture level; this subsection formalizes the preprocessing-side companion to that design.
- Whether a full masked-autoencoder-style gap-fill is implemented for the MVP or deferred to the Advanced stage (Section 35) is an open decision (Section 31).

---

# 11. Dataset Splitting Strategy

### 11.1 Training Dataset

- **Time period:** indicative 2015–2021
- **Spatial coverage:** full North Indian Ocean domain

### 11.2 Validation Dataset

- **Time period:** indicative 2022
- **Spatial coverage:** full domain

### 11.3 Test Dataset

- **Time period:** indicative 2023–2024
- **Spatial coverage:** full domain

### 11.4 Spatial Generalization Testing

An optional holdout subregion (e.g., a box within the central Bay of Bengal) never seen during training, used to test spatial generalization if time permits.

### 11.5 Temporal Generalization Testing

Strict chronological, whole-year splits with no shuffling across the train/validation/test boundary; a short buffer gap at each boundary to reduce autocorrelation-driven leakage between adjacent days.

### 11.6 Data Leakage Prevention

- Normalization statistics computed on the training set only, then applied unchanged to validation and test
- No random day-level splitting — only whole-year, chronological splits
- The GLORYS–ARGO assimilation overlap (Section 9.3) is acknowledged as a leakage risk that cannot be fully eliminated, and is mitigated (not solved) by temporal holdout and independent in-situ checks

### 11.7 Spatial / Spatiotemporal Splitting Note (Recommended)

In addition to the primary chronological/temporal split (**Mandatory**, Section 11.5), spatially or spatiotemporally blocked splits (e.g., holding out contiguous spatial tiles, not just later dates) should be considered to reduce spatial-autocorrelation leakage between train and test sets — general good practice for spatial machine-learning evaluation. This is a **Recommended** supplementary check, not a replacement for the chronological split, which remains primary.

---

# 12. Satellite Embedding Engine

### 12.1 Purpose

Learn a compact latent “ocean-state” representation from the six (or more, with lagged/derived) daily surface fields, capturing the nonlinear surface signatures of subsurface structure — thermocline tilt, mesoscale eddies, vertical mixing — while remaining robust to missing inputs, especially pre-2015 SSS.

### 12.2 Candidate Architectures

**CNN** — spatial convolutional encoder over the surface fields; captures local pattern structure cheaply.

**Vision Transformer (ViT)** — patch-based self-attention; captures longer-range, basin-scale structure (e.g., large eddies) at higher compute cost.

**Autoencoder / Masked Autoencoder (primary approach)** — self-supervised pretraining via masked reconstruction of surface fields, trainable on all available surface data without needing GLORYS labels, and naturally mask-aware for handling missing SSS.

**Graph Neural Network (GNN)** — grid cells as nodes, edges by spatial adjacency; listed as a stretch option / ablation, likely unnecessary given the regular lat–lon grid.

**Attention-based hybrid** — CNN feature extraction combined with a small transformer/attention block across the multi-day input window; candidate primary architecture alongside the MAE pretraining.

**ConvLSTM** — recurrent spatiotemporal encoder applied over a sliding window of past daily surface fields, capturing temporal evolution in addition to spatial structure; relevant if reconstruction benefits from multi-day surface-state history, following a ConvLSTM-family reconstruction approach for the northwestern Pacific (Ocean Science, 22:2161, 2026).

**Convformer (hybrid CNN+Transformer)** — combines convolutional local feature extraction with transformer-style attention for long-range spatial structure, proposed as a candidate that may capture both local fronts/eddies and basin-scale patterns better than either component alone (Remote Sensing, 16(13):2422, 2024).

### 12.3 Embedding Requirements

- **Input dimensions:** \~6 surface variables × window length (e.g., 5–10 days) × spatial patch
- **Embedding dimensions:** to be tuned (indicatively 64–256 per grid cell or patch)
- **Temporal representation:** stacked-day channels or a small temporal attention block
- **Spatial representation:** convolutional features or patch tokens
- **Multivariable fusion:** channel concatenation before the encoder
- **Missing-data handling:** input masking plus an explicit mask channel, so absent SSS or other gaps do not corrupt the embedding

### 12.4 Embedding Evaluation

- Linear-probe regression from the embedding to GLORYS temperature at a few representative depths
- Reconstruction loss on held-out, masked surface patches
- Qualitative check that embedding clusters correspond to known physical regimes (monsoon phases, eddies) via PCA/t-SNE visualization
- Ablation comparing “with embedding” against direct regression (Section 16)

**Temporal-window design (to be decided empirically):** the number of past days of surface input fed into one reconstruction (e.g., single-day vs. a 3–7 day lag window) is an architecture/training hyperparameter, not fixed by the SIH problem statement. It should be tuned and ablated (Section 16) rather than assumed; the ConvLSTM and Convformer candidates above are most sensitive to this choice.

---

# 13. Subsurface Temperature Reconstruction Model

### 13.1 Model Objective

Map the surface embedding (optionally combined with raw surface fields and a queried depth) to a subsurface temperature estimate at that depth.

### 13.2 Model Input

- Latent embedding vector from Section 12
- Raw surface fields, optionally, as a skip-connection input
- Depth value (continuous, normalized)
- Day-of-year
- Latitude/longitude or a region indicator

### 13.3 Model Output

- Predicted temperature (and uncertainty, if enabled — Section 19) at the queried depth
- A full water-column output at all 15 standard depths, generated by repeated depth queries or a multi-head output

### 13.4 Architecture

Encoder (the embedding engine, Section 12) → fusion (concatenate the embedding with a sinusoidal or learned depth encoding) → decoder (a small MLP or 1D convolutional stack) → output layer (a scalar temperature per depth, or a per-depth output vector for the 15-level case as a speed optimization).

### 13.5 Alternative Architectures

- Direct per-depth regression, with a separate output head per depth and no explicit depth-conditioning
- A plain U-Net that outputs all 15 depths directly from surface fields with no embedding stage — the primary “no embedding” ablation baseline

### 13.6 Training Strategy

- **Loss function:** masked MSE or Huber loss on the temperature anomaly relative to climatology, masked to ocean pixels and to depths shallower than local bathymetry
- **Optimizer:** Adam / AdamW
- **Learning rate:** cosine or step decay schedule
- **Batch size:** tuned to fit free-tier GPU memory (indicatively 16–64)
- **Epochs:** with early stopping on validation loss
- **Regularization:** weight decay and dropout
- **Physics-aware auxiliary loss:** a term penalizing unstable stratification (a non-monotonic warm-over-cold profile) and encouraging smooth vertical gradients

### 13.7 Joint vs. Climatology+Residual Architecture Note (Recommended)

A joint architecture (shared embedding encoder + depth-conditioned decoder predicting temperature directly) should be compared against an alternative that separately predicts a climatology term and a residual/anomaly term, as used in global satellite-to-subsurface deep-learning reconstructions (Remote Sensing, 14(13):3198, 2022). As a candidate depth-conditioning mechanism, **FiLM-style feature-wise linear modulation** (conditioning intermediate decoder features on the queried depth) is proposed for evaluation alongside simpler depth-concatenation or fixed-multi-head approaches — a candidate to compare, not a committed design.

---

# 14. Model Training Requirements

### 14.1 Training Configuration

- **Hardware:** free-tier GPU on Kaggle or Google Colab (e.g., T4 / P100)
- **Framework:** PyTorch
- **Python version:** 3.10+
- **GPU requirement:** mixed-precision (AMP) training to fit free-tier memory and time limits

### 14.2 Loss Functions

- **Primary:** masked MSE/Huber on temperature anomaly (Section 13.6)
- **Auxiliary:** stratification-stability penalty, vertical-smoothness penalty
- **Depth-weighted loss (optional):** to prevent the model from ignoring deep, low-variance levels

### 14.3 Hyperparameters

- Embedding dimension, input window length, learning rate, batch size, and the weighting between primary and physics-aware loss terms
- Manual/grid search given limited free-tier compute — no large-scale automated hyperparameter search

### 14.4 Reproducibility

- Fixed random seeds
- Environment captured via `requirements.txt` / a conda environment file
- Model checkpoints versioned by configuration hash
- Dataset versions tagged by preprocessing-pipeline commit
- Lightweight experiment tracking via logged CSV/JSON per run, or a free-tier experiment tracker

### 14.5 Depth-Aware Loss & Regularization Notes (Recommended)

- A depth-aware weighted loss (weighting shallower, better-observed depths differently from deeper, sparser ones) should be documented once its exact weighting scheme is decided.
- Vertical-smoothness / physical-plausibility regularization terms (penalizing implausible depth profiles) are a **Recommended** addition to the loss function; see Section 19 for the physical-plausibility diagnostics these terms relate to.

---

# 15. Baseline Models

| Baseline | Input | Output | Purpose |
| --- | --- | --- | --- |
| 1. Climatology | Day-of-year, location | Climatological mean temperature per depth | Floor that any learned model must beat |
| 2. Persistence | Most recent available estimate | Same field carried forward | Tests whether the model adds information beyond “yesterday's state” |
| 3. Random Forest / classical regression | Raw surface fields (no embedding) | Per-depth temperature | Classical ML baseline |
| 4. Plain U-Net (no embedding stage) | Raw surface fields | All 15 depths directly | Main deep-learning baseline / “no embedding” ablation |
| 5. Surface Quasi-Geostrophic (SQG) reconstruction — classical physical baseline (Recommended comparison) | SST + SSH (surface buoyancy/streamfunction proxies) | Interior/subsurface structure implied by quasi-geostrophic dynamics | Classical physics-based comparison point — tests where purely data-driven reconstruction adds skill beyond QG theory; not the primary method (Shan et al., "Testing the Limits of Surface QG Theory With Deep Learning", JGR: Machine Learning and Computation, 2026, doi:10.1029/2026JH001410) |

### Comparison Requirements

- Identical train/validation/test splits for every baseline and for OceanEmbed
- Identical evaluation region and masks
- Identical evaluation period
- Identical metrics (RMSE, correlation, bias — per depth, region and season)

---

# 16. Ablation Studies

### Input Ablations

- SST only
- SST + SSS
- SST + SSH
- SST + SSH + currents
- All six surface variables
- With / without the multi-day lag window

### Architecture Ablations

- No embedding (direct regression) vs. CNN embedding vs. ViT embedding vs. autoencoder/MAE embedding vs. GNN embedding (stretch) vs. attention-based hybrid embedding

### Other Ablations

- With / without depth-conditioning (continuous depth input vs. fixed 15-head output)
- With / without the physics-aware loss term
- With / without the uncertainty/ensemble head
- With / without anomaly-from-climatology framing (raw temperature vs. anomaly target)

### Additional Ablations (Recommended additions, this revision)

- Architecture ablation extended to include ConvLSTM and Convformer candidates (Section 12.2) alongside the architectures already listed above.
- Temporal-window-length ablation (above, Input Ablations) should be run at multiple window lengths (e.g., 1, 3, 7 days), not just presence/absence of the lag window, informed by TS-Cast-style temporal-window studies (Ocean Science, 22:2161, 2026).
- Loss-term ablation already includes with/without the physics-aware loss term above; this should explicitly cover the vertical-smoothness/physical-plausibility term proposed in Section 14.5 once implemented.

---

# 17. Output Requirements

### 17.1 Primary Output

Daily subsurface temperature field, 0.25° × 0.25°, at the 15 standard depths, over the North Indian Ocean domain.

### 17.2 Output Format

- **Primary format:** NetCDF4, CF-1.8 conventions
- **Internal/dashboard format:** Zarr
- **Naming convention:** `oceanembed_temp_<YYYYMMDD>.nc`
- **Metadata:** variable units, coordinate reference, model version, generation timestamp

### 17.3 Output Variables

- Predicted temperature (°C) per depth
- Per-depth uncertainty/confidence, from ensemble standard deviation, if implemented (Section 19)
- Quality flag (e.g., masked / land / below-bathymetry)
- Derived products: mixed layer depth (MLD), 20 °C isotherm depth (D20), tropical cyclone heat potential (TCHP)

### 17.3b Additional Derived Products (Recommended / Future Scope)

**Thermocline depth**, **mixed layer depth (MLD)**, and **20°C isotherm depth (D20)** as optional derived output products, computed from the reconstructed temperature profile — relevant to the same tropical Indian Ocean basin family as attention-based thermocline reconstruction work (Ocean Modelling, 196:102537, 2025). MLD and D20 are already named as Objective 3.2 deliverables (**Recommended** for the PoC timeline); a full thermocline-depth product beyond MLD/D20 is **Future Scope** if PoC time is constrained.

### 17.4 Visualization

- Surface (0 m) maps by date
- Depth-wise horizontal maps via a depth slider
- Vertical temperature profiles at a selected location
- Time-series of temperature at a chosen depth and location
- Error/uncertainty maps against ARGO
- Regional summary views for the Bay of Bengal and Arabian Sea

---

# 18. Evaluation and Validation Framework

### 18.1 Primary Metrics

- **RMSE** — computed per depth, against both the held-out GLORYS test set and independent ARGO profiles
- **Correlation** (Pearson) — per depth, region and season
- **Bias** (mean error) — per depth, region and season
- **Additional:** MAE, R², and a skill score relative to the climatology baseline

### 18.2 Evaluation Dimensions

- **Spatial:** grid-cell and regional aggregates
- **Temporal:** daily, seasonal and annual performance
- **Depth-wise:** each of the 15 standard depths reported separately — skill is expected to degrade below roughly 300 m, and this is reported honestly rather than masked by an aggregate number
- **Regional:** Bay of Bengal and Arabian Sea reported separately, since their governing dynamics differ (e.g., Bay of Bengal salinity stratification vs. Arabian Sea upwelling)

### 18.3 Independent ARGO Validation

Methodology and matching criteria as in Section 9.2; the same metrics as Section 18.1 computed against matched ARGO profiles, reported as a skill-metric table and plot per depth and region, with the GLORYS-assimilation caveat (Section 9.3) stated alongside every result.

### 18.4 Stratified Evaluation Breakdown (Recommended)

In addition to a single aggregate RMSE, evaluation metrics should be broken down by: (a) region — Bay of Bengal vs. Arabian Sea; (b) season — monsoon vs. non-monsoon; and (c) data-availability regime — near-ARGO vs. far-from-ARGO grid cells. This surfaces where skill is uneven rather than masking it behind one aggregate number.

---

# 19. Uncertainty / Confidence Estimation

Uncertainty estimation is a required, differentiating part of OceanEmbed, not an optional extra.

- **Method:** a deep ensemble (N independently initialized/trained models) or Monte-Carlo dropout at inference, producing a per-pixel, per-depth prediction mean and standard deviation
- **Prediction intervals:** derived from ensemble spread (e.g., approximately a 95% interval from ±1.96σ)
- **Confidence maps:** rendered alongside the temperature maps in the dashboard
- **Uncertainty validation:** checking that ensemble spread correlates with actual error magnitude against ARGO, via a reliability/calibration plot

### Physical-Plausibility Diagnostic Note (Future Scope candidate)

Deviations of a reconstruction from the vertical structure implied by classical Surface Quasi-Geostrophic (SQG) theory can serve as a diagnostic for physically implausible reconstructions, per recent work testing where SQG theory breaks down relative to deep-learning reconstructions (Shan et al., JGR: Machine Learning and Computation, 2026, doi:10.1029/2026JH001410). This is proposed as a **Future Scope** diagnostic check, not a committed SQG-residual correction step. Any uncertainty estimates produced by the model should additionally be calibration-checked — e.g., verifying that a stated 90% prediction interval actually covers the true (ARGO) value roughly 90% of the time — labeled **Recommended**.

---

# 20. PoC Requirements

### 20.1 PoC Region

Bay of Bengal and Arabian Sea sub-boxes within the full North Indian Ocean domain.

### 20.2 PoC Dataset

A representative multi-year subset (indicatively 2018–2023) that includes at least one well-documented historical cyclone period in the Bay of Bengal, to demonstrate the cyclone heat potential product.

### 20.3 PoC Functionality

An end-to-end, reproducible run of: ingestion → preprocessing → embedding → reconstruction → validation → visualization, executable via a single script or notebook.

### 20.4 PoC Success Criteria

- The model beats climatology and persistence baselines on RMSE and correlation for at least the upper 300 m
- ARGO validation is completed with reported skill metrics
- The dashboard runs and displays depth-wise maps, vertical profiles, and at least one derived product (MLD, D20 or TCHP) for the demo period
- The full PoC pipeline runs within free-tier compute and time constraints, with run time documented

---

# 21. System Architecture

### 21.1 High-Level Architecture

Data sources (Copernicus Marine, NASA PO.DAAC, INCOIS) → ingestion (scripted downloaders) → preprocessing/harmonization (regridding, QC, normalization) → satellite embedding engine → subsurface reconstruction model → validation (against the GLORYS test set and ARGO) → output store (NetCDF/Zarr) → visualization (Streamlit dashboard).

### 21.2 Component Diagram

&#91;embedded content: OceanEmbed pipeline · 5 stages\]

### 21.3 Data Flow

Raw, multi-source files → a harmonized, daily 0.25° NetCDF/Zarr store → training/inference tensors → model predictions → post-processed outputs (NetCDF plus derived products) → dashboard queries.

### 21.4 Training vs. Inference Pipelines (context note)

The architecture above supports two conceptually distinct pipelines built from the same components: a **training pipeline** (batch processing of the full historical harmonized dataset against GLORYS targets, run offline on free-tier compute) and an **inference/serving pipeline** (applying the trained embedding encoder and decoder to newly arrived daily surface observations to produce the day's subsurface field for the dashboard). Both pipelines share the harmonization and embedding-engine code paths; only the training pipeline touches GLORYS labels and the loss/optimizer.

---

# 22. Software Requirements

### 22.1 Technology Stack

- **Language:** Python 3.10+
- **ML framework:** PyTorch
- **Data processing:** xarray, dask, zarr, xESMF
- **Geospatial/oceanographic:** netCDF4, cartopy
- **Visualization:** Plotly / Matplotlib, Streamlit
- **Data access:** `copernicusmarine`, `earthaccess`
- **Compute (training):** Kaggle Notebooks (free GPU, \~30 hrs/week) and/or Google Colab free tier, as swappable/parallel compute — no paid infrastructure required.
- **Version control:** Git + GitHub (free public/private repo).
- **Experiment tracking:** Weights & Biases free tier, with a plain CSV/JSON run-log as a no-signup fallback.
- **Config management:** Hydra or plain YAML files under `/configs` for reproducible run parameters.
- **CI (Optional):** GitHub Actions free tier for linting and lightweight tests on push.
- **Hosting/deployment:** Hugging Face Spaces (free tier) for the public Streamlit dashboard demo; Streamlit Community Cloud as an alternative.

### 22.1b Environment Bootstrap (Getting Started)

Step-by-step setup to go from a clean machine to a working OceanEmbed environment (elaborated further as Chunk 1 in Section 35):

1. Clone the repository and create a Python 3.10+ environment (`conda create -n oceanembed python=3.10` or `python -m venv .venv`).
2. Install dependencies from `requirements.txt` / `environment.yml` — PyTorch, xarray, dask, zarr, xESMF, netCDF4, cartopy, streamlit, `copernicusmarine`, `earthaccess`.
3. Register free accounts and store credentials as environment variables (never committed): Copernicus Marine Service (satellite inputs + GLORYS), NASA Earthdata (OSTIA/SMAP/SMOS/ASCAT-family products), INCOIS LAS (ARGO / independent validation access).
4. Provision free GPU compute: enable the GPU accelerator on a Kaggle notebook (P100/T4, \~30 GPU-hrs/week) and/or open a Google Colab notebook (T4, free tier); confirm with a smoke test that `torch.cuda.is_available()` returns `True`.
5. Run a one-day, one-channel download-and-regrid smoke test for the PoC region to confirm the xESMF 0.25° regridding pipeline works end-to-end before scaling up.
6. Initialize the repository skeleton per Section 22.2 (`/data`, `/models`, `/training`, `/evaluation`, `/dashboard`, `/configs`) and commit it.
7. Stand up experiment tracking (W&B free tier or the CSV/JSON fallback) and confirm one dummy run logs successfully.

### 22.2 Code Structure

Modular repository: `/data` (download and preprocessing scripts), `/models` (embedding and reconstruction architectures), `/training` (training loops, configs), `/evaluation` (metrics, ARGO matching), `/dashboard` (Streamlit app), `/configs` (YAML run configurations). Configuration management via per-experiment YAML files.

### 22.3 Pipeline Automation

- Scripted, parameterized preprocessing — a single command from raw downloads to a harmonized store
- A training pipeline driven by config files, with checkpointing and resume
- An evaluation pipeline that runs automatically against a fixed test configuration
- An inference pipeline that generates output fields for a chosen day or period on demand

---

# 23. Performance Requirements

### Data Processing

Preprocessing one year of daily fields is expected to complete within free-tier session limits (target: a few hours, chunked and resumable). Expected data volume: low tens of GB for the full domain across the six surface variables plus the target, over multiple years.

### Model Training

Target training time per experiment fits within a single free Kaggle/Colab session (roughly ≤ 9–12 hours), with checkpointing to span multiple sessions where needed. Hardware constraint: a single free-tier GPU (typically ≤ 16 GB VRAM).

### Inference

Generating one day's full-domain, 15-depth field should run in well under a minute on a free-tier GPU or CPU; batch inference for a full year should complete within a single session.

### Storage

- Raw downloaded data: tens of GB
- Processed Zarr store: a similar order of magnitude, chunked by time
- Trained model checkpoints: a few hundred MB per run

---

# 24. Non-Functional Requirements

### Reliability

The pipeline resumes cleanly after an interrupted free-tier session, via checkpointing at each stage.

### Reproducibility

Fixed seeds, and versioned configs, data and models, as in Section 14.4.

### Scalability

Code is grid-resolution- and region-agnostic, so the system can later extend to the full Indian Ocean or other basins without redesign.

### Maintainability

Modular code structure (Section 22.2), documented configs, and unit tests for core preprocessing and metric functions.

### Portability

Runs on free-tier Kaggle/Colab as well as a local machine with a GPU; no dependency on paid cloud services.

### Accessibility

The dashboard is usable by non-ML domain scientists — clear labels, units and plain-language legends — and documentation is written for both ML and oceanography audiences.

### Note on Numeric Targets

Where a numeric target above is not yet backed by a measured run (e.g., specific latency, throughput or uptime figures), it is labeled **proposed, pending validation** rather than a fixed commitment, and will be confirmed once implementation and the PoC run produce real measurements.

---

# 25. Risks and Limitations

| Risk / Limitation | Impact | Probability | Mitigation |
| --- | --- | --- | --- |
| Missing satellite observations (e.g., SSS gaps/RFI) | Medium — degrades input quality, especially in the Bay of Bengal | High | Mask-aware model, SSS ablation, training window starting 2015 when SMAP is available |
| Dataset temporal/resolution mismatch across products | Medium | High | Conservative regridding via xESMF, daily aggregation of sub-daily products |
| Sparse ARGO observations | Medium — noisy validation statistics per cell | High | Aggregate by region, season and depth bin; report confidence intervals |
| Model extrapolation at depth (weak skill below \~300 m) | High — affects the core deliverable's usefulness at depth | High | Anomaly-from-climatology framing, honest per-depth reporting, no overstated deep-water claims |
| Distribution shift between train and test years (e.g., ENSO/IOD phase) | Medium | Medium | Multi-year training window spanning different climate phases where feasible; monitor test-period skill separately |
| Coastal / land effects | Medium | Medium | Land mask on inputs and loss; exclude high land-fraction cells |
| Validation leakage (GLORYS assimilates ARGO) | High — could overstate skill | High | Temporal holdout, supplementary independent in-situ checks, explicit caveat in all reporting (Section 9.3) |
| Free-tier compute limits (session timeouts, GPU-hour caps) | Medium | High | Small model, mixed precision, per-epoch checkpointing, chunked Zarr storage |
| Slow or unreliable data downloads | Medium | Medium | Scripted, resumable downloads started early; region/date subsetting at source |
| Domain shift between training region/period and PoC deployment period | Medium — model trained on one time window may underperform on a different period/regime | Medium | Monitor PoC-period skill separately from test-period skill; prefer a multi-year, multi-regime training window (see ENSO/IOD row above) |
| ARGO–GLORYS assimilation overlap undermining validation independence in some regions | High — could overstate validation skill in heavily-assimilated regions | High | Explicit independence caveat (Section 9.3), QC-flag and delayed-mode filtering (Section 9), report skill separately where possible |
| Overstating novelty given active prior literature on satellite-to-subsurface deep learning | Medium — credibility risk with evaluators familiar with the field | Medium | Clear related-work section (Section 34) framing OceanEmbed as an incremental, PoC-scale contribution, not a first-of-its-kind claim |
| Uncalibrated uncertainty estimates being over-trusted | Medium — a stated confidence interval that doesn't reflect true coverage could mislead downstream users | Medium | Calibration checks on prediction-interval coverage (Section 19) before uncertainty outputs are presented as decision-grade |
| Reanalysis (GLORYS) bias propagating into “ground truth” training labels | Medium — model can inherit and reproduce GLORYS's own systematic biases | Medium | Climatology-anomaly target framing (Section 8.4, Section 13.7) to reduce sensitivity to mean-state bias; document known GLORYS bias regions where available |
| Coarse ARGO validation cadence (10-day gridded composite, not daily) | Medium — validation is matched to a 10-day window rather than a single day, smoothing short-lived signals (e.g., a fast-moving eddy or storm response) out of the independent check | Medium | Documented explicitly in Section 9.1/9.2 as a PoC-timeline choice; raw per-float ARGO profiles (via `argopy`/ARGO GDAC) remain the more rigorous option for a later stage |

---

# 26. Assumptions

- Copernicus Marine Service and NASA PO.DAAC/Earthdata accounts and API access remain freely available for the project's duration.
- 0.25° is an acceptable operating resolution for the targeted disaster-management use cases (cyclone heat potential, mixed layer depth), even though native SST is finer.
- GLORYS temperature is an acceptable training target, despite carrying its own model and assimilation uncertainty.
- Free-tier compute (Kaggle/Colab) provides sufficient GPU hours to train the chosen model sizes within the project timeline.
- SMAP-era (2015 onward) data availability is sufficient for a meaningful training set, despite the shorter SSS record.
- The team will set up the necessary free accounts and credentials before large-scale downloads begin.

---

# 27. Constraints

- **Geographic:** North Indian Ocean only (5–30°N, 45–105°E) for this phase
- **Spatial:** 0.25° × 0.25°
- **Temporal:** Daily
- **Depth:** 0–1000 m, at the 15 standard depths specified in Section 4.4
- **Data:** free/open datasets only (Copernicus Marine, NASA PO.DAAC, INCOIS) — no proprietary or paid sources
- **Compute:** free-tier GPUs only (Kaggle, Google Colab) — no paid cloud compute
- **Time:** \[project/hackathon deadline\]
- **Other:** no dedicated funding — all tooling, storage and hosting (e.g., a Hugging Face Spaces demo) must stay within free tiers

---

# 28. Dependencies

| Dependency | Owner / Source | Required For | Status |
| --- | --- | --- | --- |
| OSTIA | Copernicus Marine Service | SST | \[Status\] |
| SMAP / SMOS | Copernicus Marine Service | SSS | \[Status\] |
| DUACS | Copernicus Marine Service | SSH/SLA | \[Status\] |
| OSCAR | NASA PO.DAAC | Currents | \[Status\] |
| CCMP / ASCAT | NASA PO.DAAC | Winds | \[Status\] |
| GLORYS | Copernicus Marine Service | Training target | \[Status\] |
| ARGO | INCOIS (Gridded ARGO, LAS) | Independent validation | \[Status\] |

---

# 29. Success Criteria

### Data Pipeline

- [ ] All required datasets acquired
- [ ] All datasets harmonized to a common 0.25° grid
- [ ] Daily harmonized dataset generated
- [ ] Quality-control pipeline completed

### ML Pipeline

- [ ] Satellite embedding model implemented
- [ ] Reconstruction model implemented
- [ ] Baseline models implemented
- [ ] Training pipeline reproducible
- [ ] Inference pipeline functional

### Validation

- [ ] Independent ARGO validation completed
- [ ] RMSE calculated
- [ ] Correlation calculated
- [ ] Bias calculated
- [ ] Depth-wise evaluation completed
- [ ] Regional evaluation completed

### PoC

- [ ] Bay of Bengal / Arabian Sea PoC completed
- [ ] Daily 0.25° outputs generated
- [ ] Visualization available
- [ ] Results documented

### Priority & Verification Notes (this revision)

Every checklist item above corresponds to either a **Mandatory** SIH requirement (region, resolution, cadence, 15 depths, GLORYS target, ARGO/INCOIS-LAS validation, PoC region, 7 input channels, and the pipelines that produce them) or a **Recommended**/**Optional** quality bar added by this PRD. Verification for Mandatory items is by direct inspection of pipeline output against the stated numbers (e.g., grid resolution, depth count, region bounds); verification for Recommended/Optional items is by the ablation and evaluation methodology in Sections 16 and 18. A full Requirement ID / Priority / Verification Method / Acceptance Criterion / Evidence table for the core Mandatory requirements is provided in Section 38 (Requirements Traceability Matrix).

---

# 30. Analytics / Experiment Tracking

- **Experiments to track:** each training run, identified by architecture, input ablation and hyperparameter set
- **Metrics to log:** training and validation loss, RMSE/correlation/bias per depth on the validation set
- **Model and dataset versions:** tracked via configuration hash and preprocessing-pipeline commit (Section 14.4)
- **Training runs:** logged via CSV/JSON, or a free-tier experiment tracker
- **Validation results:** stored per run to support the baseline-comparison table (Section 15)

---

# 31. Open Questions

| Question | Owner | Priority | Status |
| --- | --- | --- | --- |
| Which satellite product will be selected where multiple options exist (e.g., ASCAT vs. CCMP)? | \[Owner\] | P0 | Open |
| What interpolation/regridding method should be used at each stage? | \[Owner\] | P0 | Open |
| What exact training/validation/test periods should be used? | \[Owner\] | P0 | Open |
| Which embedding architecture will be primary? | \[Owner\] | P1 | Open |
| What baseline models are required beyond Section 15? | \[Owner\] | P1 | Open |
| What minimum skill constitutes PoC success? | \[Owner\] | P1 | Open |
| Is uncertainty estimation required for the first PoC, or a later milestone? | \[Owner\] | P1 | Open |
| What is the project/hackathon deadline? | \[Owner\] | P0 | Open |
| Which primary embedding architecture will be used — CNN, ViT, Convformer, ConvLSTM, or GNN — pending empirical comparison (Section 12, Section 16)? | \[Owner\] | P1 | Open |
| Will a masked-autoencoder-style missing-data handling approach (Section 10.8) be implemented for the MVP, or deferred to the Advanced stage (Section 35)? | \[Owner\] | P1 | Open |
| Is spatial/spatiotemporal blocked splitting (Section 11.7) needed at the PoC's spatial scale, or is chronological splitting alone sufficient? | \[Owner\] | P2 | Open |
| Is the Surface-QG physical-plausibility diagnostic (Section 19) in scope for this hackathon timeline, or Future Scope only? | \[Owner\] | P2 | Open |
| Citation needing confirmation: full author lists for the seven research citations added in this revision were not verified (only title/journal/year/DOI were confirmed via web search) — should full author lists be added before submission? | \[Owner\] | P2 | Open |

---

# 32. Deliverables

### Data

- [ ] Data acquisition pipeline
- [ ] Preprocessing pipeline
- [ ] Harmonized dataset
- [ ] Dataset documentation

### ML

- [ ] Satellite embedding engine
- [ ] Reconstruction model
- [ ] Baseline models
- [ ] Training pipeline
- [ ] Inference pipeline

### Validation

- [ ] Evaluation framework
- [ ] ARGO validation
- [ ] Performance report

### Product / PoC

- [ ] Working PoC
- [ ] Visualization interface
- [ ] Sample reconstructed fields
- [ ] Technical documentation
- [ ] User documentation

---

# 33. Future Scope

- Extension beyond the North Indian Ocean to the full Indian Ocean or global coverage
- Additional satellite variables (e.g., chlorophyll, precipitation)
- Additional ocean variables — notably salinity reconstruction
- Higher spatial resolution
- Higher (sub-daily) temporal resolution
- Additional or fully continuous depth levels, exploiting the depth-conditioned decoder
- Real-time / near-real-time inference using operational feeds (INSAT-3D, SCATSAT, Oceansat)
- Operational deployment feeding INCOIS systems
- Refined, uncertainty-aware reconstruction
- Transfer learning to other ocean basins
- ARGO assimilation into the model
- Short-term (7–14 day) forecasting
- A reusable ocean foundation model

---

# 34. References and Data Sources

### Datasets

- GLORYS Global Ocean Reanalysis (training target) — <https://doi.org/10.48670/moi-00021>
- SST, OSTIA — <https://doi.org/10.48670/moi-00168>
- SSS, SMAP / SMOS — <https://doi.org/10.48670/moi-00051>
- SSH / SLA, DUACS — <https://doi.org/10.48670/moi-00145>
- Surface currents, OSCAR L4 — [podaac.jpl.nasa.gov/dataset/OSCAR\_L4\_OC\_FINAL\_V2.0](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0)
- Winds, CCMP 10 m L4 v3.1 — [podaac.jpl.nasa.gov/dataset/CCMP\_WINDS\_10M6HR\_L4\_V3.1](https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1)
- Winds (alternative), ASCAT L2 Coastal — [podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal](https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal)
- In-situ validation: Gridded ARGO, INCOIS Live Access Server (LAS)

### Related Work (research grounding for this revision)

1. **Neural Network Approach to Retrieving Ocean Subsurface Temperatures from Surface Parameters Observed by Satellites** — Water, 13(3):388, 2021, doi:10.3390/w13030388. Found that a feed-forward/CNN mapping from multi-channel surface satellite fields to subsurface temperature profiles is viable. Relevance: motivates OceanEmbed's core surface-to-subsurface supervised learning framing (Section 2). **Inspired-by** the framing; **not reproduced** directly.
2. **Subsurface Temperature Reconstruction for the Global Ocean from 1993 to 2020 Using Satellite Observations and Deep Learning** — Remote Sensing, 14(13):3198, 2022, doi:10.3390/rs14133198. Found that a long-record, globally-trained deep learning reconstruction validated well against ARGO-type independent data, using a climatology-anomaly target. Relevance: supports the independent-validation design (Section 9) and the climatology-anomaly mitigation for reanalysis bias (Section 8.4, 13.7). **Inspired-by** the anomaly framing and validation design; **not reproduced**.
3. **Convformer: A Model for Reconstructing Ocean Subsurface Temperature and Salinity Fields Based on Multi-Source Remote Sensing Observations** — Remote Sensing, 16(13):2422, 2024. Proposed a convolutional-transformer hybrid for subsurface reconstruction. Relevance: candidate hybrid CNN+attention embedding architecture (Section 12.2). **Inspired-by** the architecture idea; **not reproduced** (candidate only, not implemented).
4. **Attention-enhanced deep learning model for reconstruction and downscaling of thermocline depth in the tropical Indian Ocean** — Ocean Modelling, 196:102537, 2025, doi:10.1016/j.ocemod.2025.102537. Found an attention-enhanced CNN improved thermocline-depth reconstruction accuracy in the tropical Indian Ocean. Relevance: directly relevant to the PoC region (same basin family); supports thermocline/MLD/D20 derived products (Section 17.3b) and attention-based decoder candidates. **Inspired-by** the derived-product idea and attention mechanism; **not reproduced**.
5. **TS-Cast: Deep Learning for Subsurface Ocean Reconstruction from Satellite Observations in the Northwestern Pacific** — Ocean Science, 22:2161, 2026. Proposed a ConvLSTM-family spatiotemporal model for joint temperature-salinity reconstruction. Relevance: candidate ConvLSTM embedding engine (Section 12.2) and motivates temporal-window-length ablation (Section 16). **Inspired-by** the recurrent spatiotemporal architecture; **not reproduced**.
6. **Satellite-based reconstruction of high-resolution ocean subsurface temperature using spatiotemporal graph attention networks** — International Journal of Applied Earth Observation and Geoinformation, 149:105237, 2026, doi:10.1016/j.jag.2026.105237. Proposed a graph-attention network handling irregular/sparse observations for subsurface reconstruction. Relevance: supports the GNN candidate already in Section 12.2 and informs missing-data handling (Section 10.8). **Inspired-by** the sparse-data handling approach; **not reproduced**.
7. **Shan et al., "Testing the Limits of Surface QG Theory With Deep Learning"** — Journal of Geophysical Research: Machine Learning and Computation, 2026, doi:10.1029/2026JH001410. Tested classical Surface Quasi-Geostrophic (SQG) theory against deep-learning reconstructions and identified where SQG breaks down. Relevance: motivates a classical SQG physical baseline (Section 15) and a physical-plausibility diagnostic note (Section 19). **Inspired-by** the comparison framing; SQG itself is **not implemented** as the primary method.

Note: for citations 1–6 above, only title, journal, year and DOI/URL were confirmed via web search; full author lists were not independently verified and are flagged in Section 31 (Open Questions) pending confirmation before final submission.

---

# 35. Work Breakdown — 8 Implementation Chunks

The project is split into eight sequential chunks, from environment setup through to a hosted PoC demo; each chunk is scoped to fit a single team sprint on free-tier compute, ending with the frontend dashboard and then deployment.

| Chunk | Focus | Key Deliverable(s) | Depends On |
| --- | --- | --- | --- |
| 1 | Environment & Data Access Setup | Free accounts (Copernicus Marine, NASA Earthdata, INCOIS LAS), repo scaffolding, Kaggle/Colab environment | — |
| 2 | Data Acquisition & Harmonization Pipeline | Scripted downloaders, regridding to 0.25°, land/seafloor masks, harmonized NetCDF/Zarr store | Chunk 1 |
| 3 | Target & Validation Data Preparation | GLORYS regridded to the 15 standard depths, climatology, ARGO matching pipeline | Chunk 2 |
| 4 | Baseline Models & Dataset Splitting | Chronological train/validation/test splits, climatology/persistence/random-forest/U-Net baselines running | Chunk 3 |
| 5 | Satellite Embedding Engine | Masked-autoencoder / CNN-ViT embedding, mask-aware training, embedding evaluation | Chunk 3 |
| 6 | Reconstruction Model, Training & Evaluation | Depth-conditioned decoder, physics-aware loss, ablations, RMSE/correlation/bias vs. baselines and ARGO | Chunks 4, 5 |
| 7 | Frontend / Dashboard | Streamlit dashboard: depth slider, vertical profiles, time series, uncertainty maps | Chunk 6 |
| 8 | Deployment & PoC Demo | Hosted demo (e.g., Hugging Face Spaces), Bay of Bengal / Arabian Sea PoC run, final documentation | Chunk 7 |

### Detailed Feature Breakdown per Chunk

**Chunk 1 — Environment & Data Access Setup**

- Set up a Python 3.10+ environment (conda/venv) with pinned dependencies (PyTorch, xarray, dask, zarr, xESMF, netCDF4, cartopy, streamlit).
- Register free-tier accounts — Copernicus Marine Service, NASA Earthdata, INCOIS LAS — credentials stored as environment variables, never committed.
- Provision free GPU compute: Kaggle Notebooks (\~30 GPU-hrs/week) and/or Google Colab free tier.
- Initialize the Git repository and modular folder structure (`/data`, `/models`, `/training`, `/evaluation`, `/dashboard`, `/configs`).
- Stand up lightweight experiment tracking (W&B free tier, or a CSV/JSON run-log fallback).
- Smoke test: confirm GPU visibility and one authenticated download from each data source.
- *Tech: Python, conda/venv, Git/GitHub, Kaggle/Colab, W&B (optional). Full step list in Section 22.1b.*

**Chunk 2 — Data Acquisition & Harmonization Pipeline**

- Scripted downloaders for each of the 7 surface input channels against Copernicus Marine / NASA Earthdata APIs, with retry/backoff and provenance logging.
- Spatial regridding of every input to the 0.25° PoC-region grid via xESMF conservative/area-weighted remapping.
- Land/seafloor masking and coastline handling for the Bay of Bengal / Arabian Sea domain.
- Temporal alignment of all channels to a common daily cadence.
- Harmonized storage layer (chunked NetCDF/Zarr) with source, license, and access-date metadata (Sec 7.4).
- Data-quality checks: missing-day report, spatial-coverage report, channel value-range sanity checks.
- *Tech: xarray, dask, zarr, xESMF, netCDF4, copernicusmarine/earthaccess.*

**Chunk 3 — Target & Validation Data Preparation**

- Acquire and regrid GLORYS to the 0.25° grid and 15 standard depths (Sec 4.4) as the training target.
- Compute a per-depth, per-cell climatology to support anomaly-based target framing (Sec 8.4).
- Build the ARGO ingestion/QC pipeline (real-time vs. delayed-mode flags) and match profiles in space/time for independent validation (Sec 9).
- Flag the GLORYS–ARGO assimilation-overlap caveat by region (Sec 9.3) so evaluation code can separate independent vs. overlap-affected profiles.
- Implement missing-data handling for target/validation gaps (Sec 10.8).
- *Tech: copernicusmarine, xarray/xESMF, ARGO NetCDF parsing (e.g. argopy).*

**Chunk 4 — Baseline Models & Dataset Splitting**

- Implement the chronological train/validation/test split (Sec 11.1–11.5), plus an optional spatial/spatiotemporal-blocked split (Sec 11.7) for leakage-robustness checks.
- Implement baselines: climatology-only, persistence, random forest, simple U-Net (Sec 15 matrix).
- Implement a minimal classical Surface-QG physical baseline for comparison (Sec 15).
- Wire all baselines into the same evaluation harness the main model will use (Sec 18), for apples-to-apples comparison from day one.
- *Tech: PyTorch/scikit-learn, the Chunk 2/3 harmonized store.*

**Chunk 5 — Satellite Embedding Engine**

- Implement and train the primary embedding architecture (start with CNN or CNN-ViT hybrid, Sec 12.2), with ConvLSTM/Convformer/GNN kept as swappable alternatives behind a config flag.
- Implement mask-aware training compatible with the missing-data strategy (Sec 10.8).
- Run the temporal-window-length ablation (Sec 12.4) to choose how many past days feed one reconstruction.
- Sanity-check embedding quality independently of the decoder before full-pipeline integration.
- *Tech: PyTorch, the `/configs` architecture-swap system.*

**Chunk 6 — Reconstruction Model, Training & Evaluation**

- Implement the depth-conditioned decoder (Sec 13.4); compare the joint encoder+decoder design against the climatology+residual-anomaly framing (Sec 13.7).
- Implement the training loss, including the depth-aware weighted loss and vertical-smoothness/physical-plausibility regularization (Sec 14.2, 14.5).
- Run the ablation suite — input-channel, architecture, temporal-window, loss-term (Sec 16).
- Compute the full evaluation suite — RMSE/correlation/bias vs. ARGO and vs. baselines — with the stratified region/season/data-availability breakdown (Sec 18.4).
- Implement the uncertainty-estimation head (deep ensembles or MC dropout) and its calibration check (Sec 19).
- *Tech: PyTorch, the Chunk 4 evaluation harness.*

**Chunk 7 — Frontend / Dashboard**

- Build the Streamlit dashboard: depth slider, vertical temperature-profile viewer, time-series viewer, uncertainty/confidence overlays (Sec 17.4).
- Add optional derived-product overlays (thermocline depth / MLD / D20) if completed in time (Sec 17.3b).
- Add a region-selection control scoped to the PoC domain.
- Add a model-vs-baseline-vs-ARGO comparison view for judge-facing demonstration.
- *Tech: Streamlit, Plotly/Matplotlib.*

**Chunk 8 — Deployment & PoC Demo**

- Package and deploy the dashboard to a free hosting tier (Hugging Face Spaces, or Streamlit Community Cloud).
- Run the end-to-end PoC over the full Bay of Bengal / Arabian Sea domain and date range (Sec 20).
- Freeze final documentation — PRD, README, model card, reproducibility notes (seeds, configs, environment versions, Sec 14.4).
- Prepare final submission artifacts — this PRD, the SIH pitch deck, and the demo link/recording.
- *Tech: Hugging Face Spaces / Streamlit Community Cloud, Git.*

### MVP vs. Advanced Staging

**MVP stage (Mandatory for the hackathon deadline):** chronological train/validation/test split; one primary embedding architecture (selected empirically, Section 31); core CNN-based encoder + simple depth-conditioned decoder; core outputs (15-depth temperature field, Section 17.1); basic Streamlit dashboard (depth slider, profiles, time series); primary evaluation (Section 18.1) against ARGO.

**Advanced stage (Recommended/Optional, time-permitting):** architecture comparison across CNN/ViT/Convformer/ConvLSTM/GNN candidates (Sections 12, 16); missing-data masked-autoencoder handling (Section 10.8); uncertainty calibration checks (Section 19); additional derived products — thermocline depth, extended MLD/D20 products (Section 17.3b); stratified region/season evaluation breakdown (Section 18.4); spatial-blocked split validation (Section 11.7); Surface-QG diagnostic check (Section 19).

---

# 36. Appendix

- Dataset metadata (per-source variable lists, units, coordinate conventions) — to be compiled during preprocessing
- Variable definitions and data dictionaries
- Coordinate conventions (latitude/longitude sign convention, depth-positive-down)
- Units reference: temperature in °C, salinity in PSU, currents/winds in m/s, sea level anomaly in m
- Model architecture diagrams (see Section 21 for the system-level view; per-model diagrams to be added as architectures are finalized)
- Experiment configuration templates
- Glossary: SST, SSS, SSH/SLA, MLD, D20, TCHP, RMSE, ARGO, GLORYS, and other acronyms used throughout this document

## 37. Change Summary

| PRD Section | Change Made | Research/Technical Rationale | Priority |
| --- | --- | --- | --- |
| Sec 1 (Doc Info) | Bumped PRD Version 0.1 → 0.2 | Reflects this research-integration revision | Mandatory |
| Sec 2.2 | Added literature-grounding sentence | Establishes credibility via prior satellite-to-subsurface DL literature (Papers 1, 2) | Recommended |
| Sec 3 | Labeled each objective/sub-objective | Clarifies which objectives are SIH-required vs. added scope | Mixed (Mandatory/Recommended/Future Scope) |
| Sec 4 | Added Non-Goals list | Clarifies out-of-scope items without changing Mandatory scope numbers | Future Scope |
| Sec 6 | Added UC-05 (thermocline/MLD use case) | Extends use cases to cover derived products (Paper 4) | Recommended |
| Sec 7 | Added per-channel licensing/access notes | Improves reproducibility documentation | Recommended |
| Sec 8 | Added GLORYS reanalysis-bias caveat + anomaly mitigation | GLORYS is a model product, not direct observation (Paper 2) | Recommended |
| Sec 9 | Extended ARGO QC / assimilation-overlap caveat | Strengthens independent-validation rigor | Recommended |
| Sec 10 | Added 10.8 Missing-Data Handling subsection | Gap-filling strategy for cloud-obscured inputs (Paper 6) | Recommended |
| Sec 11 | Added 11.7 spatial/spatiotemporal split note | Reduces spatial-autocorrelation leakage risk | Recommended |
| Sec 12 | Added ConvLSTM and Convformer candidates + temporal-window note | Broadens architecture search space (Papers 3, 5) | Recommended |
| Sec 13 | Added joint vs. climatology+residual note + FiLM candidate | Broadens decoder design options (Paper 2) | Recommended |
| Sec 14 | Added depth-aware loss + physical-plausibility regularization note | Improves training rigor, links to Sec 19 | Recommended |
| Sec 15 | Added Surface-QG classical physical baseline row | Adds a physics-based comparison point (Paper 7) | Recommended |
| Sec 16 | Added architecture/temporal-window/loss-term ablation extensions | Ensures ablation coverage matches new candidates | Recommended |
| Sec 17 | Added 17.3b thermocline/MLD/D20 derived products note | Adds decision-relevant derived products (Paper 4) | Recommended/Future Scope |
| Sec 18 | Added 18.4 stratified region/season/data-availability breakdown | Prevents a single aggregate metric masking uneven skill | Recommended |
| Sec 19 | Added SQG physical-plausibility diagnostic + calibration-check note | Known limitation and uncertainty-calibration practice (Paper 7) | Future Scope / Recommended |
| Sec 21 | Added 21.4 training vs. inference pipeline note | Clarifies pipeline split without touching the diagram widget | Recommended |
| Sec 24 | Added "proposed, pending validation" framing note | Prevents aspirational numbers being read as commitments | Recommended |
| Sec 25 | Added 5 new risk rows (domain shift, ARGO-GLORYS overlap, novelty overstatement, uncalibrated uncertainty, reanalysis bias) | Broadens risk register to cover research-integration risks | Mandatory |
| Sec 29 | Added Priority & Verification Notes | Ties checklist items to priority labels and Sec 38 | Recommended |
| Sec 31 | Added 5 new open-question rows | Surfaces decisions raised by this revision | Mandatory |
| Sec 34 | Added Related Work subsection (7 papers) | Documents research grounding with inspired-by/not-reproduced notes | Mandatory |
| Sec 35 | Added MVP vs. Advanced staging note | Clarifies delivery sequencing under hackathon time constraints | Mandatory (MVP) / Recommended (Advanced) |

## 38. Requirements Traceability Matrix

| Original SIH Requirement | Revised PRD Section | Implementation Component | Verification Method | Acceptance Evidence |
| --- | --- | --- | --- | --- |
| Region 5–30°N / 45–105°E | Sec 4.1 Geographic Scope | Data acquisition & regridding scripts (Work Breakdown Chunk 2) | Inspect grid bounds in harmonized dataset | Bounding box matches spec in harmonized NetCDF/Zarr coordinates |
| 0.25° spatial grid | Sec 4.2 Spatial Scope | Conservative (xESMF) regridding step | Inspect output grid resolution | Grid spacing = 0.25° confirmed in harmonized store metadata |
| Daily cadence | Sec 4.3 Temporal Scope | Daily aggregation of sub-daily products | Inspect output file cadence | One file per calendar day across training window |
| 15 standard depths | Sec 4.4 Depth Scope | GLORYS depth-regridding step (Chunk 3) | Inspect depth-axis length | 15-depth axis confirmed in target dataset |
| GLORYS reanalysis as training target | Sec 8 Target Dataset | Target & Validation Data Preparation (Chunk 3) | Inspect training-label source | Training labels sourced from GLORYS product (moi-00021) |
| ARGO / INCOIS-LAS as independent validation (never used in training) | Sec 9 Independent Validation | ARGO matching pipeline (Chunk 3) | Inspect train/val/test data lineage for ARGO exclusion | ARGO/INCOIS-LAS data absent from training set; used only in Sec 18 evaluation |
| PoC scoped to Bay of Bengal / Arabian Sea | Sec 20 PoC Region | PoC Demo (Chunk 8) | Inspect PoC run's spatial extent | PoC output covers only Bay of Bengal + Arabian Sea subregion |
| Exactly 7 surface input channels | Sec 7.1 Surface Input Channels | Data Acquisition & Harmonization Pipeline (Chunk 2) | Inspect input tensor channel count | Input tensor has 7 channels (SST, SSS, SSH/SLA, current U/V, wind U/V) |
| (New, Recommended) Missing-data handling for cloud-obscured inputs | Sec 10.8 | Preprocessing gap-fill step (Advanced stage) | Inspect masked/interpolated fraction in preprocessed inputs | Gap-fill strategy documented and applied where implemented |
| (New, Recommended) Classical SQG physical baseline | Sec 15 Baselines | Baseline comparison run (Advanced stage) | Compare SQG-derived reconstruction vs. ML model on held-out set | SQG baseline RMSE reported alongside ML baselines |
| (New, Recommended) Stratified region/season evaluation | Sec 18.4 | Evaluation reporting script | Inspect per-region/per-season metric breakdown | Metrics reported separately by region and season |
| (New, Recommended) Uncertainty calibration check | Sec 19 | Uncertainty/ensemble head + calibration script | Compute prediction-interval coverage vs. nominal level | Coverage statistic reported (e.g., 90% PI covers \~90% of ARGO obs) |
| (New, Mandatory/Recommended) MVP vs. Advanced staging | Sec 35 Work Breakdown | Work Breakdown chunks 1–8 | Inspect which chunks are complete at deadline | MVP chunks (1–7 core) complete; Advanced items tracked separately |
