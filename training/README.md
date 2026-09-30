# /training

Data layer, baselines, training and experiments (PRD Sections 11, 13.6, 14-16, Chunks 4 and 6).

- `data.py` — loads the store named by `paths.processed_dir` (real or synthetic), chronological splits with a buffer
  gap dropped from the end of train/validation, **train-only** normalization, and `WindowDataset` (windows never
  reach across a split boundary; missing values become 0 with an explicit mask channel).
- `baselines.py` — climatology (train-period day-of-year mean, circularly smoothed; borrows the nearest day if a
  day-of-year is absent from training), persistence (7-day lag), random forest on raw surface fields.
- `train.py` — OceanEmbed and U-Net training: masked Huber + physics-aware terms (stratification stability, vertical
  smoothness), AMP, OneCycle, early stopping, MAE pretraining, checkpoints named by config hash. Anomaly-from-
  climatology framing is **auto**: only with >= 3 training years, because with fewer the climatology is fit on the
  very days it is subtracted from and the model never sees real anomalies (this cost the one-year demo model ~5x RMSE
  before it was made automatic).
- `experiments.py` — the PRD ablation set (inputs, window, pretraining, physics loss, anomaly framing, U-Net) with
  identical budgets, resumable, plus seed repeats (`oceanembed_s1/s2`, `unet_s1/s2`) of the two headline models. The
  report's "Seed variability" section gives mean +/- std across seeds: a gap between two models smaller than that spread
  is not a real difference (single-seed differences of ~0.01-0.02 C were within noise on the first synthetic runs).
- `run_logger.py` — W&B if `WANDB_API_KEY` is set, otherwise JSONL under `runs/`. Every run logs its `data_source`.

All models are scored on the same eval days, masks and metrics, against GLORYS only. Tests: `test_training_data.py`,
`test_train.py`.
