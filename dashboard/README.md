# /dashboard

Streamlit app (PRD Section 17.4, Chunk 7): `streamlit run dashboard/app.py`.

- Tabs: maps (temperature, uncertainty, difference vs GLORYS, GLORYS, MLD, D20, TCHP), vertical profile with a
  +/-1.96 std band, time series, Bay of Bengal / Arabian Sea summaries, model performance (the evaluation report), about.
- Reads `<processed_dir>/outputs/oceanembed_predictions.zarr` (from `python -m evaluation.predict`). Override with the
  `OCEANEMBED_PREDICTIONS` and `OCEANEMBED_RUNS` env vars, or `OCEANEMBED_CONFIG` for a different config.
- A red banner is always shown for synthetic data. "Error" views compare with GLORYS only.
- `data.py` holds the pure helpers (no Streamlit import) so they are unit-tested; `tests/test_dashboard.py` also drives
  the app headlessly with Streamlit's `AppTest`.
