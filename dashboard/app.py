"""OceanEmbed dashboard (PRD 17.4, Chunk 7).

    streamlit run dashboard/app.py

Reads the predictions Zarr from `python -m evaluation.predict` (path from the demo config, or the
OCEANEMBED_PREDICTIONS env var). A red banner is always shown when the data are synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # so `streamlit run dashboard/app.py` finds the repo packages

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from dashboard import data as D

st.set_page_config(page_title="OceanEmbed", layout="wide")


@st.cache_resource
def load(path: str):
    return D.open_predictions(Path(path))


def plot_map(da, title: str, cmap: str, unit: str, marker: tuple[float, float] | None, symmetric: bool = False):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vmax = float(abs(da).max()) if symmetric and da.notnull().any() else None
    im = da.plot(ax=ax, cmap=cmap, add_colorbar=False, vmin=-vmax if vmax else None, vmax=vmax)
    fig.colorbar(im, ax=ax, label=unit, shrink=0.85)
    ax.set_facecolor("#cfcfcf")                                    # land / no data
    if marker:
        ax.plot(marker[1], marker[0], "k+", ms=14, mew=2)
    ax.set_title(title)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout()
    return fig


path = D.predictions_path()
st.title("OceanEmbed — subsurface ocean temperature, North Indian Ocean")
if not path.exists():
    st.error(f"No predictions found at `{path}`. Run `python -m evaluation.predict` first "
             "(after training a model with `python -m training.train`).")
    st.stop()

ds = load(str(path))
warning = D.data_banner(ds)
if warning:
    st.error(f"{warning}. Do not read these maps or scores as real ocean results.")
note = D.data_note(ds)
if note:
    st.info(note)
st.caption("Reconstruction of daily 0.25 deg temperature at 15 depths from six surface satellite variables. "
           "Comparisons are against GLORYS only (GLORYS assimilates ARGO), so agreement is not independent validation. "
           f"Model version: {ds.attrs.get('model_version', 'unknown')}.")

times = [str(t)[:10] for t in ds.time.values]
depths = [float(z) for z in ds.depth.values]
with st.sidebar:
    st.header("Controls")
    date = st.select_slider("Date", options=times, value=times[len(times) // 2])
    depth = st.select_slider("Depth (m)", options=depths, value=depths[0])
    label = st.selectbox("Map variable", list(D.MAP_VARIABLES))
    dlat, dlon = D.default_location(ds)
    lat = st.number_input("Latitude (N)", 5.0, 30.0, dlat, 0.25)
    lon = st.number_input("Longitude (E)", 45.0, 105.0, dlon, 0.25)
t_idx, z_idx = times.index(date), depths.index(depth)
var, cmap, unit = D.MAP_VARIABLES[label]
marker = D.nearest_cell(ds, lat, lon)
if not D.is_ocean(ds, lat, lon):
    st.sidebar.warning("That point is on land — pick an ocean location for profiles/time series.")

tab_map, tab_profile, tab_series, tab_region, tab_perf, tab_about = st.tabs(
    ["Maps", "Vertical profile", "Time series", "Regional summary", "Model performance", "About"])

with tab_map:
    field = D.map_field(ds, var, t_idx, z_idx)
    where = "" if var in D.DEPTH_FREE else f" at {depth:g} m"
    st.pyplot(plot_map(field, f"{label}{where} — {date}", cmap, unit, marker, symmetric=(var == "error")))
    st.caption("Grey = land or below the local seafloor. Black + marks the selected point.")

with tab_profile:
    prof = D.profile(ds, t_idx, lat, lon)
    if prof["temp"].isna().all():
        st.info("No data at this point (land).")
    else:
        fig, ax = plt.subplots(figsize=(4.5, 5))
        ax.plot(prof["temp"], prof["depth_m"], "o-", label="reconstruction")
        ax.fill_betweenx(prof["depth_m"], prof["temp"] - 1.96 * prof["std"], prof["temp"] + 1.96 * prof["std"], alpha=0.25,
                         label="+/-1.96 std (MC dropout)")
        ax.plot(prof["glorys"], prof["depth_m"], "s--", ms=3, label="GLORYS")
        ax.invert_yaxis(); ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("temperature (deg C)"); ax.set_ylabel("depth (m)"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"{marker[0]:.2f}N {marker[1]:.2f}E — {date}")
        st.pyplot(fig)

with tab_series:
    ts = D.timeseries(ds, z_idx, lat, lon)
    if ts["temp"].isna().all():
        st.info("No data at this point (land).")
    else:
        st.line_chart(ts[["temp", "glorys"]], y_label=f"temperature at {depth:g} m (deg C)")
        st.caption("Reconstruction vs GLORYS at the selected point and depth (evaluation days only).")

with tab_region:
    cols = st.columns(2)
    for col, region in zip(cols, D.REGION_LABELS):
        with col:
            st.subheader(D.REGION_LABELS[region])
            summ = D.regional_summary(ds, region, z_idx)
            st.line_chart(summ[["reconstruction", "glorys"]], y_label=f"regional mean at {depth:g} m (deg C)")
            st.metric("Mean daily RMSE vs GLORYS", f"{summ['rmse'].mean():.3f} deg C")

with tab_perf:
    runs = D.runs_path()
    md = sorted(runs.glob("report_*.md"))
    if not md:
        st.info("No evaluation report yet. Run `python -m evaluation.report` after the experiments finish.")
    else:
        text = md[0].read_text(encoding="utf-8")
        st.markdown("\n".join(line for line in text.splitlines() if not line.startswith("![")))
        for fig_name in ("rmse_by_depth.png", "reliability.png"):
            if (runs / "figures" / fig_name).exists():
                st.image(str(runs / "figures" / fig_name))

with tab_about:
    st.markdown(
        "**What this is.** OceanEmbed learns an embedding of six daily surface satellite variables (SST, SSS, SSH, "
        "currents, winds) and decodes it to temperature at 15 depths.\n\n"
        "**How to read it.** The uncertainty layer is a Monte-Carlo-dropout spread; check the reliability plot in "
        "*Model performance* before treating it as a calibrated interval. MLD, D20 and TCHP are coarse estimates "
        "from 15 vertical levels.\n\n"
        "**Limits.** Scores are against GLORYS only. A model trained on few years, or on synthetic data, "
        "demonstrates the pipeline rather than real-ocean skill.")
