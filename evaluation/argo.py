"""INCOIS ARGO 10-day gridded VAM product: fetch, QC, and matching to the model grid
(PRD Sections 9.1-9.3, Chunk 3).

Independent validation, not training data — never touched by the training pipeline. GLORYS
assimilates ARGO observations, so agreement here is not by itself fully independent proof of
skill (PRD Section 9.3); see `attach_independence_caveat()`.
"""

from __future__ import annotations

import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

from data.catalog import ArgoConfig, Domain
from data.qc import mask_invalid_values
from data.regrid import standardize_latlon, target_grid
from data.retry import retry
from data.vertical import interpolate_to_standard_depths

INDEPENDENCE_CAVEAT = (
    "GLORYS (the training target) assimilates ARGO observations, so strong agreement between "
    "model output and this ARGO product is not by itself fully independent proof of skill "
    "(PRD Section 9.3). Per-float assimilated-vs-not-assimilated stratification is a "
    "Recommended, not Mandatory, PRD item and is deferred at this PoC scale."
)


def _erddap_url(argo_cfg: ArgoConfig, domain: Domain, start: str, end: str) -> str:
    # ERDDAP griddap requires an explicit bound per dimension for every requested variable;
    # ARGO's own native depth coverage (ZAX) is 5-2000m (confirmed via the dataset's metadata),
    # requested in full here since vertical matching to the 15 standard depths happens downstream.
    #
    # Unlike the NOAA ERDDAP server data/bathymetry.py targets (which requires [ ] literal,
    # unescaped), INCOIS's ERDDAP runs a stricter Tomcat that rejects raw [ ] in the request
    # target with an HTTP 400 "Invalid character" error (confirmed live) — so brackets are
    # percent-encoded here specifically for this server. Parentheses/colons are left as-is,
    # matching the working query confirmed live against this same server.
    constraint = (
        f"%5B({start}T00:00:00Z):({end}T00:00:00Z)%5D"
        f"%5B(5.0):(2000.0)%5D"
        f"%5B({domain.lat_min}):({domain.lat_max})%5D"
        f"%5B({domain.lon_min}):({domain.lon_max})%5D"
    )
    vars_ = ",".join(f"{v}{constraint}" for v in argo_cfg.variables.values())
    return f"{argo_cfg.erddap_base_url}.nc?{vars_}"


INSECURE_TLS_ENV = "ARGO_ALLOW_INSECURE_TLS"


def _download_with_tls_fallback(url: str) -> bytes:
    # erddap.incois.gov.in fails standard certificate verification (confirmed live, `curl -k`
    # required) — a server-side cert-chain issue. Verified first; an unverified retry happens
    # only if the user explicitly opts in via ARGO_ALLOW_INSECURE_TLS=1, so it is never silent.
    # urllib wraps the underlying ssl.SSLCertVerificationError in a urllib.error.URLError
    # (confirmed live — catching only the ssl exception type missed it), so both are checked.
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            return resp.read()
    except (ssl.SSLCertVerificationError, urllib.error.URLError) as exc:
        is_cert_error = isinstance(exc, ssl.SSLCertVerificationError) or isinstance(
            getattr(exc, "reason", None), ssl.SSLCertVerificationError
        )
        if not is_cert_error:
            raise
        if os.environ.get(INSECURE_TLS_ENV) != "1":
            raise RuntimeError(
                "TLS certificate verification failed for erddap.incois.gov.in (known server-side "
                f"cert-chain issue). To accept this for the public, read-only ARGO fetch, set "
                f"{INSECURE_TLS_ENV}=1 (e.g. in .env)."
            ) from exc
        print("  [warn] TLS verification disabled for erddap.incois.gov.in ("
              f"{INSECURE_TLS_ENV}=1); data integrity is not authenticated.")
        with urllib.request.urlopen(url, timeout=180, context=ssl._create_unverified_context()) as resp:
            return resp.read()


def fetch_argo(argo_cfg: ArgoConfig, domain: Domain, start: str, end: str, cache_dir: Path) -> xr.Dataset:
    """Subset+download the INCOIS ARGO 10-day VAM product for [start, end], caching locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"argo_{start}_{end}.nc"

    if not cache_path.exists():
        url = _erddap_url(argo_cfg, domain, start, end)

        def _fetch() -> bytes:
            return _download_with_tls_fallback(url)

        def _on_retry(attempt: int, exc: Exception) -> None:
            print(f"  [retry {attempt}] ARGO ERDDAP fetch failed: {exc}")

        content = retry(_fetch, on_retry=_on_retry)
        cache_path.write_bytes(content)

    ds = xr.open_dataset(cache_path)
    ds = standardize_latlon(ds)
    if "ZAX" in ds.dims:
        ds = ds.rename({"ZAX": "depth"})
    return ds


def qc_argo(ds: xr.Dataset, argo_cfg: ArgoConfig, max_error_threshold: float | None = None) -> xr.Dataset:
    """Physical range check + error-field masking (PRD Section 9.2's QC step, adapted for a
    gridded analyzed product that doesn't carry per-profile QC flags)."""
    threshold = argo_cfg.max_error_threshold if max_error_threshold is None else max_error_threshold
    v = argo_cfg.variables
    out = mask_invalid_values(ds, [v["temp"], v["sal"]])

    for value_key, error_key in (("temp", "temp_error"), ("sal", "sal_error")):
        value_var, error_var = v[value_key], v.get(error_key)
        if error_var and error_var in out and value_var in out:
            out[value_var] = out[value_var].where(out[error_var] <= threshold)
    return out


def match_spatial(ds: xr.Dataset, domain: Domain) -> xr.Dataset:
    """Nearest-neighbor match of ARGO's 1deg grid onto the domain's 0.25deg target grid.

    Nearest-neighbor (not conservative regridding) is deliberate here: this is independent
    *validation* data being matched to the nearest cell (PRD Section 9.2's own wording), not a
    training input being resampled — distinct from data/regrid.py's "never nearest-neighbour"
    rule, which applies specifically to training-input regridding.
    """
    tgt = target_grid(domain)
    return ds.reindex(lat=tgt["lat"], lon=tgt["lon"], method="nearest")


def match_temporal(ds: xr.Dataset, target_dates: xr.DataArray | np.ndarray) -> xr.Dataset:
    """Match each target (daily) date to the nearest 10-day composite window (PRD Section 9.2 —
    not a daily match, since the gridded ARGO product is itself a 10-day average)."""
    return ds.reindex(time=target_dates, method="nearest")


def match_depth(ds: xr.Dataset, depths_m: list[int]) -> xr.Dataset:
    """Linear interpolation of ARGO's native depth levels onto the 15 standard depths."""
    return interpolate_to_standard_depths(ds, depths_m, depth_dim="depth")


def attach_independence_caveat(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    out.attrs["independence_caveat"] = INDEPENDENCE_CAVEAT
    return out
