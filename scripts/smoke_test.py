"""Chunk 1 smoke test (PRD Section 22.1b, step 4-5).

Confirms the environment is set up correctly before any large-scale
downloads or training begin:
  - required Python packages import
  - GPU is visible to PyTorch (if available)
  - Copernicus Marine / NASA Earthdata credentials are present in the
    environment (does not perform a live download)
  - a minimal in-memory conservative regrid runs via xESMF, if installed

Run: python scripts/smoke_test.py
"""

from __future__ import annotations

import importlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = [
    "torch",
    "xarray",
    "dask",
    "zarr",
    "netCDF4",
    "numpy",
    "pandas",
]

OPTIONAL_PACKAGES = [
    "xesmf",
    "cartopy",
    "copernicusmarine",
    "earthaccess",
    "streamlit",
]

CREDENTIAL_VARS = [
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "EARTHDATA_USERNAME",
    "EARTHDATA_PASSWORD",
]


def check_imports(packages: list[str], required: bool) -> list[str]:
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  [ok]   {pkg}")
        except Exception as exc:
            level = "MISSING (required)" if required else "missing (optional)"
            print(f"  [{'FAIL' if required else 'warn'}] {pkg}: {level} — {exc}")
            missing.append(pkg)
    return missing


def check_gpu() -> None:
    import torch

    available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() = {available}")
    if not available:
        print("  no GPU visible — CPU-only session (fine for smoke test, not for training)")
        return

    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    # is_available() can be True while the installed build still lacks kernels for this
    # GPU's architecture (seen with new archs like Blackwell/sm_120) — a real op catches that.
    try:
        x = torch.randn(1000, 1000, device="cuda")
        (x @ x).sum().item()
        print("  [ok]   GPU matmul ran successfully")
    except RuntimeError as exc:
        print(f"  [FAIL] GPU visible but a real op failed — install a PyTorch build that supports this GPU: {exc}")


def check_credentials() -> list[str]:
    missing = []
    for var in CREDENTIAL_VARS:
        if os.environ.get(var):
            print(f"  [ok]   {var} is set")
        else:
            print(f"  [warn] {var} is not set — copy .env.example to .env and fill it in")
            missing.append(var)
    return missing


def check_regrid() -> bool:
    try:
        import numpy as np
        import xarray as xr
        import xesmf as xe
    except ImportError:
        print("  [skip] xesmf not installed — install via conda-forge (see environment.yml)")
        return True

    src = xr.Dataset(
        {"dummy": (("lat", "lon"), np.random.rand(20, 20))},
        coords={"lat": np.linspace(5, 30, 20), "lon": np.linspace(45, 105, 20)},
    )
    target_grid = xr.Dataset(
        coords={
            "lat": np.arange(5, 30, 0.25),
            "lon": np.arange(45, 105, 0.25),
        }
    )
    regridder = xe.Regridder(src, target_grid, "conservative")
    out = regridder(src)
    ok = out["dummy"].shape[0] > src["dummy"].shape[0]
    print(f"  [ok]   conservative regrid ran, output shape {out['dummy'].shape}")
    return ok


def main() -> int:
    print("== Required packages ==")
    missing_required = check_imports(REQUIRED_PACKAGES, required=True)

    print("\n== Optional packages ==")
    check_imports(OPTIONAL_PACKAGES, required=False)

    print("\n== GPU ==")
    if "torch" not in missing_required:
        check_gpu()
    else:
        print("  skipped (torch missing)")

    print("\n== Credentials (.env) ==")
    missing_creds = check_credentials()

    print("\n== xESMF regrid smoke test ==")
    regrid_ok = check_regrid()

    print("\n== Summary ==")
    if missing_required:
        print(f"FAIL — missing required packages: {missing_required}")
        return 1
    if missing_creds:
        print(f"PASS with warnings — missing credentials: {missing_creds}")
    if not regrid_ok:
        print("FAIL — regrid smoke test did not produce expected output")
        return 1
    print("Environment looks ready for Chunk 2 (Data Acquisition & Harmonization).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
