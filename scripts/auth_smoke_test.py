"""Authenticated-download smoke test (PRD Section 22.1b step 5, Chunk 1 exit criterion).

Confirms real credentials work against each live data source with a minimal,
cheap request — not a full download. Run inside the `oceanembed` conda env:

    conda run -n oceanembed python scripts/auth_smoke_test.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()


def check_copernicus_marine() -> bool:
    try:
        import copernicusmarine

        # describe() hits the catalogue API and requires valid credentials
        # to resolve private/large datasets; a lightweight metadata call.
        copernicusmarine.describe(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m")
        print("  [ok]   Copernicus Marine: catalogue query succeeded")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] Copernicus Marine: {exc}")
        return False


def check_earthdata() -> bool:
    try:
        import earthaccess

        auth = earthaccess.login(strategy="environment")
        if auth.authenticated:
            print("  [ok]   NASA Earthdata: login succeeded")
            return True
        print("  [FAIL] NASA Earthdata: login returned not-authenticated")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] NASA Earthdata: {exc}")
        return False


def main() -> int:
    print("== Authenticated data-source checks ==")
    results = {
        "copernicus_marine": check_copernicus_marine(),
        "earthdata": check_earthdata(),
    }
    print("\n(INCOIS LAS has no standard Python auth client yet — verify manually at https://las.incois.gov.in/)")

    if all(results.values()):
        print("\nPASS — credentials confirmed working against live services.")
        return 0
    print(f"\nFAIL — {[k for k, v in results.items() if not v]} did not authenticate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
