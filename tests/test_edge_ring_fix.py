"""Regression tests for the edge-ring bug: Copernicus subsets were cut exactly at the domain bounds, so the target
cells centred on the domain edge were only partly covered by the source, and the conservative regrid returned a
fraction of the true value there (SST/SSS/SSH x0.5, GLORYS x0.67, corners x0.25)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data.catalog import Channel, Domain
from data.qc import edge_ratio_report, run_channel_qc
from data.regrid import COVERAGE_MIN, _weights_path, regrid_dataset
from data.sources import copernicus

DOMAIN = Domain(lat_min=5.0, lat_max=10.0, lon_min=45.0, lon_max=50.0, resolution_deg=0.25)   # 21 x 21 target cells


def _src(lo_lat: float, hi_lat: float, lo_lon: float, hi_lon: float, value: float = 20.0) -> xr.Dataset:
    """A 0.05 deg source whose CELLS span exactly [lo, hi] (centres are half a cell inside), like OSTIA/GLORYS subsets."""
    lat = np.arange(lo_lat + 0.025, hi_lat, 0.05)
    lon = np.arange(lo_lon + 0.025, hi_lon, 0.05)
    return xr.Dataset({"sst": (("lat", "lon"), np.full((len(lat), len(lon)), value))}, coords={"lat": lat, "lon": lon})


def test_the_old_behaviour_would_have_halved_the_edge(tmp_path: Path):
    """Documents the bug itself: a source cropped exactly to the domain covers half of each edge target cell."""
    src = _src(5.0, 10.0, 45.0, 50.0)
    from data.regrid import build_regridder, source_coverage
    cov = source_coverage(src, build_regridder(src, DOMAIN, tmp_path))
    assert cov.isel(lat=0, lon=10).item() == pytest.approx(0.5, abs=1e-3)          # an edge cell: half covered
    assert cov.isel(lat=0, lon=0).item() == pytest.approx(0.25, abs=1e-3)          # a corner: a quarter covered
    assert cov.isel(lat=10, lon=10).item() == pytest.approx(1.0, abs=1e-3)         # interior: fully covered


def test_partly_covered_edge_cells_become_nan_not_scaled_values(tmp_path: Path, capsys):
    out = regrid_dataset(_src(5.0, 10.0, 45.0, 50.0, value=20.0), DOMAIN, cache_dir=tmp_path)
    sst = out["sst"].values
    ring = np.zeros_like(sst, dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    assert np.isnan(sst[ring]).all()                                               # NOT 10.0 (half) or 5.0 (corner)
    assert np.allclose(sst[~ring], 20.0)                                           # interior untouched
    assert out.attrs["regrid_partial_cells_masked"] == int(ring.sum()) == 80
    assert "partly covered" in capsys.readouterr().out                             # never silent


def test_a_source_with_a_margin_covers_every_edge_cell(tmp_path: Path):
    out = regrid_dataset(_src(4.5, 10.5, 44.5, 50.5, value=20.0), DOMAIN, cache_dir=tmp_path)   # the 0.5 deg margin
    assert out.attrs["regrid_partial_cells_masked"] == 0
    assert np.allclose(out["sst"].values, 20.0) and np.isfinite(out["sst"].values).all()         # edge ring correct too
    assert COVERAGE_MIN > 0.99


def test_weights_cache_key_depends_on_the_source_extent(tmp_path: Path):
    a = _src(5.0, 10.0, 45.0, 50.0)
    b = _src(4.5, 9.5, 44.5, 49.5)                                                 # same shape, shifted extent
    assert a.sizes == b.sizes
    assert _weights_path(tmp_path, a, DOMAIN) != _weights_path(tmp_path, b, DOMAIN)
    assert _weights_path(tmp_path, a, DOMAIN) == _weights_path(tmp_path, _src(5.0, 10.0, 45.0, 50.0), DOMAIN)


def _field(edge_scale: float = 1.0) -> xr.Dataset:
    lat, lon = np.arange(5, 8.01, 0.25), np.arange(45, 48.01, 0.25)
    a = np.full((len(lat), len(lon)), 27.0)
    a[0, :] *= edge_scale
    return xr.Dataset({"analysed_sst": (("time", "lat", "lon"), np.stack([a, a]))},
                      coords={"time": [0, 1], "lat": lat, "lon": lon})


def test_qc_edge_check_flags_a_halved_edge_and_passes_a_clean_field():
    bad = edge_ratio_report(_field(0.5), "analysed_sst")
    assert bad["checked"] and not bad["pass"] and bad["suspect_sides"] == ["south"]
    assert bad["median_edge_over_inner"]["south"] == pytest.approx(0.5)
    good = edge_ratio_report(_field(1.0), "analysed_sst")
    assert good["pass"] and good["suspect_sides"] == []
    assert edge_ratio_report(_field(1.05), "analysed_sst")["pass"]                 # ordinary edge gradients are tolerated


def test_qc_edge_check_skips_fields_that_change_sign():
    ds = _field(1.0)
    ds["analysed_sst"] = ds["analysed_sst"] - 27.0                                 # zero / signed field: ratios are meaningless
    assert not edge_ratio_report(ds, "analysed_sst")["checked"]


def test_run_channel_qc_reports_and_warns_on_a_bad_edge(capsys):
    ch = Channel(name="sst", label="sst", source="copernicus", variables=["analysed_sst"],
                 native_resolution_deg=0.05, units="C")
    rep = run_channel_qc(_field(0.5), ch)
    assert rep["variables"]["analysed_sst"]["edge_check"]["pass"] is False
    assert "partly-covered edge cells" in capsys.readouterr().out
    assert run_channel_qc(_field(1.0), ch)["variables"]["analysed_sst"]["edge_check"]["pass"] is True


def test_download_requests_a_margin_and_never_reuses_an_edge_cropped_file(monkeypatch, tmp_path: Path):
    seen = {}
    monkeypatch.setattr(copernicus.copernicusmarine, "subset", lambda **kw: seen.update(kw))
    ch = Channel(name="sst", label="sst", source="copernicus", variables=["analysed_sst"],
                 native_resolution_deg=0.05, units="C", dataset_id="some-dataset")
    path = copernicus.download(ch, "2022-01-01", "2022-03-31", Domain(5.0, 30.0, 45.0, 105.0, 0.25), tmp_path)
    assert (seen["minimum_latitude"], seen["maximum_latitude"]) == (4.5, 30.5)
    assert (seen["minimum_longitude"], seen["maximum_longitude"]) == (44.5, 105.5)
    assert path.name == "sst_2022-01-01_2022-03-31_m0.5.nc" and seen["output_filename"] == path.name
    assert path.name != "sst_2022-01-01_2022-03-31.nc"                             # the old, edge-corrupting cache name
    copernicus.download(ch, "2022-01-01", "2022-03-31", Domain(-89.9, 89.9, 45.0, 105.0, 0.25), tmp_path)
    assert seen["minimum_latitude"] == -90.0 and seen["maximum_latitude"] == 90.0    # margin never leaves the globe
