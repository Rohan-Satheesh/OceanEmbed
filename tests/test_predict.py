from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from data import synthetic as syn
from data.catalog import load_base_config
from evaluation import predict as P
from training import train as T

REPO = Path(__file__).resolve().parents[1]
CFG0 = load_base_config(override=REPO / "configs" / "demo.yaml")
PERIODS = {"train": ["2022-01-01", "2022-01-31"], "validation": ["2022-02-01", "2022-02-14"],
           "test": ["2022-02-15", "2022-02-28"]}
TINY = dict(epochs=2, pretrain_epochs=1, window=2, base=8, embed_dim=16, film_ch=8, batch_size=4, eval_stride=5,
            patience=3)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("predstore")
    syn.generate("2022-01-01", "2022-02-28", root / "syn", CFG0.domain, CFG0.depths_m, seed=5, processed_dir=root)
    cfg = replace(CFG0, periods=PERIODS, paths={**CFG0.paths, "processed_dir": "syn", "checkpoints_dir": "ckpt"})
    res = T.fit(cfg, T.TrainConfig(name="pred_t", **TINY), root=root, device="cpu", log=False)
    return cfg, root, res


def test_quality_flag_and_static_geometry():
    y = np.full((3, 2, 2), np.nan)
    y[0, 0, 0] = y[1, 0, 0] = 1.0                    # ocean cell, valid at 2 of 3 depths (seafloor between)
    y[:, 1, 1] = 5.0                                 # fully valid column
    valid = P.static_geometry(y[None].repeat(4, axis=0))
    flag = P.quality_flag(valid)
    assert flag[:, 1, 1].tolist() == [0, 0, 0] and flag[:, 0, 0].tolist() == [0, 0, 2] and flag[:, 0, 1].tolist() == [1, 1, 1]


def test_checkpoint_lookup(trained):
    cfg, root, res = trained
    assert P.find_checkpoint(cfg, "pred_t", root) == Path(res["checkpoint"])
    with pytest.raises(FileNotFoundError):
        P.find_checkpoint(cfg, "does_not_exist", root)


def test_run_inference_writes_cf_netcdf_and_zarr(trained):
    cfg, root, res = trained
    zpath = P.run_inference(cfg, Path(res["checkpoint"]), "test", stride=4, mc=3, root=root, device="cpu")
    nc_files = sorted((root / "syn" / "outputs" / "netcdf").glob("oceanembed_temp_*.nc"))
    assert len(nc_files) >= 2 and all(f.name.startswith("oceanembed_temp_2022") for f in nc_files)
    assert [f.name for f in nc_files] == ["oceanembed_temp_20220215.nc", "oceanembed_temp_20220219.nc",
                                          "oceanembed_temp_20220223.nc", "oceanembed_temp_20220227.nc"]
    d = xr.open_dataset(nc_files[0])
    assert d.attrs["Conventions"] == "CF-1.8" and d.attrs["data_source"] == "synthetic"
    assert "synthetic_warning" in d.attrs and d.attrs["model_version"] == f"pred_t-{res['config_hash']}"
    assert "GLORYS only" in d.attrs["reference_note"]
    assert d["temp"].dims == ("time", "depth", "lat", "lon") and d["mld"].dims == ("time", "lat", "lon")
    assert d["quality_flag"].dims == ("depth", "lat", "lon")
    assert set(np.unique(d["quality_flag"].values)) <= {0, 1, 2}
    pred_ok = d["quality_flag"].values == 0
    assert np.isfinite(d["temp"].values[0][pred_ok]).all() and np.isnan(d["temp"].values[0][~pred_ok]).all()
    assert (d["temp_std"].values[0][pred_ok] >= 0).all()
    z = xr.open_zarr(zpath, consolidated=False)
    for v in ("temp", "temp_std", "mld", "d20", "tchp", "glorys", "quality_flag"):
        assert v in z
    assert z.sizes["time"] == len(nc_files) == 4 and z["quality_flag"].dims == ("depth", "lat", "lon")
    assert (z["mld"].values[np.isfinite(z["mld"].values)] >= 0).all()
    assert (z["tchp"].values[np.isfinite(z["tchp"].values)] >= 0).all()
