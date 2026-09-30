import pytest

from data.sources import earthdata


def test_download_all_retries_until_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(earthdata, "_is_valid_nc", lambda p: True)
    calls = []
    def fake(granules, local_path):
        calls.append(1)
        return ["a.nc"] if len(calls) == 1 else ["a.nc", "b.nc"]   # first pass drops one file
    monkeypatch.setattr(earthdata.earthaccess, "download", fake)
    out = earthdata._download_all(["g1", "g2"], tmp_path, "x", lambda *a: None)
    assert len(out) == 2 and len(calls) == 2


def test_download_all_raises_if_still_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(earthdata, "_is_valid_nc", lambda p: True)
    monkeypatch.setattr(earthdata.earthaccess, "download", lambda g, local_path: ["a.nc"])
    with pytest.raises(RuntimeError, match="1 of 2|only 1 of 2"):
        earthdata._download_all(["g1", "g2"], tmp_path, "x", lambda *a: None, attempts=2)


def test_download_all_deletes_corrupt_file_and_refetches(monkeypatch, tmp_path):
    from pathlib import Path
    f = tmp_path / "a.nc"
    f.write_bytes(b"not a netcdf file")           # truncated download left in the cache

    def fake(granules, local_path):
        if f.exists() and f.read_bytes() != b"good":
            return [str(f)]                        # earthaccess "skips" the existing corrupt file
        f.write_bytes(b"good")
        return [str(f)]

    monkeypatch.setattr(earthdata.earthaccess, "download", fake)
    monkeypatch.setattr(earthdata, "_is_valid_nc", lambda p: Path(p).read_bytes() == b"good")
    out = earthdata._download_all(["g1"], tmp_path, "x", lambda *a: None)
    assert out == [f] and f.read_bytes() == b"good"
