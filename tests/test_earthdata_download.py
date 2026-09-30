import pytest

from data.sources import earthdata


def test_download_all_retries_until_complete(monkeypatch, tmp_path):
    calls = []
    def fake(granules, local_path):
        calls.append(1)
        return ["a.nc"] if len(calls) == 1 else ["a.nc", "b.nc"]   # first pass drops one file
    monkeypatch.setattr(earthdata.earthaccess, "download", fake)
    out = earthdata._download_all(["g1", "g2"], tmp_path, "x", lambda *a: None)
    assert len(out) == 2 and len(calls) == 2


def test_download_all_raises_if_still_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(earthdata.earthaccess, "download", lambda g, local_path: ["a.nc"])
    with pytest.raises(RuntimeError, match="1 of 2|only 1 of 2"):
        earthdata._download_all(["g1", "g2"], tmp_path, "x", lambda *a: None, attempts=2)
