import numpy as np
import pytest

from evaluation import derived as Dv
from evaluation import uncertainty as U

Z = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)


def _col(f, h=2, w=2):
    return np.broadcast_to(f(Z)[:, None, None], (len(Z), h, w)).copy()


def test_d20_linear_profile():
    t = _col(lambda z: 30.0 - 0.1 * z)                       # 30 C at surface, 20 C at 100 m
    assert np.allclose(Dv.isotherm_depth(t, Z), 100.0, atol=1e-6)


def test_d20_edge_cases():
    cold = _col(lambda z: 18.0 - 0.001 * z)                   # surface already below 20 C
    assert np.allclose(Dv.isotherm_depth(cold, Z), 0.0)
    warm = _col(lambda z: 30.0 - 0.001 * z)                   # never cools to 20 C
    assert np.isnan(Dv.isotherm_depth(warm, Z)).all()
    t = _col(lambda z: 30.0 - 0.1 * z); t[:, 0, 0] = np.nan   # invalid column stays NaN
    assert np.isnan(Dv.isotherm_depth(t, Z)[0, 0])


def test_mld_step_profile_and_uniform_column():
    step = _col(lambda z: np.where(z <= 50, 29.0, 29.0 - 0.1 * (z - 50)))    # cools right below 50 m
    mld = Dv.mixed_layer_depth(step, Z)
    assert 50.0 <= mld[0, 0] <= 75.0
    uniform = _col(lambda z: 0 * z + 25.0)
    assert np.allclose(Dv.mixed_layer_depth(uniform, Z), 1000.0)              # never cools -> deepest level


def test_tchp_matches_analytic_integral_and_ignores_inversions():
    t = _col(lambda z: 30.0 - 0.04 * z)                        # 26 C at 100 m; excess is a triangle of area 0.5*4*100
    expected = Dv.RHO_SEAWATER * Dv.CP_SEAWATER * (0.5 * 4.0 * 100.0) * 1e-7
    assert np.allclose(Dv.tchp(t, Z), expected, rtol=1e-6)
    cold_top = _col(lambda z: 25.0 + 0 * z)                    # SST < 26 -> no heat potential
    assert np.allclose(Dv.tchp(cold_top, Z), 0.0)
    # 28 C down to 10 m, 24 C from 20 m, warm 28 C again below 100 m: only the connected surface layer counts.
    inv = _col(lambda z: np.where(z < 20, 28.0, np.where(z < 100, 24.0, 28.0)))
    #   0-5 m: 10, 5-10 m: 10, 10-20 m: crosses 26 C halfway -> 0.5*2*0.5*10 = 5  => 25 K m
    assert np.allclose(Dv.tchp(inv, Z), Dv.RHO_SEAWATER * Dv.CP_SEAWATER * 25.0 * 1e-7, rtol=1e-6)


def test_calibration_perfect_vs_overconfident():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(20, 3, 20, 20))
    sigma = rng.uniform(0.5, 2.0, size=true.shape)
    mean = true + rng.normal(size=true.shape) * sigma          # errors truly ~ N(0, sigma)
    good = U.calibration(mean, sigma, true)
    assert good["coverage_95"] == pytest.approx(0.95, abs=0.01) and good["coverage_90"] == pytest.approx(0.90, abs=0.01)
    assert good["spread_error_spearman"] > 0.2
    over = U.calibration(mean, sigma * 0.3, true)              # claims far less uncertainty than reality
    assert over["coverage_95"] < 0.6
    assert all(b["rmse"] > 0 for b in good["reliability"])


def test_calibration_ignores_nan_cells():
    true = np.full((2, 1, 4, 4), np.nan); true[0, 0, :2] = 1.0
    r = U.calibration(np.zeros_like(true), np.ones_like(true), true)
    assert r["n"] == 8 and r["rmse"] == pytest.approx(1.0)
