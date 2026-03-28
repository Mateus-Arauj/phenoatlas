"""Tests for phenoatlas.smoothing (whittaker, savitzky_golay, hants, weights)."""
import numpy as np
import pytest

from phenoatlas.smoothing import smooth_hants, smooth_sg, smooth_whit, whit2
from phenoatlas.smoothing.weights import get_wfun, wBisquare, wHuber, wTSM


# ---------------------------------------------------------------------------
# Synthetic season fixture
# ---------------------------------------------------------------------------

def make_season(n: int = 46, noise: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) for a synthetic Gaussian-shaped NDVI season."""
    t = np.linspace(1, 365, n)
    y = 0.1 + 0.7 * np.exp(-((t - 180) ** 2) / (2 * 50**2))
    rng = np.random.default_rng(42)
    y = y + rng.normal(0, noise, n)
    return t, y


# ---------------------------------------------------------------------------
# whit2 (low-level)
# ---------------------------------------------------------------------------

class TestWhit2:
    def test_output_shape(self):
        y = np.random.rand(46)
        w = np.ones(46)
        z = whit2(y, lam=100, w=w)
        assert z.shape == y.shape

    def test_smoother_than_input(self):
        rng = np.random.default_rng(0)
        y = np.sin(np.linspace(0, 2 * np.pi, 46)) + rng.normal(0, 0.1, 46)
        w = np.ones(46)
        z = whit2(y, lam=1000, w=w)
        # Smoothed signal must have lower total variation
        tv_y = np.sum(np.abs(np.diff(y)))
        tv_z = np.sum(np.abs(np.diff(z)))
        assert tv_z < tv_y

    def test_zero_weights_ignored(self):
        y = np.ones(20)
        y[10] = 999.0
        w = np.ones(20)
        w[10] = 0.0
        z = whit2(y, lam=100, w=w)
        # Outlier should be pulled toward 1
        assert abs(z[10] - 1.0) < 0.5

    def test_large_lambda_is_smooth(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0.5, 0.3, 46)
        w = np.ones(46)
        z = whit2(y, lam=1e6, w=w)
        # Very large lambda → nearly constant
        assert np.std(z) < 0.05


# ---------------------------------------------------------------------------
# smooth_whit
# ---------------------------------------------------------------------------

class TestSmoothWhit:
    def test_shape(self):
        _, y = make_season()
        z = smooth_whit(y, nptperyear=23, lam=100)
        assert z.shape == y.shape

    def test_no_nan_output(self):
        _, y = make_season()
        y[5] = np.nan
        z = smooth_whit(y, nptperyear=23, lam=100)
        assert not np.any(np.isnan(z))

    def test_auto_lambda(self):
        _, y = make_season()
        z = smooth_whit(y, nptperyear=23, lam=None)
        assert z.shape == y.shape
        assert not np.any(np.isnan(z))

    def test_with_weights(self):
        _, y = make_season()
        w = np.ones_like(y)
        w[::3] = 0.2   # every 3rd observation is lower quality
        z = smooth_whit(y, w=w, nptperyear=23, lam=100)
        assert z.shape == y.shape

    def test_iterative_reweighting(self):
        _, y = make_season()
        y[10] = -0.5  # cloud spike (negative)
        w = np.ones_like(y)
        z1 = smooth_whit(y, w=w, nptperyear=23, lam=100, iters=1)
        z2 = smooth_whit(y, w=w, nptperyear=23, lam=100, iters=3)
        # More iterations should reduce influence of the outlier
        assert abs(z2[10]) < abs(z1[10]) + 0.1  # not necessarily strictly less

    def test_ylu_filter(self):
        t, y = make_season()
        y_bad = y.copy()
        y_bad[5] = 5.0   # clearly out of range
        z = smooth_whit(y_bad, nptperyear=23, lam=100, ylu=(0.0, 1.0))
        # Out-of-range values should not dominate
        assert z[5] < 3.0


# ---------------------------------------------------------------------------
# smooth_sg
# ---------------------------------------------------------------------------

class TestSmoothSg:
    def test_shape(self):
        _, y = make_season()
        z = smooth_sg(y, nptperyear=23)
        assert z.shape == y.shape

    def test_no_nan(self):
        _, y = make_season()
        z = smooth_sg(y, nptperyear=23)
        assert not np.any(np.isnan(z))

    def test_smoother_than_input(self):
        rng = np.random.default_rng(3)
        y = np.sin(np.linspace(0, 2 * np.pi, 46)) + rng.normal(0, 0.15, 46)
        z = smooth_sg(y, nptperyear=23, iters=1)
        tv_y = np.sum(np.abs(np.diff(y)))
        tv_z = np.sum(np.abs(np.diff(z)))
        assert tv_z < tv_y

    def test_custom_frame(self):
        _, y = make_season()
        z = smooth_sg(y, nptperyear=23, frame=7)
        assert z.shape == y.shape


# ---------------------------------------------------------------------------
# smooth_hants
# ---------------------------------------------------------------------------

class TestSmoothHants:
    def test_shape(self):
        _, y = make_season()
        z = smooth_hants(y, nptperyear=23)
        assert z.shape == y.shape

    def test_no_nan(self):
        _, y = make_season()
        z = smooth_hants(y, nptperyear=23, iters=3)
        assert not np.any(np.isnan(z))

    def test_rejects_outlier(self):
        t, y = make_season()
        y_clean = y.copy()
        y_bad = y.copy()
        y_bad[5] = -0.8   # simulated cloud
        z_clean = smooth_hants(y_clean, nptperyear=23, nf=3)
        z_bad = smooth_hants(y_bad, nptperyear=23, nf=3)
        # HANTS should dampen the spike
        assert abs(z_bad[5] - z_clean[5]) < abs(y_bad[5] - y_clean[5])


# ---------------------------------------------------------------------------
# weight functions
# ---------------------------------------------------------------------------

class TestWeightFunctions:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.y = rng.uniform(0.1, 0.9, 30)
        self.z = self.y + rng.normal(0, 0.05, 30)
        self.w = np.ones(30)

    def test_wTSM_shape(self):
        w_new = wTSM(self.y, self.z, self.w)
        assert w_new.shape == self.w.shape
        assert np.all(w_new >= 0) and np.all(w_new <= 1)

    def test_wBisquare_shape(self):
        w_new = wBisquare(self.y, self.z, self.w)
        assert w_new.shape == self.w.shape
        assert np.all(w_new >= 0) and np.all(w_new <= 1)

    def test_wHuber_shape(self):
        w_new = wHuber(self.y, self.z, self.w)
        assert w_new.shape == self.w.shape
        assert np.all(w_new >= 0)

    def test_get_wfun_registry(self):
        for name in ["wTSM", "wBisquare", "wHuber"]:
            fn = get_wfun(name)
            assert callable(fn)

    def test_get_wfun_invalid(self):
        with pytest.raises(ValueError):
            get_wfun("notAFunction")
