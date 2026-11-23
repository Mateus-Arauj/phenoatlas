"""Tests for phenoatlas.utils (dates, signal, stats, validation)."""
import numpy as np
import pandas as pd
import pytest

from phenoatlas.utils.dates import date2doy, doy2date, make_date, year_of
from phenoatlas.utils.signal import D1, D2, curvature, movmean
from phenoatlas.utils.stats import cv_coef, gof, r2_sign
from phenoatlas.utils.validation import check_input


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

class TestDate2Doy:
    def test_jan_first(self):
        result = date2doy(pd.to_datetime(["2020-01-01"]))
        assert result[0] == 1

    def test_july_4(self):
        # 2020 is a leap year: Jan(31) Feb(29) Mar(31) Apr(30) May(31) Jun(30) + 4 = 186
        result = date2doy(pd.to_datetime(["2020-07-04"]))
        assert result[0] == 186

    def test_dec_31_non_leap(self):
        result = date2doy(pd.to_datetime(["2019-12-31"]))
        assert result[0] == 365

    def test_dec_31_leap(self):
        result = date2doy(pd.to_datetime(["2020-12-31"]))
        assert result[0] == 366

    def test_array_output(self):
        dates = pd.to_datetime(["2021-01-01", "2021-06-15"])
        result = date2doy(dates)
        assert result.dtype == np.int32
        assert len(result) == 2


class TestDoy2Date:
    def test_doy_1(self):
        result = doy2date([1], 2020)
        assert str(result[0]) == "2020-01-01"

    def test_doy_186_leap(self):
        result = doy2date([186], 2020)
        assert str(result[0]) == "2020-07-04"

    def test_broadcast_year(self):
        result = doy2date([1, 186], 2020)
        assert len(result) == 2

    def test_per_element_year(self):
        result = doy2date([1, 186], [2020, 2021])
        assert str(result[0]) == "2020-01-01"
        assert str(result[1]) == "2021-07-05"

    def test_roundtrip(self):
        dates = pd.to_datetime(["2021-03-15", "2021-09-22"])
        doy = date2doy(dates)
        year = year_of(dates)
        recovered = doy2date(doy, year)
        assert str(recovered[0]) == "2021-03-15"
        assert str(recovered[1]) == "2021-09-22"


class TestMakeDate:
    def test_same_as_doy2date(self):
        a = make_date(2020, [1, 100])
        b = doy2date([1, 100], 2020)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------

class TestD1:
    def test_constant(self):
        y = np.ones(10)
        dy = D1(y)
        np.testing.assert_allclose(dy, 0.0, atol=1e-12)

    def test_linear(self):
        t = np.arange(10, dtype=float)
        y = 3 * t + 5
        dy = D1(y, t)
        np.testing.assert_allclose(dy, 3.0, atol=1e-10)

    def test_shape_preserved(self):
        y = np.random.rand(20)
        assert D1(y).shape == y.shape

    def test_quadratic(self):
        # y = t^2 → dy/dt = 2t (central diff should be exact)
        t = np.arange(5, dtype=float)
        y = t**2
        dy = D1(y, t)
        # interior: central diff for t^2 gives 2t exactly
        np.testing.assert_allclose(dy[1:-1], 2 * t[1:-1], atol=1e-10)


class TestD2:
    def test_constant(self):
        np.testing.assert_allclose(D2(np.ones(10)), 0.0, atol=1e-10)

    def test_linear(self):
        t = np.arange(10, dtype=float)
        y = 3 * t
        np.testing.assert_allclose(D2(y, t), 0.0, atol=1e-10)

    def test_shape(self):
        y = np.random.rand(15)
        assert D2(y).shape == y.shape


class TestCurvature:
    def test_linear_curvature_zero(self):
        t = np.linspace(0, 10, 50)
        y = 2 * t + 1
        kappa = curvature(y, t)
        # Linear → κ ≈ 0
        np.testing.assert_allclose(kappa[2:-2], 0.0, atol=1e-8)

    def test_shape(self):
        y = np.sin(np.linspace(0, np.pi, 30))
        assert curvature(y).shape == y.shape


class TestMovmean:
    def test_constant(self):
        y = np.ones(10) * 3.0
        result = movmean(y, k=3)
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_shape(self):
        y = np.random.rand(20)
        assert movmean(y, k=5).shape == y.shape

    def test_weighted(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        result = movmean(y, k=3, w=w)
        # Interior points match unweighted moving average
        np.testing.assert_allclose(result[1:-1], [2.0, 3.0, 4.0], atol=1e-10)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestGof:
    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = gof(y, y)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-10)
        assert metrics["bias"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["n"] == 3

    def test_constant_obs(self):
        # When obs is constant, ss_tot = 0 → R2 = NaN
        y_obs = np.ones(5)
        y_pred = np.ones(5) * 1.1
        metrics = gof(y_obs, y_pred)
        assert np.isnan(metrics["R2"])

    def test_zero_weights_excluded(self):
        y_obs = np.array([1.0, 2.0, 999.0])
        y_pred = np.array([1.1, 1.9, 999.0])
        w = np.array([1.0, 1.0, 0.0])  # exclude last
        metrics = gof(y_obs, y_pred, w)
        assert metrics["n"] == 2

    def test_all_nan_weights_returns_nan(self):
        y_obs = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        w = np.zeros(2)
        metrics = gof(y_obs, y_pred, w)
        assert metrics["n"] == 0
        assert np.isnan(metrics["RMSE"])

    def test_rmse_positive(self):
        y_obs = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.9, 2.2])
        metrics = gof(y_obs, y_pred)
        assert metrics["RMSE"] > 0

    def test_weighted_r2_respects_weights(self):
        y_obs = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        y_pred = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        w_unweighted = np.ones(5)
        w_downweighted = np.array([1.0, 1.0, 1.0, 1.0, 1e-4])
        r2_unw = gof(y_obs, y_pred, w_unweighted)["R2"]
        r2_w = gof(y_obs, y_pred, w_downweighted)["R2"]
        assert r2_w > 0.99
        assert r2_w > r2_unw


class TestR2Sign:
    def test_good_fit(self):
        t = np.linspace(0, 1, 20)
        y = t**2
        r2 = r2_sign(y, y + 0.01 * np.random.randn(20))
        assert r2 > 0.9

    def test_bad_fit_is_negative(self):
        y_obs = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 10.0, 10.0, 10.0])
        assert r2_sign(y_obs, y_pred) < 0.0


class TestCvCoef:
    def test_perfect_fit_zero(self):
        y = np.linspace(0.2, 0.8, 10)
        assert cv_coef(y, y) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TestCheckInput:
    def test_default_weights(self):
        y = np.array([0.1, np.nan, 0.3])
        y_out, t_out, w_out = check_input(y)
        assert w_out[1] == 0.0   # NaN → weight 0
        assert y_out[1] == 0.0   # NaN replaced by 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_input([1, 2, 3], t=[1, 2])

    def test_ylu_filters(self):
        y = np.array([-0.5, 0.3, 1.5])
        _, _, w = check_input(y, ylu=(0.0, 1.0))
        assert w[0] == 0.0  # -0.5 < lower
        assert w[2] == 0.0  # 1.5 > upper
        assert w[1] == 1.0

    def test_weights_clipped(self):
        y = np.ones(5)
        w = np.array([2.0, -0.1, 0.5, 0.0, 1.0])
        _, _, w_out = check_input(y, w=w)
        assert w_out.max() <= 1.0
        assert w_out.min() >= 0.0
