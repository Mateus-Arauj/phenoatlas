"""Tests for phenoatlas.fitting (curves, init_params, optimizer, curvefit)."""
import numpy as np
import pandas as pd
import pytest

from phenoatlas.fitting import (
    FittingResult,
    curvefit,
    curvefits,
    doubleLog_AG,
    doubleLog_Beck,
    doubleLog_Elmore,
    doubleLog_Gu,
    doubleLog_Klos,
    doubleLog_Zhang,
)
from phenoatlas.fitting.init_params import (
    get_init,
    init_AG,
    init_Beck,
    init_Elmore,
    init_Gu,
    init_Klos,
    init_Zhang,
)
from phenoatlas.fitting.optimizer import i_optim, make_residual_fun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T = np.linspace(1, 365, 46)
P_BECK = np.array([0.1, 0.85, 90.0, 0.12, 270.0, 0.10])


def synth_season(curve_fn, p, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(0)
    return curve_fn(T, p) + rng.normal(0, noise, len(T))


# ---------------------------------------------------------------------------
# Curve functions — shape & range
# ---------------------------------------------------------------------------

class TestCurveFunctions:
    def _check(self, fn, p):
        y = fn(T, np.array(p))
        assert y.shape == T.shape, f"{fn.__name__} wrong shape"
        assert np.all(np.isfinite(y)), f"{fn.__name__} produced NaN/Inf"

    def test_Beck(self):
        self._check(doubleLog_Beck, [0.1, 0.85, 90.0, 0.12, 270.0, 0.10])

    def test_Elmore(self):
        self._check(doubleLog_Elmore, [0.1, 0.85, 90.0, 0.12, 270.0, 0.10, 0.0])

    def test_Zhang(self):
        # p = [t0, mn, mx, sos, rsp, eos, rau]  — t0 first (matches phenofit R)
        self._check(doubleLog_Zhang, [180.0, 0.1, 0.85, 90.0, 0.12, 270.0, 0.10])

    def test_AG(self):
        # p = [t0, mn, mx, rsp, a3, rau, a5]  — rate params (matches phenofit R)
        self._check(doubleLog_AG, [180.0, 0.1, 0.75, 0.04, 2.0, 0.04, 2.0])

    def test_Gu(self):
        # p = [y0, a1, a2, sos, rsp, eos, rau, c1, c2]  — 9 params (matches phenofit R)
        self._check(doubleLog_Gu, [0.1, 0.75, 0.75, 90.0, 0.12, 270.0, 0.10, 1.0, 1.0])

    def test_Klos(self):
        self._check(
            doubleLog_Klos,
            [
                0.0, 0.0, 0.1, 0.0, 0.2, 0.05, 0.04,
                100.0, 260.0, 0.5, 0.5, 2.0, 2.0,
            ],
        )

    def test_Beck_monotone_at_extremes(self):
        y = doubleLog_Beck(T, P_BECK)
        assert y[:3].mean() < 0.3
        assert y.max() > 0.5

    def test_Beck_extreme_params_no_overflow(self):
        p_extreme = np.array([0.0, 1.0, 50.0, 10.0, 300.0, 10.0])
        y = doubleLog_Beck(T, p_extreme)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Parameter initializers
# ---------------------------------------------------------------------------

class TestInitParams:
    def _run(self, init_fn, y, t, w, n_params):
        init = init_fn(y, t, w)
        assert "p0" in init and "lower" in init and "upper" in init
        assert len(init["p0"]) == n_params
        assert len(init["lower"]) == n_params
        assert len(init["upper"]) == n_params
        assert np.all(init["lower"] <= init["p0"])
        assert np.all(init["p0"] <= init["upper"])

    def setup_method(self):
        self.y = synth_season(doubleLog_Beck, P_BECK)
        self.t = T
        self.w = np.ones_like(T)

    def test_Beck(self):     self._run(init_Beck, self.y, self.t, self.w, 6)
    def test_Elmore(self):   self._run(init_Elmore, self.y, self.t, self.w, 7)
    def test_Zhang(self):    self._run(init_Zhang, self.y, self.t, self.w, 7)
    def test_AG(self):       self._run(init_AG, self.y, self.t, self.w, 7)
    def test_Gu(self):       self._run(init_Gu, self.y, self.t, self.w, 9)
    def test_Klos(self):     self._run(init_Klos, self.y, self.t, self.w, 13)

    def test_get_init_registry(self):
        for m in ["Beck", "Elmore", "Zhang", "AG", "Gu", "Klos"]:
            assert callable(get_init(m))

    def test_get_init_invalid(self):
        with pytest.raises(ValueError):
            get_init("NonExistentMethod")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_recovers_known_params(self):
        y = synth_season(doubleLog_Beck, P_BECK, noise=0.0)
        w = np.ones_like(T)
        init = init_Beck(y, T, w)
        obj = make_residual_fun(doubleLog_Beck, y, T, w)
        result = i_optim(obj, init["p0"], init["lower"], init["upper"])
        assert result.fun < 0.1

    def test_result_within_bounds(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        w = np.ones_like(T)
        init = init_Beck(y, T, w)
        obj = make_residual_fun(doubleLog_Beck, y, T, w)
        result = i_optim(obj, init["p0"], init["lower"], init["upper"])
        assert np.all(result.x >= init["lower"])
        assert np.all(result.x <= init["upper"])


# ---------------------------------------------------------------------------
# curvefit
# ---------------------------------------------------------------------------

class TestCurvefit:
    def test_returns_fitting_result(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        assert isinstance(fit, FittingResult)

    def test_all_default_methods_run(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T)
        assert len(fit.methods) >= 1

    def test_get_param_shape(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        p = fit.get_param("Beck")
        assert p.shape == (6,)

    def test_get_fitted_shape_matches_tout(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        assert fit.get_fitted("Beck").shape == fit.tout.shape

    def test_gof_keys(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        g = fit.get_gof("Beck")
        assert all(k in g for k in ["RMSE", "R2", "NSE", "bias", "n"])

    def test_good_fit_low_rmse(self):
        y = synth_season(doubleLog_Beck, P_BECK, noise=0.01)
        fit = curvefit(y, T, methods=["Beck"])
        assert fit.get_gof("Beck")["RMSE"] < 0.05

    def test_custom_tout(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        tout = np.linspace(1, 365, 200)
        fit = curvefit(y, T, tout=tout, methods=["Beck"])
        assert fit.get_fitted("Beck").shape == (200,)

    def test_invalid_method_skipped(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck", "NonExistent"])
        assert "Beck" in fit.methods
        assert "NonExistent" not in fit.methods

    def test_to_dataframe(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        df = fit.to_dataframe()
        assert "t" in df.columns
        assert "Beck" in df.columns
        assert len(df) == len(fit.tout)

    def test_repr_contains_method(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        assert "Beck" in repr(fit)

    def test_unknown_method_raises_key_error(self):
        y = synth_season(doubleLog_Beck, P_BECK)
        fit = curvefit(y, T, methods=["Beck"])
        with pytest.raises(KeyError):
            fit.get_param("Elmore")


# ---------------------------------------------------------------------------
# curvefits (batch)
# ---------------------------------------------------------------------------

class TestCurvefits:
    def test_returns_dict_of_fitting_results(self):
        rows = []
        for s in [1, 2]:
            y = synth_season(doubleLog_Beck, P_BECK)
            for ti, yi in zip(T, y):
                rows.append({"season": s, "t": ti, "y": yi, "w": 1.0})
        df = pd.DataFrame(rows)
        results = curvefits(df, methods=["Beck"])
        assert len(results) == 2
        assert all(isinstance(v, FittingResult) for v in results.values())

    def test_no_w_column_uses_default(self):
        df = pd.DataFrame({
            "season": 1,
            "t": T,
            "y": synth_season(doubleLog_Beck, P_BECK),
        })
        results = curvefits(df, methods=["Beck"])
        assert 1 in results
