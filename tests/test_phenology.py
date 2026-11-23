"""Tests for phenoatlas.phenology (pheno_trs, pheno_der, pheno_gu, pheno_zhang, get_pheno)."""
import numpy as np
import pandas as pd
import pytest

from phenoatlas.fitting import curvefit
from phenoatlas.fitting.curves import doubleLog_Beck
from phenoatlas.phenology import get_pheno, pheno_der, pheno_gu, pheno_trs, pheno_zhang


# ---------------------------------------------------------------------------
# Shared fixture: synthetic Gaussian-shaped season
# ---------------------------------------------------------------------------

T_DENSE = np.linspace(1, 365, 365)
Y_SYNTH = 0.1 + 0.75 * np.exp(-((T_DENSE - 180) ** 2) / (2 * 50**2))

T_OBS = np.linspace(1, 365, 46)
P_BECK = np.array([0.1, 0.85, 90.0, 0.12, 270.0, 0.10])
Y_OBS = doubleLog_Beck(T_OBS, P_BECK)


# ---------------------------------------------------------------------------
# pheno_trs
# ---------------------------------------------------------------------------

class TestPhenoTrs:
    def test_keys_present(self):
        result = pheno_trs(T_DENSE, Y_SYNTH)
        expected_keys = [
            "TRS2.SOS", "TRS2.EOS",
            "TRS5.SOS", "TRS5.EOS",
            "TRS6.SOS", "TRS6.EOS",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_sos_before_peak(self):
        result = pheno_trs(T_DENSE, Y_SYNTH)
        assert result["TRS5.SOS"] < 180

    def test_eos_after_peak(self):
        result = pheno_trs(T_DENSE, Y_SYNTH)
        assert result["TRS5.EOS"] > 180

    def test_sos_before_eos(self):
        result = pheno_trs(T_DENSE, Y_SYNTH)
        assert result["TRS5.SOS"] < result["TRS5.EOS"]

    def test_higher_threshold_narrower(self):
        r = pheno_trs(T_DENSE, Y_SYNTH)
        # TRS6 (60%) should be narrower (later SOS, earlier EOS) than TRS2 (20%)
        assert r["TRS6.SOS"] >= r["TRS2.SOS"]
        assert r["TRS6.EOS"] <= r["TRS2.EOS"]

    def test_constant_signal_returns_nan(self):
        y_flat = np.ones(100) * 0.5
        t = np.linspace(1, 365, 100)
        result = pheno_trs(t, y_flat)
        # All NaN when amplitude is zero
        assert np.isnan(result["TRS5.SOS"]) or np.isnan(result["TRS5.EOS"])

    def test_custom_thresholds(self):
        result = pheno_trs(T_DENSE, Y_SYNTH, thresholds=(0.1, 0.9))
        # Should have keys for 0.1 and 0.9 thresholds
        assert len(result) >= 4


# ---------------------------------------------------------------------------
# pheno_der
# ---------------------------------------------------------------------------

class TestPhenoDer:
    def test_keys_present(self):
        result = pheno_der(T_DENSE, Y_SYNTH)
        assert "DER.SOS" in result
        assert "DER.POS" in result
        assert "DER.EOS" in result

    def test_sos_before_pos(self):
        result = pheno_der(T_DENSE, Y_SYNTH)
        assert result["DER.SOS"] < result["DER.POS"]

    def test_pos_before_eos(self):
        result = pheno_der(T_DENSE, Y_SYNTH)
        assert result["DER.POS"] < result["DER.EOS"]

    def test_pos_near_peak(self):
        result = pheno_der(T_DENSE, Y_SYNTH)
        # POS should be near the actual Gaussian peak at t=180
        assert abs(result["DER.POS"] - 180) < 20

    def test_float_output(self):
        result = pheno_der(T_DENSE, Y_SYNTH)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"


# ---------------------------------------------------------------------------
# pheno_gu
# ---------------------------------------------------------------------------

class TestPhenoGu:
    def test_keys_present(self):
        result = pheno_gu(T_DENSE, Y_SYNTH)
        assert "UD" in result
        assert "SD" in result
        assert "DD" in result
        assert "RD" in result

    def test_ud_before_sd(self):
        result = pheno_gu(T_DENSE, Y_SYNTH)
        assert result["UD"] <= result["SD"]

    def test_dd_after_sd(self):
        result = pheno_gu(T_DENSE, Y_SYNTH)
        assert result["DD"] >= result["SD"]

    def test_float_output(self):
        result = pheno_gu(T_DENSE, Y_SYNTH)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# pheno_zhang
# ---------------------------------------------------------------------------

class TestPhenoZhang:
    def test_keys_present(self):
        result = pheno_zhang(T_DENSE, Y_SYNTH)
        assert "GreenUp" in result
        assert "Maturity" in result
        assert "Senescence" in result
        assert "Dormancy" in result

    def test_order(self):
        result = pheno_zhang(T_DENSE, Y_SYNTH)
        finite = [
            result[name]
            for name in ("GreenUp", "Maturity", "Senescence", "Dormancy")
            if np.isfinite(result[name])
        ]
        assert finite == sorted(finite)

    def test_float_output(self):
        result = pheno_zhang(T_DENSE, Y_SYNTH)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# get_pheno (dispatcher)
# ---------------------------------------------------------------------------

class TestGetPheno:
    def test_returns_dataframe(self):
        fit = curvefit(Y_OBS, T_OBS, methods=["Beck"])
        df = get_pheno(fit)
        assert isinstance(df, pd.DataFrame)

    def test_columns_include_trs5(self):
        fit = curvefit(Y_OBS, T_OBS, methods=["Beck"])
        df = get_pheno(fit, methods=["TRS"])
        assert "TRS5.SOS" in df.columns
        assert "TRS5.EOS" in df.columns

    def test_columns_include_der(self):
        fit = curvefit(Y_OBS, T_OBS, methods=["Beck"])
        df = get_pheno(fit, methods=["DER"])
        assert "DER.POS" in df.columns

    def test_fit_method_column(self):
        fit = curvefit(Y_OBS, T_OBS, methods=["Beck", "Elmore"])
        df = get_pheno(fit, methods=["TRS"])
        assert "fit_method" in df.columns
        assert set(df["fit_method"].unique()) >= {"Beck", "Elmore"}

    def test_batch_input(self):
        fits = {
            "2020": curvefit(Y_OBS, T_OBS, methods=["Beck"]),
            "2021": curvefit(Y_OBS, T_OBS, methods=["Beck"]),
        }
        df = get_pheno(fits, methods=["TRS"])
        assert "season" in df.columns
        assert len(df) == 2  # one row per (season × fit_method)

    def test_all_methods(self):
        fit = curvefit(Y_OBS, T_OBS, methods=["Beck"])
        df = get_pheno(fit)
        # All four methods should produce columns
        assert any("TRS" in c for c in df.columns)
        assert any("DER" in c for c in df.columns)
        assert any(c in df.columns for c in ["UD", "SD"])
        assert any(c in df.columns for c in ["GreenUp", "Maturity"])

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            get_pheno("not_a_fitting_result")

    def test_dense_tout_recovers_steep_winter_crop_eos(self):
        t = np.linspace(1, 365, 46)
        p_winter = np.array([0.08, 0.80, 65.0, 0.18, 175.0, 0.20])
        y = doubleLog_Beck(t, p_winter)
        tout_daily = np.arange(1, 366, dtype=float)
        fit = curvefit(y, t, tout=tout_daily, methods=["Beck"])
        df = get_pheno(fit, methods=["TRS", "DER"])
        assert abs(df.iloc[0]["TRS5.EOS"] - 175.0) <= 5.0
        assert abs(df.iloc[0]["DER.EOS"] - 175.0) <= 5.0
