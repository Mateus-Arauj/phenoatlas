"""Tests for curve fitting module."""
import pytest
import numpy as np
from phenoatlas.fitting import curvefit, doubleLog_Beck


@pytest.mark.skip(reason="Not yet implemented")
def test_doubleLog_Beck_shape():
    """doubleLog_Beck returns array matching input time shape."""
    t = np.linspace(1, 365, 23)
    p = np.array([0.1, 0.9, 100.0, 0.1, 280.0, 0.1])
    result = doubleLog_Beck(t, p)
    assert result.shape == t.shape


@pytest.mark.skip(reason="Not yet implemented")
def test_curvefit_returns_result():
    """curvefit returns FittingResult for a synthetic season."""
    t = np.linspace(1, 365, 23)
    p_true = np.array([0.1, 0.9, 100.0, 0.1, 280.0, 0.1])
    from phenoatlas.fitting.curves import doubleLog_Beck
    y = doubleLog_Beck(t, p_true) + np.random.normal(0, 0.02, 23)
    result = curvefit(y, t, methods=["Beck"])
    assert "Beck" in result.models
