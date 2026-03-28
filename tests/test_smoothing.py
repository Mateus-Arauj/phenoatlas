"""Tests for smoothing module."""
import pytest
import numpy as np
from phenoatlas.smoothing import smooth_whit, smooth_sg, smooth_hants


@pytest.mark.skip(reason="Not yet implemented")
def test_smooth_whit_basic():
    """smooth_whit returns array of same shape as input."""
    y = np.random.rand(92)
    result = smooth_whit(y, nptperyear=23)
    assert result.shape == y.shape


@pytest.mark.skip(reason="Not yet implemented")
def test_smooth_whit_handles_nan():
    """smooth_whit handles NaN values via weights."""
    y = np.random.rand(92)
    y[10] = np.nan
    w = (~np.isnan(y)).astype(float)
    y[10] = 0.0
    result = smooth_whit(y, w=w, nptperyear=23)
    assert not np.any(np.isnan(result))
