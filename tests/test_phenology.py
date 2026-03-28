"""Tests for phenology extraction module."""
import pytest
import numpy as np


@pytest.mark.skip(reason="Not yet implemented")
def test_pheno_trs_basic():
    """pheno_trs extracts SOS before POS before EOS."""
    from phenoatlas.phenology import pheno_trs
    t = np.linspace(1, 365, 365)
    y = np.clip(np.exp(-((t - 180) ** 2) / (2 * 40**2)), 0, 1)
    result = pheno_trs(t, y)
    assert result["TRS5.SOS"] < 180 < result["TRS5.EOS"]
