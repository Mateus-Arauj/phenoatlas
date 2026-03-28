"""Tests for season detection module."""
import pytest
import numpy as np


@pytest.mark.skip(reason="Not yet implemented")
def test_find_peaks_detects_single_peak():
    """find_peaks detects a clear unimodal peak."""
    from phenoatlas.season import find_peaks
    t = np.linspace(0, 365, 46)
    y = np.exp(-((t - 180) ** 2) / (2 * 40**2))
    peaks, troughs = find_peaks(y)
    assert len(peaks) == 1
