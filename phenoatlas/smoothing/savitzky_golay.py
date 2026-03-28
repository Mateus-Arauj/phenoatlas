"""
Weighted Savitzky-Golay filter for time-series smoothing.

Port of smooth_wSG() / rcpp_wSG() from phenofit (R).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def smooth_sg(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    frame: int | None = None,
    degree: int = 4,
    iters: int = 2,
) -> np.ndarray:
    """
    Weighted Savitzky-Golay filter with iterative reweighting.

    Port of smooth_wSG() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    w : array-like of shape (n,) or None
        Initial weights. If None, all weights are set to 1.
    nptperyear : int
        Number of observations per year.
    frame : int or None
        Window size (number of points). If None, derived from nptperyear.
    degree : int
        Polynomial degree for Savitzky-Golay filter.
    iters : int
        Number of iterative reweighting steps.

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError
