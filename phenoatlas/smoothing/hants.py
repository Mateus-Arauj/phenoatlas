"""
Harmonic ANalysis of Time Series (HANTS) for time-series reconstruction.

Port of smooth_wHANTS() from phenofit (R).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def smooth_hants(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    nf: int = 3,
    iters: int = 3,
    fet: float = 0.05,
    delta: float = 0.1,
) -> np.ndarray:
    """
    HANTS smoother for vegetation index time series.

    Port of smooth_wHANTS() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    w : array-like of shape (n,) or None
        Initial weights. If None, all weights are set to 1.
    nptperyear : int
        Number of observations per year.
    nf : int
        Number of frequencies to fit.
    iters : int
        Number of HANTS iterations.
    fet : float
        Fit error tolerance.
    delta : float
        Dampening factor.

    Returns
    -------
    np.ndarray of shape (n,)
        Reconstructed vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError
