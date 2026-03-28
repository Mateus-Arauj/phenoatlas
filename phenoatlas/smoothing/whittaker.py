"""
Weighted Whittaker smoother for irregularly-sampled time series.

Port of smooth_wWHIT() from phenofit (R).

References
----------
Eilers, P.H.C. (2003). A perfect smoother. Analytical Chemistry, 75(14), 3631-3636.
Eilers, P.H.C. & Marx, B.D. (2004). Splines, knots, and penalties. Wiley Interdisciplinary Reviews.
Frasso, G. & Eilers, P.H.C. (2015). L- and V-curves for optimal smoothing. Statistical Modelling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def whit2(y: np.ndarray, lam: float, w: np.ndarray) -> np.ndarray:
    """
    Core Whittaker smoother with second-order finite difference penalty.

    Parameters
    ----------
    y : array-like of shape (n,)
        Input signal (may contain NaN for missing values).
    lam : float
        Smoothing parameter lambda. Larger values = smoother result.
    w : array-like of shape (n,)
        Weights for each observation (0 = ignored, 1 = fully trusted).

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed signal.

    Notes
    -----
    Minimizes: sum(w * (y - z)^2) + lam * sum(diff(z, 2)^2)
    Port of whit2() from phenofit (R), originally calling C routine "smooth2".
    """
    # TODO: implement
    raise NotImplementedError


def smooth_whit(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    lam: float | None = None,
    nptperyear: int = 23,
    ylu: tuple[float, float] | None = None,
    iters: int = 2,
    wfun: str = "wTSM",
) -> np.ndarray:
    """
    Weighted Whittaker smoother with iterative reweighting.

    Port of smooth_wWHIT() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    w : array-like of shape (n,) or None
        Initial weights. If None, all weights are set to 1.
    lam : float or None
        Smoothing parameter. If None, automatically selected via V-curve method.
    nptperyear : int
        Number of observations per year (e.g. 23 for MODIS 16-day, 46 for 8-day).
    ylu : (float, float) or None
        Lower and upper bounds for valid vegetation index values.
    iters : int
        Number of iterative reweighting steps.
    wfun : str
        Weight-updating function name. One of: 'wTSM', 'wBisquare', 'wHuber'.

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError
