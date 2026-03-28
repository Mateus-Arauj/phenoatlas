"""
Weighted Savitzky-Golay filter for time-series smoothing.

Port of smooth_wSG() / rcpp_wSG() from phenofit (R).

References
----------
Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation of data
by simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import savgol_filter

from phenoatlas.smoothing.weights import get_wfun
from phenoatlas.utils.validation import check_input


def _sg_coeffs(window: int, degree: int) -> np.ndarray:
    """
    Compute Savitzky-Golay convolution coefficients for the centre point.

    Parameters
    ----------
    window : int
        Window size (must be odd, >= degree + 1).
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray of shape (window,)
        Convolution coefficients.
    """
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    # Vandermonde matrix
    V = np.vander(x, degree + 1, increasing=True)
    # Least-squares coefficients
    coeffs = np.linalg.pinv(V)[0]  # row 0 = smoothing coefficients
    return coeffs


def smooth_sg(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    frame: int | None = None,
    degree: int = 4,
    iters: int = 2,
    wfun: str = "wTSM",
    ylu: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Weighted Savitzky-Golay filter with iterative reweighting.

    Port of smooth_wSG() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    w : array-like of shape (n,) or None
        Initial weights in [0, 1]. If None, 1 for finite values, 0 for NaN.
    nptperyear : int
        Number of observations per year. Used to derive default frame size.
    frame : int or None
        Window size (must be odd). If None, defaults to
        ``max(degree + 1, 2 * (nptperyear // 4) + 1)``.
    degree : int
        Polynomial degree (must be < frame).
    iters : int
        Number of iterative reweighting steps.
    wfun : str
        Weight-update function: 'wTSM', 'wBisquare', or 'wHuber'.
    ylu : (float, float) or None
        Valid range; out-of-range observations receive w = 0.

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed vegetation index values.
    """
    y, t, w0 = check_input(y, w=w, ylu=ylu, nptperyear=nptperyear)

    # Derive default window size
    if frame is None:
        frame = max(degree + 1, 2 * (nptperyear // 4) + 1)
    if frame % 2 == 0:
        frame += 1  # must be odd
    frame = max(frame, degree + 1)

    update_w = get_wfun(wfun)
    w_cur = w0.copy()

    z = y.copy()
    for _ in range(max(1, iters)):
        # Weighted least-squares SG: apply scipy's savgol with per-point weights
        # We implement a simple weighted version via explicit polynomial fitting
        z = _weighted_sg(y, w_cur, frame, degree)
        if iters > 1:
            w_cur = update_w(y, z, w0)

    return z


def _weighted_sg(
    y: np.ndarray,
    w: np.ndarray,
    frame: int,
    degree: int,
) -> np.ndarray:
    """
    Weighted Savitzky-Golay using per-window weighted polynomial regression.

    For each output point i, fit a degree-d polynomial to the window
    [i-half, i+half] using the observation weights, then evaluate at i.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
    w : np.ndarray of shape (n,)
    frame : int (odd)
    degree : int

    Returns
    -------
    np.ndarray of shape (n,)
    """
    n = len(y)
    half = frame // 2
    z = np.empty(n)
    x_centre = np.arange(-half, half + 1, dtype=float)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        # Local indices relative to centre
        x_local = np.arange(i0 - i, i1 - i, dtype=float)
        y_local = y[i0:i1]
        w_local = w[i0:i1]

        if w_local.sum() < degree + 1:
            # Fall back to weighted mean when not enough data
            denom = w_local.sum()
            z[i] = np.dot(w_local, y_local) / denom if denom > 0 else y[i]
            continue

        # Weighted Vandermonde
        V = np.vander(x_local, degree + 1, increasing=True)
        W = np.diag(w_local)
        try:
            VtW = V.T @ W
            coeffs = np.linalg.solve(VtW @ V, VtW @ y_local)
            z[i] = coeffs[0]  # value at x=0 (centre)
        except np.linalg.LinAlgError:
            z[i] = y[i]

    return z
