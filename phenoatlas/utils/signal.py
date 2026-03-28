"""
Signal processing utilities.

Port of D1(), D2(), movmean(), curvature() from phenofit (R).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter1d


def D1(y: ArrayLike, t: ArrayLike | None = None) -> np.ndarray:
    """
    First numerical derivative using central differences (interior) and
    one-sided differences at the boundaries.

    Matches phenofit's D1() behaviour: output has the same length as input.

    Parameters
    ----------
    y : array-like of shape (n,)
        Signal values.
    t : array-like of shape (n,) or None
        Time points. If None, unit spacing (dt = 1) is assumed.

    Returns
    -------
    np.ndarray of shape (n,)
        dy/dt estimated at each point.

    Examples
    --------
    >>> D1([0, 1, 4, 9, 16])
    array([1., 2., 4., 6., 7.])
    """
    y = np.asarray(y, dtype=float)
    if t is None:
        t = np.arange(len(y), dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    dy = np.empty_like(y)
    # interior: central difference
    dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    # boundaries: one-sided
    dy[0] = (y[1] - y[0]) / (t[1] - t[0])
    dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return dy


def D2(y: ArrayLike, t: ArrayLike | None = None) -> np.ndarray:
    """
    Second numerical derivative (central differences, same length as input).

    Parameters
    ----------
    y : array-like of shape (n,)
    t : array-like of shape (n,) or None
        Time points. If None, unit spacing is assumed.

    Returns
    -------
    np.ndarray of shape (n,)
        d²y/dt² estimated at each point.
    """
    # Apply D1 twice
    return D1(D1(y, t), t)


def curvature(y: ArrayLike, t: ArrayLike | None = None) -> np.ndarray:
    """
    Signed curvature of a curve y(t).

    κ = y'' / (1 + y'²)^(3/2)

    Port of curvature() from phenofit (R). Used by pheno_zhang() to find
    inflection points.

    Parameters
    ----------
    y : array-like of shape (n,)
    t : array-like of shape (n,) or None

    Returns
    -------
    np.ndarray of shape (n,)
        Signed curvature values.
    """
    y = np.asarray(y, dtype=float)
    d1 = D1(y, t)
    d2 = D2(y, t)
    return d2 / (1.0 + d1**2) ** 1.5


def movmean(
    y: ArrayLike,
    k: int,
    *,
    w: ArrayLike | None = None,
) -> np.ndarray:
    """
    Moving average (optionally weighted), matching MATLAB / R movmean behaviour.

    For unweighted input uses scipy's uniform_filter1d (reflect padding).
    For weighted input computes a weighted sliding sum / sliding weight sum.

    Parameters
    ----------
    y : array-like of shape (n,)
        Input signal.
    k : int
        Window size (should be odd for symmetric windows).
    w : array-like of shape (n,) or None
        Per-observation weights. If None, uniform weights are used.

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed signal.

    Examples
    --------
    >>> movmean([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    array([1.33333333, 2.        , 3.        , 4.        , 4.66666667])
    """
    y = np.asarray(y, dtype=float)
    if w is None:
        return uniform_filter1d(y, size=k, mode="reflect")

    w = np.asarray(w, dtype=float)
    yw = y * w
    half = k // 2
    # Pad both arrays with reflect mode
    yw_pad = np.pad(yw, half, mode="reflect")
    w_pad = np.pad(w, half, mode="reflect")

    kernel = np.ones(k)
    numerator = np.convolve(yw_pad, kernel, mode="valid")
    denominator = np.convolve(w_pad, kernel, mode="valid")
    # Avoid division by zero
    denominator = np.where(denominator == 0, np.nan, denominator)
    return numerator / denominator
