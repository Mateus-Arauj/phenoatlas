"""
Signal processing utilities.

Port of D1(), D2(), movmean(), curvature() from phenofit (R).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def D1(y: ArrayLike, t: ArrayLike | None = None) -> np.ndarray:
    """
    First numerical derivative.

    Parameters
    ----------
    y : array-like of shape (n,)
    t : array-like of shape (n,) or None
        Time points. If None, unit spacing is assumed.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    # TODO: implement
    raise NotImplementedError


def D2(y: ArrayLike, t: ArrayLike | None = None) -> np.ndarray:
    """
    Second numerical derivative.

    Parameters
    ----------
    y : array-like of shape (n,)
    t : array-like of shape (n,) or None

    Returns
    -------
    np.ndarray of shape (n,)
    """
    # TODO: implement
    raise NotImplementedError


def movmean(
    y: ArrayLike,
    k: int,
    *,
    w: ArrayLike | None = None,
) -> np.ndarray:
    """
    Moving average (optionally weighted).

    Parameters
    ----------
    y : array-like of shape (n,)
    k : int
        Window size.
    w : array-like of shape (n,) or None
        Weights.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    # TODO: implement
    raise NotImplementedError
