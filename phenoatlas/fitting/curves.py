"""
Double-logistic curve functions for vegetation phenology fitting.

Port of doubleLogistics.R from phenofit (R).

Each function models vegetation index as a function of day-of-year (t)
given a parameter vector (p). Used as the objective in curve fitting.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def doubleLog_Zhang(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Zhang et al. piecewise double-logistic curve.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points (day-of-year or numeric index).
    p : array-like of shape (7,)
        Parameters: [mn, mx, sos, rsp, eos, rau, t0]
        - mn  : minimum vegetation index (background)
        - mx  : maximum vegetation index (peak)
        - sos : start-of-season (greenup inflection)
        - rsp : rate of spring increase
        - eos : end-of-season (senescence inflection)
        - rau : rate of autumn decrease
        - t0  : transition point between growth/senescence phases

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError


def doubleLog_AG(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Asymmetric Gaussian (AG) double-logistic curve.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    p : array-like of shape (7,)
        Parameters: [t0, a0, a1, a2, a3, a4, a5]
        AG uses an exponential formulation to model asymmetric seasonal curves.

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError


def doubleLog_Beck(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Beck et al. double-logistic curve.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    p : array-like of shape (6,)
        Parameters: [mn, mx, sos, rsp, eos, rau]

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.

    References
    ----------
    Beck, P.S.A. et al. (2006). Improved monitoring of vegetation dynamics at very
    high latitudes: A new method using MODIS NDVI. Remote Sensing of Environment.
    """
    # TODO: implement
    raise NotImplementedError


def doubleLog_Elmore(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Elmore et al. double-logistic with linear trend component.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    p : array-like of shape (7,)
        Parameters: [mn, mx, sos, rsp, eos, rau, m]
        - m : slope of linear trend term

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError


def doubleLog_Gu(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Gu et al. power-law double-logistic curve.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    p : array-like of shape (8,)
        Parameters: [mn, mx, sos, rsp, eos, rau, b1, b2]
        - b1, b2 : power-law exponents for spring and autumn

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError


def doubleLog_Klos(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Klos et al. double-logistic with polynomial baseline.

    Note: This method is flexible but computationally slow and prone to
    non-convergence. Use with caution.

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    p : array-like of shape (9,)
        Parameters: [a0, a1, a2, sos, rsp, eos, rau, b1, b2]

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted vegetation index values.
    """
    # TODO: implement
    raise NotImplementedError
