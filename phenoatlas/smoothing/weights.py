"""
Weight-update functions for iterative Whittaker / SG smoothing.

Port of wTSM(), wBisquare(), wHuber() from phenofit (R).

Each function takes (y_obs, y_fit, w_prev) and returns updated weights.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Type alias
WeightFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def wTSM(
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    wfact: float = 0.5,
) -> np.ndarray:
    """
    Asymmetric weight update — Trust Smoothed Mean (TSM).

    Observations above the fitted curve retain their weight. Observations
    below it are multiplied by ``wfact``. This is the dominant behavior of
    phenofit's compiled rcpp_wTSM routine and preserves its asymmetric
    treatment of negative vegetation-index contamination.

    Port of wTSM() from phenofit (R).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Original observations.
    z : np.ndarray of shape (n,)
        Current smoothed values.
    w : np.ndarray of shape (n,)
        Previous weights.
    wfact : float
        Multiplicative penalty for observations below the fit.

    Returns
    -------
    np.ndarray of shape (n,)
        Updated weights in [0, 1].
    """
    return np.where(y >= z, w, w * wfact).astype(float)


def wBisquare(
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    c: float = 4.685,
) -> np.ndarray:
    """
    Symmetric bisquare (Tukey biweight) weight update.

    Port of wBisquare() from phenofit (R).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
    z : np.ndarray of shape (n,)
    w : np.ndarray of shape (n,)
    c : float
        Tuning constant (4.685 for 95% Gaussian efficiency).

    Returns
    -------
    np.ndarray of shape (n,)
    """
    residuals = y - z
    sigma = np.median(np.abs(residuals)) / 0.6745
    if sigma == 0:
        return w.copy()
    u = residuals / (c * sigma)
    bisquare = np.clip((1 - u**2) ** 2, 0.0, 1.0)
    return (w * bisquare).astype(float)


def wHuber(
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    k: float = 1.345,
) -> np.ndarray:
    """
    Huber weight update (less aggressive down-weighting than bisquare).

    Port of wHuber() from phenofit (R).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
    z : np.ndarray of shape (n,)
    w : np.ndarray of shape (n,)
    k : float
        Tuning constant (1.345 for 95% Gaussian efficiency).

    Returns
    -------
    np.ndarray of shape (n,)
    """
    residuals = y - z
    sigma = np.median(np.abs(residuals)) / 0.6745
    if sigma == 0:
        return w.copy()
    u = np.abs(residuals) / (k * sigma)
    # Huber: weight = 1 for |u| <= 1, else 1/|u|
    huber = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    return (w * huber).astype(float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, WeightFn] = {
    "wTSM": wTSM,
    "wBisquare": wBisquare,
    "wHuber": wHuber,
}


def get_wfun(name: str) -> WeightFn:
    """
    Look up a weight-update function by name.

    Parameters
    ----------
    name : str
        One of: 'wTSM', 'wBisquare', 'wHuber'.

    Returns
    -------
    callable
        The corresponding weight-update function.

    Raises
    ------
    ValueError
        If name is not in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown weight function '{name}'. "
            f"Choose from: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]
