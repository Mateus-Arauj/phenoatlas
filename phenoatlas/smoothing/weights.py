"""
Weight-update functions for iterative Whittaker / SG smoothing.

Port of wTSM(), wBisquare(), wHuber() from phenofit (R).

Each function takes (y_obs, y_fit, w_prev) and returns updated weights.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

# Type alias
WeightFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def wTSM(
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    *,
    c: float = 6.0,
) -> np.ndarray:
    """
    Asymmetric weight update — Trust Smoothed Mean (TSM).

    Observations *above* the smoothed curve (potential cloud contamination
    in vegetation indices is usually negative, i.e. values are too low) keep
    their original weight. Observations *below* the curve are down-weighted
    using a bisquare-like kernel.

    Port of wTSM() from phenofit (R).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Original observations.
    z : np.ndarray of shape (n,)
        Current smoothed values.
    w : np.ndarray of shape (n,)
        Previous weights.
    c : float
        Tuning constant (default 6, matching phenofit).

    Returns
    -------
    np.ndarray of shape (n,)
        Updated weights in [0, 1].
    """
    residuals = y - z
    # Robust scale estimate (MAD-based, only positive residuals)
    pos = residuals[residuals > 0]
    if pos.size == 0:
        return w.copy()
    sigma = np.median(np.abs(pos)) / 0.6745  # MAD → std estimate
    if sigma == 0:
        return w.copy()

    u = residuals / (c * sigma)
    # Bisquare kernel for negative residuals (values below smoother)
    w_new = np.where(
        residuals >= 0,
        w,  # keep original weight for observations above curve
        w * np.clip((1 - u**2) ** 2, 0.0, 1.0),
    )
    return w_new.astype(float)


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
