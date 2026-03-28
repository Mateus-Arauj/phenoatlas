"""
Goodness-of-fit and statistical utilities.

Port of GOF(), R2_sign(), cv_coef() from phenofit (R).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def gof(
    y_obs: ArrayLike,
    y_pred: ArrayLike,
    w: ArrayLike | None = None,
) -> dict[str, float]:
    """
    Compute goodness-of-fit metrics between observed and predicted values.

    Port of GOF() from phenofit (R).

    Parameters
    ----------
    y_obs : array-like of shape (n,)
        Observed values.
    y_pred : array-like of shape (n,)
        Predicted / fitted values.
    w : array-like of shape (n,) or None
        Non-negative weights. If None, all observations receive weight 1.
        Observations where w == 0 are excluded.

    Returns
    -------
    dict with keys:
        'RMSE'  : Root Mean Squared Error
        'R2'    : Coefficient of determination (can be negative for bad fits)
        'NSE'   : Nash–Sutcliffe Efficiency
        'bias'  : Mean bias (pred − obs)
        'n'     : Number of observations used (after weight filtering)

    Examples
    --------
    >>> gof([1, 2, 3], [1.1, 1.9, 3.2])
    {'RMSE': ..., 'R2': ..., 'NSE': ..., 'bias': ..., 'n': 3}
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if w is None:
        w = np.ones_like(y_obs)
    else:
        w = np.asarray(w, dtype=float)

    # Only use observations with positive weight and no NaN
    mask = (w > 0) & np.isfinite(y_obs) & np.isfinite(y_pred)
    y_obs = y_obs[mask]
    y_pred = y_pred[mask]
    w = w[mask]

    n = int(mask.sum())
    if n == 0:
        return {"RMSE": np.nan, "R2": np.nan, "NSE": np.nan, "bias": np.nan, "n": 0}

    w_sum = w.sum()
    residuals = y_pred - y_obs

    # Weighted mean of observed values
    obs_mean = np.dot(w, y_obs) / w_sum

    # Weighted SS
    ss_res = np.dot(w, residuals**2)
    ss_tot = np.dot(w, (y_obs - obs_mean) ** 2)

    rmse = np.sqrt(ss_res / w_sum)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan  # same formula
    bias = np.dot(w, residuals) / w_sum

    return {"RMSE": float(rmse), "R2": float(r2), "NSE": float(nse), "bias": float(bias), "n": n}


def r2_sign(y_obs: ArrayLike, y_pred: ArrayLike, w: ArrayLike | None = None) -> float:
    """
    Signed R² — returns negative R² when residual variance exceeds total variance.

    Port of R2_sign() from phenofit (R).

    Parameters
    ----------
    y_obs, y_pred : array-like of shape (n,)
    w : array-like or None

    Returns
    -------
    float
    """
    return gof(y_obs, y_pred, w)["R2"]


def cv_coef(y_obs: ArrayLike, y_pred: ArrayLike, w: ArrayLike | None = None) -> float:
    """
    Coefficient of variation of the RMSE (RMSE / mean(obs)).

    Port of cv_coef() from phenofit (R).

    Parameters
    ----------
    y_obs, y_pred : array-like of shape (n,)
    w : array-like or None

    Returns
    -------
    float
    """
    metrics = gof(y_obs, y_pred, w)
    y_obs = np.asarray(y_obs, dtype=float)
    obs_mean = np.nanmean(y_obs)
    if obs_mean == 0 or np.isnan(obs_mean):
        return np.nan
    return float(metrics["RMSE"] / obs_mean)
