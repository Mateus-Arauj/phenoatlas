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
    Compute goodness-of-fit metrics.

    Port of GOF() from phenofit (R).

    Parameters
    ----------
    y_obs : array-like of shape (n,)
        Observed values.
    y_pred : array-like of shape (n,)
        Predicted values.
    w : array-like of shape (n,) or None
        Weights.

    Returns
    -------
    dict with keys: 'RMSE', 'R2', 'NSE', 'bias', 'n'
    """
    # TODO: implement
    raise NotImplementedError
