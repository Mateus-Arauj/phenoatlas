"""
Input validation helpers.

Port of check_input(), check_ylu() from phenofit (R).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def check_input(
    y: ArrayLike,
    t: ArrayLike | None = None,
    w: ArrayLike | None = None,
    *,
    ylu: tuple[float, float] | None = None,
    nptperyear: int = 23,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and coerce vegetation index time-series inputs.

    Port of check_input() from phenofit (R).

    Rules applied
    -------------
    - y is cast to float64; NaN is allowed (handled via weights).
    - t defaults to integer indices 0..n-1 if not provided.
    - w defaults to 1.0 for finite y, 0.0 for NaN y.
    - Values outside ylu bounds are zeroed in w.
    - w is clipped to [0, 1].

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index values (e.g. NDVI, EVI). May contain NaN.
    t : array-like of shape (n,) or None
        Time points (numeric DOY or integer index).
    w : array-like of shape (n,) or None
        Observation weights in [0, 1]. If None, derived from finiteness of y.
    ylu : (float, float) or None
        (lower, upper) valid range for y. Values outside are down-weighted to 0.
    nptperyear : int
        Metadata only; not used for validation but kept for API consistency
        with phenofit.

    Returns
    -------
    y : np.ndarray of float64, shape (n,)
        Cleaned values (NaN replaced by 0 where w == 0).
    t : np.ndarray of float64, shape (n,)
        Time points.
    w : np.ndarray of float64, shape (n,)
        Weights in [0, 1].

    Raises
    ------
    ValueError
        If y and t (or y and w) have incompatible shapes.
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    if t is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    if t.shape != y.shape:
        raise ValueError(f"t shape {t.shape} does not match y shape {y.shape}")

    if w is None:
        w = np.where(np.isfinite(y), 1.0, 0.0)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != y.shape:
            raise ValueError(f"w shape {w.shape} does not match y shape {y.shape}")
        # Down-weight NaN observations
        w = np.where(np.isfinite(y), w, 0.0)

    # Apply value-range filter
    if ylu is not None:
        lo, hi = float(ylu[0]), float(ylu[1])
        out_of_range = (y < lo) | (y > hi)
        w = np.where(out_of_range, 0.0, w)

    # Replace NaN in y with 0 to avoid propagation in linear algebra
    y = np.where(np.isfinite(y), y, 0.0)

    # Clip weights to valid range
    w = np.clip(w, 0.0, 1.0)

    return y, t, w
