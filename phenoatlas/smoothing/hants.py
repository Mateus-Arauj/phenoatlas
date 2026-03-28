"""
Harmonic ANalysis of Time Series (HANTS) for time-series reconstruction.

Port of smooth_wHANTS() from phenofit (R).

References
----------
Roerink, G.J., Menenti, M., & Verhoef, W. (2000). Reconstructing cloudfree
NDVI composites using Fourier analysis of time series. International Journal
of Remote Sensing, 21(9), 1911-1917.

Verhoef, W., Menenti, M., & Azzali, S. (1996). A colour composite of NOAA-AVHRR
NDVI based on time series analysis (1981–1992). International Journal of Remote
Sensing, 17(2), 231-235.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from phenoatlas.utils.validation import check_input


def smooth_hants(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    nf: int = 3,
    iters: int = 3,
    fet: float = 0.05,
    delta: float = 0.1,
    ylu: tuple[float, float] | None = None,
    low: bool = True,
) -> np.ndarray:
    """
    HANTS smoother for vegetation index time series.

    Fits a sum of harmonics (DC + nf sine/cosine pairs) while iteratively
    rejecting outliers that fall below the fitted curve (cloud/snow contamination).

    Port of smooth_wHANTS() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    w : array-like of shape (n,) or None
        Initial weights in [0, 1].
    nptperyear : int
        Number of observations per year.
    nf : int
        Number of frequencies to fit (excluding DC). Typical: 3–6.
    iters : int
        Maximum number of outlier-rejection iterations.
    fet : float
        Fit error tolerance. Observations with |residual| > fet * amplitude
        are rejected on each iteration.
    delta : float
        Suppression factor for outlier residuals (0 = hard rejection).
    ylu : (float, float) or None
        Valid value range; out-of-range → w = 0.
    low : bool
        If True, only reject values *below* the fit (asymmetric rejection,
        typical for vegetation indices). If False, symmetric rejection.

    Returns
    -------
    np.ndarray of shape (n,)
        Reconstructed (HANTS-fitted) vegetation index values.
    """
    y, t, w0 = check_input(y, w=w, ylu=ylu, nptperyear=nptperyear)
    n = len(y)

    # Build design matrix: [1, cos(2πf·t/n), sin(2πf·t/n), ...]
    # t is assumed to be 0..n-1 within one year cycle
    t_norm = np.arange(n, dtype=float)
    cols = [np.ones(n)]
    for f in range(1, nf + 1):
        angle = 2.0 * np.pi * f * t_norm / nptperyear
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    A = np.column_stack(cols)  # shape (n, 1 + 2*nf)

    w_cur = w0.copy()
    z = y.copy()

    for _ in range(iters):
        # Weighted least-squares: A'WA c = A'Wy
        W = np.diag(w_cur)
        AtW = A.T @ W
        try:
            coeffs = np.linalg.solve(AtW @ A + 1e-10 * np.eye(A.shape[1]), AtW @ y)
        except np.linalg.LinAlgError:
            break
        z = A @ coeffs

        # Compute residuals
        residuals = y - z
        amplitude = z.max() - z.min()
        threshold = fet * max(amplitude, 1e-6)

        # Update weights: reject outliers below (or around) the fit
        if low:
            outlier = (residuals < -threshold) & (w_cur > 0)
        else:
            outlier = (np.abs(residuals) > threshold) & (w_cur > 0)

        if not np.any(outlier):
            break  # converged

        # Soft rejection: reduce weight proportional to residual magnitude
        w_cur = np.where(
            outlier,
            w_cur * delta,
            w_cur,
        )

    return z
