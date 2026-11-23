"""
Weighted Whittaker smoother for irregularly-sampled time series.

Port of smooth_wWHIT() / whit2() from phenofit (R).

References
----------
Eilers, P.H.C. (2003). A perfect smoother. Analytical Chemistry, 75(14), 3631-3636.
Eilers, P.H.C. & Marx, B.D. (2004). Splines, knots, and penalties.
Frasso, G. & Eilers, P.H.C. (2015). L- and V-curves for optimal smoothing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse.linalg import spsolve

from phenoatlas.utils.validation import check_input


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def whit2(y: np.ndarray, lam: float, w: np.ndarray) -> np.ndarray:
    """
    Core Whittaker smoother with second-order finite difference penalty.

    Solves: (W + λ·D2'D2) z = W·y
    where W = diag(w) and D2 is the second-order difference matrix.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Input signal. NaN values should be zeroed before calling (use w=0).
    lam : float
        Smoothing parameter λ. Larger → smoother.
    w : np.ndarray of shape (n,)
        Non-negative weights (0 = ignored).

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed signal z.

    Notes
    -----
    Minimizes: Σ wᵢ(yᵢ − zᵢ)² + λ Σ (Δ²zᵢ)²
    Port of whit2() / C routine "smooth2" from phenofit (R).
    """
    n = len(y)
    W = sparse.diags(w, format="csc")

    # Second-order difference matrix D2 of shape (n-2, n)
    e = np.ones(n)
    D = sparse.spdiags([e, -2 * e, e], [0, 1, 2], n - 2, n, format="csc")

    A = W + lam * D.T @ D
    b = W @ y
    z = spsolve(A, b)
    return np.asarray(z, dtype=float)


# ---------------------------------------------------------------------------
# V-curve lambda optimisation
# ---------------------------------------------------------------------------

def _vcurve_lambda(
    y: np.ndarray,
    w: np.ndarray,
    lam_range: np.ndarray,
) -> float:
    """
    Select optimal λ via V-curve (minimum of log(||r||) vs log(||z''||)).

    Parameters
    ----------
    y : np.ndarray of shape (n,)
    w : np.ndarray of shape (n,)
    lam_range : np.ndarray
        Candidate λ values (log-spaced recommended).

    Returns
    -------
    float
        Optimal λ.
    """
    n = len(y)
    e = np.ones(n)
    D = sparse.spdiags([e, -2 * e, e], [0, 1, 2], n - 2, n, format="csc")

    log_rss = []
    log_pen = []
    for lam in lam_range:
        z = whit2(y, lam, w)
        rss = float(np.dot(w, (y - z) ** 2))
        pen = float(np.dot(D @ z, D @ z))
        log_rss.append(np.log(rss + 1e-12))
        log_pen.append(np.log(pen + 1e-12))

    log_rss = np.array(log_rss)
    log_pen = np.array(log_pen)

    if len(lam_range) < 2:
        return float(lam_range[0])

    # V-curve (Frasso & Eilers, 2015): minimum Euclidean step distance
    # along the L-curve (log_rss, log_pen) across log-spaced lambdas.
    dist = np.sqrt(np.diff(log_rss) ** 2 + np.diff(log_pen) ** 2)
    best_idx = int(np.argmin(dist))
    return float(np.sqrt(lam_range[best_idx] * lam_range[best_idx + 1]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_whit(
    y: ArrayLike,
    w: ArrayLike | None = None,
    *,
    lam: float | None = None,
    nptperyear: int = 23,
    ylu: tuple[float, float] | None = None,
    iters: int = 2,
    wfun: str = "wTSM",
) -> np.ndarray:
    """
    Weighted Whittaker smoother with iterative reweighting.

    Port of smooth_wWHIT() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series. NaN values are handled via weights.
    w : array-like of shape (n,) or None
        Initial weights in [0, 1]. If None, 1 for finite values, 0 for NaN.
    lam : float or None
        Smoothing parameter λ. If None, auto-selected via V-curve.
        Typical values: 10–10 000 for MODIS 16-day data.
    nptperyear : int
        Number of observations per year (e.g. 23 for MODIS 16-day).
        Used to set the V-curve search range when lam is None.
    ylu : (float, float) or None
        (lower, upper) valid range for y. Out-of-range → w = 0.
    iters : int
        Number of iterative reweighting steps (1 = no reweighting).
    wfun : str
        Weight-update function after each smoothing step.
        One of: 'wTSM', 'wBisquare', 'wHuber'.

    Returns
    -------
    np.ndarray of shape (n,)
        Smoothed vegetation index values.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.1, 0.3, 0.7, 0.9, 0.8, 0.4, 0.15])
    >>> smooth_whit(y, nptperyear=7, lam=10)
    array([...])
    """
    from phenoatlas.smoothing.weights import get_wfun

    y, t, w0 = check_input(y, w=w, ylu=ylu, nptperyear=nptperyear)

    # Determine lambda
    if lam is None:
        lam_range = np.logspace(
            np.log10(0.5 * nptperyear),
            np.log10(1000 * nptperyear),
            num=40,
        )
        lam = _vcurve_lambda(y, w0, lam_range)

    update_w = get_wfun(wfun)
    w_cur = w0.copy()

    z = y.copy()
    n_iters = max(1, iters)
    for step in range(n_iters):
        z = whit2(y, lam, w_cur)
        if step < n_iters - 1:
            w_cur = update_w(y, z, w_cur)

    return z
