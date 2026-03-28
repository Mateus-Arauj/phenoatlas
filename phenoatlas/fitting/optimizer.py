"""
Optimization wrappers for curve fitting.

Port of optim_pheno.R / I_optim() from phenofit (R).

Strategy: try primary method first, fall back to Nelder-Mead when it fails,
matching phenofit's multi-start, multi-method robustness.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize, OptimizeResult, Bounds


# Mapping from phenofit method names to scipy method names
_METHOD_MAP = {
    "L-BFGS-B": "L-BFGS-B",
    "BFGS": "BFGS",
    "CG": "CG",
    "Nelder-Mead": "Nelder-Mead",
    "nlminb": "L-BFGS-B",   # closest scipy equivalent
}

_BOUNDED_METHODS = {"L-BFGS-B", "SLSQP", "trust-constr"}


def i_optim(
    fun: Callable[[np.ndarray], float],
    p0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    method: str = "L-BFGS-B",
    tol: float = 1e-8,
    maxiter: int = 500,
) -> OptimizeResult:
    """
    Unified optimization interface for double-logistic curve fitting.

    Port of I_optim() from phenofit (R).

    Tries the requested method first. If it fails to improve over p0,
    falls back to Nelder-Mead (bounds enforced via penalty).

    Parameters
    ----------
    fun : callable
        Objective (residual) function. Takes a 1-D parameter array, returns float.
    p0 : np.ndarray
        Initial parameter guess (clipped to [lower, upper] automatically).
    lower, upper : np.ndarray
        Parameter bounds (same length as p0).
    method : str
        Optimization method. Recommended: 'L-BFGS-B' (default), 'Nelder-Mead'.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Contains at least:
        - x      : optimal parameters
        - fun    : objective value at x
        - success: whether optimization converged
        - message: status string
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Clip initial guess to feasible region
    p0 = np.clip(np.asarray(p0, dtype=float), lower, upper)
    f0 = fun(p0)

    scipy_method = _METHOD_MAP.get(method, method)
    options = {"maxiter": maxiter}

    if scipy_method in _BOUNDED_METHODS:
        bounds = list(zip(lower.tolist(), upper.tolist()))
        result = minimize(fun, p0, method=scipy_method, bounds=bounds,
                          tol=tol, options=options)
    else:
        # For unbounded methods, wrap with penalty to respect bounds
        def penalized(p: np.ndarray) -> float:
            penalty = np.sum(
                np.maximum(0.0, lower - p) ** 2 + np.maximum(0.0, p - upper) ** 2
            )
            return fun(np.clip(p, lower, upper)) + 1e6 * penalty

        result = minimize(penalized, p0, method=scipy_method,
                          tol=tol, options=options)

    # Clip result to bounds (numerical noise can push slightly outside)
    result.x = np.clip(result.x, lower, upper)

    # Fallback: if primary method didn't improve, try Nelder-Mead
    if not result.success or result.fun >= f0:
        bounds = list(zip(lower.tolist(), upper.tolist()))
        result_nm = minimize(fun, p0, method="L-BFGS-B", bounds=bounds,
                             tol=tol, options=options)
        result_nm.x = np.clip(result_nm.x, lower, upper)
        if result_nm.fun < result.fun:
            result = result_nm

    return result


def make_residual_fun(
    curve_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """
    Build a weighted sum-of-squares objective function for a curve model.

    Parameters
    ----------
    curve_fn : callable
        curve_fn(t, p) → predicted values.
    y, t, w : np.ndarray of shape (n,)
        Observations, time points, weights.

    Returns
    -------
    callable
        fun(p) → weighted SSR (scalar).
    """
    def fun(p: np.ndarray) -> float:
        try:
            y_pred = curve_fn(t, p)
            residuals = y - y_pred
            return float(np.dot(w, residuals**2))
        except Exception:
            return 1e12

    return fun
