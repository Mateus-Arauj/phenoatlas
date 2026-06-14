"""
Optimization wrappers for curve fitting.

Port of optim_pheno.R / I_optim() from phenofit (R).

Strategy: optimize every supplied starting point and retain the lowest
objective, matching phenofit's multi-start I_optim() behavior.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize

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
    maxiter: int = 1000,
) -> OptimizeResult:
    """
    Unified optimization interface for double-logistic curve fitting.

    Port of I_optim() from phenofit (R).

    ``p0`` may be a single parameter vector or a matrix whose rows are
    independent starting points.

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

    starts = np.asarray(p0, dtype=float)
    if starts.ndim == 1:
        starts = starts[None, :]
    starts = np.clip(starts, lower, upper)
    scipy_method = _METHOD_MAP.get(method, method)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    options = {
        "maxiter": maxiter,
        "ftol": 1e-14,
        "gtol": tol,
        "maxls": 50,
    }

    results: list[OptimizeResult] = []
    for start in starts:
        if scipy_method in _BOUNDED_METHODS:
            result = minimize(
                fun,
                start,
                method=scipy_method,
                bounds=bounds,
                options=options,
            )
        else:
            def penalized(p: np.ndarray) -> float:
                below = np.maximum(0.0, lower - p)
                above = np.maximum(0.0, p - upper)
                penalty = np.sum(
                    np.where(np.isfinite(below), below**2, 0.0)
                    + np.where(np.isfinite(above), above**2, 0.0)
                )
                return fun(np.clip(p, lower, upper)) + 1e6 * penalty

            method_options = {"maxiter": maxiter}
            result = minimize(
                penalized,
                start,
                method=scipy_method,
                tol=tol,
                options=method_options,
            )

        result.x = np.clip(result.x, lower, upper)
        result.fun = fun(result.x)
        results.append(result)

    finite = [result for result in results if np.isfinite(result.fun)]
    if finite:
        return min(finite, key=lambda result: result.fun)
    return results[0]


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
            if not np.all(np.isfinite(y_pred)):
                return 1e12
            residuals = y - y_pred
            value = float(np.dot(w, residuals**2))
            return value if np.isfinite(value) else 1e12
        except Exception:
            return 1e12

    return fun


def i_optim_lsq(
    curve_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    prior: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_nfev: int = 10000,
) -> OptimizeResult:
    """Multi-start nonlinear least squares for the 13-parameter Klos model."""
    starts = np.atleast_2d(np.asarray(prior, dtype=float))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    sqrt_w = np.sqrt(np.asarray(w, dtype=float))
    results: list[OptimizeResult] = []

    def residuals(param: np.ndarray) -> np.ndarray:
        predicted = curve_fn(t, param)
        if not np.all(np.isfinite(predicted)):
            return np.full_like(y, 100.0)
        result = sqrt_w * (y - predicted)
        if not np.all(np.isfinite(result)):
            return np.full_like(y, 100.0)
        return result

    for start in starts:
        result = least_squares(
            residuals,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        result.fun = float(np.dot(result.fun, result.fun))
        results.append(result)

    return min(results, key=lambda result: result.fun)
