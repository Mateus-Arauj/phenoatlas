"""
Optimization wrappers for curve fitting.

Port of optim_pheno.R / I_optim() from phenofit (R).
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.optimize import minimize, OptimizeResult


def i_optim(
    fun: Callable[[np.ndarray], float],
    p0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    method: str = "L-BFGS-B",
) -> OptimizeResult:
    """
    Unified optimization interface for curve fitting.

    Port of I_optim() from phenofit (R).

    Parameters
    ----------
    fun : callable
        Objective function to minimize. Takes parameter vector, returns scalar.
    p0 : np.ndarray
        Initial parameter guess.
    lower, upper : np.ndarray
        Parameter bounds.
    method : str
        Optimization method. One of: 'L-BFGS-B', 'Nelder-Mead', 'BFGS', 'nlminb'.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result including optimal parameters and convergence info.
    """
    # TODO: implement
    raise NotImplementedError
