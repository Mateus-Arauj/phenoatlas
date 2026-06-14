"""
Main curve-fitting engine.

Port of curvefit.R / curvefits.R from phenofit (R).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from phenoatlas.fitting.curves import CURVE_REGISTRY
from phenoatlas.fitting.init_params import get_init
from phenoatlas.fitting.optimizer import i_optim, i_optim_lsq, make_residual_fun
from phenoatlas.smoothing.weights import get_wfun
from phenoatlas.utils.stats import gof as compute_gof
from phenoatlas.utils.validation import check_input

FittingMethod = Literal["AG", "Beck", "Elmore", "Gu", "Klos", "Zhang"]

# Match phenofit's default method set and order.
_DEFAULT_METHODS: list[FittingMethod] = [
    "AG", "Beck", "Elmore", "Gu", "Klos", "Zhang",
]

# All phenophase names (mirrors phenofit R)
PHENOPHASE_NAMES = [
    "TRS2.SOS", "TRS2.EOS",
    "TRS5.SOS", "TRS5.EOS",
    "TRS6.SOS", "TRS6.EOS",
    "DER.SOS",  "DER.POS", "DER.EOS",
    "UD", "SD", "DD", "RD",
    "GreenUp", "Maturity", "Senescence", "Dormancy",
]


class FittingResult:
    """
    Container for curve fitting results for a single growing season.

    Mirrors the fFITs S3 object from phenofit (R).

    Attributes
    ----------
    data : pd.DataFrame
        Original input data with columns ['t', 'y', 'w'].
    tout : np.ndarray
        Dense output time points (for smooth curve prediction).
    models : dict[str, dict]
        Per-method results. Each entry has:
        - 'param'    : np.ndarray  — fitted parameters
        - 'fitted'   : np.ndarray  — predicted values at tout
        - 'gof'      : dict        — RMSE, R2, NSE, bias, n
        - 'success'  : bool        — whether optimizer converged
        - 'message'  : str         — optimizer status
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tout: np.ndarray,
        models: dict[str, dict],
    ) -> None:
        self.data = data
        self.tout = tout
        self.models = models

    # ------------------------------------------------------------------
    # Accessor helpers (mirror get_param / get_fitting / get_GOF in R)
    # ------------------------------------------------------------------

    def get_param(self, method: str) -> np.ndarray:
        """Return fitted parameters for a given method."""
        self._check_method(method)
        return self.models[method]["param"]

    def get_fitted(self, method: str) -> np.ndarray:
        """Return fitted values at tout for a given method."""
        self._check_method(method)
        return self.models[method]["fitted"]

    def get_gof(self, method: str) -> dict[str, float]:
        """Return goodness-of-fit statistics for a given method."""
        self._check_method(method)
        return self.models[method]["gof"]

    # alias matching phenofit naming
    gof = get_gof

    @property
    def methods(self) -> list[str]:
        """List of methods that were fitted."""
        return list(self.models.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with fitted curves for all methods at tout.

        Returns
        -------
        pd.DataFrame
            Columns: ['t', method1, method2, ...]
        """
        df = pd.DataFrame({"t": self.tout})
        for m, info in self.models.items():
            df[m] = info["fitted"]
        return df

    def _check_method(self, method: str) -> None:
        if method not in self.models:
            raise KeyError(
                f"Method '{method}' not in fitted models. "
                f"Available: {self.methods}"
            )

    def __repr__(self) -> str:
        methods_str = ", ".join(self.methods)
        return (
            f"FittingResult(n_obs={len(self.data)}, "
            f"tout=[{self.tout[0]:.0f}..{self.tout[-1]:.0f}], "
            f"methods=[{methods_str}])"
        )


# ---------------------------------------------------------------------------
# Single-season fitting
# ---------------------------------------------------------------------------

def curvefit(
    y: ArrayLike,
    t: ArrayLike | None = None,
    tout: ArrayLike | None = None,
    *,
    methods: list[FittingMethod] | None = None,
    w: ArrayLike | None = None,
    ylu: tuple[float, float] | None = None,
    nptperyear: int = 23,
    opt_method: str = "L-BFGS-B",
    iters: int = 2,
    wfun: str = "wTSM",
) -> FittingResult:
    """
    Fit double-logistic curve(s) to a single growing season.

    Port of curvefit() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index values for one growing season.
    t : array-like of shape (n,) or None
        Time points (DOY or numeric index). Defaults to 0..n-1.
    tout : array-like or None
        Output time points for the fitted curve. Defaults to ``t``, matching
        phenofit.
    methods : list of str or None
        Fitting methods. Defaults to all six phenofit methods.
        Available: 'AG', 'Beck', 'Elmore', 'Gu', 'Klos', 'Zhang'.
    w : array-like of shape (n,) or None
        Observation weights [0, 1]. Defaults to 1 for finite y, 0 for NaN.
    ylu : (float, float) or None
        Valid range for y. Out-of-range → w = 0.
    nptperyear : int
        Number of observations per year (metadata only in this context).
    opt_method : str
        Optimizer: 'L-BFGS-B' (default), 'Nelder-Mead', 'BFGS'.
    iters : int
        Number of fit/reweight iterations. phenofit defaults to 2.
    wfun : str
        Iterative weight function registered in phenoatlas.smoothing.weights.

    Returns
    -------
    FittingResult

    Examples
    --------
    >>> import numpy as np
    >>> from phenoatlas.fitting.curves import doubleLog_Beck
    >>> t = np.linspace(1, 365, 23)
    >>> p_true = [0.1, 0.8, 100, 0.1, 280, 0.1]
    >>> y = doubleLog_Beck(t, p_true) + np.random.normal(0, 0.02, 23)
    >>> result = curvefit(y, t, methods=["Beck"])
    >>> result.get_param("Beck")
    array([...])
    """
    if methods is None:
        methods = _DEFAULT_METHODS

    original_valid = np.isfinite(np.asarray(y, dtype=float))
    y_arr, t_arr, w_arr = check_input(y, t, w, ylu=ylu, nptperyear=nptperyear)

    if tout is None:
        tout_arr = t_arr.copy()
    else:
        tout_arr = np.asarray(tout, dtype=float)

    data = pd.DataFrame({"t": t_arr, "y": y_arr, "w": w_arr})
    fitted_models: dict[str, dict] = {}
    fit_ylu = (
        (float(np.nanmin(y_arr)), float(np.nanmax(y_arr)))
        if ylu is None
        else ylu
    )
    update_w = get_wfun(wfun)

    for method in methods:
        curve_fn = CURVE_REGISTRY.get(method)
        if curve_fn is None:
            continue
        try:
            init_fn = get_init(method)
        except ValueError:
            continue

        init = None
        try:
            init = init_fn(y_arr, t_arr, w_arr)
            obj_fn = make_residual_fun(curve_fn, y_arr, t_arr, w_arr)
            prior = init.get("prior", init["p0"])
            w_cur = w_arr.copy()
            fitted_iters: list[np.ndarray] = []
            weight_iters: list[np.ndarray] = []
            result = None

            for _ in range(max(1, iters)):
                weight_iters.append(w_cur.copy())
                obj_fn = make_residual_fun(curve_fn, y_arr, t_arr, w_cur)
                if method == "Klos":
                    result = i_optim_lsq(
                        curve_fn,
                        y_arr,
                        t_arr,
                        w_cur,
                        prior,
                        init["lower"],
                        init["upper"],
                    )
                else:
                    result = i_optim(
                        obj_fn,
                        prior,
                        init["lower"],
                        init["upper"],
                        method=opt_method,
                    )
                param = result.x
                fitted_iter = curve_fn(tout_arr, param)
                fitted_iter = np.maximum(fitted_iter, fit_ylu[0])
                fitted_iters.append(fitted_iter)

                y_fit_obs = curve_fn(t_arr, param)
                w_cur = update_w(y_arr, y_fit_obs, w_cur)
                prior_2d = np.atleast_2d(prior)
                prior = np.unique(
                    np.vstack([prior_2d[:-1], param]),
                    axis=0,
                )

            fitted = fitted_iters[-1]
            y_fit_obs = np.maximum(curve_fn(t_arr, param), fit_ylu[0])
            # phenofit's get_GOF() evaluates all finite observations, even
            # those excluded from optimization by zero weights.
            gof_stats = compute_gof(
                y_arr[original_valid],
                y_fit_obs[original_valid],
            )
        except Exception as exc:
            param = init["p0"] if init is not None else np.full(6, np.nan)
            fitted = np.full(len(tout_arr), np.nan)
            gof_stats = {"RMSE": np.nan, "R2": np.nan, "NSE": np.nan,
                         "bias": np.nan, "n": 0}
            fitted_iters = []
            weight_iters = []
            result_success = False
            result_message = str(exc)
        else:
            result_success = result.success
            result_message = result.message if hasattr(result, "message") else ""

        fitted_models[method] = {
            "param": param,
            "fitted": fitted,
            "gof": gof_stats,
            "success": result_success,
            "message": result_message,
            "fitted_iters": fitted_iters,
            "weight_iters": weight_iters,
        }

    return FittingResult(data=data, tout=tout_arr, models=fitted_models)


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def curvefits(
    df: pd.DataFrame,
    *,
    methods: list[FittingMethod] | None = None,
    t_col: str = "t",
    y_col: str = "y",
    w_col: str = "w",
    season_col: str = "season",
    ylu: tuple[float, float] | None = None,
    nptperyear: int = 23,
    opt_method: str = "L-BFGS-B",
    iters: int = 2,
    wfun: str = "wTSM",
) -> dict[str, FittingResult]:
    """
    Batch curve fitting across multiple growing seasons.

    Port of curvefits() from phenofit (R).

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Required columns: t_col, y_col, season_col.
        Optional: w_col (default weight = 1).
    methods : list of str or None
        Fitting methods.
    t_col, y_col, w_col, season_col : str
        Column names.
    ylu : (float, float) or None
        Valid value range.
    nptperyear : int
        Number of observations per year.
    opt_method : str
        Optimizer.

    Returns
    -------
    dict[season_id, FittingResult]

    Examples
    --------
    >>> results = curvefits(df, methods=["Beck"], season_col="year")
    >>> for season, fit in results.items():
    ...     print(season, fit.get_gof("Beck")["RMSE"])
    """
    if w_col not in df.columns:
        df = df.copy()
        df[w_col] = 1.0

    results: dict[str, FittingResult] = {}
    for season_id, group in df.groupby(season_col, sort=True):
        y = group[y_col].to_numpy()
        t = group[t_col].to_numpy()
        w = group[w_col].to_numpy()
        results[season_id] = curvefit(
            y, t,
            methods=methods,
            w=w,
            ylu=ylu,
            nptperyear=nptperyear,
            opt_method=opt_method,
            iters=iters,
            wfun=wfun,
        )

    return results
