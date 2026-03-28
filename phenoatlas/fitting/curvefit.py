"""
Main curve-fitting engine.

Port of curvefit.R / curvefits.R from phenofit (R).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

FittingMethod = Literal["AG", "Beck", "Elmore", "Gu", "Klos", "Zhang"]

_ALL_METHODS: list[FittingMethod] = ["AG", "Beck", "Elmore", "Gu", "Klos", "Zhang"]

# Phenophase names (mirrors phenofit)
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
        Output time points used for interpolation.
    models : dict[str, dict]
        Fitted parameters and predictions per method.
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

    def get_param(self, method: str) -> np.ndarray:
        """Return fitted parameters for a given method."""
        return self.models[method]["param"]

    def get_fitted(self, method: str) -> np.ndarray:
        """Return fitted values at tout for a given method."""
        return self.models[method]["fitted"]

    def gof(self, method: str) -> dict[str, float]:
        """Return goodness-of-fit statistics for a given method."""
        return self.models[method]["gof"]


def curvefit(
    y: ArrayLike,
    t: ArrayLike | None = None,
    tout: ArrayLike | None = None,
    *,
    methods: list[FittingMethod] | None = None,
    w: ArrayLike | None = None,
) -> FittingResult:
    """
    Fit double-logistic curve(s) to a single growing season.

    Port of curvefit() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index values for one growing season.
    t : array-like of shape (n,) or None
        Time points (DOY or numeric index). Defaults to [0, 1, ..., n-1].
    tout : array-like or None
        Output time points for prediction. Defaults to t.
    methods : list of str or None
        Fitting methods to apply. Defaults to ['AG', 'Beck', 'Elmore', 'Gu', 'Zhang'].
    w : array-like of shape (n,) or None
        Observation weights. Defaults to uniform weights.

    Returns
    -------
    FittingResult
        Container with fitted parameters and predictions for each method.
    """
    # TODO: implement
    raise NotImplementedError


def curvefits(
    df: pd.DataFrame,
    *,
    methods: list[FittingMethod] | None = None,
    t_col: str = "t",
    y_col: str = "y",
    w_col: str = "w",
    season_col: str = "season",
) -> dict[str, FittingResult]:
    """
    Batch curve fitting across multiple growing seasons.

    Port of curvefits() from phenofit (R).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns for time, vegetation index, weights, and season ID.
    methods : list of str or None
        Fitting methods to apply.
    t_col, y_col, w_col, season_col : str
        Column names for time, VI, weights, and season ID.

    Returns
    -------
    dict mapping season ID -> FittingResult
    """
    # TODO: implement
    raise NotImplementedError
