"""
Phenological date extraction.

Port of get_pheno.R, PhenoTrs.R, PhenoDeriv.R, PhenoGu.R from phenofit (R).

Phenophases extracted:
- SOS : Start Of Season (greenup)
- POS : Peak Of Season (maturity)
- EOS : End Of Season (dormancy)
- LOS : Length Of Season
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def pheno_trs(
    t: ArrayLike,
    y: ArrayLike,
    *,
    thresholds: list[float] = (0.2, 0.5, 0.6),
) -> dict[str, float]:
    """
    Threshold-based phenology extraction.

    Port of PhenoTrs() from phenofit (R).

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points (dense output from curve fitting).
    y : array-like of shape (n,)
        Fitted vegetation index values.
    thresholds : list of float
        Relative amplitude thresholds (fraction of max - min).

    Returns
    -------
    dict
        Keys: 'TRS2.SOS', 'TRS2.EOS', 'TRS5.SOS', 'TRS5.EOS', 'TRS6.SOS', 'TRS6.EOS'
        Values: time points (same units as t) or NaN if not detected.
    """
    # TODO: implement
    raise NotImplementedError


def pheno_der(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Derivative-based phenology extraction.

    Port of PhenoDeriv() from phenofit (R).

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    y : array-like of shape (n,)
        Fitted vegetation index values.

    Returns
    -------
    dict
        Keys: 'DER.SOS', 'DER.POS', 'DER.EOS'
        Values: time points or NaN.
    """
    # TODO: implement
    raise NotImplementedError


def pheno_gu(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Gu et al. geometric phenology extraction.

    Port of PhenoGu() from phenofit (R).

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    y : array-like of shape (n,)
        Fitted values.

    Returns
    -------
    dict
        Keys: 'UD', 'SD', 'DD', 'RD' (up-derivative, slope-derivative, etc.)
    """
    # TODO: implement
    raise NotImplementedError


def pheno_zhang(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Zhang et al. curvature-inflection phenology extraction.

    Port of Zhang inflection method from phenofit (R).

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points.
    y : array-like of shape (n,)
        Fitted values.

    Returns
    -------
    dict
        Keys: 'GreenUp', 'Maturity', 'Senescence', 'Dormancy'
        Values: time points or NaN.
    """
    # TODO: implement
    raise NotImplementedError


def get_pheno(
    fit_result,
    *,
    methods: list[str] | None = None,
    as_doy: bool = True,
) -> pd.DataFrame:
    """
    Extract phenological dates from a FittingResult.

    Port of get_pheno.fFITs() from phenofit (R).

    Parameters
    ----------
    fit_result : FittingResult or dict of FittingResult
        Output from curvefit() or curvefits().
    methods : list of str or None
        Phenology extraction methods: 'TRS', 'DER', 'GU', 'ZHANG'.
        Defaults to all four.
    as_doy : bool
        If True, return phenological dates as day-of-year integers.
        If False, return as datetime objects.

    Returns
    -------
    pd.DataFrame
        Rows = growing seasons, columns = phenophase names.
    """
    # TODO: implement
    raise NotImplementedError
