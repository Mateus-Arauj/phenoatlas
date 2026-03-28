"""
Date and DOY conversion utilities.

Port of date2doy(), doy2date(), make_date() from phenofit (R).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def date2doy(dates: ArrayLike) -> np.ndarray:
    """
    Convert dates to day-of-year (DOY).

    Parameters
    ----------
    dates : array-like of datetime-like
        Input dates.

    Returns
    -------
    np.ndarray of int
        Day-of-year values (1–366).
    """
    # TODO: implement
    raise NotImplementedError


def doy2date(doy: ArrayLike, year: int | ArrayLike) -> np.ndarray:
    """
    Convert day-of-year + year to date.

    Parameters
    ----------
    doy : array-like of int
        Day-of-year values (1–366).
    year : int or array-like of int
        Year(s) corresponding to each DOY.

    Returns
    -------
    np.ndarray of datetime64
        Corresponding dates.
    """
    # TODO: implement
    raise NotImplementedError
