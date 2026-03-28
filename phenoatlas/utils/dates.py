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
        Input dates. Accepts strings, numpy datetime64, pandas Timestamp,
        or Python datetime objects.

    Returns
    -------
    np.ndarray of int32
        Day-of-year values (1–366).

    Examples
    --------
    >>> import pandas as pd
    >>> date2doy(pd.to_datetime(["2020-01-01", "2020-07-04"]))
    array([  1, 186], dtype=int32)
    """
    dti = pd.DatetimeIndex(pd.to_datetime(dates))
    return dti.day_of_year.to_numpy(dtype=np.int32)


def doy2date(doy: ArrayLike, year: int | ArrayLike) -> np.ndarray:
    """
    Convert day-of-year + year to numpy datetime64[D] dates.

    Parameters
    ----------
    doy : array-like of int
        Day-of-year values (1–366).
    year : int or array-like of int
        Year(s) corresponding to each DOY. A scalar broadcasts to all DOY.

    Returns
    -------
    np.ndarray of datetime64[D]
        Corresponding calendar dates.

    Examples
    --------
    >>> doy2date([1, 186], 2020)
    array(['2020-01-01', '2020-07-04'], dtype='datetime64[D]')
    >>> doy2date([1, 186], [2020, 2021])
    array(['2020-01-01', '2021-07-05'], dtype='datetime64[D]')
    """
    doy = np.asarray(doy, dtype=np.int32)
    year = np.broadcast_to(np.asarray(year, dtype=np.int32), doy.shape).copy()
    jan1 = np.array(
        [np.datetime64(f"{y}-01-01", "D") for y in year],
        dtype="datetime64[D]",
    )
    return jan1 + (doy - 1).astype("timedelta64[D]")


def make_date(year: int | ArrayLike, doy: int | ArrayLike) -> np.ndarray:
    """
    Build dates from year + DOY with argument order matching R's make_date().

    Parameters
    ----------
    year : int or array-like of int
    doy  : int or array-like of int

    Returns
    -------
    np.ndarray of datetime64[D]
    """
    return doy2date(doy, year)


def year_of(dates: ArrayLike) -> np.ndarray:
    """
    Extract the year component from an array of dates.

    Parameters
    ----------
    dates : array-like of datetime-like

    Returns
    -------
    np.ndarray of int32
    """
    return pd.DatetimeIndex(pd.to_datetime(dates)).year.to_numpy(dtype=np.int32)
