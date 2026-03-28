"""
Growing season detection.

Port of season_mov.R from phenofit (R).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def find_peaks(
    y: ArrayLike,
    *,
    min_distance: int = 35,
    r_max: float = 0.1,
    r_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and troughs in a smoothed time series.

    Port of findpeaks() / find_season logic from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Smoothed vegetation index values.
    min_distance : int
        Minimum number of points between peaks.
    r_max : float
        Minimum relative height for a peak (fraction of amplitude).
    r_min : float
        Maximum relative height for a trough (fraction of amplitude).

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    troughs : np.ndarray
        Indices of detected troughs.
    """
    # TODO: implement
    raise NotImplementedError


def season_mov(
    y: ArrayLike,
    t: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    smooth_method: str = "whittaker",
    lam: float | None = None,
    iters: int = 2,
    len_min: int = 45,
    len_max: int = 650,
    r_max: float = 0.1,
    r_min: float = 0.1,
    calendar_year: bool = False,
) -> list[dict]:
    """
    Detect growing seasons via moving-window smoothing and peak detection.

    Port of season_mov() from phenofit (R).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series.
    t : array-like of shape (n,)
        Time points (datetime or numeric DOY).
    w : array-like of shape (n,) or None
        Observation weights.
    nptperyear : int
        Number of observations per year.
    smooth_method : str
        Smoothing method: 'whittaker', 'sg', or 'hants'.
    lam : float or None
        Smoothing lambda (None = auto via V-curve).
    iters : int
        Number of smoothing iterations.
    len_min, len_max : int
        Min/max season length in days.
    r_max, r_min : float
        Peak/trough amplitude thresholds.
    calendar_year : bool
        If True, use fixed calendar-year boundaries instead of detected peaks.

    Returns
    -------
    list of dict
        Each dict has keys: 'season_id', 't_start', 't_end', 'peak_t', 'year'.
    """
    # TODO: implement
    raise NotImplementedError


def season_cal(
    t: ArrayLike,
    *,
    sos_doy: int = 1,
    eos_doy: int = 365,
) -> list[dict]:
    """
    Calendar-based season division.

    Port of season_calendar() from phenofit (R).

    Parameters
    ----------
    t : array-like
        Time points (datetime).
    sos_doy, eos_doy : int
        Day-of-year boundaries for each season.

    Returns
    -------
    list of dict
        Season boundary records.
    """
    # TODO: implement
    raise NotImplementedError
