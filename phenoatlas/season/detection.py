"""
Growing season detection from vegetation index time series.

Port of season_mov.R, find_season.R from phenofit (R).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from phenoatlas.utils.validation import check_input


# ---------------------------------------------------------------------------
# Peak / trough detection
# ---------------------------------------------------------------------------

def find_peaks(
    y: ArrayLike,
    *,
    min_distance: int = 35,
    r_max: float = 0.1,
    r_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks and troughs in a smoothed vegetation index time series.

    Port of findpeaks() / find_season logic from phenofit (R).

    A peak is a local maximum whose height above the global minimum exceeds
    r_max × amplitude. A trough is a local minimum whose depth below the
    global maximum exceeds r_min × amplitude (from above).

    Parameters
    ----------
    y : array-like of shape (n,)
        Smoothed vegetation index values.
    min_distance : int
        Minimum number of indices between two peaks (or two troughs).
    r_max : float
        A peak must exceed (y_min + r_max × amplitude).
    r_min : float
        A trough must fall below (y_max − r_min × amplitude).

    Returns
    -------
    peaks : np.ndarray of int
        Indices of detected peaks.
    troughs : np.ndarray of int
        Indices of detected troughs.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    amplitude = y.max() - y.min()
    if amplitude == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    peak_threshold = y.min() + r_max * amplitude
    trough_threshold = y.max() - r_min * amplitude

    def _find_extrema(sign: float) -> np.ndarray:
        """sign=1 → peaks, sign=-1 → troughs."""
        candidates = []
        for i in range(1, n - 1):
            if sign * y[i] > sign * y[i - 1] and sign * y[i] >= sign * y[i + 1]:
                candidates.append(i)
        if not candidates:
            return np.array([], dtype=int)
        candidates = np.array(candidates, dtype=int)
        # Apply amplitude threshold
        if sign == 1:
            candidates = candidates[y[candidates] >= peak_threshold]
        else:
            candidates = candidates[y[candidates] <= trough_threshold]
        # Enforce minimum distance: keep only highest-amplitude within window
        if len(candidates) <= 1:
            return candidates
        kept = [candidates[0]]
        for idx in candidates[1:]:
            if idx - kept[-1] >= min_distance:
                kept.append(idx)
            else:
                # Keep the more extreme of the two
                if sign * y[idx] > sign * y[kept[-1]]:
                    kept[-1] = idx
        return np.array(kept, dtype=int)

    peaks = _find_extrema(1)
    troughs = _find_extrema(-1)
    return peaks, troughs


# ---------------------------------------------------------------------------
# Season detection — moving window
# ---------------------------------------------------------------------------

def season_mov(
    y: ArrayLike,
    t: ArrayLike,
    w: ArrayLike | None = None,
    *,
    nptperyear: int = 23,
    smooth_method: str = "whittaker",
    lam: float | None = None,
    iters: int = 2,
    len_min: int | None = None,
    len_max: int | None = None,
    r_max: float = 0.1,
    r_min: float = 0.1,
    calendar_year: bool = False,
) -> list[dict]:
    """
    Detect growing seasons via smoothing and peak detection.

    Port of season_mov() from phenofit (R).

    Each detected season is bounded by consecutive troughs (or calendar
    year boundaries when calendar_year=True).

    Parameters
    ----------
    y : array-like of shape (n,)
        Vegetation index time series (may contain NaN).
    t : array-like of shape (n,)
        Time points as numeric DOY or pandas/numpy datetime.
    w : array-like of shape (n,) or None
        Observation weights [0, 1].
    nptperyear : int
        Number of observations per year (e.g. 23 = MODIS 16-day).
    smooth_method : str
        'whittaker' (default), 'sg', or 'hants'.
    lam : float or None
        Smoothing lambda. None = auto via V-curve.
    iters : int
        Smoothing iterations.
    len_min, len_max : int or None
        Minimum / maximum season length in time-point units (indices).
        If None, defaults scale with ``nptperyear`` (approx. 45–650 days).
    r_max, r_min : float
        Amplitude thresholds for peak / trough detection.
    calendar_year : bool
        If True, use calendar-year boundaries instead of detected troughs.

    Returns
    -------
    list of dict, each with keys:
        'season_id'  : int (1-based)
        'idx_start'  : int  — index into original array
        'idx_end'    : int
        't_start'    : float or datetime — start boundary
        't_end'      : float or datetime
        'peak_idx'   : int or None — index of detected peak
        'peak_t'     : float or datetime or None
    """
    t_orig = np.asarray(t)
    t_is_dt = (
        t_orig.size > 0
        and (
            np.issubdtype(t_orig.dtype, np.datetime64)
            or hasattr(t_orig[0], "day")
        )
    )
    if t_is_dt:
        t_dt = pd.DatetimeIndex(pd.to_datetime(t_orig))
        t_for_check = (t_dt - t_dt[0]).total_seconds().to_numpy() / 86400.0
    else:
        t_for_check = t_orig

    y_arr, t_num, w_arr = check_input(y, t_for_check, w, nptperyear=nptperyear)

    if len_min is None:
        len_min = max(3, int(round(45.0 * nptperyear / 365.0)))
    if len_max is None:
        len_max = max(nptperyear * 2, int(round(650.0 * nptperyear / 365.0)))

    # Smooth
    z = _apply_smoother(y_arr, w_arr, nptperyear, smooth_method, lam, iters)

    if calendar_year:
        seasons = _season_calendar(z, t_orig, t_num, nptperyear)
    else:
        seasons = _season_peaks(z, t_orig, t_num, r_max, r_min, nptperyear)

    # Filter by length
    seasons = [
        s for s in seasons
        if len_min <= (s["idx_end"] - s["idx_start"]) <= len_max
    ]

    # Assign season IDs and add peak info
    for i, s in enumerate(seasons, start=1):
        s["season_id"] = i
        seg = z[s["idx_start"]: s["idx_end"] + 1]
        if len(seg) > 0:
            local_peak = int(np.argmax(seg))
            s["peak_idx"] = s["idx_start"] + local_peak
            s["peak_t"] = t_orig[s["peak_idx"]]
        else:
            s["peak_idx"] = None
            s["peak_t"] = None

    return seasons


def _apply_smoother(
    y: np.ndarray,
    w: np.ndarray,
    nptperyear: int,
    method: str,
    lam: float | None,
    iters: int,
) -> np.ndarray:
    if method == "whittaker":
        from phenoatlas.smoothing.whittaker import smooth_whit
        return smooth_whit(y, w, lam=lam, nptperyear=nptperyear, iters=iters)
    elif method == "sg":
        from phenoatlas.smoothing.savitzky_golay import smooth_sg
        return smooth_sg(y, w, nptperyear=nptperyear, iters=iters)
    elif method == "hants":
        from phenoatlas.smoothing.hants import smooth_hants
        return smooth_hants(y, w, nptperyear=nptperyear)
    else:
        raise ValueError(f"Unknown smooth_method '{method}'. Use 'whittaker', 'sg', or 'hants'.")


def _season_peaks(
    z: np.ndarray,
    t_orig: np.ndarray,
    t_num: np.ndarray,
    r_max: float,
    r_min: float,
    nptperyear: int,
) -> list[dict]:
    """Divide time series into seasons using detected troughs as boundaries."""
    n = len(z)
    min_dist = max(1, nptperyear // 4)
    peaks, troughs = find_peaks(z, min_distance=min_dist, r_max=r_max, r_min=r_min)

    # Build season boundaries from troughs (+ series endpoints as virtual troughs)
    boundaries = np.concatenate([[0], troughs, [n - 1]])
    boundaries = np.unique(boundaries)

    seasons = []
    for i in range(len(boundaries) - 1):
        i0 = int(boundaries[i])
        i1 = int(boundaries[i + 1])
        seasons.append({
            "idx_start": i0,
            "idx_end": i1,
            "t_start": t_orig[i0],
            "t_end": t_orig[i1],
        })

    return seasons


def _season_calendar(
    z: np.ndarray,
    t_orig: np.ndarray,
    t_num: np.ndarray,
    nptperyear: int,
) -> list[dict]:
    """Divide time series by calendar year."""
    try:
        years = pd.DatetimeIndex(pd.to_datetime(t_orig)).year.to_numpy()
    except Exception:
        # Fallback: infer years from position
        years = 1 + np.arange(len(z)) // nptperyear

    seasons = []
    for yr in np.unique(years):
        mask = years == yr
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        seasons.append({
            "idx_start": int(idxs[0]),
            "idx_end": int(idxs[-1]),
            "t_start": t_orig[idxs[0]],
            "t_end": t_orig[idxs[-1]],
            "year": int(yr),
        })

    return seasons


# ---------------------------------------------------------------------------
# Calendar-based season division (simple)
# ---------------------------------------------------------------------------

def season_cal(
    t: ArrayLike,
    *,
    sos_doy: int = 1,
    eos_doy: int = 365,
) -> list[dict]:
    """
    Calendar-based season division with fixed DOY boundaries per year.

    Port of season_calendar() from phenofit (R).

    Parameters
    ----------
    t : array-like of datetime-like
        Time points.
    sos_doy : int
        Day-of-year for season start (default 1 = Jan 1).
    eos_doy : int
        Day-of-year for season end (default 365 = Dec 31).

    Returns
    -------
    list of dict, each with keys:
        'season_id', 'year', 't_start', 't_end', 'idx_start', 'idx_end'
    """
    t_dt = pd.to_datetime(np.asarray(t))
    doy = t_dt.day_of_year.to_numpy()
    year = t_dt.year.to_numpy()

    seasons = []
    sid = 1
    for yr in np.unique(year):
        mask = (year == yr) & (doy >= sos_doy) & (doy <= eos_doy)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        seasons.append({
            "season_id": sid,
            "year": int(yr),
            "idx_start": int(idxs[0]),
            "idx_end": int(idxs[-1]),
            "t_start": t_dt[idxs[0]],
            "t_end": t_dt[idxs[-1]],
        })
        sid += 1

    return seasons
