"""
Season validation helpers.

Port of check_season_dt(), check_season_list() from phenofit (R).
"""
from __future__ import annotations

import numpy as np


def check_season(
    season: dict,
    y: np.ndarray,
    *,
    w: np.ndarray | None = None,
    amp_min: float = 0.05,
    coverage_min: float = 0.5,
    len_min: int = 10,
    len_max: int = 650,
) -> bool:
    """
    Validate a single season dict returned by season_mov().

    Port of check_season_dt() from phenofit (R).

    A season is valid when:
    1. Its length (in points) is within [len_min, len_max].
    2. The fraction of valid (finite and positive-weight) observations ≥ coverage_min.
    3. The amplitude (max - min) within the season ≥ amp_min.

    Parameters
    ----------
    season : dict
        Season dict with 'idx_start', 'idx_end'.
    y : np.ndarray of shape (n,)
        Full vegetation index array.
    w : np.ndarray of shape (n,) or None
        Optional observation weights. When provided, observations with w <= 0
        are treated as invalid for coverage calculation.
    amp_min : float
        Minimum required amplitude.
    coverage_min : float
        Minimum fraction of valid observations required.
    len_min, len_max : int
        Acceptable season length range in number of points.

    Returns
    -------
    bool
        True if the season passes all checks.
    """
    i0 = season["idx_start"]
    i1 = season["idx_end"] + 1
    seg = np.asarray(y[i0:i1], dtype=float)
    length = len(seg)

    if not (len_min <= length <= len_max):
        return False

    if w is not None:
        w_seg = np.asarray(w[i0:i1], dtype=float)
        valid = np.isfinite(seg) & (w_seg > 0)
    else:
        valid = np.isfinite(seg)
    coverage = float(valid.sum()) / length
    if coverage < coverage_min:
        return False

    if not np.any(valid):
        return False
    amplitude = float(np.nanmax(seg[valid]) - np.nanmin(seg[valid]))
    if amplitude < amp_min:
        return False

    return True


def filter_seasons(
    seasons: list[dict],
    y: np.ndarray,
    **kwargs,
) -> list[dict]:
    """
    Filter a list of season dicts using check_season().

    Parameters
    ----------
    seasons : list of dict
    y : np.ndarray
    **kwargs : passed to check_season()

    Returns
    -------
    list of dict — only valid seasons.
    """
    return [s for s in seasons if check_season(s, y, **kwargs)]
