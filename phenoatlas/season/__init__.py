"""
Growing season detection from vegetation index time series.

Functions
---------
season_mov    : Detect growing seasons using moving-window smoothing + peak detection.
season_cal    : Calendar-based season division (fixed DOY boundaries per year).
find_peaks    : Find peaks and troughs in a smoothed time series.
check_season  : Validate a single season dict.
filter_seasons: Filter a list of seasons by quality criteria.
"""

from phenoatlas.season.detection import find_peaks, season_cal, season_mov
from phenoatlas.season.validation import check_season, filter_seasons

__all__ = [
    "season_mov",
    "season_cal",
    "find_peaks",
    "check_season",
    "filter_seasons",
]
