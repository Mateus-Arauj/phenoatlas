"""
Growing season detection from vegetation index time series.

Functions
---------
season_mov    : Detect growing seasons using moving-window approach.
season_cal    : Calendar-based season division (fixed DOY boundaries).
find_peaks    : Find peaks and troughs in a time series.
"""

from phenoatlas.season.detection import season_mov, season_cal, find_peaks

__all__ = ["season_mov", "season_cal", "find_peaks"]
