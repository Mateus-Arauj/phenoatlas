"""
Utility functions for phenoatlas.

Functions
---------
date2doy    : Convert dates to day-of-year.
doy2date    : Convert day-of-year to dates.
movmean     : Moving average with optional weights.
D1, D2      : First and second numerical derivatives.
gof         : Goodness-of-fit metrics (RMSE, R2, NSE).
"""

from phenoatlas.utils.dates import date2doy, doy2date
from phenoatlas.utils.signal import movmean, D1, D2
from phenoatlas.utils.stats import gof

__all__ = ["date2doy", "doy2date", "movmean", "D1", "D2", "gof"]
