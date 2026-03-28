"""
Utility functions for phenoatlas.

Functions
---------
date2doy    : Convert dates to day-of-year.
doy2date    : Convert day-of-year to dates.
make_date   : Build dates from year + DOY (R make_date order).
year_of     : Extract year from dates.
D1, D2      : First and second numerical derivatives.
curvature   : Signed curvature κ = y'' / (1 + y'²)^1.5
movmean     : Moving average with optional weights.
gof         : Goodness-of-fit metrics (RMSE, R2, NSE, bias).
r2_sign     : Signed R².
cv_coef     : CV of RMSE.
check_input : Validate and coerce time-series input.
"""

from phenoatlas.utils.dates import date2doy, doy2date, make_date, year_of
from phenoatlas.utils.signal import D1, D2, curvature, movmean
from phenoatlas.utils.stats import cv_coef, gof, r2_sign
from phenoatlas.utils.validation import check_input

__all__ = [
    # dates
    "date2doy",
    "doy2date",
    "make_date",
    "year_of",
    # signal
    "D1",
    "D2",
    "curvature",
    "movmean",
    # stats
    "gof",
    "r2_sign",
    "cv_coef",
    # validation
    "check_input",
]
