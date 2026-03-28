"""
Smoothing methods for vegetation index time-series.

Functions
---------
smooth_whit     : Weighted Whittaker smoother (iterative, V-curve lambda selection)
smooth_sg       : Weighted Savitzky-Golay filter
smooth_hants    : Harmonic ANalysis of Time Series (HANTS)
whit2           : Core Whittaker solver (low-level)

Weight functions (for iterative reweighting)
--------------------------------------------
wTSM            : Asymmetric TSM down-weighting (default)
wBisquare       : Symmetric Tukey bisquare
wHuber          : Huber M-estimator weights
get_wfun        : Look up a weight function by name
"""

from phenoatlas.smoothing.hants import smooth_hants
from phenoatlas.smoothing.savitzky_golay import smooth_sg
from phenoatlas.smoothing.weights import get_wfun, wBisquare, wHuber, wTSM
from phenoatlas.smoothing.whittaker import smooth_whit, whit2

__all__ = [
    "smooth_whit",
    "whit2",
    "smooth_sg",
    "smooth_hants",
    "wTSM",
    "wBisquare",
    "wHuber",
    "get_wfun",
]
