"""
Smoothing methods for vegetation index time-series.

Functions
---------
smooth_whit     : Weighted Whittaker smoother (iterative, with V-curve lambda optimization)
smooth_sg       : Weighted Savitzky-Golay filter
smooth_hants    : Harmonic ANalysis of Time Series (HANTS)
"""

from phenoatlas.smoothing.whittaker import smooth_whit
from phenoatlas.smoothing.savitzky_golay import smooth_sg
from phenoatlas.smoothing.hants import smooth_hants

__all__ = ["smooth_whit", "smooth_sg", "smooth_hants"]
