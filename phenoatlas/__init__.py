"""
phenoatlas: Vegetation phenology extraction from remote sensing time-series.

Python port of the R package phenofit (eco-hydro/phenofit).

Modules
-------
smoothing   : Time-series smoothing (Whittaker, Savitzky-Golay, HANTS)
fitting     : Double-logistic curve fitting (AG, Beck, Elmore, Gu, Klos, Zhang)
season      : Growing season detection
phenology   : Phenological date extraction (TRS, DER, Gu, Zhang methods)
qc          : Quality control for remote sensing products (MODIS, Sentinel-2, SPOT)
utils       : Date utilities and helper functions
"""

__version__ = "0.1.0"
__author__ = "phenoatlas contributors"

from phenoatlas.smoothing import smooth_whit, smooth_sg, smooth_hants
from phenoatlas.fitting import curvefit, curvefits
from phenoatlas.season import season_mov
from phenoatlas.phenology import get_pheno

__all__ = [
    "smooth_whit",
    "smooth_sg",
    "smooth_hants",
    "curvefit",
    "curvefits",
    "season_mov",
    "get_pheno",
]
