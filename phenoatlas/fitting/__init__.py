"""
Double-logistic curve fitting for vegetation phenology.

Functions
---------
curvefit     : Fit one or more double-logistic models to a single growing season.
curvefits    : Batch fit across multiple growing seasons (wraps curvefit).

Curve Models
------------
doubleLog_AG     : Asymmetric Gaussian (AG)
doubleLog_Beck   : Beck et al. logistics
doubleLog_Elmore : Elmore et al. logistics
doubleLog_Gu     : Gu et al. logistics
doubleLog_Klos   : Klos logistics (flexible but slow)
doubleLog_Zhang  : Zhang et al. piecewise logistics
"""

from phenoatlas.fitting.curvefit import curvefit, curvefits
from phenoatlas.fitting.curves import (
    doubleLog_AG,
    doubleLog_Beck,
    doubleLog_Elmore,
    doubleLog_Gu,
    doubleLog_Klos,
    doubleLog_Zhang,
)

__all__ = [
    "curvefit",
    "curvefits",
    "doubleLog_AG",
    "doubleLog_Beck",
    "doubleLog_Elmore",
    "doubleLog_Gu",
    "doubleLog_Klos",
    "doubleLog_Zhang",
]
