"""
Double-logistic curve fitting for vegetation phenology.

Functions
---------
curvefit     : Fit one or more double-logistic models to a single growing season.
curvefits    : Batch fit across multiple growing seasons (wraps curvefit).
FittingResult: Result container (mirrors fFITs S3 in phenofit R).

Curve Models
------------
doubleLog_AG     : Asymmetric Gaussian (AG / TIMESAT formulation)
doubleLog_Beck   : Beck et al. logistics
doubleLog_Elmore : Elmore et al. logistics (linear trend)
doubleLog_Gu     : Gu et al. power-law logistics
doubleLog_Klos   : Klos logistics (polynomial baseline, slow)
doubleLog_Zhang  : Zhang et al. piecewise logistics

Utilities
---------
CURVE_REGISTRY  : dict mapping method name → curve function
PHENOPHASE_NAMES: list of all phenophase output names
"""

from phenoatlas.fitting.curvefit import (
    FittingResult,
    PHENOPHASE_NAMES,
    curvefit,
    curvefits,
)
from phenoatlas.fitting.curves import (
    CURVE_REGISTRY,
    doubleLog_AG,
    doubleLog_Beck,
    doubleLog_Elmore,
    doubleLog_Gu,
    doubleLog_Klos,
    doubleLog_Zhang,
)

__all__ = [
    # Main API
    "curvefit",
    "curvefits",
    "FittingResult",
    "PHENOPHASE_NAMES",
    # Curve functions
    "doubleLog_AG",
    "doubleLog_Beck",
    "doubleLog_Elmore",
    "doubleLog_Gu",
    "doubleLog_Klos",
    "doubleLog_Zhang",
    "CURVE_REGISTRY",
]
