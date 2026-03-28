"""
Phenological date extraction from fitted double-logistic curves.

Functions
---------
get_pheno    : Extract all phenological dates from a FittingResult (or dict thereof).

Extraction methods (can also be used directly on t, y arrays)
---------------------
pheno_trs    : Threshold-based — TRS2/TRS5/TRS6 × SOS/EOS
pheno_der    : Derivative-based — DER.SOS, DER.POS, DER.EOS
pheno_gu     : Gu et al. geometric — UD, SD, DD, RD
pheno_zhang  : Zhang et al. curvature-inflection — GreenUp, Maturity, Senescence, Dormancy
"""

from phenoatlas.phenology.extraction import (
    get_pheno,
    pheno_der,
    pheno_gu,
    pheno_trs,
    pheno_zhang,
)

__all__ = [
    "get_pheno",
    "pheno_trs",
    "pheno_der",
    "pheno_gu",
    "pheno_zhang",
]
