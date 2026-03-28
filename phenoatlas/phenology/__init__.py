"""
Phenological date extraction from fitted curves.

Functions
---------
get_pheno    : Extract phenological dates from FittingResult objects.

Extraction Methods
------------------
pheno_trs    : Threshold-based (20%, 50%, 60% of amplitude)
pheno_der    : Derivative-based (rate-of-change peaks)
pheno_gu     : Gu et al. geometric method
pheno_zhang  : Zhang et al. curvature inflection method
"""

from phenoatlas.phenology.extraction import get_pheno, pheno_trs, pheno_der, pheno_gu, pheno_zhang

__all__ = ["get_pheno", "pheno_trs", "pheno_der", "pheno_gu", "pheno_zhang"]
