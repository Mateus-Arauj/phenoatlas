"""
Quality control decoders for remote sensing vegetation index products.

Each function converts raw QC flags to per-observation weights in [0, 1].

Functions
---------
qc_modis_vi          : MODIS MOD13/MYD13 SummaryQA → weights
qc_modis_vi_detailed : MODIS MOD13 DetailedQA 16-bit full decode → weights
qc_modis_lai         : MODIS MOD15 FparLai_QC → weights
qc_sentinel2         : Sentinel-2 SCL (0–11) → weights
qc_ndvi3g            : GIMMS NDVI3g quality flags → weights
qc_spot              : SPOT VGT / PROBA-V status map → weights
"""

from phenoatlas.qc.quality import (
    qc_modis_lai,
    qc_modis_vi,
    qc_modis_vi_detailed,
    qc_ndvi3g,
    qc_sentinel2,
    qc_spot,
)

__all__ = [
    "qc_modis_vi",
    "qc_modis_vi_detailed",
    "qc_modis_lai",
    "qc_sentinel2",
    "qc_ndvi3g",
    "qc_spot",
]
