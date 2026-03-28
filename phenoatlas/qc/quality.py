"""
Quality control flag decoders for remote sensing vegetation index products.

Port of qc_*.R functions from phenofit (R).

Each function converts raw QC bitfields/codes to per-observation weights in [0, 1]:
  0.0 = unusable (cloud, snow, bad retrieval)
  0.5 = marginal quality
  1.0 = best quality

References
----------
MODIS VI QC: https://lpdaac.usgs.gov/documents/103/MOD13_User_Guide_V6.pdf
MODIS LAI:   https://lpdaac.usgs.gov/documents/624/MOD15_User_Guide_V61.pdf
Sentinel-2 SCL: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
GIMMS NDVI3g: Tucker et al. (2005), Pinzon & Tucker (2014)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# MODIS MOD13/MYD13 VI — 16-bit QC
# ---------------------------------------------------------------------------

def qc_modis_vi(
    qc: ArrayLike,
    *,
    level: int = 3,
) -> np.ndarray:
    """
    Decode MODIS MOD13/MYD13 VI quality flags to weights.

    Port of qc_StateQA() / qc_5l() from phenofit (R).

    Bit layout (SummaryQA, bits 0-1):
    - 00 (0) : Good data        → weight 1.0
    - 01 (1) : Marginal data    → weight 0.5 (if level ≥ 2)
    - 10 (2) : Snow/ice         → weight 0.0
    - 11 (3) : Cloudy           → weight 0.0

    Parameters
    ----------
    qc : array-like of uint16
        Raw SummaryQA values from MODIS VI products (bits 0-1).
    level : int (1–5)
        Strictness threshold:
        1 → only QA=0 gets weight 1 (rest 0)
        2 → QA=0 → 1.0, QA=1 → 0.5 (rest 0)
        3 → QA=0 → 1.0, QA=1 → 0.5 (same as level 2; default)
        4–5 → same as 3 (more granular in full bit decoding TBD)

    Returns
    -------
    np.ndarray of float64
        Weights in [0, 1].
    """
    qc = np.asarray(qc, dtype=np.uint16)
    summary = qc & 0b11   # bits 0-1

    w = np.zeros(qc.shape, dtype=float)
    w[summary == 0] = 1.0
    if level >= 2:
        w[summary == 1] = 0.5
    return w


def qc_modis_vi_detailed(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Detailed decode of MODIS MOD13 VI QC (16-bit DetailedQA).

    Extracts VI usefulness (bits 2-5), aerosol (bits 6-7), adjacency (bit 8),
    mixed clouds (bit 10), land/water (bits 11-13), possible snow (bit 14),
    possible shadow (bit 15).

    Parameters
    ----------
    qc : array-like of uint16

    Returns
    -------
    np.ndarray of float
        Weights: 1.0 for high-quality, declining for increasing contamination.
    """
    qc = np.asarray(qc, dtype=np.uint16)

    # VI usefulness (bits 2–5): 0=highest, 12=not useful, 13=not produced
    usefulness = (qc >> 2) & 0b1111
    # Aerosol quality (bits 6–7): 0=climatology, 1=low, 2=average, 3=high
    aerosol = (qc >> 6) & 0b11
    # Adjacent cloud detected (bit 8)
    adj_cloud = (qc >> 8) & 1
    # Mixed clouds (bit 10)
    mixed_cloud = (qc >> 10) & 1
    # Possible snow/ice (bit 14)
    snow = (qc >> 14) & 1
    # Possible shadow (bit 15)
    shadow = (qc >> 15) & 1

    # Base weight from usefulness
    w = np.where(usefulness <= 2, 1.0,
        np.where(usefulness <= 5, 0.8,
        np.where(usefulness <= 10, 0.5, 0.0)))

    # Penalize for aerosol, clouds, snow
    w = np.where(aerosol == 3, w * 0.5, w)
    w = np.where(adj_cloud == 1, w * 0.7, w)
    w = np.where(mixed_cloud == 1, w * 0.5, w)
    w = np.where(snow == 1, 0.0, w)
    w = np.where(shadow == 1, w * 0.7, w)

    return np.clip(w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MODIS MOD15 LAI/FPAR — 8-bit QC
# ---------------------------------------------------------------------------

def qc_modis_lai(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Decode MODIS MOD15 LAI/FPAR quality flags.

    Port of qc_FparLai() from phenofit (R).

    Bit layout (FparLai_QC):
    - Bit 0    : SCF_QC (0=main, 1=back-up algorithm)
    - Bits 1-3 : SCF_QC detail (0=good, 1=other problem, >2=problem)
    - Bit 5    : Pixel not produced (0=produced, 1=not)
    - Bit 6    : Extra QC (0=main, 1=geometry problem)
    - Bit 7    : Snow/ice (1=yes)

    Parameters
    ----------
    qc : array-like of uint8

    Returns
    -------
    np.ndarray of float
    """
    qc = np.asarray(qc, dtype=np.uint8)

    scf_main = (qc & 0b1) == 0          # bit 0: main algorithm
    scf_good = ((qc >> 1) & 0b111) == 0  # bits 1-3: best quality
    not_produced = (qc >> 5) & 1
    snow = (qc >> 7) & 1

    w = np.where(
        (not_produced == 1) | (snow == 1), 0.0,
        np.where(scf_main & scf_good, 1.0,
        np.where(scf_main, 0.7, 0.5))
    )
    return w.astype(float)


# ---------------------------------------------------------------------------
# Sentinel-2 SCL
# ---------------------------------------------------------------------------

# SCL class → weight mapping
# Reference: https://sentinels.copernicus.eu/web/sentinel/...
_SCL_WEIGHTS = {
    0: 0.0,   # No data
    1: 0.0,   # Saturated / Defective
    2: 0.0,   # Dark Area Pixels
    3: 0.0,   # Cloud Shadows
    4: 1.0,   # Vegetation
    5: 1.0,   # Not Vegetated (bare soil)
    6: 0.5,   # Water
    7: 0.0,   # Unclassified
    8: 0.0,   # Cloud (medium prob)
    9: 0.0,   # Cloud (high prob)
    10: 0.0,  # Thin Cirrus
    11: 0.0,  # Snow / Ice
}


def qc_sentinel2(
    scl: ArrayLike,
) -> np.ndarray:
    """
    Decode Sentinel-2 Scene Classification Layer (SCL) to weights.

    Port of qc_sentinel2() from phenofit (R).

    Parameters
    ----------
    scl : array-like of uint8
        SCL values (0–11) from Sentinel-2 L2A products.

    Returns
    -------
    np.ndarray of float
        Weights: 1.0 = vegetation/bare, 0.0 = cloud/shadow/snow/water.
    """
    scl = np.asarray(scl, dtype=np.uint8)
    w = np.zeros(scl.shape, dtype=float)
    for cls, weight in _SCL_WEIGHTS.items():
        w[scl == cls] = weight
    return w


# ---------------------------------------------------------------------------
# GIMMS NDVI3g
# ---------------------------------------------------------------------------

def qc_ndvi3g(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Decode GIMMS NDVI3g quality flags to weights.

    Port of qc_NDVI3g() from phenofit (R).

    NDVI3g flags:
    - 1 : good quality (no constraint)
    - 2 : good quality (NDVI ≤ 0)
    - 3 : marginal quality (cloudy)
    - 4 : snow/ice
    - 5 : missing
    - 6 : not produced (high probability of cloud)

    Parameters
    ----------
    qc : array-like of int (1–6)

    Returns
    -------
    np.ndarray of float
    """
    qc = np.asarray(qc, dtype=np.int8)

    w = np.select(
        [qc == 1, qc == 2, qc == 3, qc >= 4],
        [1.0,      0.8,    0.5,      0.0],
        default=0.0,
    )
    return w.astype(float)


# ---------------------------------------------------------------------------
# SPOT VGT / VEGETATION
# ---------------------------------------------------------------------------

def qc_spot(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Decode SPOT VEGETATION / PROBA-V status map flags.

    Port of qc_SPOT() from phenofit (R).

    SM (Status Map) flags:
    - 0 : good (land, clear sky)        → 1.0
    - 1 : good (land + cloud shadow)    → 0.5
    - 2 : cloudy                        → 0.0
    - 3 : snow/ice                      → 0.0
    - 4 : radiometric saturation        → 0.0
    - 5 : missing                       → 0.0

    Parameters
    ----------
    qc : array-like of int

    Returns
    -------
    np.ndarray of float
    """
    qc = np.asarray(qc, dtype=np.int8)
    w = np.select(
        [qc == 0, qc == 1, qc >= 2],
        [1.0,     0.5,     0.0],
        default=0.0,
    )
    return w.astype(float)
