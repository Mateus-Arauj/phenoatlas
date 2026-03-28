"""
Quality control flag decoders for remote sensing products.

Port of qc_*.R functions from phenofit (R).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def qc_modis_vi(
    qc: ArrayLike,
    *,
    level: int = 3,
) -> np.ndarray:
    """
    Decode MODIS MOD13/MYD13 VI quality flags.

    Port of qc_StateQA() / qc_5l() from phenofit (R).

    Parameters
    ----------
    qc : array-like of uint16
        Raw QC bitfield values from MODIS VI products.
    level : int
        Quality threshold (1-5). Higher = stricter filtering.

    Returns
    -------
    np.ndarray of float
        Weights in [0, 1] where 0 = bad quality, 1 = best quality.
    """
    # TODO: implement
    raise NotImplementedError


def qc_modis_lai(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Decode MODIS LAI/FPAR (MOD15) quality flags.

    Port of qc_FparLai() from phenofit (R).

    Parameters
    ----------
    qc : array-like of uint8
        Raw QC values from MODIS LAI/FPAR products.

    Returns
    -------
    np.ndarray of float
        Weights in [0, 1].
    """
    # TODO: implement
    raise NotImplementedError


def qc_sentinel2(
    scl: ArrayLike,
) -> np.ndarray:
    """
    Decode Sentinel-2 Scene Classification Layer (SCL) to weights.

    Port of qc_sentinel2() from phenofit (R).

    Parameters
    ----------
    scl : array-like of uint8
        SCL values (0-11) from Sentinel-2 L2A products.

    Returns
    -------
    np.ndarray of float
        Weights in [0, 1]. Cloud/shadow = 0, vegetation = 1.
    """
    # TODO: implement
    raise NotImplementedError


def qc_ndvi3g(
    qc: ArrayLike,
) -> np.ndarray:
    """
    Decode GIMMS NDVI3g quality flags.

    Port of qc_NDVI3g() from phenofit (R).

    Parameters
    ----------
    qc : array-like
        Raw quality flags from NDVI3g product.

    Returns
    -------
    np.ndarray of float
        Weights in [0, 1].
    """
    # TODO: implement
    raise NotImplementedError
