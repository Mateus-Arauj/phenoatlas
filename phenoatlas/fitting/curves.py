"""
Double-logistic curve functions for vegetation phenology fitting.

Port of doubleLogistics.R from phenofit (R).

Each function takes (t, p) and returns predicted vegetation index values.
All functions clamp internal exponentials to avoid overflow.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Maximum exponent to prevent overflow in exp()
_EXP_MAX = 500.0


def _logistic(t: np.ndarray, xmid: float, scale: float) -> np.ndarray:
    """Standard logistic function 1 / (1 + exp(-(t - xmid) / scale))."""
    return 1.0 / (1.0 + np.exp(np.clip(-(t - xmid) / scale, -_EXP_MAX, _EXP_MAX)))


# ---------------------------------------------------------------------------
# Beck et al. (2006) — 6 parameters
# ---------------------------------------------------------------------------

def doubleLog_Beck(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Beck et al. double-logistic curve.

    VI(t) = mn + (mx − mn) · [1/(1+exp(−rsp(t−sos))) + 1/(1+exp(rau(t−eos))) − 1]

    Parameters
    ----------
    t : array-like of shape (n,)
        Time points (day-of-year or numeric index).
    p : array-like of shape (6,)
        [mn, mx, sos, rsp, eos, rau]
        - mn  : background (minimum) VI
        - mx  : maximum VI
        - sos : spring inflection (start-of-season)
        - rsp : spring logistic rate (> 0)
        - eos : autumn inflection (end-of-season)
        - rau : autumn logistic rate (> 0)

    Returns
    -------
    np.ndarray of shape (n,)

    References
    ----------
    Beck, P.S.A. et al. (2006). Remote Sensing of Environment.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mn, mx, sos, rsp, eos, rau = p
    amplitude = mx - mn
    spring = 1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    autumn = 1.0 / (1.0 + np.exp(np.clip(rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    return mn + amplitude * (spring + autumn - 1.0)


# ---------------------------------------------------------------------------
# Elmore et al. (2012) — 7 parameters
# ---------------------------------------------------------------------------

def doubleLog_Elmore(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Elmore et al. double-logistic with linear trend.

    VI(t) = mn + (mx − mn·m·t) · [1/(1+exp(−rsp(t−sos))) + 1/(1+exp(rau(t−eos))) − 1]

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (7,)
        [mn, mx, sos, rsp, eos, rau, m]
        - m : slope of linear background term

    Returns
    -------
    np.ndarray of shape (n,)
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mn, mx, sos, rsp, eos, rau, m = p
    spring = 1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    autumn = 1.0 / (1.0 + np.exp(np.clip(rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    return mn + (mx - mn * m * t) * (spring + autumn - 1.0)


# ---------------------------------------------------------------------------
# Zhang et al. (2003) — 7 parameters (piecewise)
# ---------------------------------------------------------------------------

def doubleLog_Zhang(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Zhang et al. piecewise double-logistic curve.

    Growth phase  (t ≤ t0): VI = mn + (mx−mn) / (1 + exp(−rsp(t−sos)))
    Senescence    (t >  t0): VI = mn + (mx−mn) / (1 + exp( rau(t−eos)))

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (7,)
        [mn, mx, sos, rsp, eos, rau, t0]
        - t0 : transition day between growth and senescence phases

    Returns
    -------
    np.ndarray of shape (n,)

    References
    ----------
    Zhang, X. et al. (2003). Monitoring vegetation phenology using MODIS.
    Remote Sensing of Environment, 84(3), 471-475.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mn, mx, sos, rsp, eos, rau, t0 = p
    amplitude = mx - mn
    growth = mn + amplitude / (
        1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX))
    )
    senescence = mn + amplitude / (
        1.0 + np.exp(np.clip(rau * (t - eos), -_EXP_MAX, _EXP_MAX))
    )
    return np.where(t <= t0, growth, senescence)


# ---------------------------------------------------------------------------
# Asymmetric Gaussian — AG (TIMESAT) — 6 parameters
# ---------------------------------------------------------------------------

def doubleLog_AG(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Asymmetric Gaussian (AG) double-logistic curve (TIMESAT formulation).

    VI(t) = c1 + c2 · exp(−|t−t0|^a / b)   (left or right half)

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (6,)
        [t0, c1, c2, a1, b1, a2, b2] — but here we follow phenofit's 6-param:
        [t0, c1, c2, b1, b2, a]
        Reformulation used in phenofit (R) doubleLog.AG:
        [t0, a0, a1, a2, a3, a4, a5]  (7 params)
        We use a simplified version consistent with the R source:

        p = [t0, c1, c2, b_left, b_right, a_left, a_right]  (7 params)
        - t0      : peak location (DOY)
        - c1      : background value
        - c2      : amplitude (mx - mn)
        - b_left  : width of spring limb (> 0)
        - b_right : width of autumn limb (> 0)
        - a_left  : shape exponent for spring (> 0)
        - a_right : shape exponent for autumn (> 0)

    Returns
    -------
    np.ndarray of shape (n,)

    References
    ----------
    Jönsson, P. & Eklundh, L. (2002). Seasonality extraction by function fitting
    to time-series of satellite sensor data. IEEE TGRS.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    t0, c1, c2, b_left, b_right, a_left, a_right = p

    left_mask = t <= t0
    z = np.where(
        left_mask,
        c1 + c2 * np.exp(-np.abs((t - t0) / np.maximum(b_left, 1e-6)) ** a_left),
        c1 + c2 * np.exp(-np.abs((t - t0) / np.maximum(b_right, 1e-6)) ** a_right),
    )
    return z


# ---------------------------------------------------------------------------
# Gu et al. (2009) — 8 parameters (power-law logistic)
# ---------------------------------------------------------------------------

def doubleLog_Gu(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Gu et al. power-law double-logistic curve.

    Extends Beck by replacing logistic rates with power-law exponents:
    VI = mn + (mx−mn) · [(1/(1+exp(−rsp(t−sos))))^b1
                         + (1/(1+exp( rau(t−eos))))^b2 − 1]

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (8,)
        [mn, mx, sos, rsp, eos, rau, b1, b2]
        - b1 : spring power-law exponent
        - b2 : autumn power-law exponent

    Returns
    -------
    np.ndarray of shape (n,)

    References
    ----------
    Gu, Y. et al. (2009). A five-year analysis of MODIS NDVI and NDWI for
    grassland drought assessment over the central Great Plains. Geophysical
    Research Letters.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mn, mx, sos, rsp, eos, rau, b1, b2 = p
    amplitude = mx - mn
    spring = (
        1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    ) ** b1
    autumn = (
        1.0 / (1.0 + np.exp(np.clip(rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    ) ** b2
    return mn + amplitude * (spring + autumn - 1.0)


# ---------------------------------------------------------------------------
# Klos et al. (2018) — 9 parameters (polynomial baseline)
# ---------------------------------------------------------------------------

def doubleLog_Klos(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Klos et al. double-logistic with quadratic polynomial baseline.

    VI = (a0 + a1·t + a2·t²)
         + (1 − a0 − a1·t − a2·t²) · [L_spring + L_autumn − 1]

    where L_spring, L_autumn follow Gu formulation.

    Note: Prone to non-convergence. Use Beck or Elmore as first choice.

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (9,)
        [a0, a1, a2, sos, rsp, eos, rau, b1, b2]
        - a0, a1, a2 : polynomial baseline coefficients

    Returns
    -------
    np.ndarray of shape (n,)
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    a0, a1, a2, sos, rsp, eos, rau, b1, b2 = p
    baseline = a0 + a1 * t + a2 * t**2
    amplitude = 1.0 - baseline
    spring = (
        1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    ) ** b1
    autumn = (
        1.0 / (1.0 + np.exp(np.clip(rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    ) ** b2
    return baseline + amplitude * (spring + autumn - 1.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CURVE_REGISTRY: dict[str, callable] = {
    "Beck": doubleLog_Beck,
    "Elmore": doubleLog_Elmore,
    "Zhang": doubleLog_Zhang,
    "AG": doubleLog_AG,
    "Gu": doubleLog_Gu,
    "Klos": doubleLog_Klos,
}
