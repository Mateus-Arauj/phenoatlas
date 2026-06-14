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
    if not np.all(np.isfinite(p)) or eos < sos:
        return np.full_like(t, 99.0)
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

    VI(t) = mn + (mx − m7·t) · [1/(1+exp(−rsp(t−sos))) − 1/(1+exp(−rau(t−eos)))]

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (7,)
        [mn, mx, sos, rsp, eos, rau, m7]
        - m7 : slope of linear amplitude term (phenofit R: par[7])

    Returns
    -------
    np.ndarray of shape (n,)
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mn, mx, sos, rsp, eos, rau, m7 = p
    spring = 1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    autumn = 1.0 / (1.0 + np.exp(np.clip(-rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    return mn + (mx - m7 * t) * (spring - autumn)


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
        [t0, mn, mx, sos, rsp, eos, rau]
        Param order matches phenofit R: par[1]=t0, par[2]=mn, ...

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
    t0, mn, mx, sos, rsp, eos, rau = p
    if t0 - sos <= 1.0 or t0 - eos >= -1.0:
        return np.full_like(t, 99.0)
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
    Asymmetric Gaussian (AG) curve (TIMESAT / phenofit formulation).

    Growth   (t ≤ t0): VI = mn + (mx−mn) · exp(−((t0−t)·rsp)^a3)
    Senescence (t > t0): VI = mn + (mx−mn) · exp(−((t−t0)·rau)^a5)

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (7,)
        [t0, mn, mx, rsp, a3, rau, a5]
        Param order matches phenofit R: par[1]=t0, par[2]=mn, par[3]=mx,
        par[4]=rsp (spring rate), par[5]=a3 (spring shape exponent),
        par[6]=rau (autumn rate), par[7]=a5 (autumn shape exponent).

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
    t0, mn, mx, rsp, a3, rau, a5 = p
    amplitude = mx - mn

    left_mask = t <= t0
    result = np.empty_like(t)
    result[left_mask] = mn + amplitude * np.exp(
        -np.clip((t0 - t[left_mask]) * rsp, 0.0, _EXP_MAX) ** a3
    )
    result[~left_mask] = mn + amplitude * np.exp(
        -np.clip((t[~left_mask] - t0) * rau, 0.0, _EXP_MAX) ** a5
    )
    return result


# ---------------------------------------------------------------------------
# Gu et al. (2009) — 8 parameters (power-law logistic)
# ---------------------------------------------------------------------------

def doubleLog_Gu(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Gu et al. power-law double-logistic curve.

    VI(t) = y0 + a1·L(rsp(t−sos))^c1 − a2·L(rau(t−eos))^c2
    where L(x) = 1/(1+exp(−x))

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (9,)
        [y0, a1, a2, sos, rsp, eos, rau, c1, c2]
        Param order matches phenofit R: par[1]=y0 (background),
        par[2]=a1 (spring amplitude), par[3]=a2 (autumn amplitude),
        par[4]=sos, par[5]=rsp, par[6]=eos, par[7]=rau,
        par[8]=c1 (spring power), par[9]=c2 (autumn power).

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
    y0, a1, a2, sos, rsp, eos, rau, c1, c2 = p
    L_spring = 1.0 / (1.0 + np.exp(np.clip(-rsp * (t - sos), -_EXP_MAX, _EXP_MAX)))
    L_autumn = 1.0 / (1.0 + np.exp(np.clip(-rau * (t - eos), -_EXP_MAX, _EXP_MAX)))
    return y0 + a1 * L_spring**c1 - a2 * L_autumn**c2


# ---------------------------------------------------------------------------
# Klos et al. (2018) — 13 parameters
# ---------------------------------------------------------------------------

def doubleLog_Klos(t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Klos et al. double-logistic with linear background and quadratic amplitude.

    This is the 13-parameter formulation used by phenofit:

    VI = (a1*t + b1) + (a2*t² + b2*t + c)
         * ((1 + q1*exp(-B1*(t-m1)))^-v1
            - (1 + q2*exp(-B2*(t-m2)))^-v2)

    Parameters
    ----------
    t : array-like of shape (n,)
    p : array-like of shape (13,)
        [a1, a2, b1, b2, c, B1, B2, m1, m2, q1, q2, v1, v2]

    Returns
    -------
    np.ndarray of shape (n,)
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    a1, a2, b1, b2, c, B1, B2, m1, m2, q1, q2, v1, v2 = p
    background = a1 * t + b1
    amplitude = a2 * t**2 + b2 * t + c
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spring = (
            1.0
            + q1 * np.exp(np.clip(-B1 * (t - m1), -_EXP_MAX, _EXP_MAX))
        ) ** (-v1)
        autumn = (
            1.0
            + q2 * np.exp(np.clip(-B2 * (t - m2), -_EXP_MAX, _EXP_MAX))
        ) ** (-v2)
    return background + amplitude * (spring - autumn)


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
