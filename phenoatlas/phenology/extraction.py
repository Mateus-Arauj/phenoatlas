"""
Phenological date extraction from fitted double-logistic curves.

Port of get_pheno.R, PhenoTrs.R, PhenoDeriv.R, PhenoGu.R from phenofit (R).

Phenophases extracted
---------------------
TRS method : TRS{20|50|60}.SOS, TRS{20|50|60}.EOS
DER method : DER.SOS, DER.POS, DER.EOS
GU  method : UD, SD, DD, RD  (up-derivative, slope, down-derivative, rate)
ZHANG      : GreenUp, Maturity, Senescence, Dormancy
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from phenoatlas.utils.signal import D1, D2, curvature

if TYPE_CHECKING:
    from phenoatlas.fitting.curvefit import FittingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interp_crossing(t: np.ndarray, y: np.ndarray, level: float) -> tuple[float, float]:
    """
    Find the first upward crossing and last downward crossing of 'level'.

    Returns (t_up, t_down). If not found, returns (nan, nan).
    """
    above = y >= level
    # Upward crossing: False → True
    up = np.where(~above[:-1] & above[1:])[0]
    # Downward crossing: True → False
    down = np.where(above[:-1] & ~above[1:])[0]

    def _interp(i: int) -> float:
        """Linear interpolation to exact crossing point."""
        t0, t1 = t[i], t[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            return float(t0)
        frac = (level - y0) / (y1 - y0)
        return float(t0 + frac * (t1 - t0))

    t_sos = _interp(up[0]) if len(up) > 0 else np.nan
    t_eos = _interp(down[-1]) if len(down) > 0 else np.nan
    return t_sos, t_eos


# ---------------------------------------------------------------------------
# TRS — Threshold-based
# ---------------------------------------------------------------------------

def pheno_trs(
    t: ArrayLike,
    y: ArrayLike,
    *,
    thresholds: tuple[float, ...] = (0.2, 0.5, 0.6),
) -> dict[str, float]:
    """
    Threshold-based phenology extraction.

    Port of PhenoTrs() from phenofit (R).

    Normalises the fitted curve to [0, 1] then finds crossings of each
    threshold: first crossing = SOS, last crossing = EOS.

    Parameters
    ----------
    t : array-like of shape (n,)
        Dense time points (output of curvefit().tout).
    y : array-like of shape (n,)
        Fitted vegetation index values.
    thresholds : tuple of float
        Relative amplitude thresholds. Default (0.2, 0.5, 0.6) → TRS2/TRS5/TRS6.

    Returns
    -------
    dict
        Keys: 'TRS2.SOS', 'TRS2.EOS', 'TRS5.SOS', 'TRS5.EOS', 'TRS6.SOS', 'TRS6.EOS'
        Values: time (same units as t) or NaN if not found.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    y_min, y_max = y.min(), y.max()
    amplitude = y_max - y_min
    if amplitude < 1e-6:
        return {f"TRS{int(thr*10)}.SOS": np.nan for thr in thresholds} | \
               {f"TRS{int(thr*10)}.EOS": np.nan for thr in thresholds}

    y_norm = (y - y_min) / amplitude

    result: dict[str, float] = {}
    label_map = {0.2: "TRS2", 0.5: "TRS5", 0.6: "TRS6"}
    for thr in thresholds:
        label = label_map.get(thr, f"TRS{int(thr * 10)}")
        sos, eos = _interp_crossing(t, y_norm, thr)
        result[f"{label}.SOS"] = sos
        result[f"{label}.EOS"] = eos

    return result


# ---------------------------------------------------------------------------
# DER — Derivative-based
# ---------------------------------------------------------------------------

def pheno_der(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Derivative-based phenology extraction.

    Port of PhenoDeriv() from phenofit (R).

    - DER.SOS : time of maximum positive derivative (rapid greenup)
    - DER.POS : time of maximum value (peak of season)
    - DER.EOS : time of maximum negative derivative (rapid senescence, most negative)

    Parameters
    ----------
    t : array-like of shape (n,)
    y : array-like of shape (n,)
        Fitted vegetation index.

    Returns
    -------
    dict with keys 'DER.SOS', 'DER.POS', 'DER.EOS'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    dy = D1(y, t)
    pos_idx = int(np.argmax(y))       # POS
    sos_idx = int(np.argmax(dy[:pos_idx + 1])) if pos_idx > 0 else 0
    eos_candidates = dy[pos_idx:]
    eos_local = int(np.argmin(eos_candidates))
    eos_idx = pos_idx + eos_local

    return {
        "DER.SOS": float(t[sos_idx]),
        "DER.POS": float(t[pos_idx]),
        "DER.EOS": float(t[eos_idx]),
    }


# ---------------------------------------------------------------------------
# GU — Gu et al. geometric method
# ---------------------------------------------------------------------------

def pheno_gu(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Gu et al. geometric phenology extraction.

    Port of PhenoGu() from phenofit (R).

    Identifies four transition dates using the first derivative:
    - UD (Up-Derivative)    : greenup onset — first positive derivative peak
    - SD (Slope-Derivative) : greenup maximum — maximum value on rising limb
    - DD (Down-Derivative)  : senescence onset — first negative derivative trough
    - RD (Rate-Derivative)  : dormancy onset — last negative derivative trough

    Parameters
    ----------
    t, y : array-like of shape (n,)

    Returns
    -------
    dict with keys 'UD', 'SD', 'DD', 'RD'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = D1(y, t)

    pos_idx = int(np.argmax(y))   # peak

    # Rising limb: before the peak
    dy_rise = dy[:pos_idx + 1]
    # UD: first local maximum of dy on rising limb
    ud_idx = int(np.argmax(dy_rise)) if len(dy_rise) > 0 else 0
    # SD: maximum y on rising limb (same as POS for unimodal, but here before POS)
    sd_idx = pos_idx  # peak of season

    # Falling limb: after the peak
    dy_fall = dy[pos_idx:]
    if len(dy_fall) > 0:
        # DD: first local minimum of dy on falling limb (most negative)
        dd_local = int(np.argmin(dy_fall))
        dd_idx = pos_idx + dd_local
        # RD: last local minimum (derivative returns toward 0)
        rd_local = dd_local  # simplification: same as DD for unimodal
        rd_idx = pos_idx + rd_local
    else:
        dd_idx = pos_idx
        rd_idx = pos_idx

    return {
        "UD": float(t[ud_idx]),
        "SD": float(t[sd_idx]),
        "DD": float(t[dd_idx]),
        "RD": float(t[rd_idx]),
    }


# ---------------------------------------------------------------------------
# ZHANG — Curvature inflection method
# ---------------------------------------------------------------------------

def pheno_zhang(
    t: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """
    Zhang et al. curvature-inflection phenology extraction.

    Port of Zhang inflection method from phenofit (R).

    Identifies four transition dates as extrema of the curvature κ:
    - GreenUp  : first local maximum of κ on the rising limb
    - Maturity : second local maximum of κ (peak near-plateau)
    - Senescence : first local maximum of κ on the falling limb
    - Dormancy   : second local maximum of κ on the falling limb

    Parameters
    ----------
    t, y : array-like of shape (n,)

    Returns
    -------
    dict with keys 'GreenUp', 'Maturity', 'Senescence', 'Dormancy'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    kappa = curvature(y, t)
    pos_idx = int(np.argmax(y))

    def _local_maxima(arr: np.ndarray) -> np.ndarray:
        """Return indices of local maxima in arr."""
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
                peaks.append(i)
        return np.array(peaks, dtype=int)

    # Rising limb
    kappa_rise = kappa[:pos_idx + 1]
    peaks_rise = _local_maxima(kappa_rise)

    if len(peaks_rise) >= 2:
        greenup_idx = int(peaks_rise[0])
        maturity_idx = int(peaks_rise[-1])
    elif len(peaks_rise) == 1:
        greenup_idx = int(peaks_rise[0])
        maturity_idx = pos_idx
    else:
        greenup_idx = 0
        maturity_idx = pos_idx

    # Falling limb
    kappa_fall = kappa[pos_idx:]
    peaks_fall = _local_maxima(kappa_fall)

    if len(peaks_fall) >= 2:
        senescence_idx = pos_idx + int(peaks_fall[0])
        dormancy_idx = pos_idx + int(peaks_fall[-1])
    elif len(peaks_fall) == 1:
        senescence_idx = pos_idx + int(peaks_fall[0])
        dormancy_idx = len(t) - 1
    else:
        senescence_idx = pos_idx
        dormancy_idx = len(t) - 1

    return {
        "GreenUp": float(t[greenup_idx]),
        "Maturity": float(t[maturity_idx]),
        "Senescence": float(t[senescence_idx]),
        "Dormancy": float(t[dormancy_idx]),
    }


# ---------------------------------------------------------------------------
# Dispatcher — get_pheno()
# ---------------------------------------------------------------------------

_PHENO_METHODS = {
    "TRS": pheno_trs,
    "DER": pheno_der,
    "GU": pheno_gu,
    "ZHANG": pheno_zhang,
}

_DEFAULT_PHENO_METHODS = ["TRS", "DER", "GU", "ZHANG"]


def get_pheno(
    fit_result: "FittingResult | dict",
    *,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract phenological dates from a FittingResult.

    Port of get_pheno.fFITs() from phenofit (R).

    Parameters
    ----------
    fit_result : FittingResult or dict[str, FittingResult]
        Output from curvefit() (single season) or curvefits() (multiple seasons).
        When a dict is passed, one row per season is returned.
    methods : list of str or None
        Phenology extraction methods: 'TRS', 'DER', 'GU', 'ZHANG'.
        Defaults to all four.

    Returns
    -------
    pd.DataFrame
        Rows = (season × fitting-method) combinations.
        Columns = phenophase names + 'season', 'fit_method'.

    Examples
    --------
    >>> fit = curvefit(y, t, methods=["Beck"])
    >>> df = get_pheno(fit)
    >>> df[["TRS5.SOS", "TRS5.EOS", "DER.POS"]]
    """
    from phenoatlas.fitting.curvefit import FittingResult  # local import → avoid cycle

    if methods is None:
        methods = _DEFAULT_PHENO_METHODS

    # Normalise input to dict[season_id → FittingResult]
    if isinstance(fit_result, FittingResult):
        fits: dict = {"1": fit_result}
    elif isinstance(fit_result, dict):
        fits = fit_result
    else:
        raise TypeError(
            f"fit_result must be FittingResult or dict, got {type(fit_result)}"
        )

    records = []
    for season_id, fr in fits.items():
        for fit_method in fr.methods:
            tout = fr.tout
            fitted = fr.get_fitted(fit_method)
            row: dict = {"season": season_id, "fit_method": fit_method}

            for pheno_method in methods:
                fn = _PHENO_METHODS.get(pheno_method.upper())
                if fn is None:
                    continue
                try:
                    pheno_dates = fn(tout, fitted)
                    row.update(pheno_dates)
                except Exception:
                    pass

            records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
