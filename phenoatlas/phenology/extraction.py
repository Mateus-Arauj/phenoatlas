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

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import make_smoothing_spline

from phenoatlas.fitting.curves import CURVE_REGISTRY
from phenoatlas.utils.signal import D1, D2

if TYPE_CHECKING:
    from phenoatlas.fitting.curvefit import FittingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_index(y: np.ndarray) -> int:
    """Return the first maximum, matching R's which.max()."""
    return int(np.nanargmax(y))


def _r_round(value: float) -> float:
    """Use R-compatible round-to-even behavior and preserve NaN."""
    return float(np.rint(value)) if np.isfinite(value) else np.nan


def _median_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _find_peak(
    values: np.ndarray,
    *,
    nups: int,
    ndowns: int,
) -> int | None:
    """Small numeric equivalent of phenofit's findpeaks() for DER extraction."""
    values = np.asarray(values, dtype=float)
    if values.size < max(2, nups + ndowns + 1):
        return None

    candidates: list[int] = []
    for i in range(values.size):
        left = 0
        j = i
        while j > 0 and values[j] > values[j - 1]:
            left += 1
            j -= 1

        right = 0
        j = i
        while j < values.size - 1 and values[j] > values[j + 1]:
            right += 1
            j += 1

        if left >= nups and right >= ndowns:
            candidates.append(i)

    if not candidates:
        return None
    return max(candidates, key=lambda idx: values[idx])


def _richardson_first(fun, x: np.ndarray) -> np.ndarray:
    """Pointwise Richardson derivative matching numDeriv::grad defaults."""
    x = np.asarray(x, dtype=float)
    zero_tol = np.sqrt(np.finfo(float).eps / 7e-7)
    h = np.abs(1e-4 * x) + 1e-4 * (np.abs(x) < zero_tol)
    estimates = []
    for _ in range(4):
        estimates.append((fun(x + h) - fun(x - h)) / (2.0 * h))
        h = h / 2.0

    table = np.asarray(estimates)
    for order in range(1, 4):
        factor = 4**order
        table = (table[1:] * factor - table[:-1]) / (factor - 1)
    return np.asarray(table[0], dtype=float)


def _curve_derivatives(
    curve_fn,
    t: np.ndarray,
    param: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical first and second curve derivatives used by phenofit."""
    def first(x: np.ndarray) -> np.ndarray:
        return _richardson_first(lambda q: curve_fn(q, param), x)

    der1 = first(t)
    der2 = _richardson_first(first, t)
    return der1, der2


def _smoother_trace(n: int, lam: float) -> float:
    """Trace of scipy's natural cubic smoothing-spline operator."""
    x = np.arange(1, n + 1, dtype=float)
    trace = 0.0
    for i in range(n):
        basis = np.zeros(n)
        basis[i] = 1.0
        trace += float(make_smoothing_spline(x, basis, lam=lam)(x)[i])
    return trace


@lru_cache(maxsize=16)
def _lambda_for_df(n: int) -> float:
    """Find the smoothing penalty whose effective df equals 10% of n."""
    target = min(float(n), max(2.01, 0.1 * n))
    lower, upper = 1e-8, 1e8
    for _ in range(16):
        middle = np.sqrt(lower * upper)
        if _smoother_trace(n, middle) > target:
            lower = middle
        else:
            upper = middle
    return float(np.sqrt(lower * upper))


def _find_peaks_r(
    values: np.ndarray,
    *,
    nups: int = 1,
    ndowns: int = 1,
    min_height: float = -np.inf,
    min_distance: int = 1,
    npeaks: int = 0,
) -> np.ndarray:
    """Subset of phenofit's regex-based findpeaks() used by PhenoKl."""
    values = np.asarray(values, dtype=float)
    signs = np.sign(np.diff(values))
    candidates: list[int] = []
    i = 0

    while i < len(signs):
        if signs[i] <= 0:
            i += 1
            continue

        rise_start = i
        while i < len(signs) and signs[i] > 0:
            i += 1
        rise_end = i
        if rise_end - rise_start < nups:
            continue

        while i < len(signs) and signs[i] == 0:
            i += 1
        fall_start = i
        while i < len(signs) and signs[i] < 0:
            i += 1
        fall_count = i - fall_start
        if fall_count < ndowns:
            continue

        segment_end = i if ndowns > 0 else rise_end
        segment = values[rise_start:segment_end + 1]
        peak = rise_start + int(np.floor(np.median(np.flatnonzero(
            segment == np.nanmax(segment)
        ))))
        if values[peak] >= min_height:
            candidates.append(peak)

    if not candidates:
        return np.array([], dtype=int)

    candidates.sort()
    while len(candidates) > 1:
        distances = np.diff(candidates)
        close = np.flatnonzero(distances < min_distance)
        if close.size == 0:
            break
        left = int(close[0])
        if values[candidates[left]] <= values[candidates[left + 1]]:
            delete = left
        else:
            delete = left + 1
        del candidates[delete]

    candidates.sort(key=lambda idx: values[idx], reverse=True)
    if npeaks > 0:
        candidates = candidates[:npeaks]
    return np.asarray(candidates, dtype=int)


def _fill_zhang_positions(positions: np.ndarray, half_length: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 1:
        position_r = positions[0] + 1
        if position_r < half_length - position_r:
            return np.array([positions[0], np.nan])
        return np.array([np.nan, positions[0]])
    if positions.size == 0:
        return np.array([np.nan, np.nan])
    return positions[:2]


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

    Uses phenofit's asymmetric White threshold method. The rising and falling
    limbs have independent baselines, and dates are medians of samples within
    a threshold band rather than interpolated crossing points.

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

    if np.all(np.isnan(y)):
        return {
            key: np.nan
            for thr in thresholds
            for key in (f"TRS{int(thr * 10)}.SOS", f"TRS{int(thr * 10)}.EOS")
        }

    peak_idx = _peak_index(y)
    n = len(y)
    if peak_idx + 1 < 5 or peak_idx + 1 > n - 5:
        return {
            key: np.nan
            for thr in thresholds
            for key in (f"TRS{int(thr * 10)}.SOS", f"TRS{int(thr * 10)}.EOS")
        }

    peak = float(np.nanmax(y))
    pos = float(t[peak_idx])
    mn_before = float(np.nanmin(y[:peak_idx + 1]))
    mn_after = float(np.nanmin(y[peak_idx + 1:]))
    baseline = np.concatenate([
        np.full(peak_idx + 1, mn_before),
        np.full(n - peak_idx - 1, mn_after),
    ])
    amplitude = peak - baseline
    ratio = np.divide(
        y - baseline,
        amplitude,
        out=np.full_like(y, np.nan),
        where=amplitude != 0,
    )
    delta = np.diff(ratio, prepend=np.nan)
    greenup = np.full(n, np.nan)
    greenup[delta > 0] = 1.0
    greenup[delta < 0] = 0.0

    result: dict[str, float] = {}
    label_map = {0.2: "TRS2", 0.5: "TRS5", 0.6: "TRS6"}
    for thr in thresholds:
        label = label_map.get(thr, f"TRS{int(thr * 10)}")
        in_band = (ratio >= thr - 0.05) & (ratio <= thr + 0.05)
        sos = _median_or_nan(t[(greenup == 1) & in_band & (t < pos)])
        eos = _median_or_nan(t[(greenup == 0) & in_band & (t > pos)])
        sos = _r_round(sos)
        eos = _r_round(eos)

        if not np.isfinite(sos) or not np.isfinite(eos):
            wide = (ratio >= thr - 0.1) & (ratio <= thr + 0.1)
            if not np.isfinite(sos):
                sos = _r_round(
                    _median_or_nan(t[(greenup == 1) & wide & (t < pos)])
                )
            if not np.isfinite(eos):
                eos = _r_round(
                    _median_or_nan(t[(greenup == 0) & wide & (t > pos)])
                )
            if not np.isfinite(sos):
                sos = _r_round((t[0] + pos) / 2.0)
            if not np.isfinite(eos):
                eos = _r_round((t[-1] + pos) / 2.0)

        result[f"{label}.SOS"] = sos
        result[f"{label}.EOS"] = eos

    return result


# ---------------------------------------------------------------------------
# DER — Derivative-based
# ---------------------------------------------------------------------------

def pheno_der(
    t: ArrayLike,
    y: ArrayLike,
    *,
    der1: ArrayLike | None = None,
) -> dict[str, float]:
    """
    Derivative-based phenology extraction.

    Port of PhenoDeriv() from phenofit (R).

    Peaks must have five rising and five falling derivative samples, matching
    phenofit's PhenoDeriv.default(). If no qualifying peak exists, phenofit's
    midpoint fallback is used.

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

    if np.all(np.isnan(y)):
        return {"DER.SOS": np.nan, "DER.POS": np.nan, "DER.EOS": np.nan}

    pos_idx = _peak_index(y)
    n = len(t)
    if pos_idx + 1 < 5 or pos_idx + 1 > n - 5:
        return {"DER.SOS": np.nan, "DER.POS": np.nan, "DER.EOS": np.nan}

    dy = D1(y, t) if der1 is None else np.asarray(der1, dtype=float)
    pos = float(t[pos_idx])
    rise_end = pos_idx - 4
    fall_start = pos_idx + 5

    sos_local = _find_peak(dy[:rise_end], nups=5, ndowns=5)
    eos_local = _find_peak(-dy[fall_start:], nups=5, ndowns=5)

    if sos_local is None:
        sos_local = _find_peak(dy[:rise_end], nups=5, ndowns=0)
    if eos_local is None:
        eos_local = _find_peak(-dy[fall_start:], nups=5, ndowns=0)

    sos = (
        float(t[sos_local])
        if sos_local is not None
        else _r_round((t[0] + pos) / 2.0)
    )
    eos = (
        float(t[fall_start + eos_local])
        if eos_local is not None
        else _r_round((t[-1] + pos) / 2.0)
    )

    return {
        "DER.SOS": sos,
        "DER.POS": pos,
        "DER.EOS": eos,
    }


# ---------------------------------------------------------------------------
# GU — Gu et al. geometric method
# ---------------------------------------------------------------------------

def pheno_gu(
    t: ArrayLike,
    y: ArrayLike,
    *,
    der1: ArrayLike | None = None,
) -> dict[str, float]:
    """
    Gu et al. geometric phenology extraction.

    Port of PhenoGu() from phenofit (R).

    Reproduces phenofit's tangent-intersection construction around the maximum
    spring and autumn rates.

    Parameters
    ----------
    t, y : array-like of shape (n,)

    Returns
    -------
    dict with keys 'UD', 'SD', 'DD', 'RD'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(np.isnan(y)):
        return {name: np.nan for name in ("UD", "SD", "DD", "RD")}

    pos_idx = _peak_index(y)
    n = len(t)
    if pos_idx + 1 < 5 or pos_idx + 1 > n - 5:
        return {name: np.nan for name in ("UD", "SD", "DD", "RD")}

    dy = D1(y, t) if der1 is None else np.asarray(der1, dtype=float)
    rise_end = pos_idx - 4
    fall_start = pos_idx + 5
    sos_idx = int(np.argmax(dy[:rise_end]))
    eos_local = int(np.argmin(dy[fall_start:]))
    # This offset intentionally follows phenofit's one-based index arithmetic.
    eos_idx = min(eos_local + pos_idx + 1, n - 1)

    sos = float(t[sos_idx])
    eos = float(t[eos_idx])
    rsp = float(dy[sos_idx])
    rau = float(dy[eos_idx])
    rise_intercept = float(y[sos_idx] - rsp * sos)
    fall_intercept = float(y[eos_idx] - rau * eos)
    baseline = float(np.nanmin(y))
    maximum = float(np.nanmax(y))

    with np.errstate(divide="ignore", invalid="ignore"):
        ud = (baseline - rise_intercept) / rsp
        sd = (maximum - rise_intercept) / rsp
        dd = (maximum - fall_intercept) / rau
        rd = (baseline - fall_intercept) / rau

    plateau = (t >= sd) & (t <= dd)
    if np.count_nonzero(plateau) > 3 and not np.all(np.isnan(y[plateau])):
        design = np.column_stack([np.ones(np.count_nonzero(plateau)), t[plateau]])
        intercept, slope = np.linalg.lstsq(design, y[plateau], rcond=None)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = (fall_intercept - intercept) / (slope - rau)

    values = np.rint([ud, sd, dd, rd])
    values = np.clip(values, t[0], t[-1])
    return dict(zip(("UD", "SD", "DD", "RD"), values.astype(float)))


# ---------------------------------------------------------------------------
# ZHANG — Curvature inflection method
# ---------------------------------------------------------------------------

def pheno_zhang(
    t: ArrayLike,
    y: ArrayLike,
    *,
    der1: ArrayLike | None = None,
    der2: ArrayLike | None = None,
) -> dict[str, float]:
    """
    Zhang et al. curvature-inflection phenology extraction.

    Port of Zhang inflection method from phenofit (R).

    Identifies four transition dates from extrema of the smoothed curvature
    rate, matching phenofit's PhenoKl implementation.

    Parameters
    ----------
    t, y : array-like of shape (n,)

    Returns
    -------
    dict with keys 'GreenUp', 'Maturity', 'Senescence', 'Dormancy'
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.all(np.isnan(y)):
        return {
            name: np.nan
            for name in ("GreenUp", "Maturity", "Senescence", "Dormancy")
        }

    pos_idx = _peak_index(y)
    n = len(t)
    if pos_idx + 1 < 5 or pos_idx + 1 > n - 5:
        return {
            name: np.nan
            for name in ("GreenUp", "Maturity", "Senescence", "Dormancy")
        }

    if der1 is None or der2 is None:
        first = D1(y, t)
        second = D2(y, t)
    else:
        first = np.asarray(der1, dtype=float)
        second = np.asarray(der2, dtype=float)

    kappa = second / (1.0 + first**2) ** 1.5
    x = np.arange(1, n + 1, dtype=float)
    spline = make_smoothing_spline(x, kappa, lam=_lambda_for_df(n))
    curvature_rate = spline.derivative(1)(x)
    amplitude = float(np.ptp(curvature_rate))

    ascending = curvature_rate[:pos_idx + 1]
    asc_positions = _find_peaks_r(
        ascending,
        ndowns=0,
        min_distance=15,
        min_height=amplitude * 0.025 + float(np.min(ascending)),
        npeaks=2,
    )
    asc_positions = np.sort(asc_positions)
    asc_positions = _fill_zhang_positions(asc_positions, len(ascending))

    descending = curvature_rate[pos_idx:]
    desc_positions = _find_peaks_r(
        -descending,
        min_distance=15,
        min_height=amplitude * 0.025 + float(np.min(-descending)),
        npeaks=2,
    )
    desc_positions = np.sort(desc_positions)
    desc_positions = _fill_zhang_positions(desc_positions, len(descending))

    metrics = np.concatenate([
        np.where(
            np.isnan(asc_positions),
            np.nan,
            t[np.nan_to_num(asc_positions, nan=0).astype(int)],
        ),
        np.where(
            np.isnan(desc_positions),
            np.nan,
            t[pos_idx + np.nan_to_num(desc_positions, nan=0).astype(int)],
        ),
    ])
    return dict(zip(
        ("GreenUp", "Maturity", "Senescence", "Dormancy"),
        metrics.astype(float),
    ))


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
    need_derivatives = any(
        method.upper() in {"DER", "GU", "ZHANG"}
        for method in methods
    )

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
            curve_fn = CURVE_REGISTRY.get(fit_method)
            derivatives = None
            if curve_fn is not None and need_derivatives:
                try:
                    derivatives = _curve_derivatives(
                        curve_fn,
                        tout,
                        fr.get_param(fit_method),
                    )
                except Exception:
                    derivatives = None

            for pheno_method in methods:
                fn = _PHENO_METHODS.get(pheno_method.upper())
                if fn is None:
                    continue
                try:
                    if derivatives is not None and pheno_method.upper() == "DER":
                        pheno_dates = fn(tout, fitted, der1=derivatives[0])
                    elif derivatives is not None and pheno_method.upper() == "GU":
                        pheno_dates = fn(tout, fitted, der1=derivatives[0])
                    elif derivatives is not None and pheno_method.upper() == "ZHANG":
                        pheno_dates = fn(
                            tout,
                            fitted,
                            der1=derivatives[0],
                            der2=derivatives[1],
                        )
                    else:
                        pheno_dates = fn(tout, fitted)
                    row.update(pheno_dates)
                except Exception:
                    pass

            records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
