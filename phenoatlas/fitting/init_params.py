"""
Parameter initializers and bounds for each double-logistic fitting method.

Port of init_*.R from phenofit (R).

Each initializer returns a dict with:
    'p0'    : np.ndarray — initial parameter guess
    'lower' : np.ndarray — lower bounds
    'upper' : np.ndarray — upper bounds
    'names' : list[str]  — parameter names (for diagnostics)
"""
from __future__ import annotations

import numpy as np


def _season_stats(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Derive basic season statistics used by all initializers.

    Parameters
    ----------
    y, t, w : np.ndarray of shape (n,)

    Returns
    -------
    dict with keys: mn, mx, t_mn, t_mx, t_mid, amplitude, t_min, t_max, n
    """
    # Use weighted observations only
    mask = w > 0
    if mask.sum() < 2:
        mask = np.ones(len(y), dtype=bool)

    y_m = y[mask]
    t_m = t[mask]

    mn = float(np.percentile(y_m, 10))   # background ~ 10th pct
    mx = float(np.percentile(y_m, 90))   # peak ~ 90th pct
    t_mx = float(t_m[np.argmax(y_m)])    # time of approximate peak
    t_mn = float(t_m[np.argmin(y_m)])    # time of approximate minimum

    # Mid-season split
    t_mid = float(0.5 * (t_m[0] + t_m[-1]))

    return {
        "mn": mn,
        "mx": mx,
        "t_mx": t_mx,
        "t_mn": t_mn,
        "t_mid": t_mid,
        "amplitude": max(mx - mn, 0.05),
        "t_min": float(t_m.min()),
        "t_max": float(t_m.max()),
        "n": int(mask.sum()),
    }


def init_Beck(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Beck double-logistic (6 params).

    p = [mn, mx, sos, rsp, eos, rau]
    """
    s = _season_stats(y, t, w)
    t_range = s["t_max"] - s["t_min"]

    sos0 = s["t_min"] + 0.25 * t_range
    eos0 = s["t_min"] + 0.75 * t_range

    p0 = np.array([s["mn"], s["mx"], sos0, 0.1, eos0, 0.1])
    lower = np.array([s["mn"] - 0.2, s["mn"], s["t_min"], 0.001, sos0, 0.001])
    upper = np.array([s["mx"], s["mx"] + 0.2, eos0, 1.0, s["t_max"], 1.0])

    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["mn", "mx", "sos", "rsp", "eos", "rau"],
    }


def init_Elmore(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Elmore double-logistic (7 params).

    p = [mn, mx, sos, rsp, eos, rau, m7]
    """
    base = init_Beck(y, t, w)
    p0 = np.append(base["p0"], 0.0)         # m7 = 0 (no linear trend)
    lower = np.append(base["lower"], -0.01)
    upper = np.append(base["upper"], 0.01)
    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["mn", "mx", "sos", "rsp", "eos", "rau", "m7"],
    }


def init_Zhang(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Zhang piecewise double-logistic (7 params).

    p = [t0, mn, mx, sos, rsp, eos, rau]   — t0 first, matching phenofit R.
    """
    base = init_Beck(y, t, w)
    s = _season_stats(y, t, w)
    t0_0 = s["t_mx"]  # transition near peak

    beck_p0 = base["p0"]    # [mn, mx, sos, rsp, eos, rau]
    beck_lo = base["lower"]
    beck_hi = base["upper"]

    sos0, eos0 = beck_p0[2], beck_p0[4]

    p0 = np.array([t0_0, beck_p0[0], beck_p0[1], beck_p0[2],
                   beck_p0[3], beck_p0[4], beck_p0[5]])
    lower = np.array([sos0, beck_lo[0], beck_lo[1], beck_lo[2],
                      beck_lo[3], beck_lo[4], beck_lo[5]])
    upper = np.array([eos0, beck_hi[0], beck_hi[1], beck_hi[2],
                      beck_hi[3], beck_hi[4], beck_hi[5]])
    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["t0", "mn", "mx", "sos", "rsp", "eos", "rau"],
    }


def init_AG(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Asymmetric Gaussian (7 params).

    p = [t0, mn, mx, rsp, a3, rau, a5]   — rate params, matching phenofit R.
    """
    s = _season_stats(y, t, w)
    t_range = s["t_max"] - s["t_min"]
    rate0 = 4.0 / max(t_range, 1.0)   # approximate rate ~ 4 half-widths per range

    p0 = np.array([
        s["t_mx"],        # t0: peak location
        s["mn"],          # mn: background
        s["mx"],          # mx: maximum
        rate0,            # rsp: spring rate
        2.0,              # a3: spring shape exponent
        rate0,            # rau: autumn rate
        2.0,              # a5: autumn shape exponent
    ])
    lower = np.array([s["t_min"], 0.0,    0.0,    1e-4, 0.5, 1e-4, 0.5])
    upper = np.array([s["t_max"], s["mx"], s["mx"] + 0.2, 1.0, 8.0, 1.0, 8.0])
    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["t0", "mn", "mx", "rsp", "a3", "rau", "a5"],
    }


def init_Gu(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Gu power-law double-logistic (9 params).

    p = [y0, a1, a2, sos, rsp, eos, rau, c1, c2]   — matching phenofit R.
    """
    base = init_Beck(y, t, w)
    s = _season_stats(y, t, w)
    t_range = s["t_max"] - s["t_min"]
    sos0 = s["t_min"] + 0.25 * t_range
    eos0 = s["t_min"] + 0.75 * t_range

    y0_0  = s["mn"]
    a1_0  = s["mx"] - s["mn"]   # spring amplitude
    a2_0  = s["mx"] - s["mn"]   # autumn amplitude (symmetric start)

    p0 = np.array([y0_0, a1_0, a2_0, sos0, 0.1, eos0, 0.1, 1.0, 1.0])
    lower = np.array([0.0, 0.0, 0.0, s["t_min"], 0.001, sos0, 0.001, 0.1, 0.1])
    upper = np.array([s["mx"], 1.5, 1.5, eos0, 1.0, s["t_max"], 1.0, 10.0, 10.0])
    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["y0", "a1", "a2", "sos", "rsp", "eos", "rau", "c1", "c2"],
    }


def init_Klos(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """
    Initial parameters and bounds for Klos polynomial double-logistic (9 params).

    p = [a0, a1, a2, sos, rsp, eos, rau, b1, b2]
    """
    s = _season_stats(y, t, w)
    t_range = s["t_max"] - s["t_min"]
    sos0 = s["t_min"] + 0.25 * t_range
    eos0 = s["t_min"] + 0.75 * t_range

    p0 = np.array([s["mn"], 0.0, 0.0, sos0, 0.1, eos0, 0.1, 1.0, 1.0])
    lower = np.array([0.0, -0.1, -0.01, s["t_min"], 0.001, sos0, 0.001, 0.1, 0.1])
    upper = np.array([1.0, 0.1, 0.01, eos0, 1.0, s["t_max"], 1.0, 10.0, 10.0])
    return {
        "p0": p0,
        "lower": lower,
        "upper": upper,
        "names": ["a0", "a1", "a2", "sos", "rsp", "eos", "rau", "b1", "b2"],
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INIT_REGISTRY: dict[str, callable] = {
    "Beck": init_Beck,
    "Elmore": init_Elmore,
    "Zhang": init_Zhang,
    "AG": init_AG,
    "Gu": init_Gu,
    "Klos": init_Klos,
}


def get_init(method: str) -> callable:
    """
    Look up an initializer function by method name.

    Parameters
    ----------
    method : str
        One of: 'Beck', 'Elmore', 'Zhang', 'AG', 'Gu', 'Klos'.

    Returns
    -------
    callable
        initializer(y, t, w) → dict with p0, lower, upper, names.

    Raises
    ------
    ValueError
    """
    if method not in INIT_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(INIT_REGISTRY)}"
        )
    return INIT_REGISTRY[method]
