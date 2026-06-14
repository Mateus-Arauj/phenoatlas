"""
Parameter initializers and bounds for each double-logistic fitting method.

Direct port of phenofit's init_param() and init_*() functions.  Each
initializer exposes both the first starting point as ``p0`` for backward
compatibility and the complete multi-start matrix as ``prior``.
"""
from __future__ import annotations

import numpy as np


def _init_param(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Build the shared statistics used by phenofit's model initializers."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)

    w_min = 0.5
    if np.mean(w >= w_min) < 0.4:
        w_min = 0.0

    valid = (w >= w_min) & np.isfinite(y)
    if not np.any(valid):
        valid = np.isfinite(y)

    mx = float(np.max(y[valid]))
    mn = float(np.min(y[valid]))
    doy_mx = float(t[np.nanargmax(y)])
    doy = np.array([(doy_mx + t[0]) / 2.0, (t[-1] + doy_mx) / 2.0])
    t1 = (doy_mx - doy[0]) / 3.0
    t2 = (doy[1] - doy_mx) / 3.0
    amplitude = mx - mn
    delta_y = amplitude * 0.1
    half = (float(np.max(t)) - float(np.min(t))) / 2.0
    delta_t = half / 4.0
    k = 4.0 / half * 2.67

    limits = {
        "t0": (doy_mx - delta_t, doy_mx + delta_t),
        "mn": (mn - delta_y, mn + delta_y),
        "mx": (mx - 2.0 * delta_y, mx + 2.0 * delta_y),
        "r": (k / 1.2, k * 5.0),
        "sos": (float(np.min(t)), min(doy_mx + delta_t, float(np.max(t)))),
        "eos": (doy_mx - delta_t, float(np.max(t))),
    }
    return {
        "mn": mn,
        "mx": mx,
        "amplitude": amplitude,
        "doy": doy,
        "doy_mx": doy_mx,
        "t1": t1,
        "t2": t2,
        "k": k,
        "half": half,
        "limits": limits,
    }


def _result(
    prior: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    names: list[str],
) -> dict:
    prior = np.asarray(prior, dtype=float)
    if prior.ndim == 1:
        prior = prior[None, :]
    return {
        "p0": prior[0].copy(),
        "prior": prior,
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
        "names": names,
    }


def _bounds(e: dict, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([e["limits"][key][0] for key in keys], dtype=float)
    upper = np.array([e["limits"][key][1] for key in keys], dtype=float)
    return lower, upper


def init_Beck(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for Beck: [mn, mx, sos, rsp, eos, rau]."""
    e = _init_param(y, t, w)
    mn, mx, doy, k = e["mn"], e["mx"], e["doy"], e["k"]
    prior = np.array([
        [mn, mx, doy[0], k, doy[1], k],
        [mn, mx, doy[0] + e["t1"], 2 * k, doy[1] - e["t2"], 2 * k],
    ])
    lower, upper = _bounds(e, ["mn", "mx", "sos", "r", "eos", "r"])
    return _result(
        prior, lower, upper, ["mn", "mx", "sos", "rsp", "eos", "rau"]
    )


def init_Elmore(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for Elmore: [mn, mx, sos, rsp, eos, rau, m7]."""
    e = _init_param(y, t, w)
    mn, mx, doy, k = e["mn"], e["mx"], e["doy"], e["k"]
    prior = np.array([
        [
            mn, mx - mn, doy[0] + e["t1"], 2.5 * k,
            doy[1] - e["t2"], 2.5 * k, 0.002,
        ],
        [
            mn, mx - mn, doy[0] - e["t1"], 0.25 * k,
            doy[1] + e["t2"], 0.25 * k, 0.001,
        ],
    ])
    lower, upper = _bounds(e, ["mn", "mx", "sos", "r", "eos", "r"])
    return _result(
        prior,
        np.append(lower, 0.0),
        np.append(upper, 1.0),
        ["mn", "mx", "sos", "rsp", "eos", "rau", "m7"],
    )


def init_Zhang(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for Zhang: [t0, mn, mx, sos, rsp, eos, rau]."""
    e = _init_param(y, t, w)
    mn, mx, doy, k, peak = (
        e["mn"], e["mx"], e["doy"], e["k"], e["doy_mx"]
    )
    prior = np.array([
        [peak, mn, mx, doy[0], k, doy[1], k],
        [
            peak, mn, mx, doy[0] + e["t1"], 2 * k,
            doy[1] - e["t1"], 2 * k,
        ],
        [peak, mn, mx, doy[0] - e["t1"], k, doy[1] + e["t2"], k],
    ])
    lower, upper = _bounds(
        e, ["t0", "mn", "mx", "sos", "r", "eos", "r"]
    )
    return _result(
        prior,
        lower,
        upper,
        ["t0", "mn", "mx", "sos", "rsp", "eos", "rau"],
    )


def init_AG(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for AG: [t0, mn, mx, rsp, a3, rau, a5]."""
    e = _init_param(y, t, w)
    peak, mn, mx, half = e["doy_mx"], e["mn"], e["mx"], e["half"]
    prior = np.array([
        [peak, mn, mx, 1 / half, 2, 1 / half, 2],
        [peak, mn, mx, 1 / (0.8 * half), 3, 1 / (0.8 * half), 3],
    ])
    limits = e["limits"]
    lower = np.array([
        limits["t0"][0], limits["mn"][0], limits["mx"][0],
        1 / (1.4 * half), 2, 1 / (1.4 * half), 2,
    ])
    upper = np.array([
        limits["t0"][1], limits["mn"][1], limits["mx"][1],
        1 / (0.1 * half), 6, 1 / (0.1 * half), 6,
    ])
    return _result(
        prior, lower, upper, ["t0", "mn", "mx", "rsp", "a3", "rau", "a5"]
    )


def init_Gu(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for Gu: [y0, a1, a2, sos, rsp, eos, rau, c1, c2]."""
    e = _init_param(y, t, w)
    mn, a, doy, k = e["mn"], e["amplitude"], e["doy"], e["k"]
    prior = np.array([
        [mn, a, a, doy[0], k, doy[1], k, 2, 2],
        [
            mn, a, a, doy[0] + e["t1"], 2 * k,
            doy[1] - e["t2"], 2 * k, 0.5, 0.5,
        ],
        [
            mn, a, a, doy[0] + e["t1"], 3 * k,
            doy[1] - e["t2"], 3 * k, 5, 5,
        ],
    ])
    lower, upper = _bounds(
        e, ["mn", "mx", "mx", "sos", "r", "eos", "r"]
    )
    return _result(
        prior,
        np.append(lower, [0.0, 0.0]),
        np.append(upper, [np.inf, np.inf]),
        ["y0", "a1", "a2", "sos", "rsp", "eos", "rau", "c1", "c2"],
    )


def init_Klos(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> dict:
    """Initial values for phenofit's 13-parameter Klos curve."""
    e = _init_param(y, t, w)
    doy, peak = e["doy"], e["doy_mx"]
    b1, b2, c = e["mn"], 0.0, 0.2 * e["mx"]
    rate1 = 4.0 / (peak - doy[0])
    rate2 = 3.2 / (doy[1] - peak)
    m1 = doy[0] + 0.5 * (peak - doy[0])
    m2 = peak + 0.5 * (doy[1] - peak)
    common = [0.0, 0.0, b1, b2, c, rate1, rate2]
    prior = np.array([
        common + [m1, m2, 0.5, 0.5, 2, 2],
        [0, 0, b1, 0.01, 0, rate1, rate2, m1, doy[1], 0.5, 1, 2, 4],
        common + [doy[0], m2, 0.5, 0.5, 2, 2],
        common + [m1, doy[1], 0.5, 0.5, 2, 2],
        common + [doy[0], m2, 0.5, 0.5, 2, 2],
    ])
    names = [
        "a1", "a2", "b1", "b2", "c", "B1", "B2",
        "m1", "m2", "q1", "q2", "v1", "v2",
    ]
    return _result(
        prior,
        np.full(13, -np.inf),
        np.full(13, np.inf),
        names,
    )


INIT_REGISTRY: dict[str, callable] = {
    "Beck": init_Beck,
    "Elmore": init_Elmore,
    "Zhang": init_Zhang,
    "AG": init_AG,
    "Gu": init_Gu,
    "Klos": init_Klos,
}


def get_init(method: str) -> callable:
    """Look up an initializer by phenofit method name."""
    if method not in INIT_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(INIT_REGISTRY)}"
        )
    return INIT_REGISTRY[method]
