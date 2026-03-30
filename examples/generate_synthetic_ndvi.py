"""
Generate realistic synthetic NDVI time series for multiple vegetation scenarios.

Scenarios
---------
1. temperate_forest   – broad summer season, smooth transitions
2. winter_crop        – sharp rise, early senescence, partial cloud cover
3. double_season      – two peaks (Mediterranean / tropical grassland)
4. arid_shrubland     – low-amplitude, noise-dominated, cloud gaps

Outputs
-------
validation/synthetic_ndvi.csv
    Columns: scenario, t, y, w, y_true
    - t      : day-of-year (1–365, 46 evenly-spaced observations)
    - y      : noisy NDVI (cloud-contaminated observations visible)
    - w      : quality weight  (0 = cloud-masked, 1 = clear)
    - y_true : noise-free curve (for reference only)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Ensure output directory exists
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "validation")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "synthetic_ndvi.csv")


# ---------------------------------------------------------------------------
# Curve primitives
# ---------------------------------------------------------------------------

def beck(t: np.ndarray, mn: float, mx: float, sos: float, rsp: float,
         eos: float, rau: float) -> np.ndarray:
    """Beck et al. double-logistic."""
    spring = 1.0 / (1.0 + np.exp(-rsp * (t - sos)))
    autumn = 1.0 / (1.0 + np.exp(rau * (t - eos)))
    return mn + (mx - mn) * (spring + autumn - 1.0)


def double_peak(t: np.ndarray,
                mn: float, mx1: float, mx2: float,
                sos1: float, rsp1: float, eos1: float, rau1: float,
                sos2: float, rsp2: float, eos2: float, rau2: float) -> np.ndarray:
    """Superposition of two Beck curves (double-season)."""
    p1 = beck(t, mn, mx1, sos1, rsp1, eos1, rau1)
    p2 = beck(t, mn, mx2, sos2, rsp2, eos2, rau2)
    return np.maximum(p1, p2)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)
N = 46
T = np.linspace(1, 365, N)


def _make_clouds(prob_cloud: float = 0.15) -> np.ndarray:
    """Return a weight array where cloud pixels have w=0."""
    w = np.ones(N)
    cloud_idx = RNG.random(N) < prob_cloud
    w[cloud_idx] = 0.0
    return w


def scenario_temperate_forest() -> pd.DataFrame:
    """
    Temperate broadleaf forest.
    Broad growing season (Apr–Oct), smooth logistic transitions.
    Typical of temperate deciduous forests in central Europe or eastern USA.
    """
    y_true = beck(T, mn=0.10, mx=0.88, sos=100.0, rsp=0.10,
                  eos=285.0, rau=0.09)
    noise = RNG.normal(0, 0.015, N)
    y = np.clip(y_true + noise, 0.0, 1.0)
    w = _make_clouds(prob_cloud=0.10)
    return pd.DataFrame({"scenario": "temperate_forest", "t": T, "y": y,
                         "w": w, "y_true": y_true})


def scenario_winter_crop() -> pd.DataFrame:
    """
    Winter wheat / rapeseed.
    Fast green-up in early spring, early senescence before July.
    High cloud contamination during sowing (autumn).
    """
    y_true = beck(T, mn=0.08, mx=0.80, sos=65.0, rsp=0.18,
                  eos=175.0, rau=0.20)
    noise = RNG.normal(0, 0.025, N)
    y = np.clip(y_true + noise, 0.0, 1.0)
    # More clouds in autumn–winter (t > 240)
    w = np.ones(N)
    w[T > 240] = RNG.choice([0.0, 1.0], size=np.sum(T > 240),
                             p=[0.30, 0.70])
    return pd.DataFrame({"scenario": "winter_crop", "t": T, "y": y,
                         "w": w, "y_true": y_true})


def scenario_double_season() -> pd.DataFrame:
    """
    Mediterranean / savanna grassland — two distinct green-up periods.
    Spring flush (Mar–May) and autumn regrowth (Oct–Nov) after summer drought.
    """
    y_true = double_peak(
        T, mn=0.06,
        mx1=0.72, sos1=70.0,  rsp1=0.16, eos1=155.0, rau1=0.18,
        mx2=0.55, sos2=260.0, rsp2=0.20, eos2=330.0, rau2=0.22,
    )
    noise = RNG.normal(0, 0.02, N)
    y = np.clip(y_true + noise, 0.0, 1.0)
    w = _make_clouds(prob_cloud=0.12)
    return pd.DataFrame({"scenario": "double_season", "t": T, "y": y,
                         "w": w, "y_true": y_true})


def scenario_arid_shrubland() -> pd.DataFrame:
    """
    Arid shrubland / semi-desert.
    Low-amplitude response, heavy noise, frequent cloud/dust masking.
    Tests robustness of the fitting pipeline.
    """
    y_true = beck(T, mn=0.06, mx=0.38, sos=80.0, rsp=0.08,
                  eos=240.0, rau=0.07)
    noise = RNG.normal(0, 0.030, N)       # proportionally large noise
    contamination = RNG.uniform(-0.05, 0.05, N) * (RNG.random(N) < 0.08)
    y = np.clip(y_true + noise + contamination, 0.0, 1.0)
    w = _make_clouds(prob_cloud=0.20)
    return pd.DataFrame({"scenario": "arid_shrubland", "t": T, "y": y,
                         "w": w, "y_true": y_true})


# ---------------------------------------------------------------------------
# Assemble and save
# ---------------------------------------------------------------------------

def main() -> None:
    parts = [
        scenario_temperate_forest(),
        scenario_winter_crop(),
        scenario_double_season(),
        scenario_arid_shrubland(),
    ]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print(f"Scenarios: {df['scenario'].unique().tolist()}")
    print(f"Observations per scenario: {df.groupby('scenario').size().to_dict()}")


if __name__ == "__main__":
    main()
