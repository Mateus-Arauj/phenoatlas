"""
compare_synthetic.py
====================
Run phenoatlas on the synthetic NDVI CSV, then compare with R (phenofit)
results saved in validation/r_results/.

Usage
-----
    # 1. Generate synthetic data
    python examples/generate_synthetic_ndvi.py

    # 2. Run R reference
    Rscript validation/r/compare_synthetic.R

    # 3. Compare
    python validation/compare_synthetic.py

Outputs
-------
Prints a side-by-side comparison table to stdout and saves:
    validation/comparison_gof.csv
    validation/comparison_pheno.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make sure phenoatlas is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phenoatlas.fitting.curvefit import curvefit
from phenoatlas.phenology.extraction import get_pheno

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.join(os.path.dirname(__file__), "..")
IN_CSV   = os.path.join(ROOT, "validation", "synthetic_ndvi.csv")
R_DIR    = os.path.join(ROOT, "validation", "r_results")
OUT_DIR  = os.path.join(ROOT, "validation")

METHODS  = ["Beck", "Elmore", "Zhang", "AG", "Gu"]


# ---------------------------------------------------------------------------
# Run phenoatlas
# ---------------------------------------------------------------------------

def run_python(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit all scenarios with phenoatlas, return (gof_df, pheno_df).
    """
    gof_rows   = []
    pheno_rows = []

    for scenario, grp in df.groupby("scenario"):
        t = grp["t"].to_numpy()
        y = grp["y"].to_numpy()
        w = grp["w"].to_numpy()

        fit = curvefit(y, t, methods=METHODS, w=w)

        # GOF
        for m in fit.methods:
            g = fit.get_gof(m)
            gof_rows.append({
                "scenario": scenario,
                "meth": m,
                "RMSE": g["RMSE"],
                "R2":   g["R2"],
            })

        # Phenology (Beck only, mirrors R script)
        fit_beck = curvefit(y, t, methods=["Beck"], w=w)
        pheno_df = get_pheno(fit_beck)
        if not pheno_df.empty:
            row = pheno_df.iloc[0].to_dict()
            row["scenario"] = scenario
            pheno_rows.append(row)

    return pd.DataFrame(gof_rows), pd.DataFrame(pheno_rows)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _diff_str(py_val: float, r_val: float) -> str:
    if np.isnan(py_val) or np.isnan(r_val):
        return "  NA"
    diff = abs(py_val - r_val)
    flag = " ✓" if diff < 0.02 else (" !" if diff < 0.05 else " ✗")
    return f"{diff:+.4f}{flag}"


def compare_gof(py_gof: pd.DataFrame, r_gof: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Python and R GOF tables, compute delta.
    """
    py_gof = py_gof.rename(columns={"RMSE": "RMSE_py", "R2": "R2_py"})
    r_gof  = r_gof[["scenario", "meth", "RMSE", "R2"]].rename(
        columns={"RMSE": "RMSE_r", "R2": "R2_r"}
    )
    merged = pd.merge(py_gof, r_gof, on=["scenario", "meth"], how="outer")
    merged["ΔRMSE"] = (merged["RMSE_py"] - merged["RMSE_r"]).abs()
    merged["ΔR2"]   = (merged["R2_py"]   - merged["R2_r"]).abs()
    return merged.sort_values(["scenario", "meth"]).reset_index(drop=True)


def compare_pheno(py_pheno: pd.DataFrame, r_pheno: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Python and R phenology tables, show DOY differences.
    """
    PHENO_COLS = ["TRS2.SOS", "TRS2.EOS", "TRS5.SOS", "TRS5.EOS",
                  "DER.SOS", "DER.POS", "DER.EOS"]

    py_sub = py_pheno[["scenario"] + [c for c in PHENO_COLS if c in py_pheno.columns]]
    r_sub  = r_pheno[ ["scenario"] + [c for c in PHENO_COLS if c in r_pheno.columns]]

    merged = pd.merge(py_sub, r_sub, on="scenario", suffixes=("_py", "_r"), how="outer")

    diff_rows = []
    for _, row in merged.iterrows():
        d = {"scenario": row["scenario"]}
        for col in PHENO_COLS:
            py_v = row.get(f"{col}_py", np.nan)
            r_v  = row.get(f"{col}_r",  np.nan)
            d[f"{col}_py"] = py_v
            d[f"{col}_r"]  = r_v
            try:
                d[f"Δ{col}"] = abs(float(py_v) - float(r_v))
            except (TypeError, ValueError):
                d[f"Δ{col}"] = np.nan
        diff_rows.append(d)

    return pd.DataFrame(diff_rows)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_gof_table(cmp: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("GOODNESS-OF-FIT COMPARISON  (Python phenoatlas  vs  R phenofit)")
    print("=" * 80)
    print(f"{'Scenario':<22} {'Method':<8} {'RMSE_py':>8} {'RMSE_r':>8} "
          f"{'ΔRMSE':>8} {'R2_py':>7} {'R2_r':>7} {'ΔR2':>7}")
    print("-" * 80)
    for _, row in cmp.iterrows():
        rmse_py = f"{row['RMSE_py']:.4f}" if pd.notna(row.get('RMSE_py')) else "   NA"
        rmse_r  = f"{row['RMSE_r']:.4f}"  if pd.notna(row.get('RMSE_r'))  else "   NA"
        drmse   = f"{row['ΔRMSE']:.4f}"   if pd.notna(row.get('ΔRMSE'))   else "   NA"
        r2_py   = f"{row['R2_py']:.4f}"   if pd.notna(row.get('R2_py'))   else "   NA"
        r2_r    = f"{row['R2_r']:.4f}"    if pd.notna(row.get('R2_r'))    else "   NA"
        dr2     = f"{row['ΔR2']:.4f}"     if pd.notna(row.get('ΔR2'))     else "   NA"

        flag = ""
        if pd.notna(row.get('ΔRMSE')) and row['ΔRMSE'] > 0.05:
            flag = " ← large diff"
        print(f"{str(row['scenario']):<22} {str(row['meth']):<8} "
              f"{rmse_py:>8} {rmse_r:>8} {drmse:>8} "
              f"{r2_py:>7} {r2_r:>7} {dr2:>7}{flag}")


def print_pheno_table(cmp: pd.DataFrame) -> None:
    PHENO_COLS = ["TRS2.SOS", "TRS2.EOS", "TRS5.SOS", "TRS5.EOS",
                  "DER.SOS", "DER.POS", "DER.EOS"]

    print("\n" + "=" * 90)
    print("PHENOLOGY DATE COMPARISON  (Python phenoatlas  vs  R phenofit)  — unit: DOY")
    print("Tolerance threshold: ≤5 DOY = ✓,  ≤15 DOY = !,  >15 DOY = ✗")
    print("=" * 90)

    for _, row in cmp.iterrows():
        print(f"\n  Scenario: {row['scenario']}")
        print(f"  {'Phenophase':<14} {'Python':>8} {'R':>8} {'|Δ|':>7}  Status")
        print(f"  {'-'*50}")
        for col in PHENO_COLS:
            py_v = row.get(f"{col}_py", np.nan)
            r_v  = row.get(f"{col}_r",  np.nan)
            delta = row.get(f"Δ{col}", np.nan)
            try:
                py_s  = f"{float(py_v):8.1f}"
                r_s   = f"{float(r_v):8.1f}"
                d_s   = f"{float(delta):7.1f}"
                status = ("✓" if delta <= 5 else "!" if delta <= 15 else "✗") if pd.notna(delta) else "NA"
            except (TypeError, ValueError):
                py_s = r_s = d_s = "      NA"
                status = "NA"
            print(f"  {col:<14} {py_s} {r_s} {d_s}  {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load synthetic data
    if not os.path.exists(IN_CSV):
        print(f"Input not found: {IN_CSV}")
        print("Run:  python examples/generate_synthetic_ndvi.py")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} rows | scenarios: {df['scenario'].unique().tolist()}")

    # 2. Run phenoatlas
    print("\nRunning phenoatlas (Python)...")
    py_gof, py_pheno = run_python(df)
    print(f"  Fitted {len(py_gof)} (scenario × method) combinations")

    # 3. Load R results
    r_gof_path   = os.path.join(R_DIR, "r_gof.csv")
    r_pheno_path = os.path.join(R_DIR, "r_pheno.csv")
    r_available  = os.path.exists(r_gof_path) and os.path.exists(r_pheno_path)

    if not r_available:
        print("\nR results not found. Run:  Rscript validation/r/compare_synthetic.R")
        print("\nPhenology output (Python only):")
        print(py_pheno[["scenario", "TRS2.SOS", "TRS2.EOS",
                         "TRS5.SOS", "TRS5.EOS",
                         "DER.SOS",  "DER.POS", "DER.EOS"]].to_string(index=False))
        print("\nGOF output (Python only):")
        print(py_gof.to_string(index=False))
        return

    r_gof   = pd.read_csv(r_gof_path)
    r_pheno = pd.read_csv(r_pheno_path)

    # 4. Compare
    cmp_gof   = compare_gof(py_gof, r_gof)
    cmp_pheno = compare_pheno(py_pheno, r_pheno)

    print_gof_table(cmp_gof)
    print_pheno_table(cmp_pheno)

    # 5. Save
    cmp_gof.to_csv(os.path.join(OUT_DIR, "comparison_gof.csv"), index=False,
                   float_format="%.4f")
    cmp_pheno.to_csv(os.path.join(OUT_DIR, "comparison_pheno.csv"), index=False,
                     float_format="%.1f")
    print(f"\nSaved: validation/comparison_gof.csv")
    print(f"Saved: validation/comparison_pheno.csv")

    # 6. Summary
    ok_rmse = (cmp_gof["ΔRMSE"] < 0.05).sum()
    total   = cmp_gof["ΔRMSE"].notna().sum()
    ok_doy  = sum(
        1 for col in ["ΔTRS2.SOS", "ΔTRS2.EOS", "ΔTRS5.SOS", "ΔTRS5.EOS",
                      "ΔDER.SOS", "ΔDER.POS", "ΔDER.EOS"]
        for v in cmp_pheno[col] if pd.notna(v) and v <= 5
    )
    total_doy = sum(
        1 for col in ["ΔTRS2.SOS", "ΔTRS2.EOS", "ΔTRS5.SOS", "ΔTRS5.EOS",
                      "ΔDER.SOS", "ΔDER.POS", "ΔDER.EOS"]
        for v in cmp_pheno[col] if pd.notna(v)
    )
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"  GOF: {ok_rmse}/{total} methods within ΔRMSE < 0.05")
    print(f"  DOY: {ok_doy}/{total_doy} phenodates within ±5 days")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
