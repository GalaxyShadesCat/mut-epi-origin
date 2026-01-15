#!/usr/bin/env python3
"""
analyse_mut_vs_dnase_results.py

Goal
----
Given that "mela" is the correct cell type, find the configuration that:
1) uses the smallest sample_size_k, and
2) makes mela the clearest winner (largest margin vs second best),
using either:
- pearson: best_celltype_linear_resid (Pearson on linear residuals)
- rf:      best_celltype_rf_resid     (Pearson on RF residual target)

Notebook usage
--------------
from scripts.analyse_mut_vs_dnase_results import analyse_results
out = analyse_results(".../results.csv", mode="auto")
out["best"]  # pandas Series of the recommended row
out["k_summary"]  # DataFrame summarising mela wins by k for each metric

CLI usage
---------
python scripts/analyse_mut_vs_dnase_results.py --results-csv path/to/results.csv --mode auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd

Mode = Literal["auto", "pearson", "rf"]

MODE_TO_SUFFIX = {
    "pearson": "linear_resid",  # “pearson” means the linear-residualised Pearson track comparison
    "rf": "rf_resid",
}


def _safe_json_load(s: Any) -> Dict[str, float]:
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return {}
    if not isinstance(d, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _extract_dnase_perm(importances_json: Any) -> Dict[str, float]:
    d = _safe_json_load(importances_json)
    return {k: v for k, v in d.items() if k.startswith("dnase_") and np.isfinite(v)}


def _top_dnase_feature(importances_json: Any) -> Tuple[Optional[str], float]:
    dn = _extract_dnase_perm(importances_json)
    if not dn:
        return None, float("nan")
    k = max(dn, key=lambda kk: dn[kk])
    return k.replace("dnase_", ""), float(dn[k])


def _cols_for(mode: Mode) -> Tuple[str, str, str]:
    """
    Return (best_celltype_col, best_value_col, best_margin_col).
    """
    if mode == "auto":
        raise ValueError("auto has no direct columns")
    suf = MODE_TO_SUFFIX[mode]
    return f"best_celltype_{suf}", f"best_celltype_{suf}_value", f"best_minus_second_{suf}"


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalise sample size
    df["sample_size_k_num"] = pd.to_numeric(df.get("sample_size_k", np.nan), errors="coerce")

    # normalise winner cols (lowercase)
    for mode in ("pearson", "rf"):
        best_cell, _, _ = _cols_for(mode)  # type: ignore[arg-type]
        if best_cell in df.columns:
            df[best_cell] = df[best_cell].fillna("").astype(str).str.strip().str.lower()

    # parse RF permutation importances (optional but useful)
    df["dnase_top_feature"] = None
    df["dnase_top_importance"] = np.nan
    df["dnase_mela_importance"] = np.nan

    if "rf_perm_importances_mean_json" in df.columns:
        for i in range(len(df)):
            top_ct, top_val = _top_dnase_feature(df.loc[i, "rf_perm_importances_mean_json"])
            df.loc[i, "dnase_top_feature"] = top_ct
            df.loc[i, "dnase_top_importance"] = top_val

            dn = _extract_dnase_perm(df.loc[i, "rf_perm_importances_mean_json"])
            if "dnase_mela" in dn:
                df.loc[i, "dnase_mela_importance"] = float(dn["dnase_mela"])

    return df


def k_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise mela win rate by k for pearson and rf.
    """
    out = []
    d = df.dropna(subset=["sample_size_k_num"]).copy()

    for mode in ("pearson", "rf"):
        best_cell, best_val, best_margin = _cols_for(mode)  # type: ignore[arg-type]
        if best_cell not in d.columns:
            continue
        for k, sub in d.groupby("sample_size_k_num"):
            wins = (sub[best_cell] == "mela").sum()
            out.append(
                {
                    "mode": mode,
                    "k": int(k),
                    "n_runs": int(len(sub)),
                    "mela_wins": int(wins),
                    "mela_win_rate": float(wins / len(sub)) if len(sub) else float("nan"),
                    "best_mela_margin_max": float(sub.loc[sub[best_cell] == "mela", best_margin].max())
                    if wins else float("nan"),
                    "best_mela_value_min": float(sub.loc[sub[best_cell] == "mela", best_val].min())
                    if wins else float("nan"),
                }
            )

    return pd.DataFrame(out).sort_values(["mode", "k"]).reset_index(drop=True)


def _best_row_for_mode(
        df: pd.DataFrame,
        mode: Mode,
        require_mela_top_perm: bool = False,
) -> pd.Series:
    """
    Pick the best configuration for a given mode:
    - restrict to mela-winning rows
    - choose smallest k
    - within that k: maximise margin, then minimise value (more negative), then maximise dnase_mela_importance
    """
    if mode == "auto":
        raise ValueError("Use analyse_results(mode='auto') instead.")

    best_cell, best_val, best_margin = _cols_for(mode)

    d = df.dropna(subset=["sample_size_k_num"]).copy()
    if best_cell not in d.columns:
        raise ValueError(f"Missing required column: {best_cell}")

    d = d.loc[d[best_cell] == "mela"].copy()
    if require_mela_top_perm:
        d = d.loc[d["dnase_top_feature"].astype(str).str.lower() == "mela"].copy()

    if d.empty:
        raise ValueError(f"No mela-winning rows found for mode={mode} (with current constraints).")

    min_k = float(d["sample_size_k_num"].min())
    d = d.loc[d["sample_size_k_num"] == min_k].copy()

    d["_rank_margin"] = pd.to_numeric(d.get(best_margin, np.nan), errors="coerce")
    d["_rank_value"] = pd.to_numeric(d.get(best_val, np.nan), errors="coerce")
    d["_rank_perm"] = pd.to_numeric(d.get("dnase_mela_importance", np.nan), errors="coerce")

    d = d.sort_values(
        by=["_rank_margin", "_rank_value", "_rank_perm"],
        ascending=[False, True, False],
        na_position="last",
    )
    return d.iloc[0]


def analyse_results(
        results_csv: str | Path,
        mode: Mode = "auto",
        require_mela_top_perm: bool = False,
) -> Dict[str, Any]:
    """
    Notebook-friendly entrypoint.

    mode:
      - "pearson": use best_celltype_linear_resid
      - "rf":      use best_celltype_rf_resid
      - "auto":    choose between pearson and rf by:
            1) smallest achievable k where mela wins for that mode
            2) if tied on k, choose the mode whose best row at that k has the larger margin

    Returns dict:
      - df: full results DataFrame
      - k_summary: summary table by k and mode
      - chosen_mode: "pearson" or "rf"
      - best: pandas Series row for the recommended configuration
      - best_cols: dict of the key columns used for that mode
    """
    df = load_results(results_csv)
    ks = k_summary(df)

    if mode != "auto":
        best = _best_row_for_mode(df, mode, require_mela_top_perm=require_mela_top_perm)
        best_cell, best_val, best_margin = _cols_for(mode)
        return {
            "df": df,
            "k_summary": ks,
            "chosen_mode": mode,
            "best": best,
            "best_cols": {"best_cell": best_cell, "best_value": best_val, "best_margin": best_margin},
        }

    # auto: compare pearson vs rf using smallest k, then margin at that k
    candidates = {}
    for m in ("pearson", "rf"):
        try:
            row = _best_row_for_mode(df, m, require_mela_top_perm=require_mela_top_perm)  # type: ignore[arg-type]
        except Exception:
            continue
        _, _, best_margin = _cols_for(m)  # type: ignore[arg-type]
        margin = float(pd.to_numeric(row.get(best_margin, np.nan), errors="coerce"))
        k = float(row["sample_size_k_num"])
        candidates[m] = (k, margin, row)

    if not candidates:
        raise ValueError("auto mode found no mela-winning rows for either pearson or rf.")

    # primary: minimise k, secondary: maximise margin
    chosen_mode = sorted(candidates.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0][0]
    best = candidates[chosen_mode][2]
    best_cell, best_val, best_margin = _cols_for(chosen_mode)  # type: ignore[arg-type]

    return {
        "df": df,
        "k_summary": ks,
        "chosen_mode": chosen_mode,
        "best": best,
        "best_cols": {"best_cell": best_cell, "best_value": best_val, "best_margin": best_margin},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "pearson", "rf"], default="auto")
    ap.add_argument("--require-mela-top-perm", action="store_true")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    out = analyse_results(args.results_csv, mode=args.mode, require_mela_top_perm=args.require_mela_top_perm)
    df: pd.DataFrame = out["df"]
    ks: pd.DataFrame = out["k_summary"]
    best: pd.Series = out["best"]
    chosen_mode: str = out["chosen_mode"]
    cols: Dict[str, str] = out["best_cols"]

    print(f"Loaded {len(df):,} runs from: {args.results_csv}")
    print("\n=== mela win summary by k (pearson vs rf) ===")
    print(ks.to_string(index=False))

    print("\n=== recommended configuration ===")
    print(f"chosen_mode: {chosen_mode}")
    print(f"min_k: {int(best['sample_size_k_num'])}")
    print(f"run_id: {best.get('run_id', '')}")
    print(f"track_strategy: {best.get('track_strategy', '')}")
    print(f"bin_size: {best.get('bin_size', '')}")
    print(f"covariates: {best.get('covariates', '')}")
    print(f"{cols['best_cell']}: {best.get(cols['best_cell'], '')}")
    print(f"{cols['best_value']}: {best.get(cols['best_value'], '')}")
    print(f"{cols['best_margin']}: {best.get(cols['best_margin'], '')}")
    print(f"dnase_top_feature (perm): {best.get('dnase_top_feature', '')}")
    print(f"dnase_mela_importance (perm): {best.get('dnase_mela_importance', np.nan)}")

    # show top-N mela-winning rows at that k for the chosen mode
    best_cell = cols["best_cell"]
    best_val = cols["best_value"]
    best_margin = cols["best_margin"]

    d = df.dropna(subset=["sample_size_k_num"]).copy()
    d = d.loc[d["sample_size_k_num"] == best["sample_size_k_num"]].copy()
    d = d.loc[d[best_cell] == "mela"].copy()
    if args.require_mela_top_perm:
        d = d.loc[d["dnase_top_feature"].astype(str).str.lower() == "mela"].copy()

    d["_rank_margin"] = pd.to_numeric(d.get(best_margin, np.nan), errors="coerce")
    d["_rank_value"] = pd.to_numeric(d.get(best_val, np.nan), errors="coerce")
    d["_rank_perm"] = pd.to_numeric(d.get("dnase_mela_importance", np.nan), errors="coerce")
    d = d.sort_values(by=["_rank_margin", "_rank_value", "_rank_perm"], ascending=[False, True, False])

    show_cols = [
        "sample_size_k_num",
        "bin_size",
        "track_strategy",
        "covariates",
        best_val,
        best_margin,
        "dnase_top_feature",
        "dnase_mela_importance",
        "run_id",
    ]
    show_cols = [c for c in show_cols if c in d.columns]

    print(f"\n=== top {args.top_n} mela-winning runs at k={int(best['sample_size_k_num'])} ({chosen_mode}) ===")
    print(d[show_cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
