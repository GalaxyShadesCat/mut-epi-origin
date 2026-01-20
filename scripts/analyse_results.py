"""
analyse_mut_vs_dnase_results.py

Goal
----
Given that "mela" is the correct cell type, analyse a results.csv produced by
grid_search.py and report:

1) The recommended "best" configuration (smallest k that still makes mela win clearly),
2) The best configuration *per sample size (k)*,
3) The best configuration *per track strategy* (and optionally per k),
4) Descriptive smmaries.

Modes
-----
- pearson: uses best_celltype_linear_resid (Pearson on linear-residualised tracks)
- rf:      uses best_celltype_rf_resid     (Pearson on RF-residualised DNase target)
- raw:     uses best_celltype_raw          (Pearson on raw tracks)  [optional]

Notebook usage
--------------
from scripts.analyse_mut_vs_dnase_results import (
    analyse_results,
    describe_recommendation,
    describe_k_breakdown,
    describe_track_breakdown,
    describe_metric_margin_summary,
    describe_downsample_accuracy,
    best_by_k,
    best_by_track,
)

out = analyse_results(".../results.csv", mode="auto")
text = describe_recommendation(out)          # a readable narrative string
text_k = describe_k_breakdown(out)           # narrative for best-by-k
text_track = describe_track_breakdown(out)   # narrative for best-by-track
text_margins = describe_metric_margin_summary(out)
text_acc = describe_downsample_accuracy(out)

# Access dataframes too:
out["k_summary"]          # mela win summary by k/mode
out["best_overall"]       # pandas Series for global recommendation
out["best_by_k_df"]       # DataFrame: best row per k
out["best_by_track_df"]   # DataFrame: best row per track strategy
out["best_by_k_track_df"] # DataFrame: best row per (k, track_strategy)
out["metric_margin_summary_df"] # Weighted margin summary for correctly classified rows
out["downsample_accuracy_overall_df"] # Accuracy by downsample target + metric
out["downsample_accuracy_track_df"]   # Accuracy by downsample target + metric + track strategy

CLI usage
---------
python scripts/analyse_mut_vs_dnase_results.py --results-csv path/to/results.csv --mode auto

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal, List

import numpy as np
import pandas as pd

Mode = Literal["auto", "pearson", "rf", "raw", "local_score"]

MODE_TO_SUFFIX = {
    "pearson": "linear_resid",
    "rf": "rf_resid",
    "raw": "raw",
    "local_score": "local_score",
}


# -------------------------
# JSON helpers
# -------------------------
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


# -------------------------
# Column mapping
# -------------------------
def _cols_for(mode: Mode) -> Tuple[str, str, str]:
    """
    Return (best_celltype_col, best_value_col, best_margin_col).
    """
    if mode == "auto":
        raise ValueError("auto has no direct columns")
    suf = MODE_TO_SUFFIX[mode]
    return f"best_celltype_{suf}", f"best_celltype_{suf}_value", f"best_minus_second_{suf}"


def _metric_modes() -> List[Mode]:
    return ["raw", "pearson", "rf", "local_score"]


def _normalise_str_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()


# -------------------------
# Loading
# -------------------------
def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # normalise sample size
    df["sample_size_k_num"] = pd.to_numeric(df.get("sample_size_k", np.nan), errors="coerce")

    # normalise track strategy + track param tag
    if "track_strategy" in df.columns:
        df["track_strategy"] = df["track_strategy"].fillna("").astype(str).str.strip()
    if "track_param_tag" in df.columns:
        df["track_param_tag"] = df["track_param_tag"].fillna("").astype(str).str.strip()

    # normalise winner cols (lowercase)
    for m in ("pearson", "rf", "raw", "local_score"):
        best_cell, _, _ = _cols_for(m)  # type: ignore[arg-type]
        _normalise_str_col(df, best_cell)

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


# -------------------------
# Summaries
# -------------------------
def k_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise mela win rate by k for each mode.
    """
    out = []
    d = df.dropna(subset=["sample_size_k_num"]).copy()

    for mode in _metric_modes():
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


def _rank_and_pick(
        d: pd.DataFrame,
        *,
        best_margin_col: str,
        best_value_col: str,
) -> pd.DataFrame:
    """
    Add ranking columns and sort so the top row is the best.
    Rule within the filtered set:
      1) maximise margin (clearer winner)
      2) minimise value (more negative association)
      3) maximise dnase_mela_importance (if present)
    """
    dd = d.copy()
    dd["_rank_margin"] = pd.to_numeric(dd.get(best_margin_col, np.nan), errors="coerce")
    dd["_rank_value"] = pd.to_numeric(dd.get(best_value_col, np.nan), errors="coerce")
    dd["_rank_perm"] = pd.to_numeric(dd.get("dnase_mela_importance", np.nan), errors="coerce")

    dd = dd.sort_values(
        by=["_rank_margin", "_rank_value", "_rank_perm"],
        ascending=[False, True, False],
        na_position="last",
    )
    return dd


def _best_rows_grouped(
        df: pd.DataFrame,
        *,
        group_cols: List[str],
        mode: Mode,
        require_mela_top_perm: bool,
) -> pd.DataFrame:
    """
    For each group (e.g. k, or track_strategy), return the single best row where mela wins.
    """
    if mode == "auto":
        raise ValueError("mode must be explicit for grouped best rows (pearson/rf/raw).")

    best_cell, best_val, best_margin = _cols_for(mode)

    d = df.dropna(subset=["sample_size_k_num"]).copy()
    if best_cell not in d.columns:
        raise ValueError(f"Missing required column: {best_cell}")

    d = d.loc[d[best_cell] == "mela"].copy()
    if require_mela_top_perm and "dnase_top_feature" in d.columns:
        d = d.loc[d["dnase_top_feature"].astype(str).str.lower() == "mela"].copy()

    if d.empty:
        return pd.DataFrame(columns=df.columns)

    out_rows = []
    for _, sub in d.groupby(group_cols, dropna=False):
        ranked = _rank_and_pick(sub, best_margin_col=best_margin, best_value_col=best_val)
        out_rows.append(ranked.iloc[0])

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    # keep a clean ordering
    sort_cols = [c for c in group_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def best_by_k(df: pd.DataFrame, mode: Mode, require_mela_top_perm: bool = False) -> pd.DataFrame:
    """
    Best mela-winning configuration per sample size k.
    """
    return _best_rows_grouped(
        df,
        group_cols=["sample_size_k_num"],
        mode=mode,
        require_mela_top_perm=require_mela_top_perm,
    )


def best_by_track(df: pd.DataFrame, mode: Mode, require_mela_top_perm: bool = False) -> pd.DataFrame:
    """
    Best mela-winning configuration per track_strategy (aggregating across all k).
    """
    return _best_rows_grouped(
        df,
        group_cols=["track_strategy"],
        mode=mode,
        require_mela_top_perm=require_mela_top_perm,
    )


def best_by_k_and_track(df: pd.DataFrame, mode: Mode, require_mela_top_perm: bool = False) -> pd.DataFrame:
    """
    Best mela-winning configuration per (k, track_strategy).
    """
    return _best_rows_grouped(
        df,
        group_cols=["sample_size_k_num", "track_strategy"],
        mode=mode,
        require_mela_top_perm=require_mela_top_perm,
    )


# -------------------------
# Choosing "best overall"
# -------------------------
def _best_row_for_mode(
        df: pd.DataFrame,
        mode: Mode,
        require_mela_top_perm: bool = False,
) -> pd.Series:
    """
    Pick the best configuration for a given mode:
      - restrict to mela-winning rows
      - choose smallest k
      - within that k: maximise margin, then minimise value, then maximise dnase_mela_importance
    """
    if mode == "auto":
        raise ValueError("Use analyse_results(mode='auto') instead.")
    best_cell, best_val, best_margin = _cols_for(mode)

    d = df.dropna(subset=["sample_size_k_num"]).copy()
    if best_cell not in d.columns:
        raise ValueError(f"Missing required column: {best_cell}")

    d = d.loc[d[best_cell] == "mela"].copy()
    if require_mela_top_perm and "dnase_top_feature" in d.columns:
        d = d.loc[d["dnase_top_feature"].astype(str).str.lower() == "mela"].copy()

    if d.empty:
        raise ValueError(f"No mela-winning rows found for mode={mode} (with current constraints).")

    min_k = float(d["sample_size_k_num"].min())
    d = d.loc[d["sample_size_k_num"] == min_k].copy()

    d = _rank_and_pick(d, best_margin_col=best_margin, best_value_col=best_val)
    return d.iloc[0]


def analyse_results(
        results_csv: str | Path,
        mode: Mode = "auto",
        require_mela_top_perm: bool = False,
) -> Dict[str, Any]:
    """
    Main notebook-friendly entrypoint.

    Returns dict:
      - df: full results DataFrame
      - k_summary: mela win summary by k/mode
      - chosen_mode: "pearson" or "rf" or "raw" (if mode != auto)
      - best_overall: pandas Series for the recommended configuration
      - best_cols: dict of key columns used for that mode
      - best_by_k_df: DataFrame best row per k (for chosen_mode)
      - best_by_track_df: DataFrame best row per track_strategy (for chosen_mode)
      - best_by_k_track_df: DataFrame best row per (k, track_strategy) (for chosen_mode)
      - metric_margin_summary_df: DataFrame of weighted margin summaries for mela-correct rows
      - downsample_accuracy_overall_df: DataFrame of accuracy by downsample target + metric
      - downsample_accuracy_track_df: DataFrame of accuracy by downsample target + metric + track strategy
    """
    df = load_results(results_csv)
    ks = k_summary(df)

    metric_margin_df = metric_margin_summary(df, weight_mode="metric_abs")
    downsample_overall_df, downsample_track_df = downsample_accuracy(df)

    if mode != "auto":
        best = _best_row_for_mode(df, mode, require_mela_top_perm=require_mela_top_perm)
        best_cell, best_val, best_margin = _cols_for(mode)
        return {
            "df": df,
            "k_summary": ks,
            "chosen_mode": mode,
            "best_overall": best,
            "best_cols": {"best_cell": best_cell, "best_value": best_val, "best_margin": best_margin},
            "best_by_k_df": best_by_k(df, mode, require_mela_top_perm=require_mela_top_perm),
            "best_by_track_df": best_by_track(df, mode, require_mela_top_perm=require_mela_top_perm),
            "best_by_k_track_df": best_by_k_and_track(df, mode, require_mela_top_perm=require_mela_top_perm),
            "metric_margin_summary_df": metric_margin_df,
            "downsample_accuracy_overall_df": downsample_overall_df,
            "downsample_accuracy_track_df": downsample_track_df,
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

    chosen_mode = sorted(candidates.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0][0]
    best = candidates[chosen_mode][2]
    best_cell, best_val, best_margin = _cols_for(chosen_mode)  # type: ignore[arg-type]

    return {
        "df": df,
        "k_summary": ks,
        "chosen_mode": chosen_mode,
        "best_overall": best,
        "best_cols": {"best_cell": best_cell, "best_value": best_val, "best_margin": best_margin},
        "best_by_k_df": best_by_k(df, chosen_mode, require_mela_top_perm=require_mela_top_perm),
        # type: ignore[arg-type]
        "best_by_track_df": best_by_track(df, chosen_mode, require_mela_top_perm=require_mela_top_perm),
        # type: ignore[arg-type]
        "best_by_k_track_df": best_by_k_and_track(df, chosen_mode, require_mela_top_perm=require_mela_top_perm),
        # type: ignore[arg-type]
        "metric_margin_summary_df": metric_margin_df,
        "downsample_accuracy_overall_df": downsample_overall_df,
        "downsample_accuracy_track_df": downsample_track_df,
    }


# -------------------------
# Notebook-friendly descriptions (return strings)
# -------------------------
def _fmt_float(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _row_track_label(row: pd.Series) -> str:
    ts = str(row.get("track_strategy", ""))
    tag = str(row.get("track_param_tag", "")).strip()
    if tag and tag.lower() != "raw":
        return f"{ts} ({tag})"
    return ts


def _row_track_bin(row: pd.Series) -> Any:
    ts = str(row.get("track_strategy", ""))
    return row.get(f"{ts}_bin", "NA")


def _row_track_params(row: pd.Series) -> List[str]:
    ts = str(row.get("track_strategy", ""))
    if ts == "counts_gauss":
        return [
            f"counts_gauss_sigma_bins={row.get('counts_gauss_sigma_bins', 'NA')}",
            f"counts_gauss_sigma_units={row.get('counts_gauss_sigma_units', 'NA')}",
        ]
    if ts == "inv_dist_gauss":
        return [
            f"inv_dist_gauss_sigma_bins={row.get('inv_dist_gauss_sigma_bins', 'NA')}",
            f"inv_dist_gauss_max_distance_bp={row.get('inv_dist_gauss_max_distance_bp', 'NA')}",
            f"inv_dist_gauss_sigma_units={row.get('inv_dist_gauss_sigma_units', 'NA')}",
        ]
    if ts == "exp_decay":
        return [
            f"exp_decay_decay_bp={row.get('exp_decay_decay_bp', 'NA')}",
            f"exp_decay_max_distance_bp={row.get('exp_decay_max_distance_bp', 'NA')}",
        ]
    if ts == "exp_decay_adaptive":
        return [
            f"exp_decay_adaptive_k={row.get('exp_decay_adaptive_k', 'NA')}",
            f"exp_decay_adaptive_min_bandwidth_bp={row.get('exp_decay_adaptive_min_bandwidth_bp', 'NA')}",
            f"exp_decay_adaptive_max_distance_bp={row.get('exp_decay_adaptive_max_distance_bp', 'NA')}",
        ]
    return []


def metric_margin_summary(
        df: pd.DataFrame,
        *,
        weight_mode: str = "metric_abs",
) -> pd.DataFrame:
    """
    Weighted average of best_minus_second margins for mela-correct rows.
    weight_mode:
      - metric_abs: abs(best_value) for each metric (raw weighted by raw, local by local, etc.)
      - n_bins_total: weight by n_bins_total
      - none: unweighted
    """
    weight_mode = weight_mode.strip().lower()

    rows: List[Dict[str, Any]] = []
    for mode in _metric_modes():
        best_cell, best_value, best_margin = _cols_for(mode)
        if best_cell not in df.columns or best_margin not in df.columns:
            continue
        if weight_mode == "metric_abs":
            weights = pd.to_numeric(df.get(best_value, np.nan), errors="coerce").abs().fillna(0.0)
        elif weight_mode == "n_bins_total":
            weights = pd.to_numeric(df.get("n_bins_total", np.nan), errors="coerce").fillna(0.0)
        else:
            weights = pd.Series(1.0, index=df.index)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=df.index)
        winners = df[best_cell].astype(str).str.lower() == "mela"
        margins = pd.to_numeric(df[best_margin], errors="coerce")
        valid = winners & margins.notna()
        if not valid.any():
            avg_weighted = float("nan")
            avg_unweighted = float("nan")
            n_correct = 0
        else:
            w = weights[valid]
            avg_weighted = float((margins[valid] * w).sum() / w.sum()) if w.sum() else float("nan")
            avg_unweighted = float(margins[valid].mean())
            n_correct = int(valid.sum())
        rows.append(
            {
                "metric": mode,
                "n_correct": n_correct,
                "avg_margin_weighted": avg_weighted,
                "avg_margin_unweighted": avg_unweighted,
                "weight_basis": weight_mode,
            }
        )

    return pd.DataFrame(rows)


def downsample_accuracy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accuracy summaries for downsampled runs by metric.
    Returns (overall_df, track_df).
    """
    if "downsample_target" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    d = df.copy()
    d["downsample_target_num"] = pd.to_numeric(d["downsample_target"], errors="coerce")
    if d["downsample_target_num"].notna().sum() == 0:
        return pd.DataFrame(), pd.DataFrame()

    overall_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    for mode in _metric_modes():
        best_cell, _, _ = _cols_for(mode)
        if best_cell not in d.columns:
            continue
        d["is_correct"] = d[best_cell].astype(str).str.lower() == "mela"
        for downsample_target, sub in d.groupby("downsample_target_num", dropna=False):
            n_runs = int(len(sub))
            n_correct = int(sub["is_correct"].sum())
            overall_rows.append(
                {
                    "downsample_target": int(downsample_target),
                    "metric": mode,
                    "n_runs": n_runs,
                    "n_correct": n_correct,
                    "accuracy": float(n_correct / n_runs) if n_runs else float("nan"),
                }
            )
        if "track_strategy" in d.columns:
            for (downsample_target, track_strategy), sub in d.groupby(
                ["downsample_target_num", "track_strategy"], dropna=False
            ):
                n_runs = int(len(sub))
                n_correct = int(sub["is_correct"].sum())
                track_rows.append(
                    {
                        "downsample_target": int(downsample_target),
                        "track_strategy": str(track_strategy),
                        "metric": mode,
                        "n_runs": n_runs,
                        "n_correct": n_correct,
                        "accuracy": float(n_correct / n_runs) if n_runs else float("nan"),
                    }
                )

    overall_df = pd.DataFrame(overall_rows).sort_values(
        ["downsample_target", "metric"]
    ).reset_index(drop=True)
    track_df = pd.DataFrame(track_rows).sort_values(
        ["downsample_target", "metric", "track_strategy"]
    ).reset_index(drop=True)
    return overall_df, track_df


def describe_recommendation(out: Dict[str, Any]) -> str:
    """
    Human-readable narrative for the recommended configuration.
    """
    best: pd.Series = out["best_overall"]
    cols: Dict[str, str] = out["best_cols"]
    chosen_mode: str = out["chosen_mode"]

    lines = []
    lines.append("Recommended configuration (given mela is correct)")
    lines.append("-----------------------------------------------")
    lines.append(f"Mode used to judge winners: {chosen_mode}")
    lines.append(f"Smallest k where mela wins (under this mode): {int(best.get('sample_size_k_num', -1))}")
    lines.append(f"Track: {_row_track_label(best)}")
    lines.append(f"Bin size: {_row_track_bin(best)}")
    lines.append(f"Covariates: {best.get('covariates', '')}")
    lines.append(f"Tuned params:")
    params = _row_track_params(best)
    if params:
        for item in params:
            lines.append(f"  {item}")
    lines.append("")
    lines.append("Winner strength (mela vs second place)")
    lines.append(f"  {cols['best_value']}: {_fmt_float(best.get(cols['best_value']))} (more negative is stronger)")
    lines.append(f"  {cols['best_margin']}: {_fmt_float(best.get(cols['best_margin']))} (bigger is clearer)")

    if "dnase_mela_importance" in best.index:
        lines.append("")
        lines.append("RF feature analysis (optional)")
        lines.append(f"  dnase_top_feature (perm): {best.get('dnase_top_feature', 'NA')}")
        lines.append(f"  dnase_mela_importance (perm): {_fmt_float(best.get('dnase_mela_importance'))}")

    lines.append("")
    lines.append(f"run_id: {best.get('run_id', '')}")
    return "\n".join(lines)


def describe_metric_margin_summary(out: Dict[str, Any]) -> str:
    dfm: pd.DataFrame = out.get("metric_margin_summary_df", pd.DataFrame())
    if dfm.empty:
        return "No metric margin summary available (missing columns or no mela wins)."

    lines = []
    lines.append("Weighted margins for correctly classified runs (mela wins)")
    lines.append("---------------------------------------------------------")
    weight_basis = dfm["weight_basis"].iloc[0] if "weight_basis" in dfm.columns else "metric_abs"
    if weight_basis == "metric_abs":
        lines.append("Weight basis: abs(best_value) per metric")
    elif weight_basis == "n_bins_total":
        lines.append("Weight basis: n_bins_total")
    else:
        lines.append("Weight basis: unweighted")
    for _, row in dfm.iterrows():
        lines.append(
            f"{row['metric']}: n_correct={int(row['n_correct'])} | "
            f"avg_margin_weighted={_fmt_float(row['avg_margin_weighted'])} | "
            f"avg_margin_unweighted={_fmt_float(row['avg_margin_unweighted'])}"
        )
    return "\n".join(lines)


def describe_downsample_accuracy(out: Dict[str, Any]) -> str:
    overall_df: pd.DataFrame = out.get("downsample_accuracy_overall_df", pd.DataFrame())
    track_df: pd.DataFrame = out.get("downsample_accuracy_track_df", pd.DataFrame())

    if overall_df.empty:
        return "No downsampled runs detected (no downsample_target values)."

    lines = []
    lines.append("Accuracy by mutation burden (overall)")
    lines.append("-------------------------------------")
    for downsample_target, sub in overall_df.groupby("downsample_target"):
        parts = []
        for _, row in sub.iterrows():
            parts.append(
                f"{row['metric']}={int(row['n_correct'])}/{int(row['n_runs'])}"
                f" ({_fmt_float(row['accuracy'], nd=3)})"
            )
        lines.append(f"downsample={int(downsample_target)}: " + " | ".join(parts))

    if not track_df.empty:
        lines.append("")
        lines.append("Best track strategy per burden+metric (by accuracy)")
        lines.append("---------------------------------------------------")
        for (downsample_target, metric), sub in track_df.groupby(["downsample_target", "metric"]):
            sub_sorted = sub.sort_values("accuracy", ascending=False)
            best = sub_sorted.iloc[0]
            lines.append(
                f"downsample={int(downsample_target)} metric={metric}: "
                f"{best['track_strategy']} {int(best['n_correct'])}/{int(best['n_runs'])}"
                f" ({_fmt_float(best['accuracy'], nd=3)})"
            )

    return "\n".join(lines)


def describe_k_breakdown(out: Dict[str, Any], top_n: int = 3) -> str:
    """
    Narrative: for each k, show the best mela-winning configuration and key metrics.
    """
    dfk: pd.DataFrame = out["best_by_k_df"]
    cols: Dict[str, str] = out["best_cols"]
    chosen_mode: str = out["chosen_mode"]

    if dfk.empty:
        return "No mela-winning rows found to summarise by k."

    lines = []
    lines.append(f"Best mela-winning configuration by sample size k (mode={chosen_mode})")
    lines.append("------------------------------------------------------------")

    for _, row in dfk.iterrows():
        k = int(row.get("sample_size_k_num", -1))
        lines.append(
            f"k={k}: {_row_track_label(row)} | bin={_row_track_bin(row)} | covs={row.get('covariates', '')}")
        lines.append(
            f"  margin={_fmt_float(row.get(cols['best_margin']))} | value={_fmt_float(row.get(cols['best_value']))}")
        if "dnase_mela_importance" in row.index:
            lines.append(f"  dnase_mela_importance={_fmt_float(row.get('dnase_mela_importance'))}")
        lines.append(f"  run_id={row.get('run_id', '')}")
        lines.append("")

    return "\n".join(lines)


def describe_track_breakdown(out: Dict[str, Any], top_n: int = 3) -> str:
    """
    Narrative: for each track_strategy, show the best mela-winning configuration.
    """
    dft: pd.DataFrame = out["best_by_track_df"]
    cols: Dict[str, str] = out["best_cols"]
    chosen_mode: str = out["chosen_mode"]

    if dft.empty:
        return "No mela-winning rows found to summarise by track strategy."

    lines = []
    lines.append(f"Best mela-winning configuration by track strategy (mode={chosen_mode})")
    lines.append("------------------------------------------------------------")

    for _, row in dft.iterrows():
        ts = str(row.get("track_strategy", ""))
        lines.append(
            f"{ts}: {_row_track_label(row)} | k={int(row.get('sample_size_k_num', -1))} | bin={_row_track_bin(row)}")
        lines.append(f"  covs={row.get('covariates', '')}")
        lines.append(
            f"  margin={_fmt_float(row.get(cols['best_margin']))} | value={_fmt_float(row.get(cols['best_value']))}")
        params = _row_track_params(row)
        if params:
            lines.append(f"  params: {', '.join(params)}")
        lines.append(f"  run_id={row.get('run_id', '')}")
        lines.append("")

    return "\n".join(lines)


# -------------------------
# CLI
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "pearson", "rf", "raw", "local_score"], default="auto")
    ap.add_argument("--require-mela-top-perm", action="store_true")
    args = ap.parse_args()

    out = analyse_results(args.results_csv, mode=args.mode, require_mela_top_perm=args.require_mela_top_perm)

    print(f"Loaded {len(out['df']):,} runs from: {args.results_csv}")
    print("\n=== mela win summary by k (all modes) ===")
    print(out["k_summary"].to_string(index=False))

    print("\n=== recommendation ===")
    print(describe_recommendation(out))

    print("\n=== weighted margin summary ===")
    print(describe_metric_margin_summary(out))

    print("\n=== downsample accuracy ===")
    print(describe_downsample_accuracy(out))

    print("\n=== best by k ===")
    print(describe_k_breakdown(out))

    print("\n=== best by track strategy ===")
    print(describe_track_breakdown(out))


if __name__ == "__main__":
    main()
