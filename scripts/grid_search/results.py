"""Results schema helpers for grid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.grid_search.config import _prefixed_track_params


def _build_results_columns(
    celltypes: Sequence[str],
    track_strategies: Sequence[str],
    *,
    accessibility_prefix: str = "dnase",
) -> List[str]:
    base_cols = [
        "sample_size_k",
        "repeat",
        "seed_samples",
        "n_selected_samples",
        "selected_sample_ids",
        "selected_tumour_types",
        "correct_celltypes",
        "sample_slice_start",
        "sample_slice_end",
        "downsample_target",
        "mutations_pre_downsample",
        "mutations_post_downsample",
        "track_strategy",
        "track_param_tag",
        "covariates",
        "include_trinuc",
        "n_mutations_total",
        "n_bins_total",
        "run_id",
    ]

    param_cols: List[str] = []
    seen = set()
    for strategy in track_strategies:
        sigma_units = "bins" if strategy in {"counts_gauss", "inv_dist_gauss"} else "bp"
        params = _prefixed_track_params(
            track_strategy=strategy,
            bin_size=1,
            counts_sigma_bins_run=1.0,
            inv_sigma_bins_run=1.0,
            max_distance_bp_run=1,
            exp_decay_bp_run=1.0,
            exp_max_distance_bp_run=1,
            adaptive_k_run=1,
            adaptive_min_bandwidth_bp_run=1.0,
            adaptive_max_distance_bp_run=1,
            sigma_units=sigma_units,
        )
        for key in params.keys():
            if key not in seen:
                param_cols.append(key)
                seen.add(key)

    per_celltype_cols: List[str] = []
    for ct in celltypes:
        per_celltype_cols.extend(
            [
                f"pearson_r_raw_{ct}_mean_weighted",
                f"pearson_r_linear_resid_{ct}_mean_weighted",
                f"spearman_r_raw_{ct}_mean_weighted",
                f"spearman_r_linear_resid_{ct}_mean_weighted",
                f"pearson_local_score_global_{ct}_mean_weighted",
                f"pearson_local_score_global_{ct}_mean_unweighted",
                f"pearson_local_score_negative_corr_fraction_{ct}_mean_weighted",
                f"pearson_local_score_negative_corr_fraction_{ct}_mean_unweighted",
                f"spearman_local_score_global_{ct}_mean_weighted",
                f"spearman_local_score_global_{ct}_mean_unweighted",
                f"spearman_local_score_negative_corr_fraction_{ct}_mean_weighted",
                f"spearman_local_score_negative_corr_fraction_{ct}_mean_unweighted",
                f"pearson_r_rf_resid_{ct}_mean_weighted",
                f"pearson_r_raw_{ct}_mean_unweighted",
                f"pearson_r_linear_resid_{ct}_mean_unweighted",
                f"spearman_r_raw_{ct}_mean_unweighted",
                f"spearman_r_linear_resid_{ct}_mean_unweighted",
                f"pearson_r_rf_resid_{ct}_mean_unweighted",
            ]
        )

    summary_cols = [
        "best_celltype_raw",
        "best_celltype_raw_value",
        "best_minus_second_raw",
        "best_celltype_linear_resid",
        "best_celltype_linear_resid_value",
        "best_minus_second_linear_resid",
        "best_celltype_spearman_raw",
        "best_celltype_spearman_raw_value",
        "best_minus_second_spearman_raw",
        "best_celltype_spearman_linear_resid",
        "best_celltype_spearman_linear_resid_value",
        "best_minus_second_spearman_linear_resid",
        "best_celltype_pearson_local_score",
        "best_celltype_pearson_local_score_value",
        "best_minus_second_pearson_local_score",
        "best_celltype_spearman_local_score",
        "best_celltype_spearman_local_score_value",
        "best_minus_second_spearman_local_score",
        "best_celltype_rf_resid",
        "best_celltype_rf_resid_value",
        "best_minus_second_rf_resid",
        "correct_celltype_canon",
        "pred_celltype_raw_canon",
        "pred_celltype_linear_resid_canon",
        "pred_celltype_spearman_raw_canon",
        "pred_celltype_spearman_linear_resid_canon",
        "pred_celltype_pearson_local_score_canon",
        "pred_celltype_spearman_local_score_canon",
        "pred_celltype_rf_resid_canon",
        "is_correct_raw",
        "is_correct_linear_resid",
        "is_correct_spearman_raw",
        "is_correct_spearman_linear_resid",
        "is_correct_pearson_local_score",
        "is_correct_spearman_local_score",
        "is_correct_rf_resid",
        "downsample_applied",
        "downsample_ratio",
        "rf_top_feature_perm",
        "rf_top_feature_importance_perm",
        f"rf_top_is_{accessibility_prefix}",
        "rf_perm_importances_mean_json",
        "rf_feature_sign_corr_mean_json",
        "ridge_coef_mean_json",
        "rf_r2_mean_weighted",
        "ridge_r2_mean_weighted",
        "rf_top_celltype_feature_perm",
        "rf_top_celltype_importance_perm",
    ]

    return base_cols + param_cols + per_celltype_cols + summary_cols


def canonicalise_celltype(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.lower().replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in ("myelprog", "myel_prog"):
        return "myel_prog"
    if s in ("esoepi", "eso_epi"):
        return "eso_epi"
    if s in ("neurostem", "neuro_stem"):
        return "neuro_stem"
    return s


def to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _extract_top_perm_feature(raw_json: Any) -> Tuple[Optional[str], Optional[float]]:
    if raw_json is None:
        return None, None
    if isinstance(raw_json, dict):
        data = raw_json
    else:
        s = str(raw_json).strip()
        if not s:
            return None, None
        try:
            data = json.loads(s)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, None
    if not isinstance(data, dict):
        return None, None
    items: List[Tuple[str, float]] = []
    for key, val in data.items():
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        items.append((str(key), num))
    if not items:
        return None, None
    max_val = max(val for _, val in items)
    top_keys = sorted([key for key, val in items if val == max_val])
    top_key = top_keys[0] if top_keys else None
    return top_key, float(max_val) if top_key is not None else None


def compute_derived_fields(
    row: Dict[str, Any],
    *,
    accessibility_prefix: str = "dnase",
) -> Dict[str, Any]:
    correct = canonicalise_celltype(row.get("correct_celltypes"))
    pred_raw = canonicalise_celltype(row.get("best_celltype_raw"))
    pred_linear = canonicalise_celltype(row.get("best_celltype_linear_resid"))
    pred_rf = canonicalise_celltype(row.get("best_celltype_rf_resid"))
    pred_spearman_raw = canonicalise_celltype(row.get("best_celltype_spearman_raw"))
    pred_spearman_linear = canonicalise_celltype(row.get("best_celltype_spearman_linear_resid"))
    pred_pearson_local = canonicalise_celltype(row.get("best_celltype_pearson_local_score"))
    pred_spearman_local = canonicalise_celltype(row.get("best_celltype_spearman_local_score"))

    def _is_correct(pred: Optional[str]) -> Optional[bool]:
        if correct is None or pred is None:
            return None
        return pred == correct

    pre = to_number(row.get("mutations_pre_downsample"))
    post = to_number(row.get("mutations_post_downsample"))
    if pre is not None and post is not None:
        downsample_applied = post < pre
    else:
        downsample_applied = None
    if pre is not None and post is not None and pre > 0:
        downsample_ratio = post / pre
    else:
        downsample_ratio = None

    top_key, top_val = _extract_top_perm_feature(row.get("rf_perm_importances_mean_json"))
    prefix = f"{accessibility_prefix}_"
    rf_top_is_accessibility = top_key.startswith(prefix) if top_key else None

    return {
        "correct_celltype_canon": correct,
        "pred_celltype_raw_canon": pred_raw,
        "pred_celltype_linear_resid_canon": pred_linear,
        "pred_celltype_rf_resid_canon": pred_rf,
        "pred_celltype_spearman_raw_canon": pred_spearman_raw,
        "pred_celltype_spearman_linear_resid_canon": pred_spearman_linear,
        "pred_celltype_pearson_local_score_canon": pred_pearson_local,
        "pred_celltype_spearman_local_score_canon": pred_spearman_local,
        "is_correct_raw": _is_correct(pred_raw),
        "is_correct_linear_resid": _is_correct(pred_linear),
        "is_correct_rf_resid": _is_correct(pred_rf),
        "is_correct_spearman_raw": _is_correct(pred_spearman_raw),
        "is_correct_spearman_linear_resid": _is_correct(pred_spearman_linear),
        "is_correct_pearson_local_score": _is_correct(pred_pearson_local),
        "is_correct_spearman_local_score": _is_correct(pred_spearman_local),
        "downsample_applied": downsample_applied,
        "downsample_ratio": downsample_ratio,
        "rf_top_feature_perm": top_key,
        "rf_top_feature_importance_perm": top_val,
        f"rf_top_is_{accessibility_prefix}": rf_top_is_accessibility,
    }


def _append_results_row(
    results_path: Path,
    row: Dict[str, Any],
    results_columns: List[str],
) -> List[str]:
    extra_cols = [key for key in row.keys() if key not in results_columns]
    if extra_cols:
        results_columns = results_columns + extra_cols
        if results_path.exists() and results_path.stat().st_size > 0:
            existing = pd.read_csv(results_path)
            for col in results_columns:
                if col not in existing.columns:
                    existing[col] = np.nan
            existing = existing[results_columns]
            existing.to_csv(results_path, index=False)
        else:
            pd.DataFrame(columns=results_columns).to_csv(results_path, index=False)

    df = pd.DataFrame([row], columns=results_columns)
    if results_path.exists() and results_path.stat().st_size > 0:
        df.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df.to_csv(results_path, index=False)
    return results_columns
