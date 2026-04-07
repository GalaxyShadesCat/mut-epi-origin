#!/usr/bin/env python3
"""Validate inferred cell-state scores against clinical metadata.

This script performs indirect biological validation for experiment results where
true cell-state labels are unavailable. It is flexible to any number of inferred
states, as long as the relevant score columns can be resolved from results.csv.

Main features:
- Loads single-sample rows from results.csv
- Resolves score columns for any number of states
- Runs score-level group tests and correlations
- Builds per-sample ranking summaries for any number of states
- Computes generic score contrasts for all state pairs
- Runs label-level association analyses on derived best labels
- Fits predictive models using score features plus optional metadata covariates

Examples
--------
3-state run:
python scripts/validate_state_scores.py \
  --experiment-name my_experiment \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --allow-aggregated-results

2-state FOXA2 run:
python scripts/validate_state_scores.py \
  --experiment-name my_experiment_foxa2 \
  --state-labels normal_FOXA2_pos,abnormal_FOXA2_zero \
  --state-suffixes normal_FOXA2_pos,abnormal_FOXA2_zero \
  --allow-aggregated-results

NAFLD-focused run:
python scripts/validate_state_scores.py \
  --experiment-name my_experiment_nafld \
  --metadata-path data/derived/master_metadata.csv \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --modelling-targets nafld_status \
  --group-test-vars nafld_status,obesity_class \
  --correlation-vars nafld_status \
  --covariate-cols alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class \
  --allow-aggregated-results
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
    pearsonr,
    spearmanr,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCORING_COLUMN_TEMPLATES: Dict[str, str] = {
    "rf_resid": "pearson_r_rf_resid_{cell}_mean_weighted",
    "pearson_local_score": "pearson_local_score_global_{cell}_mean_weighted",
    "spearman_local_score": "spearman_local_score_global_{cell}_mean_weighted",
    "pearson_r_linear_resid": "pearson_r_linear_resid_{cell}_mean_weighted",
    "spearman_r_linear_resid": "spearman_r_linear_resid_{cell}_mean_weighted",
}

DEFAULT_GROUP_TEST_METADATA_VARS: List[str] = [
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
    "obesity_class",
    "fibrosis_present",
]

DEFAULT_CORRELATION_METADATA_VARS: List[str] = ["fibrosis_ishak_score"]

DEFAULT_MODELLING_TARGETS: List[str] = [
    "fibrosis_ishak_score",
    "fibrosis_present",
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
]

DEFAULT_COVARIATE_COLUMNS: List[str] = [
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
    "obesity_class",
    "fibrosis_present",
    "fibrosis_ishak_score",
]

KNOWN_METADATA_COLUMNS: List[str] = [
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
    "obesity_class",
    "fibrosis_ishak_score",
    "fibrosis_present",
]

GROUP_TEST_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "score_feature",
    "metadata_variable",
    "test_type",
    "n_total",
    "n_groups",
    "statistic",
    "p_value",
    "effect_summary_json",
    "group_summary_json",
]

CORRELATION_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "score_feature",
    "metadata_variable",
    "correlation_type",
    "n",
    "statistic",
    "p_value",
]

MODEL_SUMMARY_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "target",
    "feature_set",
    "model",
    "best_params",
    "best_score",
    "score_gap",
    "cv_score_mean",
    "cv_score_std",
    "n_samples",
    "cv_folds",
    "is_best_model_for_group",
]

MODEL_PREDICTION_COLUMNS: List[str] = [
    "sample",
    "config_id",
    "scoring_system",
    "target",
    "feature_set",
    "model",
    "true_value",
    "predicted_value",
]

FEATURE_IMPORTANCE_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "target",
    "feature_set",
    "model",
    "feature",
    "importance",
]

RANKING_COLUMNS_BASE: List[str] = [
    "sample",
    "config_id",
    "scoring_system",
    "best_cell_state",
    "best_score",
    "second_best_score",
    "score_gap",
    "n_states",
    "score_entropy",
    "normalised_best_score",
]

SCORE_CONTRAST_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "contrast_feature",
    "metadata_variable",
    "test_type",
    "n_total",
    "n_groups",
    "statistic",
    "p_value",
    "effect_summary_json",
    "group_summary_json",
]

LABEL_ASSOCIATION_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "metadata_variable",
    "confidence_filter",
    "n_total",
    "n_groups",
    "n_states",
    "test_type",
    "statistic",
    "p_value",
    "counts_json",
    "proportions_json",
]

LABEL_ONE_VS_REST_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "metadata_variable",
    "target_label",
    "confidence_filter",
    "n_total",
    "n_groups",
    "test_type",
    "statistic",
    "p_value",
    "counts_json",
    "proportions_json",
]

CONFIDENT_SCORE_GAP_THRESHOLD = 0.01


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate inferred state scores against metadata in an experiment "
            "directory. Supports any number of states."
        )
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment directory name under --experiments-root.",
    )
    parser.add_argument(
        "--experiments-root",
        default="outputs/experiments",
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/derived/master_metadata.csv",
        help="Path to sample metadata CSV.",
    )
    parser.add_argument(
        "--metadata-sample-col",
        default="tumour_sample_submitter_id",
        help="Metadata sample identifier column.",
    )
    parser.add_argument(
        "--scoring-systems",
        default=",".join(SCORING_COLUMN_TEMPLATES.keys()),
        help="Comma-separated scoring systems to analyse.",
    )
    parser.add_argument(
        "--group-test-vars",
        default=",".join(DEFAULT_GROUP_TEST_METADATA_VARS),
        help="Comma-separated metadata variables used for group comparison tests. Use 'none' to disable.",
    )
    parser.add_argument(
        "--correlation-vars",
        default=",".join(DEFAULT_CORRELATION_METADATA_VARS),
        help="Comma-separated metadata variables used for correlation tests. Use 'none' to disable.",
    )
    parser.add_argument(
        "--modelling-targets",
        default=",".join(DEFAULT_MODELLING_TARGETS),
        help="Comma-separated metadata targets used in predictive modelling. Use 'none' to disable.",
    )
    parser.add_argument(
        "--covariate-cols",
        default=",".join(DEFAULT_COVARIATE_COLUMNS),
        help="Comma-separated metadata covariate columns used by modelling feature sets. Use 'none' to disable.",
    )
    parser.add_argument(
        "--state-labels",
        required=True,
        help=(
            "Comma-separated state labels used in downstream outputs, e.g. "
            "'hepatocyte_normal,hepatocyte_ac,hepatocyte_ah' or "
            "'normal_FOXA2_pos,abnormal_FOXA2_zero'."
        ),
    )
    parser.add_argument(
        "--state-suffixes",
        required=True,
        help=(
            "Comma-separated suffixes used in results.csv score column names, in "
            "the same order as --state-labels."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum number of cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed used by random forest models.",
    )
    parser.add_argument(
        "--score-gap-threshold",
        type=float,
        default=CONFIDENT_SCORE_GAP_THRESHOLD,
        help="Threshold for confident assignments based on best minus second-best score.",
    )
    parser.add_argument(
        "--no-invert-scores",
        action="store_true",
        help="Disable score inversion (default inverts so higher means stronger anti-correlation).",
    )
    parser.add_argument(
        "--allow-aggregated-results",
        action="store_true",
        help="Allow aggregated rows in results.csv (these rows are skipped).",
    )
    return parser.parse_args()


def ensure_scoring_systems(raw_value: str) -> List[str]:
    systems = [s.strip() for s in raw_value.split(",") if s.strip()]
    if not systems:
        raise ValueError("No scoring systems were provided.")
    unknown = [s for s in systems if s not in SCORING_COLUMN_TEMPLATES]
    if unknown:
        raise ValueError(
            "Unknown scoring systems: "
            + ", ".join(unknown)
            + ". Known systems: "
            + ", ".join(SCORING_COLUMN_TEMPLATES.keys())
        )
    return systems


def parse_csv_list(raw_value: str, label: str) -> List[str]:
    if raw_value.strip().lower() in {"none", "off"}:
        return []
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError(f"No values were provided for {label}.")
    return values


def parse_state_config(labels_raw: str, suffixes_raw: str) -> List[Tuple[str, str]]:
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
    suffixes = [x.strip() for x in suffixes_raw.split(",") if x.strip()]
    if not labels:
        raise ValueError("No state labels were provided.")
    if not suffixes:
        raise ValueError("No state suffixes were provided.")
    if len(labels) != len(suffixes):
        raise ValueError(
            "--state-labels and --state-suffixes must have the same number of items."
        )
    if len(set(labels)) != len(labels):
        raise ValueError("State labels must be unique.")
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("State suffixes must be unique.")
    return list(zip(labels, suffixes))


def safe_feature_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "state"


def score_col_for_state(state_label: str) -> str:
    return f"score_{safe_feature_name(state_label)}"


def ranking_score_columns(state_config: Sequence[Tuple[str, str]]) -> List[str]:
    return [score_col_for_state(label) for label, _ in state_config]


def ranking_output_columns(state_config: Sequence[Tuple[str, str]]) -> List[str]:
    return ["sample", "config_id", "scoring_system"] + ranking_score_columns(state_config) + RANKING_COLUMNS_BASE[3:]


def contrast_pairs(state_config: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    labels = [label for label, _ in state_config]
    return list(itertools.combinations(labels, 2))


def contrast_feature_name(a: str, b: str) -> str:
    return f"delta_{safe_feature_name(a)}_minus_{safe_feature_name(b)}"


def _normalise_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"na", "n/a", "none", "null", "nan"}:
        return None
    return text


def _parse_selected_sample_ids(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed: Any | None = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
        if isinstance(parsed, list):
            raw_parts = parsed
        else:
            raw_parts = re.split(r"[,;|]", text)
    out: List[str] = []
    seen: set[str] = set()
    for item in raw_parts:
        parsed_item = _normalise_text(item)
        if parsed_item is None:
            continue
        cleaned = parsed_item.strip("\"'[]() ")
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _derive_config_id(results_df: pd.DataFrame) -> pd.Series:
    if "track_strategy" in results_df.columns:
        strategy = results_df["track_strategy"].fillna("NA").astype(str).str.strip()

        strategy_bin_cols = [
            "counts_raw_bin",
            "counts_gauss_bin",
            "inv_dist_gauss_bin",
            "exp_decay_bin",
            "exp_decay_adaptive_bin",
        ]
        generic_bin_col = "bin_size" if "bin_size" in results_df.columns else None
        bin_value = pd.Series(["NA"] * len(results_df), index=results_df.index, dtype="object")

        if generic_bin_col is not None:
            generic_bin = results_df[generic_bin_col].fillna("NA").astype(str).str.strip()
            bin_value = generic_bin.copy()

        available_strategy_bin_cols = [col for col in strategy_bin_cols if col in results_df.columns]
        if available_strategy_bin_cols:
            bin_lookup = results_df[available_strategy_bin_cols].copy()
            for col in available_strategy_bin_cols:
                bin_lookup[col] = bin_lookup[col].fillna("NA").astype(str).str.strip()

            first_available_bin = bin_lookup.replace("NA", np.nan).bfill(axis=1).iloc[:, 0].fillna("NA")
            if generic_bin_col is None:
                bin_value = first_available_bin.copy()

            for col in available_strategy_bin_cols:
                strategy_name = col.removesuffix("_bin")
                mask = strategy == strategy_name
                bin_value.loc[mask] = bin_lookup.loc[mask, col]

        return "track_strategy=" + strategy + "|bin_size=" + bin_value.astype(str).str.strip()

    if "config_id" in results_df.columns:
        return results_df["config_id"].astype(str).str.strip()

    if "run_id" in results_df.columns:
        return results_df["run_id"].astype(str).str.strip()

    return pd.Series(["default"] * len(results_df), index=results_df.index, dtype="object")


def load_sample_level_results(
    experiment_dir: Path,
    scoring_systems: Sequence[str],
    state_config: Sequence[Tuple[str, str]],
    invert_scores: bool,
    allow_aggregated_results: bool,
) -> pd.DataFrame:
    results_path = experiment_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing required file: {results_path}")

    log(f"Loading results: {results_path}")
    results_df = pd.read_csv(results_path)
    if results_df.empty:
        raise ValueError("results.csv is empty.")

    sample_ids = pd.Series([[] for _ in range(len(results_df))], index=results_df.index, dtype="object")
    if "selected_sample_ids" in results_df.columns:
        sample_ids = results_df["selected_sample_ids"].apply(_parse_selected_sample_ids)
    elif "sample_id" in results_df.columns:
        sample_ids = results_df["sample_id"].apply(
            lambda v: [_normalise_text(v)] if _normalise_text(v) else []
        )
    else:
        raise ValueError(
            "results.csv must contain either 'selected_sample_ids' or 'sample_id' to identify samples."
        )

    aggregated_mask = sample_ids.apply(len) != 1
    if "n_selected_samples" in results_df.columns:
        n_selected = pd.to_numeric(results_df["n_selected_samples"], errors="coerce")
        aggregated_mask = aggregated_mask | (n_selected.notna() & (n_selected != 1))

    n_aggregated = int(aggregated_mask.sum())
    if n_aggregated > 0 and not allow_aggregated_results:
        raise ValueError(
            f"Found {n_aggregated} aggregated rows in results.csv. "
            "Use --allow-aggregated-results to skip them explicitly."
        )
    if n_aggregated > 0:
        log(f"Skipping {n_aggregated} aggregated rows due to --allow-aggregated-results.")

    usable = results_df.loc[~aggregated_mask].copy()
    if usable.empty:
        raise ValueError("No usable single-sample rows were found in results.csv.")

    usable["sample"] = sample_ids.loc[usable.index].str[0].astype(str).str.strip()
    usable = usable[usable["sample"] != ""].copy()
    if usable.empty:
        raise ValueError("No valid sample identifiers were extracted from usable rows.")

    usable["config_id"] = _derive_config_id(usable)
    usable = usable[usable["config_id"].astype(str).str.strip() != ""].copy()
    if usable.empty:
        raise ValueError("No valid config_id values were available in usable rows.")

    score_frames: List[pd.DataFrame] = []
    for scoring_system in scoring_systems:
        template = SCORING_COLUMN_TEMPLATES[scoring_system]
        col_map: Dict[str, str] = {}
        missing: List[str] = []

        for state_label, state_suffix in state_config:
            primary = template.format(cell=state_suffix)
            alternative = template.format(cell=state_label)
            if primary in usable.columns:
                resolved = primary
            elif alternative in usable.columns:
                resolved = alternative
            else:
                resolved = ""
                missing.append(f"{primary} (or {alternative})")
            col_map[score_col_for_state(state_label)] = resolved

        if missing:
            raise ValueError(
                f"Missing score columns for scoring_system {scoring_system}: "
                + ", ".join(missing)
            )

        cols = ["sample", "config_id"] + list(col_map.values())
        subset = usable[cols].rename(columns={v: k for k, v in col_map.items()}).copy()

        agg_map: Dict[str, Tuple[str, str]] = {}
        for score_col in col_map.keys():
            subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")
            if invert_scores:
                subset[score_col] = -1.0 * subset[score_col]
            agg_map[score_col] = (score_col, "mean")

        subset = subset.groupby(["sample", "config_id"], as_index=False).agg(**agg_map).copy()
        subset["scoring_system"] = scoring_system
        score_frames.append(subset)

    return pd.concat(score_frames, ignore_index=True)


def parse_ishak_to_ordinal(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text_raw = str(value).strip()
    if not text_raw:
        return np.nan
    try:
        return float(text_raw)
    except ValueError:
        pass
    text = text_raw.lower()
    if "0" in text and "fibrosis" in text and "no" in text:
        return 0.0
    if "1,2" in text or "1-2" in text:
        return 1.0
    if "3,4" in text or "3-4" in text:
        return 2.0
    if text.startswith("5") or "incomplete cirrhosis" in text:
        return 3.0
    if text.startswith("6") or "established cirrhosis" in text:
        return 4.0
    return np.nan


def load_metadata(
    metadata_path: Path,
    sample_col: str,
    requested_metadata_columns: Sequence[str],
) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    log(f"Loading metadata: {metadata_path}")

    metadata = pd.read_csv(metadata_path, dtype="object")
    if sample_col not in metadata.columns:
        raise ValueError(f"Missing required metadata sample column: {sample_col}")

    requested = [col for col in requested_metadata_columns if col != sample_col]
    keep_cols = [sample_col] + [col for col in requested if col in metadata.columns]
    out = metadata[keep_cols].copy()
    out["sample"] = out[sample_col].astype(str).str.strip()
    out = out[out["sample"] != ""].copy()
    out.drop(columns=[sample_col], inplace=True)

    for col in ["alcohol_status", "hbv_status", "hcv_status", "nafld_status", "obesity_class", "fibrosis_present"]:
        if col in out.columns:
            out[col] = out[col].map(_normalise_text)

    if "fibrosis_ishak_score" in out.columns:
        out["fibrosis_ishak_score"] = out["fibrosis_ishak_score"].apply(parse_ishak_to_ordinal)
        if "fibrosis_present" not in out.columns:
            out["fibrosis_present"] = out["fibrosis_ishak_score"].apply(
                lambda x: np.nan if pd.isna(x) else bool(x > 0.0)
            )

    out = out.drop_duplicates(subset=["sample"], keep="first")
    return out


def run_group_tests(
    score_df: pd.DataFrame,
    score_features: Sequence[str],
    metadata_vars: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = score_df.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        for score_feature in score_features:
            for metadata_var in metadata_vars:
                d = sub[[score_feature, metadata_var]].dropna().copy()
                if d.empty:
                    continue
                d["group_label"] = d[metadata_var].astype(str)
                full_group_values: Dict[str, np.ndarray] = {
                    key: vals[score_feature].astype(float).to_numpy()
                    for key, vals in d.groupby("group_label", dropna=False)
                    if len(vals) > 0
                }
                if len(full_group_values) < 2:
                    continue
                if any(len(vals) < 2 for vals in full_group_values.values()):
                    continue

                sorted_labels = sorted(full_group_values.keys())
                sorted_values = [full_group_values[label] for label in sorted_labels]
                group_summary = {
                    label: {
                        "n": int(len(vals)),
                        "median": float(np.median(vals)),
                        "mean": float(np.mean(vals)),
                    }
                    for label, vals in zip(sorted_labels, sorted_values)
                }

                if len(sorted_values) == 2:
                    stat, p_val = mannwhitneyu(sorted_values[0], sorted_values[1], alternative="two-sided")
                    test_type = "mannwhitneyu"
                    med_0 = group_summary[sorted_labels[0]]["median"]
                    med_1 = group_summary[sorted_labels[1]]["median"]
                    mean_0 = group_summary[sorted_labels[0]]["mean"]
                    mean_1 = group_summary[sorted_labels[1]]["mean"]
                    effect_summary = {
                        "group_order": sorted_labels,
                        "median_diff_group1_minus_group0": med_1 - med_0,
                        "mean_diff_group1_minus_group0": mean_1 - mean_0,
                    }
                else:
                    stat, p_val = kruskal(*sorted_values)
                    test_type = "kruskal"
                    medians = [group_summary[label]["median"] for label in sorted_labels]
                    means = [group_summary[label]["mean"] for label in sorted_labels]
                    effect_summary = {
                        "groups": sorted_labels,
                        "median_range": float(np.max(medians) - np.min(medians)),
                        "mean_range": float(np.max(means) - np.min(means)),
                    }

                rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "score_feature": score_feature,
                        "metadata_variable": metadata_var,
                        "test_type": test_type,
                        "n_total": int(sum(len(vals) for vals in sorted_values)),
                        "n_groups": int(len(sorted_values)),
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "effect_summary_json": json.dumps(effect_summary, sort_keys=True),
                        "group_summary_json": json.dumps(group_summary, sort_keys=True),
                    }
                )

    return pd.DataFrame(rows, columns=GROUP_TEST_COLUMNS)


def _encode_binary_text_series(values: pd.Series) -> pd.Series:
    negative_tokens = {"0", "false", "f", "no", "n", "absent", "negative"}
    positive_tokens = {"1", "true", "t", "yes", "y", "present", "positive"}

    def _map_value(value: Any) -> float:
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if text in negative_tokens:
            return 0.0
        if text in positive_tokens:
            return 1.0
        return np.nan

    return values.map(_map_value)


def run_correlations(
    score_df: pd.DataFrame,
    score_features: Sequence[str],
    metadata_vars: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = score_df.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        for score_feature in score_features:
            for metadata_var in metadata_vars:
                d = sub[[score_feature, metadata_var]].dropna().copy()
                if len(d) < 3:
                    continue
                x_series = pd.to_numeric(d[score_feature], errors="coerce")
                y_series = pd.to_numeric(d[metadata_var], errors="coerce")
                if y_series.isna().all():
                    y_series = _encode_binary_text_series(d[metadata_var])

                paired = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
                if len(paired) < 3:
                    continue

                x = paired["x"].to_numpy()
                y = paired["y"].to_numpy()
                if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
                    continue
                rho_s, p_s = spearmanr(x, y)
                rho_p, p_p = pearsonr(x, y)
                rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "score_feature": score_feature,
                        "metadata_variable": metadata_var,
                        "correlation_type": "spearman",
                        "n": int(len(paired)),
                        "statistic": float(rho_s),
                        "p_value": float(p_s),
                    }
                )
                rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "score_feature": score_feature,
                        "metadata_variable": metadata_var,
                        "correlation_type": "pearson",
                        "n": int(len(paired)),
                        "statistic": float(rho_p),
                        "p_value": float(p_p),
                    }
                )

    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


def add_score_contrasts(df: pd.DataFrame, state_config: Sequence[Tuple[str, str]]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    contrast_features: List[str] = []
    for label_a, label_b in contrast_pairs(state_config):
        col_a = score_col_for_state(label_a)
        col_b = score_col_for_state(label_b)
        contrast_col = contrast_feature_name(label_a, label_b)
        out[contrast_col] = out[col_a] - out[col_b]
        contrast_features.append(contrast_col)
    return out, contrast_features


def run_score_contrast_tests(
    score_df: pd.DataFrame,
    contrast_features: Sequence[str],
    metadata_vars: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = score_df.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        for contrast_feature in contrast_features:
            if contrast_feature not in sub.columns:
                continue
            for metadata_var in metadata_vars:
                d = sub[[contrast_feature, metadata_var]].dropna().copy()
                if d.empty:
                    continue
                d["group_label"] = d[metadata_var].astype(str)
                full_group_values: Dict[str, np.ndarray] = {
                    key: vals[contrast_feature].astype(float).to_numpy()
                    for key, vals in d.groupby("group_label", dropna=False)
                    if len(vals) > 0
                }
                if len(full_group_values) < 2:
                    continue
                if any(len(vals) < 2 for vals in full_group_values.values()):
                    continue

                sorted_labels = sorted(full_group_values.keys())
                sorted_values = [full_group_values[label] for label in sorted_labels]
                group_summary = {
                    label: {
                        "n": int(len(vals)),
                        "median": float(np.median(vals)),
                        "mean": float(np.mean(vals)),
                    }
                    for label, vals in zip(sorted_labels, sorted_values)
                }

                if len(sorted_values) == 2:
                    stat, p_val = mannwhitneyu(sorted_values[0], sorted_values[1], alternative="two-sided")
                    test_type = "mannwhitneyu"
                    med_0 = group_summary[sorted_labels[0]]["median"]
                    med_1 = group_summary[sorted_labels[1]]["median"]
                    mean_0 = group_summary[sorted_labels[0]]["mean"]
                    mean_1 = group_summary[sorted_labels[1]]["mean"]
                    effect_summary = {
                        "group_order": sorted_labels,
                        "median_diff_group1_minus_group0": med_1 - med_0,
                        "mean_diff_group1_minus_group0": mean_1 - mean_0,
                    }
                else:
                    stat, p_val = kruskal(*sorted_values)
                    test_type = "kruskal"
                    medians = [group_summary[label]["median"] for label in sorted_labels]
                    means = [group_summary[label]["mean"] for label in sorted_labels]
                    effect_summary = {
                        "groups": sorted_labels,
                        "median_range": float(np.max(medians) - np.min(medians)),
                        "mean_range": float(np.max(means) - np.min(means)),
                    }

                rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "contrast_feature": contrast_feature,
                        "metadata_variable": metadata_var,
                        "test_type": test_type,
                        "n_total": int(sum(len(vals) for vals in sorted_values)),
                        "n_groups": int(len(sorted_values)),
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "effect_summary_json": json.dumps(effect_summary, sort_keys=True),
                        "group_summary_json": json.dumps(group_summary, sort_keys=True),
                    }
                )

    out = pd.DataFrame(rows, columns=SCORE_CONTRAST_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(
        by=["config_id", "scoring_system", "metadata_variable", "contrast_feature", "p_value"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def _build_feature_sets(
    df: pd.DataFrame,
    target: str,
    state_score_cols: Sequence[str],
    covariate_columns: Sequence[str],
) -> Dict[str, List[str]]:
    covariate_cols = [col for col in covariate_columns if col != target]
    feature_sets = {
        "chromatin_only": list(state_score_cols),
        "covariates_only": covariate_cols,
        "full": list(state_score_cols) + covariate_cols,
    }

    deduped: Dict[str, List[str]] = {}
    seen: set[Tuple[str, ...]] = set()
    for name, cols in feature_sets.items():
        key = tuple(cols)
        if key not in seen:
            seen.add(key)
            deduped[name] = cols
    return deduped


def _build_xy(
    group_df: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    cols_needed = list(predictors) + [target, "sample"]
    d = group_df[cols_needed].copy()
    d = d.dropna(subset=[target]).copy()
    if target == "fibrosis_ishak_score":
        d[target] = pd.to_numeric(d[target], errors="coerce")
        d = d.dropna(subset=[target]).copy()
    d = d.dropna(subset=list(predictors)).copy()
    x = d[list(predictors)].copy()
    y = d[target].copy()
    return x, y


def _cv_for_target(y: pd.Series, target: str, max_folds: int, random_state: int):
    n_samples = int(len(y))
    if n_samples < 2:
        return None, None

    if target == "fibrosis_ishak_score":
        folds = min(max_folds, n_samples // 2)
        if folds < 2:
            return None, None
        return KFold(n_splits=folds, shuffle=True, random_state=random_state), folds

    class_counts = y.value_counts(dropna=False)
    if len(class_counts) < 2:
        return None, None
    min_class = int(class_counts.min())
    if min_class < 2:
        return None, None

    folds = min(max_folds, n_samples, min_class)
    if folds < 2:
        return None, None
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), folds


def run_predictive_models(
    score_df: pd.DataFrame,
    state_score_cols: Sequence[str],
    modelling_targets: Sequence[str],
    covariate_columns: Sequence[str],
    cv_folds: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    regression_models = {
        "ridge": (
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "r2",
        ),
        "random_forest": (
            RandomForestRegressor(random_state=random_state),
            {
                "model__n_estimators": [300, 800],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [5, 10],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": ["sqrt"],
                "model__bootstrap": [True],
            },
            "r2",
        ),
    }

    classification_models = {
        "ridge_classifier": (
            RidgeClassifier(),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "balanced_accuracy",
        ),
        "random_forest_classifier": (
            RandomForestClassifier(random_state=random_state),
            {
                "model__n_estimators": [300, 800],
                "model__max_depth": [3, 5, None],
                "model__min_samples_split": [5, 10],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": ["sqrt"],
                "model__bootstrap": [True],
            },
            "balanced_accuracy",
        ),
    }

    grouped = score_df.groupby(["config_id", "scoring_system"], dropna=False)
    total_groups = int(grouped.ngroups)
    fitted_models = 0
    skipped_targets = 0
    group_idx = 0

    for (config_id, scoring_system), group_df in grouped:
        group_idx += 1
        log(
            "Modelling group "
            f"{group_idx}/{total_groups}: config_id={config_id} | scoring_system={scoring_system}"
        )

        for target in modelling_targets:
            if target not in group_df.columns:
                continue

            feature_sets = _build_feature_sets(
                group_df,
                target,
                state_score_cols,
                covariate_columns,
            )
            for feature_set_name, predictors in feature_sets.items():
                x, y = _build_xy(group_df, target, predictors)
                if len(x) < 2:
                    skipped_targets += 1
                    continue

                cv, folds_used = _cv_for_target(y, target, cv_folds, random_state)
                if cv is None or folds_used is None:
                    skipped_targets += 1
                    continue

                cat_cols = [c for c in x.columns if x[c].dtype == "object" or str(x[c].dtype) == "bool"]
                num_cols = [c for c in x.columns if c not in cat_cols]
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                        ("num", "passthrough", num_cols),
                    ],
                    remainder="drop",
                )

                model_specs = regression_models if target == "fibrosis_ishak_score" else classification_models
                for model_name, (estimator, param_grid, scoring) in model_specs.items():
                    pipeline = Pipeline(
                        steps=[
                            ("preprocess", preprocessor),
                            ("model", estimator),
                        ]
                    )
                    search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=-1,
                        refit=True,
                    )
                    search.fit(x, y)
                    fitted_models += 1
                    log(
                        "  Fitted model "
                        f"{fitted_models}: target={target}, feature_set={feature_set_name}, "
                        f"model={model_name}, n_samples={len(x)}, cv_folds={folds_used}"
                    )

                    best_idx = int(search.best_index_)
                    cv_mean = float(search.cv_results_["mean_test_score"][best_idx])
                    cv_std = float(search.cv_results_["std_test_score"][best_idx])
                    best_score = float(search.best_score_)
                    best_params = {
                        k.replace("model__", ""): v for k, v in search.best_params_.items()
                    }

                    summary_rows.append(
                        {
                            "config_id": config_id,
                            "scoring_system": scoring_system,
                            "target": target,
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "best_params": json.dumps(best_params, sort_keys=True),
                            "best_score": best_score,
                            "score_gap": np.nan,
                            "cv_score_mean": cv_mean,
                            "cv_score_std": cv_std,
                            "n_samples": int(len(x)),
                            "cv_folds": int(folds_used),
                            "is_best_model_for_group": False,
                        }
                    )

                    preds = cross_val_predict(search.best_estimator_, x, y, cv=cv, n_jobs=-1)
                    samples_used = group_df.loc[x.index, "sample"].astype(str).to_numpy()
                    for idx, sample in enumerate(samples_used):
                        pred_rows.append(
                            {
                                "sample": sample,
                                "config_id": config_id,
                                "scoring_system": scoring_system,
                                "target": target,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "true_value": y.iloc[idx],
                                "predicted_value": preds[idx],
                            }
                        )

                    if "random_forest" in model_name:
                        pre = search.best_estimator_.named_steps["preprocess"]
                        model = search.best_estimator_.named_steps["model"]
                        feature_names = pre.get_feature_names_out()
                        importances = model.feature_importances_
                        for feature, importance in zip(feature_names, importances):
                            importance_rows.append(
                                {
                                    "config_id": config_id,
                                    "scoring_system": scoring_system,
                                    "target": target,
                                    "feature_set": feature_set_name,
                                    "model": model_name,
                                    "feature": str(feature),
                                    "importance": float(importance),
                                }
                            )

    summary_df = pd.DataFrame(summary_rows, columns=MODEL_SUMMARY_COLUMNS)
    if not summary_df.empty:
        group_keys = ["config_id", "scoring_system", "target", "feature_set"]
        for _, idx in summary_df.groupby(group_keys).groups.items():
            sub = summary_df.loc[idx].copy()
            sub_sorted = sub.sort_values(["best_score", "cv_score_mean", "model"], ascending=[False, False, True])
            best_row_idx = sub_sorted.index[0]
            second_score = (
                float(sub_sorted.iloc[1]["best_score"]) if len(sub_sorted) > 1 else float(sub_sorted.iloc[0]["best_score"])
            )
            best_score = float(sub_sorted.iloc[0]["best_score"])
            score_gap = best_score - second_score
            summary_df.loc[idx, "score_gap"] = score_gap
            summary_df.loc[idx, "is_best_model_for_group"] = False
            summary_df.loc[best_row_idx, "is_best_model_for_group"] = True

    preds_df = pd.DataFrame(pred_rows, columns=MODEL_PREDICTION_COLUMNS)
    imp_df = pd.DataFrame(importance_rows, columns=FEATURE_IMPORTANCE_COLUMNS)
    log(
        "Model fitting complete: "
        f"fitted_models={fitted_models}, skipped_target_feature_sets={skipped_targets}"
    )
    return summary_df, preds_df, imp_df


def _entropy_from_scores(values: np.ndarray) -> float:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = float(exp_values.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    probs = exp_values / denom
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _normalised_best_score(values: np.ndarray) -> float:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = float(exp_values.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    probs = exp_values / denom
    return float(np.max(probs))


def build_rankings(score_df: pd.DataFrame, state_config: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    score_cols = [(label, score_col_for_state(label)) for label, _ in state_config]

    for _, row in score_df.iterrows():
        score_map = {label: float(row[col]) for label, col in score_cols}
        ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        best_cell, best_score = ordered[0]
        second_best_score = ordered[1][1] if len(ordered) > 1 else ordered[0][1]
        raw_scores = np.array([score_map[label] for label, _ in score_cols], dtype=float)

        out_row: Dict[str, Any] = {
            "sample": row["sample"],
            "config_id": row["config_id"],
            "scoring_system": row["scoring_system"],
            "best_cell_state": best_cell,
            "best_score": float(best_score),
            "second_best_score": float(second_best_score),
            "score_gap": float(best_score - second_best_score),
            "n_states": int(len(score_cols)),
            "score_entropy": _entropy_from_scores(raw_scores),
            "normalised_best_score": _normalised_best_score(raw_scores),
        }
        for label, col in score_cols:
            out_row[col] = row[col]
        rows.append(out_row)

    return pd.DataFrame(rows, columns=ranking_output_columns(state_config))


def build_rankings_with_metadata(rankings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return rankings.merge(metadata, on="sample", how="left")


def _serialise_counts_and_proportions(table: pd.DataFrame) -> Tuple[str, str]:
    row_labels = sorted(table.index.astype(str).tolist())
    col_labels = sorted(table.columns.astype(str).tolist())
    stable = table.copy()
    stable.index = stable.index.astype(str)
    stable.columns = stable.columns.astype(str)
    stable = stable.reindex(index=row_labels, columns=col_labels, fill_value=0)

    counts_dict: Dict[str, Dict[str, int]] = {}
    proportions_dict: Dict[str, Dict[str, float]] = {}
    for row_label in row_labels:
        row_counts = stable.loc[row_label]
        total = float(row_counts.sum())
        counts_dict[row_label] = {col: int(row_counts[col]) for col in col_labels}
        if total > 0.0:
            proportions_dict[row_label] = {
                col: float(row_counts[col] / total) for col in col_labels
            }
        else:
            proportions_dict[row_label] = {col: float(np.nan) for col in col_labels}
    return (
        json.dumps(counts_dict, sort_keys=True),
        json.dumps(proportions_dict, sort_keys=True),
    )


def run_label_association_tests(
    rankings_with_metadata: pd.DataFrame,
    metadata_vars: Sequence[str],
    confidence_filter_name: str = "all",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = rankings_with_metadata.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        for metadata_var in metadata_vars:
            needed = ["best_cell_state", metadata_var]
            if any(col not in sub.columns for col in needed):
                continue

            d = sub[needed].dropna().copy()
            if d.empty:
                continue

            d["group_label"] = d[metadata_var].astype(str)
            d["state_label"] = d["best_cell_state"].astype(str)
            contingency = pd.crosstab(d["group_label"], d["state_label"], dropna=False)

            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            values = contingency.to_numpy(dtype=float)
            if values.sum() <= 0.0:
                continue

            if contingency.shape == (2, 2):
                _, _, _, expected = chi2_contingency(values)
                if (expected < 5).any():
                    stat, p_value = fisher_exact(values)
                    test_type = "fisher_exact_2x2"
                else:
                    stat, p_value, _, _ = chi2_contingency(values)
                    test_type = "chi2"
            else:
                stat, p_value, _, _ = chi2_contingency(values)
                test_type = "chi2"

            counts_json, proportions_json = _serialise_counts_and_proportions(contingency)
            rows.append(
                {
                    "config_id": config_id,
                    "scoring_system": scoring_system,
                    "metadata_variable": metadata_var,
                    "confidence_filter": confidence_filter_name,
                    "n_total": int(values.sum()),
                    "n_groups": int(contingency.shape[0]),
                    "n_states": int(contingency.shape[1]),
                    "test_type": test_type,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "counts_json": counts_json,
                    "proportions_json": proportions_json,
                }
            )

    out = pd.DataFrame(rows, columns=LABEL_ASSOCIATION_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(
        by=["config_id", "scoring_system", "metadata_variable", "p_value"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def run_label_one_vs_rest_tests(
    rankings_with_metadata: pd.DataFrame,
    metadata_vars: Sequence[str],
    target_labels: Sequence[str],
    confidence_filter_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = rankings_with_metadata.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        for metadata_var in metadata_vars:
            if metadata_var not in sub.columns or "best_cell_state" not in sub.columns:
                continue

            base = sub[[metadata_var, "best_cell_state"]].dropna().copy()
            if base.empty:
                continue

            base["group_label"] = base[metadata_var].astype(str)
            base["state_label"] = base["best_cell_state"].astype(str)

            for target_label in target_labels:
                d = base.copy()
                d["binary_label"] = np.where(d["state_label"] == target_label, target_label, "other")
                contingency = pd.crosstab(d["group_label"], d["binary_label"], dropna=False)

                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue

                values = contingency.to_numpy(dtype=float)
                if values.sum() <= 0.0:
                    continue

                if contingency.shape == (2, 2):
                    _, _, _, expected = chi2_contingency(values)
                    if (expected < 5).any():
                        stat, p_value = fisher_exact(values)
                        test_type = "fisher_exact_2x2"
                    else:
                        stat, p_value, _, _ = chi2_contingency(values)
                        test_type = "chi2"
                else:
                    stat, p_value, _, _ = chi2_contingency(values)
                    test_type = "chi2"

                counts_json, proportions_json = _serialise_counts_and_proportions(contingency)
                rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "metadata_variable": metadata_var,
                        "target_label": target_label,
                        "confidence_filter": confidence_filter_name,
                        "n_total": int(values.sum()),
                        "n_groups": int(contingency.shape[0]),
                        "test_type": test_type,
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "counts_json": counts_json,
                        "proportions_json": proportions_json,
                    }
                )

    out = pd.DataFrame(rows, columns=LABEL_ONE_VS_REST_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(
        by=["config_id", "scoring_system", "metadata_variable", "target_label", "p_value"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_validation_summary(
    rankings: pd.DataFrame,
    state_config: Sequence[Tuple[str, str]],
    score_gap_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = rankings.groupby(["config_id", "scoring_system"], dropna=False)

    for (config_id, scoring_system), sub in grouped:
        row: Dict[str, Any] = {
            "config_id": config_id,
            "scoring_system": scoring_system,
            "n_samples": int(len(sub)),
            "n_states": int(sub["n_states"].iloc[0]) if len(sub) else len(state_config),
            "mean_score_gap": float(sub["score_gap"].mean()),
            "median_score_gap": float(sub["score_gap"].median()),
            "mean_score_entropy": float(sub["score_entropy"].mean()),
            "median_score_entropy": float(sub["score_entropy"].median()),
            "confident_fraction": float((sub["score_gap"] >= score_gap_threshold).mean()),
        }
        state_counts = sub["best_cell_state"].value_counts(dropna=False)
        for label, _ in state_config:
            row[f"n_best_{safe_feature_name(label)}"] = int(state_counts.get(label, 0))
            row[f"prop_best_{safe_feature_name(label)}"] = float(state_counts.get(label, 0) / len(sub))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["config_id", "scoring_system"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    out = df.copy()
    if columns is not None:
        if out.empty:
            out = pd.DataFrame(columns=columns)
        else:
            for col in columns:
                if col not in out.columns:
                    out[col] = np.nan
            out = out[list(columns)]
    out.to_csv(path, index=False)
    log(f"Wrote {path.name} ({len(out)} rows)")


def main() -> None:
    args = parse_args()

    experiment_dir = Path(args.experiments_root) / args.experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {experiment_dir}")

    scoring_systems = ensure_scoring_systems(args.scoring_systems)
    group_test_vars_requested = parse_csv_list(args.group_test_vars, "--group-test-vars")
    correlation_vars_requested = parse_csv_list(args.correlation_vars, "--correlation-vars")
    modelling_targets_requested = parse_csv_list(args.modelling_targets, "--modelling-targets")
    covariate_cols_requested = parse_csv_list(args.covariate_cols, "--covariate-cols")
    state_config = parse_state_config(args.state_labels, args.state_suffixes)
    state_score_cols = ranking_score_columns(state_config)
    ranking_columns = ranking_output_columns(state_config)
    invert_scores = not args.no_invert_scores

    requested_metadata_columns = list(
        dict.fromkeys(
            group_test_vars_requested
            + correlation_vars_requested
            + modelling_targets_requested
            + covariate_cols_requested
            + KNOWN_METADATA_COLUMNS
        )
    )

    log(f"Experiment directory: {experiment_dir}")
    log(f"Scoring systems: {', '.join(scoring_systems)}")
    log(f"States: {', '.join(label for label, _ in state_config)}")
    log(f"Invert scores: {invert_scores}")

    sample_scores = load_sample_level_results(
        experiment_dir=experiment_dir,
        scoring_systems=scoring_systems,
        state_config=state_config,
        invert_scores=invert_scores,
        allow_aggregated_results=args.allow_aggregated_results,
    )

    metadata = load_metadata(
        metadata_path=Path(args.metadata_path),
        sample_col=args.metadata_sample_col,
        requested_metadata_columns=requested_metadata_columns,
    )
    merged = sample_scores.merge(metadata, on="sample", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples between results.csv sample identifiers and metadata.")

    available_metadata_cols = set(merged.columns)

    group_test_vars = [col for col in group_test_vars_requested if col in available_metadata_cols]
    correlation_vars = [col for col in correlation_vars_requested if col in available_metadata_cols]
    modelling_targets = [col for col in modelling_targets_requested if col in available_metadata_cols]
    covariate_cols = [col for col in covariate_cols_requested if col in available_metadata_cols]

    skipped_group = [col for col in group_test_vars_requested if col not in group_test_vars]
    skipped_corr = [col for col in correlation_vars_requested if col not in correlation_vars]
    skipped_targets = [col for col in modelling_targets_requested if col not in modelling_targets]
    skipped_covars = [col for col in covariate_cols_requested if col not in covariate_cols]

    if skipped_group:
        log("Skipping missing group-test variables: " + ", ".join(skipped_group))
    if skipped_corr:
        log("Skipping missing correlation variables: " + ", ".join(skipped_corr))
    if skipped_targets:
        log("Skipping missing modelling targets: " + ", ".join(skipped_targets))
    if skipped_covars:
        log("Skipping missing covariate columns: " + ", ".join(skipped_covars))

    log(
        "Merged modelling rows: "
        f"{len(merged)} (samples={merged['sample'].nunique()}, "
        f"configs={merged['config_id'].nunique()}, systems={merged['scoring_system'].nunique()})"
    )

    log("Running group comparison tests...")
    group_tests = run_group_tests(merged, state_score_cols, group_test_vars)
    log(f"Completed group comparison tests: {len(group_tests)} rows")

    log("Running correlation tests...")
    correlations = run_correlations(merged, state_score_cols, correlation_vars)
    log(f"Completed correlation tests: {len(correlations)} rows")

    log("Running predictive validation models...")
    model_summary, model_predictions, feature_importance = run_predictive_models(
        merged,
        state_score_cols=state_score_cols,
        modelling_targets=modelling_targets,
        covariate_columns=covariate_cols,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    log("Building per-sample ranking summaries...")
    rankings = build_rankings(merged, state_config)
    rankings_with_metadata = build_rankings_with_metadata(rankings, metadata)

    log("Building pairwise score contrasts...")
    merged_with_contrasts, contrast_features = add_score_contrasts(merged, state_config)

    log("Running score contrast tests...")
    score_contrast_tests = run_score_contrast_tests(
        merged_with_contrasts,
        contrast_features,
        group_test_vars,
    )
    log(f"Completed score contrast tests: {len(score_contrast_tests)} rows")

    log("Running label association tests on all assignments...")
    label_assoc_all = run_label_association_tests(
        rankings_with_metadata,
        metadata_vars=group_test_vars,
        confidence_filter_name="all",
    )
    log(f"Completed label association tests (all): {len(label_assoc_all)} rows")

    rankings_confident = rankings_with_metadata[
        rankings_with_metadata["score_gap"] >= args.score_gap_threshold
    ].copy()

    log(
        "Running label association tests on confident assignments "
        f"(score_gap >= {args.score_gap_threshold})..."
    )
    label_assoc_confident = run_label_association_tests(
        rankings_confident,
        metadata_vars=group_test_vars,
        confidence_filter_name=f"score_gap_ge_{args.score_gap_threshold:.2f}",
    )
    log(f"Completed label association tests (confident-only): {len(label_assoc_confident)} rows")

    target_labels = [label for label, _ in state_config]
    log("Running one-vs-rest label association tests...")
    label_ovr_all = run_label_one_vs_rest_tests(
        rankings_with_metadata,
        metadata_vars=group_test_vars,
        target_labels=target_labels,
        confidence_filter_name="all",
    )
    label_ovr_confident = run_label_one_vs_rest_tests(
        rankings_confident,
        metadata_vars=group_test_vars,
        target_labels=target_labels,
        confidence_filter_name=f"score_gap_ge_{args.score_gap_threshold:.2f}",
    )
    label_ovr = pd.concat([label_ovr_all, label_ovr_confident], ignore_index=True)
    log(f"Completed one-vs-rest label tests: {len(label_ovr)} rows")

    validation_summary = build_validation_summary(
        rankings=rankings,
        state_config=state_config,
        score_gap_threshold=args.score_gap_threshold,
    )

    write_csv(group_tests, experiment_dir / "validation_group_tests.csv", GROUP_TEST_COLUMNS)
    write_csv(correlations, experiment_dir / "validation_correlations.csv", CORRELATION_COLUMNS)
    write_csv(model_summary, experiment_dir / "validation_model_summary.csv", MODEL_SUMMARY_COLUMNS)
    write_csv(model_predictions, experiment_dir / "validation_model_predictions.csv", MODEL_PREDICTION_COLUMNS)
    write_csv(feature_importance, experiment_dir / "validation_feature_importance.csv", FEATURE_IMPORTANCE_COLUMNS)
    write_csv(rankings, experiment_dir / "validation_score_rankings.csv", ranking_columns)
    write_csv(score_contrast_tests, experiment_dir / "validation_score_contrasts.csv", SCORE_CONTRAST_COLUMNS)
    write_csv(label_assoc_all, experiment_dir / "validation_label_associations.csv", LABEL_ASSOCIATION_COLUMNS)
    write_csv(label_assoc_confident, experiment_dir / "validation_label_associations_confident.csv", LABEL_ASSOCIATION_COLUMNS)
    write_csv(label_ovr, experiment_dir / "validation_label_one_vs_rest.csv", LABEL_ONE_VS_REST_COLUMNS)
    write_csv(validation_summary, experiment_dir / "validation_summary.csv")

    log("\nFinal summary")
    log(f"- Samples used: {merged['sample'].nunique()}")
    log(f"- config_id groups analysed: {merged['config_id'].nunique()}")
    log(f"- Scoring systems analysed: {merged['scoring_system'].nunique()}")
    log(f"- States analysed: {len(state_config)}")
    log(f"- Group tests run: {len(group_tests)}")
    log(f"- Score contrast tests run: {len(score_contrast_tests)}")
    log(f"- Correlations run: {len(correlations)}")
    log(f"- Label association tests run: {len(label_assoc_all)}")
    log(f"- Confident-only label association tests run: {len(label_assoc_confident)}")
    log(f"- One-vs-rest label tests run: {len(label_ovr)}")
    log(f"- Models fit: {len(model_summary)}")


if __name__ == "__main__":
    main()
