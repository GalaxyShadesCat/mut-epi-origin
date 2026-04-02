#!/usr/bin/env python3
"""Run adjusted label-association validation for inferred state scores.

This script is a companion to ``validate_state_scores.py`` and keeps the
existing unadjusted outputs untouched. It focuses on adjusted one-vs-rest
association tests where each inferred best-state label is modelled with
logistic regression and likelihood-ratio tests.

Key features:
- Loads single-sample rows from results.csv
- Builds inferred best-state labels per sample/config/scoring-system
- Adds total mutation burden from results.csv and computes log1p burden
- Runs adjusted one-vs-rest association tests for requested exposure variables
- Supports a primary adjustment model and an optional sensitivity model
- Writes a tidy CSV with raw and FDR-adjusted p-values

Examples
--------
Primary + sensitivity models:
python scripts/validate_state_scores_adjusted.py \
  --experiment-name lihc_foxa2_top4 \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --allow-aggregated-results

HCV-only exposure in the primary model:
python scripts/validate_state_scores_adjusted.py \
  --experiment-name lihc_foxa2_top4 \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --exposure-vars hcv_status \
  --allow-aggregated-results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from validate_state_scores import (
    CONFIDENT_SCORE_GAP_THRESHOLD,
    KNOWN_METADATA_COLUMNS,
    SCORING_COLUMN_TEMPLATES,
    _derive_config_id,
    _normalise_text,
    _parse_selected_sample_ids,
    build_rankings,
    ensure_scoring_systems,
    load_metadata,
    load_sample_level_results,
    parse_csv_list,
    parse_state_config,
    ranking_output_columns,
)


DEFAULT_EXPOSURE_VARS: List[str] = [
    "hcv_status",
    "hbv_status",
    "alcohol_status",
    "nafld_status",
    "obesity_class",
]

DEFAULT_PRIMARY_ADJUSTMENT_VARS: List[str] = [
    "log1p_mutation_burden",
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
    "fibrosis_ishak_score",
]

DEFAULT_SENSITIVITY_EXTRA_VARS: List[str] = ["obesity_class"]

ADJUSTED_OUTPUT_COLUMNS: List[str] = [
    "config_id",
    "scoring_system",
    "confidence_filter",
    "model_set",
    "exposure_var",
    "target_label",
    "n_total",
    "n_positive",
    "n_negative",
    "n_exposure_levels",
    "n_exposure_terms",
    "adjustment_vars",
    "test_type",
    "statistic",
    "df",
    "p_value",
    "p_value_fdr",
    "coef_json",
    "odds_ratio_json",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run adjusted one-vs-rest association tests for inferred cell-state "
            "labels using mutation burden and metadata covariates."
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
        default="data/derived/master_sample_metadata_lihc_fibrosis.csv",
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
        "--state-labels",
        required=True,
        help="Comma-separated state labels used in downstream outputs.",
    )
    parser.add_argument(
        "--state-suffixes",
        required=True,
        help="Comma-separated suffixes used in results.csv score column names.",
    )
    parser.add_argument(
        "--mutation-burden-col",
        default="n_mutations_total",
        help="Mutation burden column in results.csv used for adjustment.",
    )
    parser.add_argument(
        "--exposure-vars",
        default=",".join(DEFAULT_EXPOSURE_VARS),
        help="Comma-separated exposure variables to test.",
    )
    parser.add_argument(
        "--primary-adjustment-vars",
        default=",".join(DEFAULT_PRIMARY_ADJUSTMENT_VARS),
        help="Comma-separated adjustment variables for the primary model.",
    )
    parser.add_argument(
        "--sensitivity-extra-vars",
        default=",".join(DEFAULT_SENSITIVITY_EXTRA_VARS),
        help="Additional variables appended to the primary model for sensitivity checks.",
    )
    parser.add_argument(
        "--skip-sensitivity-model",
        action="store_true",
        help="Disable the sensitivity model.",
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


def load_sample_level_mutation_burden(
    experiment_dir: Path,
    mutation_burden_col: str,
    allow_aggregated_results: bool,
) -> pd.DataFrame:
    results_path = experiment_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing required file: {results_path}")

    results_df = pd.read_csv(results_path)
    if mutation_burden_col not in results_df.columns:
        raise ValueError(
            f"Missing mutation burden column in results.csv: {mutation_burden_col}"
        )

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

    usable = results_df.loc[~aggregated_mask].copy()
    usable["sample"] = sample_ids.loc[usable.index].str[0].astype(str).str.strip()
    usable = usable[usable["sample"] != ""].copy()
    usable["config_id"] = _derive_config_id(usable)

    usable[mutation_burden_col] = pd.to_numeric(usable[mutation_burden_col], errors="coerce")
    out = (
        usable[["sample", "config_id", mutation_burden_col]]
        .dropna(subset=[mutation_burden_col])
        .groupby(["sample", "config_id"], as_index=False)[mutation_burden_col]
        .mean()
    )
    if out.empty:
        raise ValueError(
            f"No usable values were available for mutation burden column: {mutation_burden_col}"
        )
    return out


def _fit_logistic_and_loglik(
    x: pd.DataFrame,
    y: pd.Series,
) -> Tuple[LogisticRegression, float]:
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(x, y)
    probs = model.predict_proba(x)[:, 1]
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    y_array = y.to_numpy(dtype=float)
    log_lik = float(np.sum(y_array * np.log(probs) + (1.0 - y_array) * np.log(1.0 - probs)))
    return model, log_lik


def _intercept_only_loglik(y: pd.Series) -> float:
    y_array = y.to_numpy(dtype=float)
    p_hat = float(np.mean(y_array))
    p_hat = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    return float(np.sum(y_array * np.log(p_hat) + (1.0 - y_array) * np.log(1.0 - p_hat)))


def _encode_predictors(
    df: pd.DataFrame,
    predictors: Sequence[str],
) -> pd.DataFrame:
    x = df[list(predictors)].copy()
    cat_cols = [
        col
        for col in predictors
        if x[col].dtype == "object" or str(x[col].dtype) == "bool"
    ]
    for col in predictors:
        if col not in cat_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    encoded = pd.get_dummies(
        x,
        columns=cat_cols,
        drop_first=True,
        prefix_sep="__",
    )
    return encoded


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    n_tests = len(p_values)
    if n_tests == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranks = np.arange(1, n_tests + 1, dtype=float)
    raw = p_values[order] * float(n_tests) / ranks
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def add_fdr_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_value_fdr"] = np.nan
    if out.empty:
        return out

    for _, idx in out.groupby(["model_set", "confidence_filter"]).groups.items():
        p_vals = out.loc[idx, "p_value"].astype(float).to_numpy()
        out.loc[idx, "p_value_fdr"] = _bh_fdr(p_vals)
    return out


def run_adjusted_label_association_tests(
    rankings_with_covariates: pd.DataFrame,
    state_labels: Sequence[str],
    exposure_vars: Sequence[str],
    primary_adjustment_vars: Sequence[str],
    sensitivity_extra_vars: Sequence[str],
    run_sensitivity: bool,
    mutation_burden_col: str,
    score_gap_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    model_sets: List[Tuple[str, List[str]]] = [("primary", list(primary_adjustment_vars))]
    if run_sensitivity:
        sensitivity_adjustment = list(primary_adjustment_vars) + list(sensitivity_extra_vars)
        model_sets.append(("sensitivity", sensitivity_adjustment))

    confidence_sets: List[Tuple[str, pd.DataFrame]] = [
        ("all", rankings_with_covariates),
        (f"score_gap_ge_{score_gap_threshold:.2f}", rankings_with_covariates[rankings_with_covariates["score_gap"] >= score_gap_threshold].copy()),
    ]

    grouped = rankings_with_covariates.groupby(["config_id", "scoring_system"], dropna=False)
    for (config_id, scoring_system), _ in grouped:
        for confidence_name, confidence_df in confidence_sets:
            sub = confidence_df[
                (confidence_df["config_id"] == config_id)
                & (confidence_df["scoring_system"] == scoring_system)
            ].copy()
            if sub.empty:
                continue

            available_cols = set(sub.columns)
            available_exposure_vars = [var for var in exposure_vars if var in available_cols]

            for model_set_name, adjustment_vars_raw in model_sets:
                for exposure_var in available_exposure_vars:
                    for target_label in state_labels:
                        adjustment_vars = [
                            var
                            for var in adjustment_vars_raw
                            if var in available_cols and var != exposure_var
                        ]
                        predictors = [exposure_var] + adjustment_vars

                        required_cols = ["best_cell_state"] + predictors
                        d = sub[required_cols].copy()
                        for col in [mutation_burden_col, "log1p_mutation_burden", "fibrosis_ishak_score"]:
                            if col in d.columns:
                                d[col] = pd.to_numeric(d[col], errors="coerce")
                        d = d.dropna(subset=required_cols).copy()
                        if d.empty:
                            continue

                        y = (d["best_cell_state"].astype(str) == target_label).astype(int)
                        if y.nunique() < 2:
                            continue
                        if int(y.sum()) < 2 or int((1 - y).sum()) < 2:
                            continue

                        n_exposure_levels = int(d[exposure_var].astype(str).nunique())
                        if n_exposure_levels < 2:
                            continue

                        x_full = _encode_predictors(d, predictors)
                        exposure_terms = [
                            col
                            for col in x_full.columns
                            if col == exposure_var or col.startswith(f"{exposure_var}__")
                        ]
                        if not exposure_terms:
                            continue

                        try:
                            model_full, ll_full = _fit_logistic_and_loglik(x_full, y)
                        except Exception:
                            continue

                        reduced_predictors = [col for col in predictors if col != exposure_var]
                        if reduced_predictors:
                            x_reduced = _encode_predictors(d, reduced_predictors)
                            try:
                                _, ll_reduced = _fit_logistic_and_loglik(x_reduced, y)
                                reduced_n_params = 1 + x_reduced.shape[1]
                            except Exception:
                                continue
                        else:
                            ll_reduced = _intercept_only_loglik(y)
                            reduced_n_params = 1

                        full_n_params = 1 + x_full.shape[1]
                        df_lrt = int(full_n_params - reduced_n_params)
                        if df_lrt < 1:
                            continue

                        stat = float(2.0 * (ll_full - ll_reduced))
                        p_value = float(chi2.sf(stat, df_lrt))

                        coef_series = pd.Series(
                            model_full.coef_.reshape(-1),
                            index=x_full.columns,
                        )
                        exposure_coefs = {key: float(coef_series[key]) for key in exposure_terms}
                        exposure_ors = {
                            key: float(math.exp(exposure_coefs[key]))
                            for key in exposure_terms
                        }

                        rows.append(
                            {
                                "config_id": config_id,
                                "scoring_system": scoring_system,
                                "confidence_filter": confidence_name,
                                "model_set": model_set_name,
                                "exposure_var": exposure_var,
                                "target_label": target_label,
                                "n_total": int(len(d)),
                                "n_positive": int(y.sum()),
                                "n_negative": int((1 - y).sum()),
                                "n_exposure_levels": n_exposure_levels,
                                "n_exposure_terms": int(len(exposure_terms)),
                                "adjustment_vars": ",".join(adjustment_vars),
                                "test_type": "logistic_lrt",
                                "statistic": stat,
                                "df": df_lrt,
                                "p_value": p_value,
                                "p_value_fdr": np.nan,
                                "coef_json": json.dumps(exposure_coefs, sort_keys=True),
                                "odds_ratio_json": json.dumps(exposure_ors, sort_keys=True),
                            }
                        )

    out = pd.DataFrame(rows, columns=ADJUSTED_OUTPUT_COLUMNS)
    if out.empty:
        return out
    out = add_fdr_column(out)
    return out.sort_values(
        by=[
            "config_id",
            "scoring_system",
            "confidence_filter",
            "model_set",
            "exposure_var",
            "target_label",
            "p_value",
        ],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    out = df.copy()
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
    state_config = parse_state_config(args.state_labels, args.state_suffixes)
    ranking_columns = ranking_output_columns(state_config)
    invert_scores = not args.no_invert_scores

    exposure_vars_requested = parse_csv_list(args.exposure_vars, "--exposure-vars")
    primary_adjustment_vars_requested = parse_csv_list(
        args.primary_adjustment_vars,
        "--primary-adjustment-vars",
    )
    sensitivity_extra_vars_requested = parse_csv_list(
        args.sensitivity_extra_vars,
        "--sensitivity-extra-vars",
    )

    requested_metadata_columns = list(
        dict.fromkeys(
            exposure_vars_requested
            + primary_adjustment_vars_requested
            + sensitivity_extra_vars_requested
            + KNOWN_METADATA_COLUMNS
        )
    )
    requested_metadata_columns = [
        col
        for col in requested_metadata_columns
        if col not in {"log1p_mutation_burden", args.mutation_burden_col}
    ]

    log(f"Experiment directory: {experiment_dir}")
    log(f"Scoring systems: {', '.join(scoring_systems)}")
    log(f"States: {', '.join(label for label, _ in state_config)}")
    log(f"Invert scores: {invert_scores}")
    log(f"Mutation burden column: {args.mutation_burden_col}")

    sample_scores = load_sample_level_results(
        experiment_dir=experiment_dir,
        scoring_systems=scoring_systems,
        state_config=state_config,
        invert_scores=invert_scores,
        allow_aggregated_results=args.allow_aggregated_results,
    )

    sample_mutation_burden = load_sample_level_mutation_burden(
        experiment_dir=experiment_dir,
        mutation_burden_col=args.mutation_burden_col,
        allow_aggregated_results=args.allow_aggregated_results,
    )

    metadata = load_metadata(
        metadata_path=Path(args.metadata_path),
        sample_col=args.metadata_sample_col,
        requested_metadata_columns=requested_metadata_columns,
    )

    merged = sample_scores.merge(
        sample_mutation_burden,
        on=["sample", "config_id"],
        how="inner",
    )
    merged = merged.merge(metadata, on="sample", how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping samples across results, mutation burden and metadata."
        )

    merged["log1p_mutation_burden"] = np.log1p(
        pd.to_numeric(merged[args.mutation_burden_col], errors="coerce")
    )
    merged = merged.dropna(subset=["log1p_mutation_burden"]).copy()

    support_cols = list(
        dict.fromkeys(
            [args.mutation_burden_col, "log1p_mutation_burden"]
            + requested_metadata_columns
        )
    )
    support_cols = [col for col in support_cols if col in merged.columns]

    rankings = build_rankings(merged, state_config)
    support = (
        merged[["sample", "config_id", "scoring_system"] + support_cols]
        .drop_duplicates(subset=["sample", "config_id", "scoring_system"], keep="first")
        .copy()
    )
    rankings_with_covariates = rankings.merge(
        support,
        on=["sample", "config_id", "scoring_system"],
        how="left",
    )

    available_cols = set(rankings_with_covariates.columns)
    exposure_vars = [col for col in exposure_vars_requested if col in available_cols]
    primary_adjustment_vars = [
        col for col in primary_adjustment_vars_requested if col in available_cols
    ]
    sensitivity_extra_vars = [
        col for col in sensitivity_extra_vars_requested if col in available_cols
    ]

    skipped_exposures = [col for col in exposure_vars_requested if col not in exposure_vars]
    skipped_primary = [
        col
        for col in primary_adjustment_vars_requested
        if col not in primary_adjustment_vars
    ]
    skipped_sensitivity = [
        col
        for col in sensitivity_extra_vars_requested
        if col not in sensitivity_extra_vars
    ]

    if skipped_exposures:
        log("Skipping missing exposure variables: " + ", ".join(skipped_exposures))
    if skipped_primary:
        log("Skipping missing primary adjustment variables: " + ", ".join(skipped_primary))
    if skipped_sensitivity:
        log("Skipping missing sensitivity variables: " + ", ".join(skipped_sensitivity))

    log(
        "Merged modelling rows: "
        f"{len(rankings_with_covariates)} (samples={rankings_with_covariates['sample'].nunique()}, "
        f"configs={rankings_with_covariates['config_id'].nunique()}, "
        f"systems={rankings_with_covariates['scoring_system'].nunique()})"
    )

    log("Running adjusted one-vs-rest association tests...")
    adjusted = run_adjusted_label_association_tests(
        rankings_with_covariates=rankings_with_covariates,
        state_labels=[label for label, _ in state_config],
        exposure_vars=exposure_vars,
        primary_adjustment_vars=primary_adjustment_vars,
        sensitivity_extra_vars=sensitivity_extra_vars,
        run_sensitivity=not args.skip_sensitivity_model,
        mutation_burden_col=args.mutation_burden_col,
        score_gap_threshold=args.score_gap_threshold,
    )
    log(f"Completed adjusted tests: {len(adjusted)} rows")

    write_csv(
        rankings,
        experiment_dir / "validation_score_rankings_adjusted_inputs.csv",
        ranking_columns,
    )
    write_csv(
        adjusted,
        experiment_dir / "validation_label_associations_adjusted.csv",
        ADJUSTED_OUTPUT_COLUMNS,
    )

    log("\nFinal summary")
    log(f"- Samples used: {rankings_with_covariates['sample'].nunique()}")
    log(
        "- config_id groups analysed: "
        f"{rankings_with_covariates['config_id'].nunique()}"
    )
    log(
        "- Scoring systems analysed: "
        f"{rankings_with_covariates['scoring_system'].nunique()}"
    )
    log(f"- States analysed: {len(state_config)}")
    log(f"- Adjusted tests run: {len(adjusted)}")


if __name__ == "__main__":
    main()
