"""Build per-sample modelling tables from grid-search outputs and fit tuned models.

This script reads a completed grid-search experiment directory, constructs a
structured modelling matrix, runs model selection with cross-validation for
each (config_id, scoring_system) pair, and writes modelling outputs back into
the same experiment directory.

Output files:
- model_matrix.csv
- model_scores.csv
- grid_search_summary.csv
- feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict


SCORING_COLUMN_TEMPLATES = {
    "rf_resid": "pearson_r_rf_resid_{cell}_mean_weighted",
    "pearson_local_score": "pearson_local_score_global_{cell}_mean_weighted",
    "spearman_local_score": "spearman_local_score_global_{cell}_mean_weighted",
    "pearson_r_linear_resid": "pearson_r_linear_resid_{cell}_mean_weighted",
    "spearman_r_linear_resid": "spearman_r_linear_resid_{cell}_mean_weighted",
}

CELLTYPE_TO_SUFFIX = {
    "hepatocyte_normal": "normal",
    "hepatocyte_ac": "ac",
    "hepatocyte_ah": "ah",
}

CATEGORICAL_COLUMNS = [
    "alcohol_status",
    "hbv_status",
    "hcv_status",
    "nafld_status",
    "obesity_class",
]

CONFIG_KEY_CANDIDATES = [
    "track_strategy",
    "track_param_tag",
    "covariates",
    "include_trinuc",
    "downsample_target",
    "counts_raw_bin",
    "counts_gauss_bin",
    "inv_dist_gauss_bin",
    "exp_decay_bin",
    "exp_decay_adaptive_bin",
    "counts_gauss_sigma_bins",
    "counts_gauss_sigma_units",
    "inv_dist_gauss_sigma_bins",
    "inv_dist_gauss_sigma_units",
    "inv_dist_gauss_max_distance_bp",
    "exp_decay_decay_bp",
    "exp_decay_max_distance_bp",
    "exp_decay_adaptive_k",
    "exp_decay_adaptive_min_bandwidth_bp",
    "exp_decay_adaptive_max_distance_bp",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment directory name under outputs/experiments.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("outputs/experiments"),
        help="Base directory containing experiment folders.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/derived/master_sample_metadata_lihc_fibrosis.csv"),
        help="Metadata CSV with fibrosis and covariates.",
    )
    parser.add_argument(
        "--metadata-sample-col",
        type=str,
        default="tumour_sample_submitter_id",
        help="Metadata column matching sample IDs in results.csv.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="fibrosis_ishak_score",
        choices=["fibrosis_ishak_score", "fibrosis_present"],
        help="Target variable used for model fitting.",
    )
    parser.add_argument(
        "--scoring-systems",
        type=str,
        default=",".join(SCORING_COLUMN_TEMPLATES.keys()),
        help="Comma-separated scoring systems to include.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum number of cross-validation folds to use; 5 is recommended for this dataset size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed for model fitting.",
    )
    parser.add_argument(
        "--no-invert-scores",
        action="store_true",
        help=(
            "Disable score inversion. By default, chromatin score features are "
            "multiplied by -1 so higher values represent stronger anti-correlation."
        ),
    )
    parser.add_argument(
        "--allow-aggregated-results",
        action="store_true",
        help=(
            "Allow rows where selected_sample_ids contains multiple samples. "
            "By default this is disabled because aggregated runs do not provide "
            "sample-specific chromatin scores."
        ),
    )
    return parser.parse_args()


def parse_ishak_score(value: object) -> float:
    """Parse Ishak fibrosis score from numeric or labelled text."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    token = text.split("-", maxsplit=1)[0].strip()
    try:
        return float(token)
    except ValueError:
        return np.nan


def normalise_sample_id(value: object) -> str:
    """Normalise sample IDs to trimmed strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_config_id(row: pd.Series) -> str:
    """Build a deterministic config identifier from run parameters."""
    parts: List[str] = []
    for key in CONFIG_KEY_CANDIDATES:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        parts.append(f"{key}={text}")
    return "__".join(parts)


def build_scoring_rows(
    results_df: pd.DataFrame,
    scoring_systems: List[str],
    invert_scores: bool,
    allow_aggregated_results: bool,
) -> pd.DataFrame:
    """Expand results rows into one row per (sample, config_id, scoring_system)."""
    records: List[Dict[str, object]] = []
    missing_systems = [s for s in scoring_systems if s not in SCORING_COLUMN_TEMPLATES]
    if missing_systems:
        raise ValueError(f"Unknown scoring systems: {missing_systems}")

    for _, row in results_df.iterrows():
        sample = normalise_sample_id(row.get("selected_sample_ids"))
        if not sample:
            continue
        n_selected = row.get("n_selected_samples")
        if pd.notna(n_selected) and int(n_selected) != 1 and not allow_aggregated_results:
            raise ValueError(
                "results.csv contains aggregated runs (n_selected_samples != 1). "
                "Run grid search in per-sample mode or pass --allow-aggregated-results."
            )
        if "," in sample:
            if not allow_aggregated_results:
                raise ValueError(
                    "results.csv contains comma-joined selected_sample_ids. "
                    "Cannot map aggregated rows to sample-level chromatin scores."
                )
            sample = sample.split(",", maxsplit=1)[0].strip()
        config_id = build_config_id(row)
        for scoring_system in scoring_systems:
            template = SCORING_COLUMN_TEMPLATES[scoring_system]
            scores: Dict[str, float] = {}
            missing_columns: List[str] = []
            for cell, suffix in CELLTYPE_TO_SUFFIX.items():
                column = template.format(cell=cell)
                if column not in results_df.columns:
                    missing_columns.append(column)
                    continue
                value = float(row[column])
                if invert_scores:
                    value = -value
                scores[f"score_hepatocyte_{suffix}"] = value
            if missing_columns:
                raise ValueError(
                    f"Missing expected columns for scoring system '{scoring_system}': {missing_columns}"
                )
            records.append(
                {
                    "sample": sample,
                    "config_id": config_id,
                    "scoring_system": scoring_system,
                    **scores,
                }
            )
    return pd.DataFrame.from_records(records)


def load_metadata(metadata_path: Path, sample_col: str) -> pd.DataFrame:
    """Load and standardise metadata used by modelling."""
    metadata_df = pd.read_csv(metadata_path)
    if sample_col not in metadata_df.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in metadata: {metadata_path}")
    needed = [sample_col, "fibrosis_ishak_score", *CATEGORICAL_COLUMNS]
    missing = [c for c in needed if c not in metadata_df.columns]
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {missing}")
    out = metadata_df[needed].copy()
    out = out.rename(columns={sample_col: "sample"})
    out["sample"] = out["sample"].map(normalise_sample_id)
    out["fibrosis_ishak_score"] = out["fibrosis_ishak_score"].map(parse_ishak_score)
    out["fibrosis_present"] = (out["fibrosis_ishak_score"] > 0).astype(int)
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("Unknown").astype(str).str.strip()
        out.loc[out[col] == "", col] = "Unknown"
    out = out.drop_duplicates(subset=["sample"], keep="first")
    return out


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical metadata columns."""
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS, dtype=int)
    return encoded


def model_specs(target: str, random_state: int) -> List[Tuple[str, object, Dict[str, List[object]], str]]:
    """Build model definitions, grids, and scoring function.
    """
    if target == "fibrosis_ishak_score":
        rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        ridge = Ridge()

        rf_grid = {
            "n_estimators": [300, 800],
            "max_depth": [3, 5, None],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],
            "bootstrap": [True],
        }

        ridge_grid = {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        }

        return [
            ("random_forest", rf, rf_grid, "r2"),
            ("ridge", ridge, ridge_grid, "r2"),
        ]

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    ridge = RidgeClassifier()

    rf_grid = {
        "n_estimators": [300, 800],
        "max_depth": [3, 5, None],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"],
        "bootstrap": [True],
    }

    ridge_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    }

    return [
        ("random_forest", rf, rf_grid, "accuracy"),
        ("ridge", ridge, ridge_grid, "accuracy"),
    ]

def best_and_gap(mean_scores: np.ndarray) -> Tuple[float, float]:
    """Return (best_score, score_gap) where gap = best - second_best."""
    finite = np.asarray(mean_scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    finite = np.sort(finite)[::-1]
    best = float(finite[0])
    if finite.size < 2:
        return best, float("nan")
    second = float(finite[1])
    return best, float(best - second)


def fit_models(
    matrix_df: pd.DataFrame,
    target: str,
    cv_folds: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit models for each (config_id, scoring_system) and collect outputs."""
    id_cols = {"sample", "config_id", "scoring_system", "fibrosis_ishak_score", "fibrosis_present"}
    feature_cols = [c for c in matrix_df.columns if c not in id_cols]
    score_cols = [c for c in feature_cols if c.startswith("score_hepatocyte_")]
    covariate_cols = [c for c in feature_cols if c not in score_cols]
    feature_sets = [
        ("full", feature_cols),
        ("covariates_only", covariate_cols),
    ]
    model_score_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    for (config_id, scoring_system), group in matrix_df.groupby(["config_id", "scoring_system"], dropna=False):
        group_clean = group.dropna(subset=[target]).copy()
        n_samples = len(group_clean)
        if n_samples < 2:
            continue
        y = group_clean[target].to_numpy()
        folds = min(cv_folds, n_samples)
        if folds < 2:
            continue

        for feature_set_name, active_features in feature_sets:
            if not active_features:
                continue
            X = group_clean[active_features].to_numpy(dtype=float)
            for model_name, estimator, param_grid, scorer in model_specs(target=target, random_state=random_state):
                grid = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=folds,
                    scoring=scorer,
                    n_jobs=-1,
                )
                grid.fit(X, y)
                best_model = clone(grid.best_estimator_)
                predictions = cross_val_predict(best_model, X, y, cv=folds, n_jobs=-1)

                for sample, truth, pred in zip(group_clean["sample"].tolist(), y.tolist(), predictions.tolist()):
                    model_score_rows.append(
                        {
                            "sample": sample,
                            "config_id": config_id,
                            "scoring_system": scoring_system,
                            "feature_set": feature_set_name,
                            "true_fibrosis": truth,
                            "predicted_fibrosis": pred,
                            "model": model_name,
                            "target": target,
                        }
                    )

                cv_mean = float(grid.cv_results_["mean_test_score"][grid.best_index_])
                cv_std = float(grid.cv_results_["std_test_score"][grid.best_index_])
                best_score, score_gap = best_and_gap(np.asarray(grid.cv_results_["mean_test_score"], dtype=float))
                summary_rows.append(
                    {
                        "config_id": config_id,
                        "scoring_system": scoring_system,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "best_params": json.dumps(grid.best_params_, sort_keys=True),
                        "best_score": best_score,
                        "score_gap": score_gap,
                        "cv_score_mean": cv_mean,
                        "cv_score_std": cv_std,
                        "n_samples": int(n_samples),
                        "cv_folds": int(folds),
                        "target": target,
                    }
                )

                if model_name == "random_forest":
                    fitted = clone(grid.best_estimator_)
                    fitted.fit(X, y)
                    importances = fitted.feature_importances_
                    for feature, importance in zip(active_features, importances.tolist()):
                        importance_rows.append(
                            {
                                "config_id": config_id,
                                "scoring_system": scoring_system,
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "feature": feature,
                                "importance": float(importance),
                                "target": target,
                            }
                        )

    return (
        pd.DataFrame.from_records(model_score_rows),
        annotate_selected_models(pd.DataFrame.from_records(summary_rows)),
        pd.DataFrame.from_records(importance_rows),
    )


def annotate_selected_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate best model per (config_id, scoring_system, target) using score and gap."""
    if summary_df.empty:
        return summary_df
    out = summary_df.copy()
    out["is_best_model_for_group"] = False

    group_cols = ["config_id", "scoring_system", "feature_set", "target"]
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        sub = out.loc[idx].copy()
        sub["best_score_rank"] = sub["best_score"].astype(float)
        sub["score_gap_rank"] = sub["score_gap"].astype(float)
        sub = sub.sort_values(
            by=["best_score_rank", "score_gap_rank", "model"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        best_idx = sub.index[0]
        out.loc[best_idx, "is_best_model_for_group"] = True

    return out


def main() -> None:
    """Run modelling from an existing grid-search experiment directory."""
    args = parse_args()
    experiment_dir = args.experiments_root / args.experiment_name
    results_path = experiment_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found for experiment: {results_path}")

    results_df = pd.read_csv(results_path)
    scoring_systems = [s.strip() for s in args.scoring_systems.split(",") if s.strip()]
    score_df = build_scoring_rows(
        results_df=results_df,
        scoring_systems=scoring_systems,
        invert_scores=not bool(args.no_invert_scores),
        allow_aggregated_results=bool(args.allow_aggregated_results),
    )
    metadata_df = load_metadata(args.metadata_path, args.metadata_sample_col)
    merged = score_df.merge(metadata_df, on="sample", how="inner")
    if merged.empty:
        raise ValueError("No rows after joining results.csv samples with metadata samples.")

    matrix_df = one_hot_encode(merged)
    model_scores_df, summary_df, importance_df = fit_models(
        matrix_df=matrix_df,
        target=args.target,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(experiment_dir / "model_matrix.csv", index=False)
    model_scores_df.to_csv(experiment_dir / "model_scores.csv", index=False)
    summary_df.to_csv(experiment_dir / "grid_search_summary.csv", index=False)
    importance_df.to_csv(experiment_dir / "feature_importance.csv", index=False)

    print(f"Saved model_matrix.csv: {experiment_dir / 'model_matrix.csv'}")
    print(f"Saved model_scores.csv: {experiment_dir / 'model_scores.csv'}")
    print(f"Saved grid_search_summary.csv: {experiment_dir / 'grid_search_summary.csv'}")
    print(f"Saved feature_importance.csv: {experiment_dir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
