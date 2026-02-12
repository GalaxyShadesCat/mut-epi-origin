"""Hepatocyte annotation helpers for the results dashboard.

This module handles annotation label loading, canonical label mapping,
accuracy summaries, and weighted-margin summaries for hepatocyte-focused
dashboard views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_HEPA_LABELS_REL = Path("data/processed/hepa_labels/hepa_labels_from_annotations.csv")


def default_hepa_labels_path(root: Path) -> Path:
    return root / DEFAULT_HEPA_LABELS_REL


def canonical_hepa_label(value: str) -> str:
    raw = str(value).strip().lower()
    if not raw:
        return ""
    if raw in {"normal", "hepatocyte_normal"}:
        return "hepatocyte_normal"
    if raw in {"ah", "hepatocyte_ah"}:
        return "hepatocyte_ah"
    if raw in {"ac", "hepatocyte_ac"}:
        return "hepatocyte_ac"
    if raw in {"ambiguous", "hepatocyte_ambiguous"}:
        return "hepatocyte_ambiguous"
    return raw


def hepa_label_display(value: str) -> str:
    canonical = canonical_hepa_label(value)
    if canonical == "hepatocyte_normal":
        return "Normal"
    if canonical == "hepatocyte_ah":
        return "Alcoholic hepatitis (AH)"
    if canonical == "hepatocyte_ac":
        return "Alcohol-associated cirrhosis (AC)"
    if canonical == "hepatocyte_ambiguous":
        return "Ambiguous"
    return str(value)


def hepa_short_label(value: str) -> str:
    canonical = canonical_hepa_label(value)
    if canonical == "hepatocyte_normal":
        return "Normal"
    if canonical == "hepatocyte_ah":
        return "AH"
    if canonical == "hepatocyte_ac":
        return "AC"
    if canonical == "hepatocyte_ambiguous":
        return "Ambiguous"
    return str(value)


def load_hepa_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels.columns = [str(c).strip() for c in labels.columns]
    required = {"sample_id", "cell_type_label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError("Missing required hepa label columns: " + ", ".join(missing))
    out = labels[["sample_id", "cell_type_label"]].copy()
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["cell_type_label"] = out["cell_type_label"].map(canonical_hepa_label)
    out = out[out["sample_id"] != ""].drop_duplicates(subset=["sample_id"], keep="first")
    return out


def _standout_weights(margins: pd.Series, best_values: pd.Series) -> pd.Series:
    margin_vals = pd.to_numeric(margins, errors="coerce")
    best_vals = pd.to_numeric(best_values, errors="coerce")
    second_vals = best_vals - margin_vals
    eps = 1e-6
    weights = margin_vals / (second_vals.abs() + eps)
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _weighted_correct_margin(sub: pd.DataFrame) -> float:
    correct = sub[sub["is_correct"]].copy()
    margins = pd.to_numeric(correct["margin"], errors="coerce")
    valid = margins.notna()
    if not valid.any():
        return float("nan")
    margins = margins[valid]
    weights = pd.to_numeric(correct.loc[margins.index, "margin_weight"], errors="coerce")
    if weights.notna().any() and float(weights.sum()) != 0.0:
        return float((margins * weights).sum() / weights.sum())
    return float(margins.mean())


def hepa_label_prediction_rows(
    df: pd.DataFrame,
    hepa_labels: pd.DataFrame,
    metrics: dict,
    resolve_bin_sizes: Callable[[pd.DataFrame], pd.Series],
    norm_tumour_label: Callable[[str], str],
    weight_basis: str = "metric_abs",
    tumour_choice: str = "All",
    true_celltype_filter: str = "all",
) -> pd.DataFrame:
    if "sample_id" not in df.columns or "track_strategy" not in df.columns:
        return pd.DataFrame()
    if hepa_labels.empty:
        return pd.DataFrame()

    d = df.copy()
    d["sample_id"] = d["sample_id"].astype(str).str.strip()
    if tumour_choice != "All" and "selected_tumour_types" in d.columns:

        def _has_tumour(value: str) -> bool:
            for tumour in str(value).split(","):
                if norm_tumour_label(tumour) == tumour_choice:
                    return True
            return False

        d = d[d["selected_tumour_types"].apply(_has_tumour)].copy()
        if d.empty:
            return pd.DataFrame()

    label_map = hepa_labels.set_index("sample_id")["cell_type_label"]
    d = d[d["sample_id"].isin(label_map.index)].copy()
    if d.empty:
        return pd.DataFrame()

    d["bin_size"] = resolve_bin_sizes(d)
    d = d[d["bin_size"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for metric, spec in metrics.items():
        best_cell_col = spec["best_cell"]
        best_margin_col = spec["best_margin"]
        best_value_col = spec["best_value"]
        required_cols = ["sample_id", "track_strategy", "bin_size", best_cell_col]
        if weight_basis == "n_bins_total" and "n_bins_total" in d.columns:
            required_cols.append("n_bins_total")
        if best_margin_col in d.columns:
            required_cols.append(best_margin_col)
        if best_value_col in d.columns:
            required_cols.append(best_value_col)
        if best_cell_col not in d.columns:
            continue
        subset = d[required_cols].copy()
        subset["pred_celltype"] = subset[best_cell_col].map(canonical_hepa_label)
        subset["true_celltype"] = subset["sample_id"].map(label_map).map(canonical_hepa_label)
        if best_margin_col in subset.columns:
            subset["margin"] = pd.to_numeric(subset[best_margin_col], errors="coerce")
        else:
            subset["margin"] = float("nan")
        if weight_basis == "metric_abs" and best_value_col in subset.columns:
            subset["margin_weight"] = _standout_weights(subset["margin"], subset[best_value_col])
        elif weight_basis == "n_bins_total" and "n_bins_total" in subset.columns:
            subset["margin_weight"] = pd.to_numeric(subset["n_bins_total"], errors="coerce")
        else:
            subset["margin_weight"] = 1.0
        subset = subset[
            (subset["pred_celltype"] != "")
            & (subset["true_celltype"] != "")
        ].copy()
        if true_celltype_filter != "all":
            subset = subset[subset["true_celltype"] == true_celltype_filter].copy()
        if subset.empty:
            continue
        subset["metric"] = metric
        subset["is_correct"] = subset["pred_celltype"] == subset["true_celltype"]
        rows.append(subset)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def hepa_label_accuracy_rankings(
    df: pd.DataFrame,
    hepa_labels: pd.DataFrame,
    metrics: dict,
    resolve_bin_sizes: Callable[[pd.DataFrame], pd.Series],
    norm_tumour_label: Callable[[str], str],
    weight_basis: str = "metric_abs",
    tumour_choice: str = "All",
    true_celltype_filter: str = "all",
) -> pd.DataFrame:
    pred_rows = hepa_label_prediction_rows(
        df,
        hepa_labels,
        metrics=metrics,
        resolve_bin_sizes=resolve_bin_sizes,
        norm_tumour_label=norm_tumour_label,
        weight_basis=weight_basis,
        tumour_choice=tumour_choice,
        true_celltype_filter=true_celltype_filter,
    )
    if pred_rows.empty:
        return pd.DataFrame()
    out = (
        pred_rows.groupby(["track_strategy", "metric", "bin_size"], dropna=False)
        .agg(
            n_runs=("is_correct", "size"),
            n_correct=("is_correct", "sum"),
            n_samples=("sample_id", "nunique"),
        )
        .reset_index()
    )
    out["accuracy"] = out["n_correct"] / out["n_runs"]
    out["accuracy_overall"] = out["accuracy"]

    per_class = (
        pred_rows.groupby(
            ["track_strategy", "metric", "bin_size", "true_celltype"],
            dropna=False,
        )
        .agg(
            class_total=("is_correct", "size"),
            class_correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    per_class["class_accuracy"] = per_class["class_correct"] / per_class["class_total"]
    balanced = (
        per_class.groupby(["track_strategy", "metric", "bin_size"], dropna=False)
        .agg(
            accuracy_balanced=("class_accuracy", "mean"),
            n_true_celltypes=("true_celltype", "nunique"),
        )
        .reset_index()
    )
    out = out.merge(
        balanced,
        on=["track_strategy", "metric", "bin_size"],
        how="left",
    )
    margin = (
        pred_rows.groupby(["track_strategy", "metric", "bin_size"], dropna=False)
        .apply(_weighted_correct_margin, include_groups=False)
        .rename("avg_margin")
        .reset_index()
    )
    out = out.merge(
        margin,
        on=["track_strategy", "metric", "bin_size"],
        how="left",
    )
    out["accuracy_balanced"] = pd.to_numeric(out["accuracy_balanced"], errors="coerce")
    return out.sort_values(
        ["accuracy", "avg_margin", "n_runs", "n_samples", "track_strategy", "metric", "bin_size"],
        ascending=[False, False, False, False, True, True, True],
    ).reset_index(drop=True)
