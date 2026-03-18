"""Streamlit dashboard for flexible inferred-state validation outputs.

Run with:
streamlit run tools/state_validation_dashboard.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.results_dashboard.core import _style

EXPERIMENTS_DIR = ROOT / "outputs" / "experiments"
DEFAULT_METADATA_PATH = ROOT / "data" / "derived" / "master_sample_metadata_lihc_fibrosis.csv"

VALIDATION_FILES = {
    "group_tests": "validation_group_tests.csv",
    "correlations": "validation_correlations.csv",
    "model_summary": "validation_model_summary.csv",
    "score_rankings": "validation_score_rankings.csv",
    "score_contrasts": "validation_score_contrasts.csv",
    "label_associations": "validation_label_associations.csv",
    "label_associations_confident": "validation_label_associations_confident.csv",
    "label_one_vs_rest": "validation_label_one_vs_rest.csv",
}


def _clean_text(value: object) -> object:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "none", "null", "nan"}:
        return np.nan
    return text


@st.cache_data(show_spinner=False)
def list_experiment_dirs() -> List[str]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return [p.name for p in sorted(EXPERIMENTS_DIR.iterdir()) if p.is_dir()]


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_experiment_data(experiment_name: str) -> Dict[str, pd.DataFrame]:
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    out: Dict[str, pd.DataFrame] = {}
    for key, filename in VALIDATION_FILES.items():
        path = experiment_dir / filename
        out[key] = load_csv(path) if path.exists() else pd.DataFrame()
    return out


def _detect_sample_col(df: pd.DataFrame) -> str | None:
    candidates = ["sample", "tumour_sample_submitter_id", "tumor_sample_submitter_id", "tumour_sample_id", "tumor_sample_id"]
    for col in candidates:
        if col in df.columns:
            return col
    fuzzy = [c for c in df.columns if "sample" in c.lower()]
    return fuzzy[0] if fuzzy else None


@st.cache_data(show_spinner=False)
def load_metadata(path: Path, sample_col_override: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sample"])
    df = pd.read_csv(path, dtype="object")
    sample_col = sample_col_override if sample_col_override in df.columns else _detect_sample_col(df)
    if sample_col is None:
        return pd.DataFrame(columns=["sample"])
    out = df.copy()
    out["sample"] = out[sample_col].map(_clean_text)
    out = out.dropna(subset=["sample"]).copy()
    out["sample"] = out["sample"].astype(str)
    for col in out.columns:
        if col in {"sample", sample_col}:
            continue
        out[col] = out[col].map(_clean_text)
    if sample_col != "sample":
        out = out.drop(columns=[sample_col])
    return out.drop_duplicates(subset=["sample"], keep="first")


def _score_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("score_")])


def _with_ranking_metrics(rankings: pd.DataFrame) -> pd.DataFrame:
    out = rankings.copy()
    scores = _score_columns(out)
    for col in scores:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "best_score" not in out.columns and scores:
        out["best_score"] = out[scores].max(axis=1)
    if "second_best_score" not in out.columns and scores:
        out["second_best_score"] = out[scores].apply(
            lambda r: np.sort(r.to_numpy(dtype=float))[-2] if r.notna().sum() >= 2 else np.nan,
            axis=1,
        )
    if "score_gap" not in out.columns and {"best_score", "second_best_score"}.issubset(out.columns):
        out["score_gap"] = out["best_score"] - out["second_best_score"]
    return out


def _parse_json_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        parsed = []
        for value in out[col]:
            if value is None or pd.isna(value):
                parsed.append({})
            else:
                try:
                    parsed.append(json.loads(str(value)))
                except json.JSONDecodeError:
                    parsed.append({})
        out[f"{col}_parsed"] = parsed
    return out


def _summarise_tests(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    required = list(group_cols) + ["p_value"]
    if df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame(columns=list(group_cols) + ["n_tests", "n_significant", "min_p_value"])
    d = df.copy()
    d["p_value"] = pd.to_numeric(d["p_value"], errors="coerce")
    d = d.dropna(subset=["p_value"]).copy()
    if d.empty:
        return pd.DataFrame(columns=list(group_cols) + ["n_tests", "n_significant", "min_p_value"])
    return (
        d.groupby(list(group_cols), dropna=False)
        .agg(
            n_tests=("p_value", "size"),
            n_significant=("p_value", lambda s: int((s < 0.05).sum())),
            min_p_value=("p_value", "min"),
        )
        .reset_index()
        .sort_values(["n_significant", "min_p_value", "n_tests"], ascending=[False, True, False])
    )


def _filter_common(df: pd.DataFrame, scoring_system: str, config_id: str) -> pd.DataFrame:
    out = df.copy()
    if scoring_system != "All" and "scoring_system" in out.columns:
        out = out[out["scoring_system"] == scoring_system].copy()
    if config_id != "All" and "config_id" in out.columns:
        out = out[out["config_id"] == config_id].copy()
    return out


def _collect_metadata_variables(
    metadata_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    *validation_frames: pd.DataFrame,
) -> List[str]:
    """Collect metadata variable names from metadata and validation outputs only."""
    candidates: set[str] = set()

    if not metadata_df.empty:
        for col in metadata_df.columns:
            if col != "sample":
                candidates.add(str(col))

    for frame in validation_frames:
        if not frame.empty and "metadata_variable" in frame.columns:
            values = (
                frame["metadata_variable"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            candidates.update(v for v in values if v)

    # Keep only variables actually present in merged rows used for plotting/filtering.
    merged_cols = set(merged_df.columns) if not merged_df.empty else set()
    resolved = sorted([name for name in candidates if name in merged_cols])
    return resolved


def _is_numeric(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().sum() >= 3


def _contrast_series(df: pd.DataFrame, contrast_name: str) -> pd.Series | None:
    if contrast_name in df.columns:
        return pd.to_numeric(df[contrast_name], errors="coerce")
    match = re.fullmatch(r"delta_(.+)_minus_(.+)", contrast_name)
    if not match:
        return None
    left = f"score_{match.group(1)}"
    right = f"score_{match.group(2)}"
    if left not in df.columns or right not in df.columns:
        return None
    return pd.to_numeric(df[left], errors="coerce") - pd.to_numeric(df[right], errors="coerce")


def run_dashboard() -> None:
    st.set_page_config(page_title="State Validation Dashboard", layout="wide")
    _style()
    st.title("State Validation Dashboard")

    with st.sidebar:
        experiments = list_experiment_dirs()
        if not experiments:
            st.error(f"No experiment directories found under {EXPERIMENTS_DIR}")
            st.stop()
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
        experiment_name = st.selectbox("Experiment", experiments, index=0)
        metadata_path_str = st.text_input("Metadata CSV path", str(DEFAULT_METADATA_PATH))
        metadata_sample_override = st.text_input("Metadata sample column (optional)", "")

    data = load_experiment_data(experiment_name)
    rankings = _with_ranking_metrics(data["score_rankings"])
    metadata = load_metadata(Path(metadata_path_str), metadata_sample_override)
    merged = rankings.merge(metadata, on="sample", how="left") if (not rankings.empty and "sample" in rankings.columns and not metadata.empty) else rankings

    group_tests = _parse_json_columns(data["group_tests"], ["effect_summary_json", "group_summary_json"])
    correlations = data["correlations"]
    score_contrasts = _parse_json_columns(data["score_contrasts"], ["effect_summary_json", "group_summary_json"])
    label_assoc = _parse_json_columns(data["label_associations"], ["counts_json", "proportions_json"])
    label_assoc_conf = _parse_json_columns(data["label_associations_confident"], ["counts_json", "proportions_json"])
    label_ovr = _parse_json_columns(data["label_one_vs_rest"], ["counts_json", "proportions_json"])
    model_summary = data["model_summary"]

    score_features = _score_columns(merged)
    scoring_options = sorted(set(pd.concat([merged.get("scoring_system", pd.Series(dtype="object")), group_tests.get("scoring_system", pd.Series(dtype="object")), model_summary.get("scoring_system", pd.Series(dtype="object"))], ignore_index=True).dropna().astype(str)))
    config_options = sorted(set(pd.concat([merged.get("config_id", pd.Series(dtype="object")), group_tests.get("config_id", pd.Series(dtype="object")), model_summary.get("config_id", pd.Series(dtype="object"))], ignore_index=True).dropna().astype(str)))

    metadata_options = _collect_metadata_variables(
        metadata,
        merged,
        group_tests,
        correlations,
        score_contrasts,
        label_assoc,
        label_assoc_conf,
        label_ovr,
    )

    with st.sidebar:
        scoring_system = st.selectbox("Scoring system", ["All"] + scoring_options)
        config_id = st.selectbox("config_id", ["All"] + config_options)
        metadata_var = st.selectbox("Metadata variable", metadata_options if metadata_options else ["<none>"])
        score_feature = st.selectbox("Score feature", score_features if score_features else ["<none>"])
        threshold = st.number_input("Ambiguous score-gap threshold", min_value=0.0, value=0.01, step=0.005)

    filtered_scores = _filter_common(merged, scoring_system, config_id)
    filtered_group = _filter_common(group_tests, scoring_system, config_id)
    filtered_corr = _filter_common(correlations, scoring_system, config_id)
    filtered_contrast = _filter_common(score_contrasts, scoring_system, config_id)
    filtered_label = _filter_common(label_assoc, scoring_system, config_id)
    filtered_label_conf = _filter_common(label_assoc_conf, scoring_system, config_id)
    filtered_ovr = _filter_common(label_ovr, scoring_system, config_id)
    filtered_model = _filter_common(model_summary, scoring_system, config_id)

    st.caption(
        f"Current view: {len(filtered_scores)} ranking rows, "
        f"{len(filtered_group)} group tests, {len(filtered_corr)} correlations, "
        f"{len(filtered_model)} model rows."
    )

    tabs = st.tabs(["Assignment confidence", "Score-level", "Label-level", "State contrasts", "Models"])

    with tabs[0]:
        if filtered_scores.empty:
            st.info("No ranking rows are available for this selection.")
        else:
            valid_gap = pd.to_numeric(filtered_scores.get("score_gap"), errors="coerce").dropna()
            ambiguous = float((valid_gap < threshold).mean()) if not valid_gap.empty else np.nan
            st.caption(
                f"Samples: {filtered_scores['sample'].nunique() if 'sample' in filtered_scores.columns else 0}; "
                f"median score gap: {valid_gap.median():.4f}" if not valid_gap.empty else "No valid score_gap values available."
            )
            st.metric("Ambiguous assignments", "NA" if pd.isna(ambiguous) else f"{ambiguous:.1%}")
            if "best_cell_state" in filtered_scores.columns:
                state_counts = filtered_scores["best_cell_state"].fillna("missing").value_counts(dropna=False).rename_axis("best_cell_state").reset_index(name="n")
                st.altair_chart(alt.Chart(state_counts).mark_bar().encode(x="best_cell_state:N", y="n:Q", tooltip=["best_cell_state", "n"]), use_container_width=True)

    with tabs[1]:
        st.markdown("### Table 1: Group Difference Tests")
        st.caption(
            "Compares score values between metadata groups (for example yes vs no) "
            "using tests such as Mann-Whitney or Kruskal."
        )

        if "metadata_variable" in filtered_group.columns:
            filtered_group = filtered_group[filtered_group["metadata_variable"] == metadata_var].copy()
        if "score_feature" in filtered_group.columns:
            filtered_group = filtered_group[filtered_group["score_feature"] == score_feature].copy()
        st.caption(f"Group tests shown: {len(filtered_group)} rows for metadata `{metadata_var}` and score `{score_feature}`.")
        st.dataframe(filtered_group, use_container_width=True)

        st.markdown("### Table 2: Correlation Tests")
        st.caption(
            "Measures numeric correlation between a score and a metadata variable "
            "using Pearson or Spearman correlation."
        )

        if "metadata_variable" in filtered_corr.columns:
            filtered_corr = filtered_corr[filtered_corr["metadata_variable"] == metadata_var].copy()
        if "score_feature" in filtered_corr.columns:
            filtered_corr = filtered_corr[filtered_corr["score_feature"] == score_feature].copy()
        if not filtered_corr.empty:
            st.dataframe(filtered_corr, use_container_width=True)
        else:
            st.info(
                f"No correlation rows for metadata `{metadata_var}` and score `{score_feature}` "
                "in this experiment output."
            )

        if not filtered_scores.empty and metadata_var in filtered_scores.columns and score_feature in filtered_scores.columns:
            d = filtered_scores[["sample", metadata_var, score_feature]].dropna().copy()
            if not d.empty:
                if _is_numeric(d[metadata_var]):
                    d[metadata_var] = pd.to_numeric(d[metadata_var], errors="coerce")
                    plot = alt.Chart(d).mark_circle(size=65, opacity=0.6).encode(x=f"{metadata_var}:Q", y=f"{score_feature}:Q", tooltip=["sample", metadata_var, score_feature])
                    plot += plot.transform_regression(metadata_var, score_feature).mark_line(color="#d62728")
                else:
                    plot = alt.Chart(d).mark_boxplot(size=35).encode(x=f"{metadata_var}:N", y=f"{score_feature}:Q")
                st.altair_chart(plot, use_container_width=True)

    with tabs[2]:
        label_combined = pd.concat([filtered_label, filtered_label_conf], ignore_index=True)
        summary = _summarise_tests(label_combined, ["metadata_variable", "confidence_filter"])
        st.caption(f"Label-association summary rows: {len(summary)}")
        st.dataframe(summary, use_container_width=True)
        st.markdown("One-vs-rest")
        st.dataframe(filtered_ovr, use_container_width=True)

    with tabs[3]:
        contrast_options = sorted(set(filtered_contrast.get("contrast_feature", pd.Series(dtype="object")).dropna().astype(str)))
        if not contrast_options and score_features:
            suffixes = [c.removeprefix("score_") for c in score_features]
            contrast_options = [f"delta_{suffixes[i]}_minus_{suffixes[j]}" for i in range(len(suffixes)) for j in range(i + 1, len(suffixes))]
        contrast_name = st.selectbox("Contrast feature", contrast_options if contrast_options else ["<none>"])

        d_contrast = filtered_contrast.copy()
        if "contrast_feature" in d_contrast.columns and contrast_name != "<none>":
            d_contrast = d_contrast[d_contrast["contrast_feature"] == contrast_name].copy()
        if "metadata_variable" in d_contrast.columns and metadata_var != "<none>":
            d_contrast = d_contrast[d_contrast["metadata_variable"] == metadata_var].copy()
        st.caption(f"Contrast test rows shown: {len(d_contrast)}")
        st.dataframe(d_contrast, use_container_width=True)

        if not filtered_scores.empty and metadata_var in filtered_scores.columns and contrast_name != "<none>":
            s = _contrast_series(filtered_scores, contrast_name)
            if s is not None:
                d = filtered_scores[["sample", metadata_var]].copy()
                d["contrast_value"] = s
                d = d.dropna(subset=[metadata_var, "contrast_value"]).copy()
                if not d.empty:
                    if _is_numeric(d[metadata_var]):
                        d[metadata_var] = pd.to_numeric(d[metadata_var], errors="coerce")
                        plot = alt.Chart(d).mark_circle(size=65, opacity=0.6).encode(x=f"{metadata_var}:Q", y="contrast_value:Q", tooltip=["sample", metadata_var, "contrast_value"])
                        plot += plot.transform_regression(metadata_var, "contrast_value").mark_line(color="#d62728")
                    else:
                        plot = alt.Chart(d).mark_boxplot(size=35).encode(x=f"{metadata_var}:N", y="contrast_value:Q")
                    st.altair_chart(plot, use_container_width=True)

    with tabs[4]:
        st.caption(f"Model summary rows shown: {len(filtered_model)}")
        st.dataframe(filtered_model, use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
