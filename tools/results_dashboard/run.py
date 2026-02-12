"""Streamlit dashboard entry point with tabbed analysis sections.

This module orchestrates UI layout and delegates data/ranking logic to helper
modules for maintainability.
"""

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.results_dashboard.core import (
    CONFIG_PATH,
    EXPERIMENTS_DIR,
    METRICS,
    METRIC_LABELS,
    _collect_tumours_from_results,
    _format_path_label,
    _list_experiment_results,
    _load_config,
    _load_results,
    _normalize_results,
    _persist_config,
    _required_columns,
    _resolve_atac_map_path,
    _resolve_bin_sizes,
    _resolve_celltype_map_source,
    _style,
)
from tools.results_dashboard.hepa import default_hepa_labels_path
from tools.results_dashboard.sections import (
    render_accuracy_tab,
    render_bin_tab,
    render_hepa_labels_tab,
    render_hepa_state_tab,
    render_metric_tab,
    render_tumour_tab,
)

DEFAULT_HEPA_LABELS_PATH = default_hepa_labels_path(ROOT)

st.set_page_config(page_title="Results Dashboard")
_style()
st.title("Results Dashboard")
st.caption("Compare metrics, margins, and accuracy for each correct cell type.")

config = _load_config()
path: Optional[Path] = None
experiment_dir: Optional[Path] = None
with st.sidebar:
    st.header("Load data")
    experiments = _list_experiment_results()
    if not experiments:
        st.error(f"No results.csv found under {EXPERIMENTS_DIR}")
        st.stop()
    options = list(experiments.keys())
    saved_name = config.get("experiment_name")
    if saved_name not in options:
        saved_path = str(config.get("path_input", ""))
        saved_name = next(
            (name for name, rel in experiments.items() if str(rel) == saved_path),
            options[0],
        )
    experiment_name = st.selectbox(
        "Experiment",
        options=options,
        index=options.index(saved_name),
        key="experiment_name",
        on_change=_persist_config,
    )
    is_liver_experiment = "liver" in experiment_name.lower()
    path_input = str(experiments[experiment_name])
    st.session_state["path_input"] = path_input
    if not CONFIG_PATH.exists():
        _persist_config()

    weight_basis = "metric_abs"
    path = Path(path_input)
    if not path.is_absolute():
        path = ROOT / path
    experiment_dir = path.parent
    if is_liver_experiment:
        map_path = _resolve_atac_map_path(experiment_dir)
        if map_path is None:
            st.warning("Cell type map (ATAC): not found")
        else:
            scope_label = "experiment" if map_path.parent == experiment_dir else "project default"
            st.info(f"Cell type map: {scope_label} (ATAC)")
            st.caption(_format_path_label(map_path))
    else:
        map_path, map_kind, map_scope = _resolve_celltype_map_source(experiment_dir)
        if map_path is None or map_kind is None:
            st.warning("Cell type map: not found")
        else:
            kind_label = "ATAC" if map_kind == "atac" else "DNase"
            scope_label = "experiment" if map_scope == "experiment" else "project default"
            st.info(f"Cell type map: {scope_label} ({kind_label})")
            st.caption(_format_path_label(map_path))

if path is None:
    path = Path(path_input)
    if not path.is_absolute():
        path = ROOT / path
if not path.exists():
    st.error(f"File not found: {path}")
    st.stop()
if experiment_dir is None:
    experiment_dir = path.parent
df = _load_results(path)
df = _normalize_results(df)
missing = [col for col in _required_columns() if col not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
source_label = str(path)
loaded_path = str(path)
metric_labels = METRIC_LABELS.copy()

if st.session_state.get("path_input") != loaded_path:
    st.session_state["path_input"] = loaded_path
    _persist_config()

_ = source_label

total_runs = len(df)
sample_modes = (
    ", ".join(sorted(df["sample_mode"].dropna().astype(str).unique()))
    if "sample_mode" in df.columns
    else "n/a"
)
bin_sizes = sorted(_resolve_bin_sizes(df).dropna().unique().tolist())
bin_label = ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in bin_sizes) or "none"
tumour_values = sorted(_collect_tumours_from_results(df))
tumour_label = ", ".join(tumour_values) if tumour_values else "none"
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Runs</div>
            <div class='stat-value'>{total_runs:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    sample_count = int(df["sample_id"].dropna().nunique())
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Samples</div>
            <div class='stat-value'>{sample_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Tumours</div>
            <div class='stat-value'>{len(tumour_values)}</div>
            <div class='stat-meta'>{tumour_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_d:
    strategies = sorted(df["track_strategy"].dropna().unique().tolist())
    strategy_list = ", ".join(strategies) if strategies else "none"
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Tracks</div>
            <div class='stat-value'>{len(strategies)}</div>
            <div class='stat-meta'>{strategy_list}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_names = ["Accuracy vs Mutation Burden"]
if is_liver_experiment:
    tab_names.extend([
        "Hepatocyte State Breakdown",
        "Annotation Label Benchmark",
    ])
tab_names.extend([
    "Best by Bin Size",
    "Best by Tumour Type",
    "Metric Separation",
])
tabs = st.tabs(tab_names)
tab_lookup = dict(zip(tab_names, tabs))

with tab_lookup["Accuracy vs Mutation Burden"]:
    accuracy_ctx = render_accuracy_tab(
        df=df,
        experiment_dir=experiment_dir,
        weight_basis=weight_basis,
        metric_labels=metric_labels,
        metrics=METRICS,
    )

track_choice = accuracy_ctx["track_choice"]
metric_choice_label = accuracy_ctx["metric_choice_label"]
bin_choice = accuracy_ctx["bin_choice"]
tumour_choice = accuracy_ctx["tumour_choice"]
metric_label_to_key = accuracy_ctx["metric_label_to_key"]

if is_liver_experiment:
    with tab_lookup["Hepatocyte State Breakdown"]:
        render_hepa_state_tab(
            is_liver_experiment=is_liver_experiment,
            experiment_dir=experiment_dir,
            df=df,
            track_choice=track_choice,
            metric_choice_label=metric_choice_label,
            bin_choice=bin_choice,
            metric_labels=metric_labels,
            metrics=METRICS,
            weight_basis=weight_basis,
        )

    with tab_lookup["Annotation Label Benchmark"]:
        render_hepa_labels_tab(
            is_liver_experiment=is_liver_experiment,
            df=df,
            tumour_choice=tumour_choice,
            metric_labels=metric_labels,
            metrics=METRICS,
            weight_basis=weight_basis,
            default_hepa_labels_path=DEFAULT_HEPA_LABELS_PATH,
        )

with tab_lookup["Best by Bin Size"]:
    render_bin_tab(df=df, weight_basis=weight_basis, metric_labels=metric_labels)

with tab_lookup["Best by Tumour Type"]:
    render_tumour_tab(
        df=df,
        weight_basis=weight_basis,
        metric_labels=metric_labels,
        experiment_dir=experiment_dir,
    )

with tab_lookup["Metric Separation"]:
    render_metric_tab(df=df, weight_basis=weight_basis, metric_labels=metric_labels)
