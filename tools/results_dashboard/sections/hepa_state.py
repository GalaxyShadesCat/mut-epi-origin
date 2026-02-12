"""Hepatocyte state breakdown tab renderer."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tools.results_dashboard.core import (
    _build_tumour_options,
    _canonical_state,
    _format_pct,
    _hepatocyte_state_map,
    _load_atac_celltype_map,
    _load_dnase_celltype_map,
    _norm_tumour_label,
    _resolve_bin_sizes,
    _resolve_mutation_burden,
    _state_label,
    track_metric_rankings_by_state_prediction,
)


def render_hepa_state_tab(
    is_liver_experiment,
    experiment_dir,
    df,
    track_choice,
    metric_choice_label,
    bin_choice,
    metric_labels,
    metrics,
    weight_basis,
):
    METRICS = metrics
    metric_label_to_key = {metric_labels.get(m, m): m for m in METRICS}

    if is_liver_experiment:
        st.subheader("Hepatocyte state breakdown")
        atac_map = _load_atac_celltype_map(experiment_dir)
        if atac_map is None:
            st.info("Hepatocyte state summary unavailable: celltype_atac_map.json not found.")
        else:
            state_by_celltype = _hepatocyte_state_map(atac_map)
            if not state_by_celltype:
                st.info("Hepatocyte state summary unavailable: no hepatocyte states found.")
            else:
                track_choices = ["All"] + sorted(df["track_strategy"].dropna().astype(str).unique().tolist())
                metric_keys = list(METRICS.keys())
                metric_choices = ["All"] + [metric_labels.get(m, m) for m in metric_keys]
                bin_values = _resolve_bin_sizes(df).dropna().unique().tolist()
                bin_values = sorted(bin_values)
                bin_labels = [
                    int(x) if float(x).is_integer() else float(x) for x in bin_values
                ]
                bin_choices = ["All"] + [str(x) for x in bin_labels]
                top_n_choices = ["All", 3, 5, 10]
                celltype_map = _load_dnase_celltype_map(experiment_dir)
                tumour_options, tumour_label_to_key = _build_tumour_options(df, celltype_map)

                row_a, row_b = st.columns(2)
                with row_a:
                    state_track_choice = st.selectbox(
                        "Track",
                        options=track_choices,
                        index=track_choices.index(track_choice) if track_choice in track_choices else 0,
                        key="hepa_state_track_choice",
                    )
                with row_b:
                    state_metric_choice_label = st.selectbox(
                        "Metric",
                        options=metric_choices,
                        index=(
                            metric_choices.index(metric_choice_label)
                            if metric_choice_label in metric_choices
                            else 0
                        ),
                        key="hepa_state_metric_choice_label",
                    )

                row_c, row_d, row_e = st.columns(3)
                with row_c:
                    state_bin_choice = st.selectbox(
                        "Bin size",
                        options=bin_choices,
                        index=bin_choices.index(bin_choice) if bin_choice in bin_choices else 0,
                        key="hepa_state_bin_choice",
                    )
                with row_d:
                    tumour_default_label = next(
                        (label for label, key in tumour_label_to_key.items() if key == "All"),
                        tumour_options[0],
                    )
                    state_tumour_choice_label = st.selectbox(
                        "Tumour",
                        options=tumour_options,
                        index=(
                            tumour_options.index(tumour_default_label)
                            if tumour_default_label in tumour_options
                            else 0
                        ),
                        key="hepa_state_tumour_choice",
                    )
                    state_tumour_choice = tumour_label_to_key.get(state_tumour_choice_label, "All")
                with row_e:
                    state_top_n_choice = st.selectbox(
                        "Top by accuracy",
                        options=top_n_choices,
                        index=top_n_choices.index(5),
                        key="hepa_state_top_n_choice",
                    )

                def _apply_tumour_filter(frame: pd.DataFrame) -> pd.DataFrame:
                    if state_tumour_choice == "All":
                        return frame
                    if "selected_tumour_types" not in frame.columns:
                        return frame.iloc[0:0].copy()

                    def _has_tumour(value: str) -> bool:
                        for tumour in str(value).split(","):
                            if _norm_tumour_label(tumour) == state_tumour_choice:
                                return True
                        return False

                    return frame[frame["selected_tumour_types"].apply(_has_tumour)].copy()

                state_df = df.copy()
                if state_track_choice != "All":
                    state_df = state_df[
                        state_df["track_strategy"].astype(str).str.strip() == state_track_choice
                    ].copy()
                state_df["bin_size"] = _resolve_bin_sizes(state_df)
                if state_bin_choice != "All":
                    try:
                        bin_value = float(state_bin_choice)
                    except ValueError:
                        bin_value = None
                    if bin_value is not None:
                        state_df = state_df[state_df["bin_size"] == bin_value].copy()
                state_df = _apply_tumour_filter(state_df)
                if state_df.empty:
                    st.info("Hepatocyte state summary unavailable: no runs after filtering.")
                else:
                    state_metric_choice = metric_label_to_key.get(state_metric_choice_label)
                    state_rows = []
                    if state_metric_choice is None:
                        for metric, spec in METRICS.items():
                            best_cell_col = spec["best_cell"]
                            if best_cell_col not in state_df.columns:
                                continue
                            subset = state_df.copy()
                            subset["metric_label"] = metric_labels.get(metric, metric)
                            subset["pred_celltype"] = (
                                subset[best_cell_col].astype(str).str.strip().str.lower()
                            )
                            state_rows.append(subset)
                    else:
                        best_cell_col = METRICS[state_metric_choice]["best_cell"]
                        if best_cell_col in state_df.columns:
                            subset = state_df.copy()
                            subset["metric_label"] = state_metric_choice_label
                            subset["pred_celltype"] = (
                                subset[best_cell_col].astype(str).str.strip().str.lower()
                            )
                            state_rows.append(subset)
                    if not state_rows:
                        st.info("Hepatocyte state summary unavailable: selected metric not available.")
                    else:
                        state_df = pd.concat(state_rows, ignore_index=True)
                        state_df["state_raw"] = state_df["pred_celltype"].map(state_by_celltype)
                        state_df["state"] = state_df["state_raw"].map(_canonical_state)
                        state_df = state_df[state_df["state"].notna()].copy()
                        state_df = state_df[state_df["state"].astype(str).str.strip() != ""].copy()
                        if state_df.empty:
                            st.info("Hepatocyte state summary unavailable: no hepatocyte predictions.")
                        else:
                            state_df["mutation_burden"], _ = _resolve_mutation_burden(state_df)
                            total_state_samples = int(state_df["sample_id"].nunique())
                            state_counts = (
                                state_df["state"]
                                .value_counts()
                                .rename_axis("state")
                                .reset_index(name="n_runs")
                            )
                            state_order = ["Normal", "AH", "AC"]
                            state_counts["state_order"] = pd.Categorical(
                                state_counts["state"],
                                categories=state_order,
                                ordered=True,
                            )
                            state_counts = state_counts.sort_values("state_order")
                    total_state_runs = int(state_counts["n_runs"].sum())
                    state_counts["pct"] = state_counts["n_runs"] / max(total_state_runs, 1)
                    state_counts["pct_label"] = state_counts["pct"].apply(lambda v: _format_pct(v))
                    state_counts["label"] = (
                        state_counts["state"].map(_state_label).astype(str)
                        + " ("
                        + state_counts["pct_label"]
                        + ")"
                    )
                    if alt is not None:
                        pie = (
                            alt.Chart(state_counts)
                            .mark_arc(innerRadius=40)
                            .encode(
                                theta=alt.Theta("n_runs:Q", title="Runs"),
                                color=alt.Color(
                                    "state:N",
                                    title="State",
                                    sort=state_order,
                                    legend=alt.Legend(
                                        offset=36,
                                        titlePadding=10,
                                        labelPadding=8,
                                        labelLimit=0,
                                        labelExpr=(
                                            "datum.label == 'AH' ? 'Alcoholic hepatitis (AH)' : "
                                            "datum.label == 'AC' ? 'Alcohol-associated cirrhosis (AC)' : "
                                            "datum.label == 'Normal' ? 'Normal' : datum.label"
                                        )
                                    ),
                                ),
                                tooltip=[
                                    alt.Tooltip("state:N", title="State"),
                                    alt.Tooltip("n_runs:Q", title="Runs", format=",.0f"),
                                    alt.Tooltip("pct_label:N", title="Share"),
                                ],
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(pie, use_container_width=True)

                    unknown_df = df.copy()
                    if state_track_choice != "All":
                        unknown_df = unknown_df[
                            unknown_df["track_strategy"].astype(str).str.strip() == state_track_choice
                        ].copy()
                    unknown_df["bin_size"] = _resolve_bin_sizes(unknown_df)
                    if state_bin_choice != "All":
                        try:
                            bin_value = float(state_bin_choice)
                        except ValueError:
                            bin_value = None
                        if bin_value is not None:
                            unknown_df = unknown_df[unknown_df["bin_size"] == bin_value].copy()
                    unknown_df = _apply_tumour_filter(unknown_df)
                    unknown_rows = []
                    if state_metric_choice is None:
                        for metric, spec in METRICS.items():
                            best_cell_col = spec["best_cell"]
                            if best_cell_col not in unknown_df.columns:
                                continue
                            subset = unknown_df.copy()
                            subset["metric_label"] = metric_labels.get(metric, metric)
                            subset["pred_celltype"] = (
                                subset[best_cell_col].astype(str).str.strip().str.lower()
                            )
                            unknown_rows.append(subset)
                    else:
                        best_cell_col = METRICS[state_metric_choice]["best_cell"]
                        if best_cell_col in unknown_df.columns:
                            subset = unknown_df.copy()
                            subset["metric_label"] = state_metric_choice_label
                            subset["pred_celltype"] = (
                                subset[best_cell_col].astype(str).str.strip().str.lower()
                            )
                            unknown_rows.append(subset)
                    if unknown_rows:
                        unknown_df = pd.concat(unknown_rows, ignore_index=True)
                        unknown_df["state"] = unknown_df["pred_celltype"].map(state_by_celltype)
                        unknown_df = unknown_df[
                            unknown_df["pred_celltype"].astype(str).str.strip() != ""
                        ].copy()
                        unknown_df = unknown_df[unknown_df["state"].isna()].copy()
                        if not unknown_df.empty:
                            unknown_counts = (
                                unknown_df["pred_celltype"]
                                .value_counts()
                                .rename_axis("predicted_celltype")
                                .reset_index(name="n_runs")
                            )
                            st.markdown("**Unknown predictions**")
                            st.dataframe(unknown_counts, use_container_width=True)

                    rankings_source = df
                    if state_track_choice != "All":
                        rankings_source = rankings_source[
                            rankings_source["track_strategy"].astype(str).str.strip() == state_track_choice
                        ].copy()
                    rankings_source["bin_size"] = _resolve_bin_sizes(rankings_source)
                    if state_bin_choice != "All":
                        try:
                            bin_value = float(state_bin_choice)
                        except ValueError:
                            bin_value = None
                        if bin_value is not None:
                            rankings_source = rankings_source[
                                rankings_source["bin_size"] == bin_value
                            ].copy()
                    rankings_source = _apply_tumour_filter(rankings_source)
                    metrics_to_use = None
                    if state_metric_choice is not None:
                        metrics_to_use = [state_metric_choice]
                    top_n_value = None if state_top_n_choice == "All" else int(state_top_n_choice)
                    state_rankings = track_metric_rankings_by_state_prediction(
                        rankings_source,
                        state_by_celltype,
                        weight_basis,
                        top_n=top_n_value,
                        metrics_to_use=metrics_to_use,
                    )
                    if state_rankings.empty:
                        st.info("State summary unavailable: no hepatocyte state predictions found.")
                    else:
                        state_samples = (
                            state_df[["sample_id", "state"]]
                            .dropna()
                            .groupby("state")["sample_id"]
                            .nunique()
                        )
                        avg_burden_by_state = (
                            state_df.groupby("state")["mutation_burden"]
                            .mean()
                            .fillna(float("nan"))
                        )
                        state_order = ["Normal", "AH", "AC"]
                        state_rankings["state_order"] = pd.Categorical(
                            state_rankings["state"],
                            categories=state_order,
                            ordered=True,
                        )
                        state_rankings = state_rankings.sort_values(["state_order", "state", "rank"])
                        for state in state_order:
                            sub_state = state_rankings[state_rankings["state"] == state]
                            state_label = _state_label(state)
                            avg_burden = avg_burden_by_state.get(state, float("nan"))
                            avg_burden_label = (
                                f"{avg_burden:,.0f}" if np.isfinite(avg_burden) else "n/a"
                            )
                            n_samples = int(state_samples.get(state, 0))
                            st.markdown(
                                f"<div class='bin-header'>State: {state_label}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div class='section-subtitle'>Cell type: Hepatocyte · "
                                f"Avg mutation burden: {avg_burden_label} · Samples: {n_samples} of {total_state_samples}</div>",
                                unsafe_allow_html=True,
                            )
                            if sub_state.empty:
                                st.info("No predicted runs for this state with the current filters.")
                                continue
                            cols = st.columns(len(sub_state))
                            for col, (_, row) in zip(cols, sub_state.iterrows()):
                                avg_margin = row["avg_margin"]
                                metric_label = metric_labels.get(row["metric"], row["metric"])
                                bin_size = row["bin_size"]
                                bin_label = int(bin_size) if float(bin_size).is_integer() else float(bin_size)
                                n_runs = int(row.get("n_runs", 0))
                                with col:
                                    st.markdown(
                                        "<div class='section-card' style='min-height: auto; margin-bottom: 12px;'>"
                                        f"<div class='section-kicker'>Rank {int(row['rank'])}</div>"
                                        f"<div class='section-title'>{row['track_strategy']} + {metric_label}</div>"
                                        f"<div class='section-metric'>Bin size: {bin_label}</div>"
                                        f"<div class='section-metric'>Weighted avg margin: {avg_margin:.4f}</div>"
                                        f"<div class='section-metric'>Predicted runs: {n_runs}</div>"
                                        "</div>",
                                        unsafe_allow_html=True,
                                    )
