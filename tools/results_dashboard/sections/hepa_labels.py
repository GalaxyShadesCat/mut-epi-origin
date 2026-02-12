"""Annotation-label benchmark tab renderer."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tools.results_dashboard.core import (
    _format_path_label,
    _format_pct,
    _norm_tumour_label,
    _resolve_bin_sizes,
)
from tools.results_dashboard.hepa import (
    canonical_hepa_label,
    hepa_label_accuracy_rankings,
    hepa_label_display,
    hepa_label_prediction_rows,
    hepa_short_label,
    load_hepa_labels,
)


def render_hepa_labels_tab(
    is_liver_experiment,
    df,
    tumour_choice,
    metric_labels,
    metrics,
    weight_basis,
    default_hepa_labels_path,
):
    METRICS = metrics

    if is_liver_experiment:
        st.subheader("Best track + metric + bin size vs annotation-based labels")
        control_col_a, control_col_b, control_col_c = st.columns([1.2, 1.6, 1.2])
        hepa_celltype_options = {
            "All": "all",
            "Normal": "hepatocyte_normal",
            "Alcoholic hepatitis (AH)": "hepatocyte_ah",
            "Alcohol-associated cirrhosis (AC)": "hepatocyte_ac",
        }
        with control_col_c:
            include_ambiguous_labels = st.checkbox(
                "Include ambiguous annotation labels",
                value=False,
            )
        if include_ambiguous_labels:
            hepa_celltype_options["Ambiguous"] = "hepatocyte_ambiguous"
        with control_col_a:
            selected_hepa_celltype_label = st.selectbox(
                "Annotation cell type",
                options=list(hepa_celltype_options.keys()),
                index=0,
            )
        selected_hepa_celltype = hepa_celltype_options[selected_hepa_celltype_label]
        accuracy_mode_options = {
            "Overall accuracy": "accuracy_overall",
            "Balanced accuracy (equal cell-type weighting)": "accuracy_balanced",
        }
        with control_col_b:
            selected_accuracy_mode_label = st.selectbox(
                "Accuracy mode",
                options=list(accuracy_mode_options.keys()),
                index=1,
            )
        selected_accuracy_col = accuracy_mode_options[selected_accuracy_mode_label]
        labels_path = default_hepa_labels_path
        if not labels_path.exists():
            st.info(
                "Annotation-based hepa labels file not found at "
                f"`{_format_path_label(labels_path)}`."
            )
        else:
            try:
                hepa_labels_df = load_hepa_labels(labels_path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                st.info(f"Could not load annotation-based hepa labels: {exc}")
            else:
                ambiguous_mask = (
                    hepa_labels_df["cell_type_label"].astype(str) == "hepatocyte_ambiguous"
                )
                n_ambiguous = int(ambiguous_mask.sum())
                if not include_ambiguous_labels:
                    hepa_labels_df = hepa_labels_df[~ambiguous_mask].copy()
                    if n_ambiguous > 0:
                        st.caption(
                            f"Excluded ambiguous annotation labels: {n_ambiguous}"
                        )
                elif n_ambiguous > 0:
                    st.caption(
                        f"Including ambiguous annotation labels: {n_ambiguous}"
                    )
                rankings = hepa_label_accuracy_rankings(
                    df,
                    hepa_labels_df,
                    metrics=METRICS,
                    resolve_bin_sizes=_resolve_bin_sizes,
                    norm_tumour_label=_norm_tumour_label,
                    weight_basis=weight_basis,
                    tumour_choice=tumour_choice,
                    true_celltype_filter=selected_hepa_celltype,
                )
                if rankings.empty:
                    st.info(
                        "No comparable runs found between results and annotation-based labels "
                        "for the current tumour filter."
                    )
                else:
                    rankings = rankings.sort_values(
                        [
                            selected_accuracy_col,
                            "avg_margin",
                            "n_runs",
                            "n_samples",
                            "track_strategy",
                            "metric",
                            "bin_size",
                        ],
                        ascending=[False, False, False, False, True, True, True],
                    ).reset_index(drop=True)
                    top5 = rankings.head(5).copy()
                    pred_rows = hepa_label_prediction_rows(
                        df,
                        hepa_labels_df,
                        metrics=METRICS,
                        resolve_bin_sizes=_resolve_bin_sizes,
                        norm_tumour_label=_norm_tumour_label,
                        weight_basis=weight_basis,
                        tumour_choice=tumour_choice,
                        true_celltype_filter=selected_hepa_celltype,
                    )
                    selection_key = "hepa_breakdown_selection"
                    selection = st.session_state.get(selection_key)
                    valid_configs = {
                        (str(r["track_strategy"]), str(r["metric"]), float(r["bin_size"]))
                        for _, r in top5.iterrows()
                    }
                    if (
                        not selection
                        or (selection["track_strategy"], selection["metric"], float(selection["bin_size"])) not in valid_configs
                    ) and not top5.empty:
                        selection = {
                            "track_strategy": str(top5.iloc[0]["track_strategy"]),
                            "metric": str(top5.iloc[0]["metric"]),
                            "bin_size": float(top5.iloc[0]["bin_size"]),
                        }
                        st.session_state[selection_key] = selection
                    cols = st.columns(len(top5))
                    for col, (rank, (_, row)) in zip(cols, enumerate(top5.iterrows(), start=1)):
                        metric_label = metric_labels.get(row["metric"], str(row["metric"]))
                        bin_size = row["bin_size"]
                        bin_label = int(bin_size) if float(bin_size).is_integer() else float(bin_size)
                        config_mask = (
                            (pred_rows["track_strategy"].astype(str) == str(row["track_strategy"]))
                            & (pred_rows["metric"].astype(str) == str(row["metric"]))
                            & (pred_rows["bin_size"].astype(float) == float(row["bin_size"]))
                        )
                        card_rows = pred_rows[config_mask].copy()
                        by_cell = (
                            card_rows.groupby("true_celltype", dropna=False)
                            .agg(total=("is_correct", "size"), correct=("is_correct", "sum"))
                            .reset_index()
                        )
                        by_cell["acc"] = by_cell["correct"] / by_cell["total"]
                        acc_lookup = {
                            canonical_hepa_label(r["true_celltype"]): float(r["acc"])
                            for _, r in by_cell.iterrows()
                        }
                        total_lookup = {
                            canonical_hepa_label(r["true_celltype"]): int(r["total"])
                            for _, r in by_cell.iterrows()
                        }
                        correct_lookup = {
                            canonical_hepa_label(r["true_celltype"]): int(r["correct"])
                            for _, r in by_cell.iterrows()
                        }

                        def _format_celltype_accuracy(label_key: str) -> str:
                            pct = _format_pct(acc_lookup.get(label_key, float("nan")))
                            correct = correct_lookup.get(label_key, 0)
                            total = total_lookup.get(label_key, 0)
                            return f"{pct} ({correct}/{total})"

                        cell_breakdown_html = (
                            "<div class='section-metric'><strong>Cell-type accuracy</strong><br>"
                            f"Normal: {_format_celltype_accuracy('hepatocyte_normal')}<br>"
                            f"AH: {_format_celltype_accuracy('hepatocyte_ah')}<br>"
                            f"AC: {_format_celltype_accuracy('hepatocyte_ac')}"
                            "</div>"
                        )
                        overall_acc = float(row["accuracy_overall"]) if np.isfinite(row["accuracy_overall"]) else float("nan")
                        balanced_acc = float(row["accuracy_balanced"]) if np.isfinite(row["accuracy_balanced"]) else float("nan")
                        avg_margin = float(row["avg_margin"]) if np.isfinite(row["avg_margin"]) else float("nan")
                        with col:
                            st.markdown(
                                "<div class='section-card' style='min-height: auto; margin-bottom: 12px;'>"
                                f"<div class='section-kicker'>Rank {rank}</div>"
                                f"<div class='section-title'>{row['track_strategy']} + {metric_label}</div>"
                                f"<div class='section-metric'>Bin size: {bin_label}</div>"
                                f"<div class='section-metric'>Overall accuracy: {_format_pct(overall_acc)}</div>"
                                f"<div class='section-metric'>Balanced accuracy: {_format_pct(balanced_acc)}</div>"
                                f"<div class='section-metric'>Weighted avg margin: {avg_margin:.4f}</div>"
                                f"<div class='section-metric'>Correct runs: {int(row['n_correct'])}/{int(row['n_runs'])}</div>"
                                f"{cell_breakdown_html}"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                "View breakdown",
                                key=(
                                    "hepa_select_"
                                    f"{rank}_"
                                    f"{str(row['track_strategy'])}_"
                                    f"{str(row['metric'])}_"
                                    f"{float(row['bin_size'])}_"
                                    f"{selected_hepa_celltype}_"
                                    f"{selected_accuracy_col}"
                                ),
                                use_container_width=True,
                            ):
                                selection = {
                                    "track_strategy": str(row["track_strategy"]),
                                    "metric": str(row["metric"]),
                                    "bin_size": float(row["bin_size"]),
                                }
                                st.session_state[selection_key] = selection
                    if selection:
                        sel_mask = (
                            (pred_rows["track_strategy"].astype(str) == selection["track_strategy"])
                            & (pred_rows["metric"].astype(str) == selection["metric"])
                            & (pred_rows["bin_size"].astype(float) == float(selection["bin_size"]))
                        )
                        sel_rows = pred_rows[sel_mask].copy()
                        if not sel_rows.empty:
                            sel_metric_label = metric_labels.get(selection["metric"], selection["metric"])
                            sel_bin = selection["bin_size"]
                            sel_bin_label = int(sel_bin) if float(sel_bin).is_integer() else float(sel_bin)
                            st.markdown(
                                f"**Misclassification pie breakdown: {selection['track_strategy']} + "
                                f"{sel_metric_label} (bin {sel_bin_label})**"
                            )
                            pie_cols = st.columns(3)
                            true_order = ["hepatocyte_normal", "hepatocyte_ah", "hepatocyte_ac"]
                            pred_order = ["hepatocyte_normal", "hepatocyte_ah", "hepatocyte_ac"]
                            for pcol, true_label in zip(pie_cols, true_order):
                                wrong = sel_rows[
                                    (sel_rows["true_celltype"] == true_label)
                                    & (~sel_rows["is_correct"])
                                ].copy()
                                with pcol:
                                    st.markdown(f"**{hepa_label_display(true_label)}**")
                                    if wrong.empty:
                                        st.info("No wrong predictions.")
                                        continue
                                    pie_df = (
                                        wrong.groupby("pred_celltype", dropna=False)
                                        .size()
                                        .rename("n_runs")
                                        .reset_index()
                                    )
                                    pie_df["pct"] = pie_df["n_runs"] / pie_df["n_runs"].sum()
                                    pie_df["pred_label"] = pie_df["pred_celltype"].map(hepa_short_label)
                                    pie_df["pct_label"] = pie_df["pct"].apply(lambda v: _format_pct(float(v)))
                                    pie_df["pred_order"] = pd.Categorical(
                                        pie_df["pred_label"],
                                        categories=["Normal", "AH", "AC"],
                                        ordered=True,
                                    )
                                    pie_df = pie_df.sort_values("pred_order")
                                    if alt is not None:
                                        chart = (
                                            alt.Chart(pie_df)
                                            .mark_arc(innerRadius=32)
                                            .encode(
                                                theta=alt.Theta("n_runs:Q", title="Wrong runs"),
                                                color=alt.Color(
                                                    "pred_label:N",
                                                    title="Mapped to",
                                                    sort=["Normal", "AH", "AC"],
                                                    scale=alt.Scale(domain=["Normal", "AH", "AC"]),
                                                ),
                                                tooltip=[
                                                    alt.Tooltip("pred_label:N", title="Mapped to"),
                                                    alt.Tooltip("n_runs:Q", title="Wrong runs", format=",.0f"),
                                                    alt.Tooltip("pct_label:N", title="Share"),
                                                ],
                                            )
                                            .properties(height=240)
                                        )
                                        st.altair_chart(chart, use_container_width=True)
                                    st.dataframe(
                                        pie_df[["pred_label", "n_runs", "pct_label"]].rename(
                                            columns={
                                                "pred_label": "mapped_to",
                                                "n_runs": "wrong_runs",
                                                "pct_label": "share",
                                            }
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
