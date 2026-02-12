"""Best-by-tumour tab renderer."""

import numpy as np
import pandas as pd
import streamlit as st

from tools.results_dashboard.core import (
    _format_celltype_label,
    _format_pct,
    _load_dnase_celltype_map,
    track_metric_rankings_by_tumour,
)


def render_tumour_tab(df, weight_basis, metric_labels, experiment_dir):
    st.subheader("Best track + metric by tumour type")
    tumour_rankings = track_metric_rankings_by_tumour(df, weight_basis, top_n=5)
    if tumour_rankings.empty:
        st.info("Tumour-type summary unavailable: no tumour types in this results.csv.")
    else:
        tumour_samples = (
            df[["sample_id", "selected_tumour_types"]]
            .fillna("")
            .assign(
                tumour_type=lambda d: d["selected_tumour_types"]
                .astype(str)
                .str.split(",")
            )
            .explode("tumour_type")
        )
        tumour_samples["tumour_type"] = tumour_samples["tumour_type"].astype(str).str.strip()
        tumour_samples = tumour_samples[tumour_samples["tumour_type"] != ""]
        tumour_sample_counts = tumour_samples.groupby("tumour_type")["sample_id"].nunique()
        celltype_map = _load_dnase_celltype_map(experiment_dir)
        for tumour_type, sub_tumour in tumour_rankings.groupby("tumour_type"):
            correct_celltypes = sorted(sub_tumour["correct_celltype"].dropna().unique().tolist())
            correct_celltype_label = _format_celltype_label(correct_celltypes, celltype_map)
            avg_burden = sub_tumour["avg_burden"].dropna().mean()
            avg_burden_label = (
                f"{avg_burden:,.0f}" if np.isfinite(avg_burden) else "n/a"
            )
            n_samples = int(tumour_sample_counts.get(tumour_type, 0))
            st.markdown(
                f"<div class='bin-header'>Tumour: {tumour_type}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='section-subtitle'>Cell type: {correct_celltype_label} · "
                f"Avg mutation burden: {avg_burden_label} · Samples: {n_samples}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(sub_tumour))
            for col, (_, row) in zip(cols, sub_tumour.iterrows()):
                avg_margin = row["avg_margin"]
                win_rate = row.get("win_rate", float("nan"))
                metric_label = metric_labels.get(row["metric"], row["metric"])
                bin_size = row["bin_size"]
                bin_label = int(bin_size) if float(bin_size).is_integer() else float(bin_size)
                accuracy_label = (
                    f"Accuracy: {_format_pct(win_rate)}"
                    if np.isfinite(win_rate)
                    else "Accuracy: not available"
                )
                with col:
                    st.markdown(
                        "<div class='section-card' style='min-height: auto; margin-bottom: 12px;'>"
                        f"<div class='section-kicker'>Rank {int(row['rank'])}</div>"
                        f"<div class='section-title'>{row['track_strategy']} + {metric_label}</div>"
                        f"<div class='section-metric'>Bin size: {bin_label}</div>"
                        f"<div class='section-metric'>Weighted avg margin: {avg_margin:.4f}</div>"
                        f"<div class='section-metric'>{accuracy_label}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
