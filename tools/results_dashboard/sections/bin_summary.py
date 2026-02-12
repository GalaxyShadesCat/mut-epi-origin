"""Best-by-bin tab renderer."""

import numpy as np
import streamlit as st

from tools.results_dashboard.core import _format_pct, track_metric_rankings_by_bin


def render_bin_tab(df, weight_basis, metric_labels):
    st.subheader("Best track + metric by bin size")
    bin_rankings = track_metric_rankings_by_bin(df, weight_basis, top_n=5)
    if bin_rankings.empty:
        st.info("Bin-size summary unavailable: no correct predictions in this dataset.")
    else:
        for bin_size, sub in bin_rankings.groupby("bin_size"):
            bin_label = int(bin_size) if float(bin_size).is_integer() else float(bin_size)
            st.markdown(
                f"<div class='bin-header'>Bin size {bin_label}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(sub))
            for col, (_, row) in zip(cols, sub.iterrows()):
                avg_margin = row["avg_margin"]
                win_rate = row.get("win_rate", float("nan"))
                metric_label = metric_labels.get(row["metric"], row["metric"])
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
                        f"<div class='section-metric'>Weighted avg margin: {avg_margin:.4f}</div>"
                        f"<div class='section-metric'>{accuracy_label}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
