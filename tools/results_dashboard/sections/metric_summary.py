"""Metric-separation tab renderer."""

import altair as alt
import pandas as pd
import streamlit as st

from tools.results_dashboard.core import metric_margin_summary


def render_metric_tab(df, weight_basis, metric_labels):
    st.subheader("Which metric separates the correct cell type best?")
    weight_label = "margin / (abs(second_best) + eps)"
    summary_df = metric_margin_summary(df, weight_basis)
    if summary_df.empty:
        st.info("Margin summary unavailable: required columns are missing in this results.csv.")
    else:
        summary_df = summary_df.copy()
        summary_df["metric"] = summary_df["metric"].map(metric_labels).fillna(summary_df["metric"])
        summary_df["metric_plot"] = summary_df["metric"].replace(
            {
                "Pearson r (linear covariate)": "Pearson r\n(linear covariate)",
                "Spearman r (linear covariate)": "Spearman r\n(linear covariate)",
                "Pearson local score (linear covariate)": "Pearson local score\n(linear covariate)",
                "Spearman local score (linear covariate)": "Spearman local score\n(linear covariate)",
                "RF (non-linear covariate)": "RF\n(non-linear covariate)",
            }
        )
        metric_order_display = [
            "Pearson r",
            "Pearson r (linear covariate)",
            "Spearman r",
            "Spearman r (linear covariate)",
            "Pearson local score (linear covariate)",
            "Spearman local score (linear covariate)",
            "RF (non-linear covariate)",
        ]
        summary_df["metric_order"] = pd.Categorical(
            summary_df["metric"],
            categories=metric_order_display,
            ordered=True,
        )
        summary_df = summary_df.sort_values("metric_order").drop(columns=["metric_order"])
        best_metric = summary_df.sort_values(
            ["accuracy", "avg_margin_weighted"], ascending=[False, False]
        ).iloc[0]
        st.markdown(
            "For each run, we take the best-vs-second margin and divide it by the absolute value of the "
            "second-best score (plus a tiny epsilon). This boosts runs where the winner is clearly ahead "
            "of the runner-up, and downweights runs where the top two are close. We then average those "
            "weighted margins across runs."
        )
        st.dataframe(summary_df.drop(columns=["metric_plot"]), use_container_width=True)
        if alt is not None:
            metric_order = [
                "Pearson r",
                "Pearson r\n(linear covariate)",
                "Spearman r",
                "Spearman r\n(linear covariate)",
                "Pearson local score\n(linear covariate)",
                "Spearman local score\n(linear covariate)",
                "RF\n(non-linear covariate)",
            ]
            chart = (
                alt.Chart(summary_df)
                .mark_bar(color="#0f766e")
                .encode(
                    x=alt.X(
                        "metric_plot:N",
                        title="Metric",
                        sort=metric_order,
                        axis=alt.Axis(
                            labelAngle=0,
                            labelLimit=0,
                            labelPadding=8,
                            labelOverlap=False,
                            labelExpr='split(datum.label, "\\n")',
                        ),
                    ),
                    y=alt.Y("avg_margin_weighted:Q", title="Weighted avg margin"),
                    tooltip=[
                        "metric",
                        "n_runs",
                        "n_correct",
                        "accuracy",
                        "avg_margin_weighted",
                        "avg_margin_unweighted",
                    ],
                )
            )
            st.altair_chart(chart, use_container_width=True)
