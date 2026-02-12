"""Accuracy vs mutation burden tab renderer."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from typing import Set

from tools.results_dashboard.core import (
    _build_tumour_options,
    _coerce_bool,
    _format_pct,
    _load_dnase_celltype_map,
    _norm_tumour_label,
    _resolve_bin_sizes,
    _resolve_mutation_burden,
    _standout_weights,
)


def render_accuracy_tab(
    df,
    experiment_dir,
    weight_basis,
    metric_labels,
    metrics,
):
    METRICS = metrics

    st.subheader("Accuracy vs Mutation Burden")

    track_choices = ["All"] + sorted(df["track_strategy"].dropna().astype(str).unique().tolist())
    metric_keys = list(METRICS.keys())
    metric_choices = ["All"] + [metric_labels.get(m, m) for m in metric_keys]
    metric_label_to_key = {metric_labels.get(m, m): m for m in metric_keys}
    top_n_choices = ["All", 3, 5, 10]
    celltype_map = _load_dnase_celltype_map(experiment_dir)
    tumour_options, tumour_label_to_key = _build_tumour_options(df, celltype_map)

    row_a, row_b = st.columns(2)
    with row_a:
        track_choice = st.selectbox("Track", options=track_choices, index=0)
    with row_b:
        metric_choice_label = st.selectbox("Metric", options=metric_choices, index=0)

    row_c, row_d, row_e = st.columns(3)
    with row_c:
        bin_values = _resolve_bin_sizes(df).dropna().unique().tolist()
        bin_values = sorted(bin_values)
        bin_labels = [
            int(x) if float(x).is_integer() else float(x) for x in bin_values
        ]
        bin_choices = ["All"] + [str(x) for x in bin_labels]
        bin_choice = st.selectbox("Bin size", options=bin_choices, index=0)
    with row_d:
        tumour_choice_label = st.selectbox("Tumour", options=tumour_options, index=0)
        tumour_choice = tumour_label_to_key.get(tumour_choice_label, "All")
    with row_e:
        top_default_index = top_n_choices.index(5) if 5 in top_n_choices else 0
        top_n_choice = st.selectbox("Top by accuracy", options=top_n_choices, index=top_default_index)

    plot_df = df.copy()
    plot_df["mutation_burden"], burden_source = _resolve_mutation_burden(plot_df)
    plot_df = plot_df[plot_df["mutation_burden"].notna()].copy()
    if plot_df.empty:
        st.info("Mutation burden not available for this dataset.")
    else:
        if track_choice != "All":
            plot_df = plot_df[
                plot_df["track_strategy"].astype(str).str.strip() == track_choice
                ].copy()
        plot_df["bin_size"] = _resolve_bin_sizes(plot_df)
        if bin_choice != "All":
            try:
                bin_value = float(bin_choice)
            except ValueError:
                bin_value = None
            if bin_value is not None:
                plot_df = plot_df[plot_df["bin_size"] == bin_value].copy()
        if tumour_choice != "All":
            if "selected_tumour_types" not in plot_df.columns:
                st.info("Tumour filter unavailable: selected_tumour_types missing.")
                plot_df = plot_df.iloc[0:0].copy()
            else:
                def _has_tumour(value: str) -> bool:
                    for tumour in str(value).split(","):
                        if _norm_tumour_label(tumour) == tumour_choice:
                            return True
                    return False

                plot_df = plot_df[plot_df["selected_tumour_types"].apply(_has_tumour)].copy()
        if plot_df.empty:
            st.info("No data available after tumour filtering.")
        else:
            metric_choice = metric_label_to_key.get(metric_choice_label)
            accuracy_cols = []
            for metric, spec in METRICS.items():
                col = spec["is_correct"]
                if col in plot_df.columns:
                    plot_df[col] = _coerce_bool(plot_df[col]).astype(float)
                    accuracy_cols.append(col)
            if not accuracy_cols:
                st.info("Accuracy columns not available for this dataset.")
            else:
                def _series_or_nan(df: pd.DataFrame, col_name: str) -> pd.Series:
                    if col_name in df.columns:
                        return pd.to_numeric(df[col_name], errors="coerce")
                    return pd.Series(np.nan, index=df.index)

                if metric_choice is None:
                    long_rows = []
                    for metric, spec in METRICS.items():
                        col = spec["is_correct"]
                        if col not in plot_df.columns:
                            continue
                        acc = plot_df[col].fillna(0.0).astype(float)
                        margin = _series_or_nan(plot_df, spec["best_margin"])
                        best_values = _series_or_nan(plot_df, spec["best_value"])
                        if weight_basis == "metric_abs":
                            margin_weight = _standout_weights(margin, best_values)
                        elif weight_basis == "n_bins_total":
                            margin_weight = _series_or_nan(plot_df, "n_bins_total")
                        else:
                            margin_weight = pd.Series(1.0, index=plot_df.index)
                        long_rows.append(
                            plot_df.assign(
                                accuracy=acc,
                                metric_key=metric,
                                metric_label=metric_labels.get(metric, metric),
                                margin=margin,
                                margin_weight=margin_weight,
                            )
                        )
                    if not long_rows:
                        st.info("Selected metric not available for this dataset.")
                    else:
                        plot_df = pd.concat(long_rows, ignore_index=True)
                else:
                    chosen_col = METRICS[metric_choice]["is_correct"]
                    if chosen_col not in plot_df.columns:
                        st.info("Selected metric not available for this dataset.")
                    else:
                        plot_df["accuracy"] = plot_df[chosen_col].fillna(0.0).astype(float)
                        plot_df["metric_label"] = metric_choice_label
                        margin = _series_or_nan(plot_df, METRICS[metric_choice]["best_margin"])
                        best_values = _series_or_nan(plot_df, METRICS[metric_choice]["best_value"])
                        if weight_basis == "metric_abs":
                            plot_df["margin_weight"] = _standout_weights(margin, best_values)
                        elif weight_basis == "n_bins_total":
                            plot_df["margin_weight"] = _series_or_nan(plot_df, "n_bins_total")
                        else:
                            plot_df["margin_weight"] = 1.0
                        plot_df["margin"] = margin
                if "accuracy" in plot_df.columns:
                    overall_df = plot_df.copy()
                    plot_df = plot_df[plot_df["mutation_burden"] > 0].copy()
                    if plot_df.empty:
                        st.info("Mutation burden must be positive to plot on a log scale.")
                    else:
                        plot_df = plot_df.dropna(subset=["mutation_burden", "accuracy"])
                        if plot_df.empty:
                            st.info("No data available after filtering.")
                        else:
                            plot_df = plot_df[plot_df["mutation_burden"] >= 100].copy()
                            if plot_df.empty:
                                st.info("Mutation burden must be at least 100 to plot.")
                            else:
                                min_burden = float(plot_df["mutation_burden"].min())
                                max_burden = float(plot_df["mutation_burden"].max())
                                if min_burden <= 0:
                                    st.info("Mutation burden must be positive to plot on a log scale.")
                                else:
                                    min_burden = max(min_burden, 100.0)
                                    min_pow = int(np.floor(np.log10(min_burden)))
                                    max_pow = int(np.ceil(np.log10(max_burden)))
                                    bin_edges = np.unique(
                                        np.power(10, np.arange(min_pow, max_pow + 1, 1 / 3))
                                    )
                                    if len(bin_edges) < 2:
                                        bin_edges = np.array([min_burden, max_burden * 1.01])
                                    plot_df["mutation_bucket"] = pd.cut(
                                        plot_df["mutation_burden"],
                                        bins=bin_edges,
                                        include_lowest=True,
                                    )
                                    plot_df = plot_df[plot_df["mutation_bucket"].notna()].copy()
                                    if plot_df.empty:
                                        st.info("No data available after bucketing.")
                                    else:
                                        bucket = plot_df["mutation_bucket"]
                                        plot_df["bucket_low"] = bucket.apply(lambda x: float(x.left))
                                        plot_df["bucket_high"] = bucket.apply(lambda x: float(x.right))
                                        plot_df["bucket_mid"] = np.sqrt(
                                            plot_df["bucket_low"].astype(float)
                                            * plot_df["bucket_high"].astype(float)
                                        )
                                        def _count_unique_samples(series: pd.Series) -> int:
                                            unique: Set[str] = set()
                                            for raw in series.dropna():
                                                for sample_id in str(raw).split(","):
                                                    sample_id = sample_id.strip()
                                                    if sample_id:
                                                        unique.add(sample_id)
                                            return len(unique)

                                        sample_id_col = None
                                        if "selected_sample_ids" in plot_df.columns:
                                            sample_id_col = "selected_sample_ids"
                                        elif "sample_id" in plot_df.columns:
                                            plot_df["selected_sample_ids"] = plot_df["sample_id"].astype(str)
                                            sample_id_col = "selected_sample_ids"

                                        agg_spec = {
                                            "mutation_burden": ("bucket_mid", "mean"),
                                            "accuracy": ("accuracy", "mean"),
                                            "n_runs": ("accuracy", "size"),
                                            "bucket_low": ("bucket_low", "min"),
                                            "bucket_high": ("bucket_high", "max"),
                                        }
                                        if sample_id_col is not None:
                                            agg_spec["n_samples"] = (sample_id_col, _count_unique_samples)
                                        elif "n_selected_samples" in plot_df.columns:
                                            agg_spec["n_samples"] = ("n_selected_samples", "mean")
                                        else:
                                            agg_spec["n_samples"] = ("accuracy", "size")

                                        group_cols = [
                                            "track_strategy",
                                            "metric_label",
                                            "bin_size",
                                            "bucket_mid",
                                        ]

                                        def _weighted_margin(sub: pd.DataFrame) -> float:
                                            margin_vals = pd.to_numeric(sub["margin"], errors="coerce")
                                            weight_vals = pd.to_numeric(sub["margin_weight"], errors="coerce")
                                            mask = margin_vals.notna()
                                            if not mask.any():
                                                return float("nan")
                                            weights = weight_vals.where(mask)
                                            if weights.notna().any() and weights.sum() > 0:
                                                return float((margin_vals[mask] * weights[mask]).sum() / weights[mask].sum())
                                            return float(margin_vals[mask].mean())

                                        grouped = plot_df.groupby(group_cols, dropna=False)
                                        agg_df = grouped.agg(**agg_spec).reset_index()
                                        weighted_margin_df = (
                                            plot_df[group_cols + ["margin", "margin_weight"]]
                                            .groupby(group_cols, dropna=False)
                                            .apply(_weighted_margin, include_groups=False)
                                            .rename("weighted_margin")
                                            .reset_index()
                                        )
                                    plot_df = (
                                        agg_df.merge(weighted_margin_df, on=group_cols, how="left")
                                        .dropna(subset=["mutation_burden", "accuracy"])
                                    )
                                    if plot_df.empty:
                                        st.info("No data available after aggregation.")
                                    else:
                                        max_accuracy = pd.to_numeric(
                                            plot_df["accuracy"], errors="coerce"
                                        ).max()
                                        if not np.isfinite(max_accuracy) or max_accuracy <= 0:
                                            plot_df["accuracy"] = np.nan
                                        pct = (plot_df["accuracy"] * 100).round(1)
                                        pct_finite = np.isfinite(pct)
                                        is_int = np.isclose(pct % 1, 0)
                                        pct_label = np.where(
                                            pct_finite & is_int,
                                            pct.round(0).astype("Int64").astype(str) + "%",
                                            np.where(
                                                pct_finite,
                                                pct.astype(str) + "%",
                                                "n/a",
                                            ),
                                        )
                                        plot_df["accuracy_label"] = pct_label
                                        plot_df["weighted_margin_label"] = plot_df["weighted_margin"].apply(
                                            lambda v: f"{v:.4f}" if np.isfinite(v) else "n/a"
                                        )
                                        if top_n_choice != "All":
                                            config_cols = [
                                                "track_strategy",
                                                "metric_label",
                                                "bin_size",
                                            ]
                                            if metric_choice_label != "All":
                                                config_cols = ["track_strategy", "bin_size"]
                                            if bin_choice != "All":
                                                config_cols = ["track_strategy", "metric_label"]
                                            if (
                                                    metric_choice_label != "All"
                                                    and bin_choice != "All"
                                            ):
                                                config_cols = ["track_strategy"]
                                            config_scores = (
                                                overall_df.groupby(config_cols, as_index=False)
                                                .agg(avg_accuracy=("accuracy", "mean"))
                                                .sort_values("avg_accuracy", ascending=False)
                                            )
                                            top_configs = config_scores.head(int(top_n_choice))
                                            plot_df = plot_df.merge(
                                                top_configs[config_cols], on=config_cols
                                            )

                                            config_scores = (
                                                overall_df.groupby(
                                                    ["track_strategy", "metric_label", "bin_size"],
                                                    as_index=False,
                                                )
                                                .agg(avg_accuracy=("accuracy", "mean"))
                                            )
                                            overall_margin_df = (
                                                overall_df[
                                                    [
                                                        "track_strategy",
                                                        "metric_label",
                                                        "bin_size",
                                                        "margin",
                                                        "margin_weight",
                                                    ]
                                                ]
                                                .groupby(
                                                    ["track_strategy", "metric_label", "bin_size"],
                                                    dropna=False,
                                                )
                                                .apply(_weighted_margin, include_groups=False)
                                                .rename("overall_weighted_margin")
                                                .reset_index()
                                            )
                                            config_scores = config_scores.merge(
                                                overall_margin_df,
                                                on=["track_strategy", "metric_label", "bin_size"],
                                                how="left",
                                            )
                                            plot_df["bin_label"] = plot_df["bin_size"].apply(
                                                lambda x: int(x)
                                                if float(x).is_integer()
                                                else float(x)
                                            )
                                            plot_df["config_label"] = (
                                                plot_df["track_strategy"].astype(str)
                                                + " [Bin size "
                                                + plot_df["bin_label"].astype(str)
                                                + "]\n"
                                                + plot_df["metric_label"].astype(str)
                                            )
                                            track_order = [
                                                "counts_raw",
                                                "counts_gauss",
                                                "inv_dist_gauss",
                                                "exp_decay",
                                                "exp_decay_adaptive",
                                            ]
                                            metric_order = [
                                                "Pearson r",
                                                "Pearson r (linear covariate)",
                                                "Spearman r",
                                                "Spearman r (linear covariate)",
                                                "Pearson local score (linear covariate)",
                                                "Spearman local score (linear covariate)",
                                                "RF (non-linear covariate)",
                                            ]
                                            track_rank = {name: idx for idx, name in enumerate(track_order)}
                                            metric_rank = {name: idx for idx, name in enumerate(metric_order)}
                                            order_df = (
                                                plot_df[
                                                    [
                                                        "config_label",
                                                        "track_strategy",
                                                        "metric_label",
                                                        "bin_size",
                                                    ]
                                                ]
                                                .drop_duplicates()
                                                .merge(
                                                    config_scores,
                                                    on=["track_strategy", "metric_label", "bin_size"],
                                                    how="left",
                                                )
                                                .assign(
                                                    track_rank=lambda d: d["track_strategy"]
                                                    .map(track_rank)
                                                    .fillna(len(track_rank)),
                                                    metric_rank=lambda d: d["metric_label"]
                                                    .map(metric_rank)
                                                    .fillna(len(metric_rank)),
                                                    avg_accuracy=lambda d: d["avg_accuracy"].fillna(-1.0),
                                                    overall_weighted_margin=lambda d: d["overall_weighted_margin"].fillna(-1.0),
                                                )
                                                .sort_values(
                                                    [
                                                        "avg_accuracy",
                                                        "overall_weighted_margin",
                                                        "track_rank",
                                                        "metric_rank",
                                                        "bin_size",
                                                        "config_label",
                                                    ],
                                                    ascending=[False, False, True, True, True, True],
                                                )
                                            )
                                            config_domain = order_df["config_label"].tolist()
                                            plot_df = plot_df.merge(
                                                config_scores,
                                                on=["track_strategy", "metric_label", "bin_size"],
                                                how="left",
                                            )
                                            plot_df["avg_accuracy"] = plot_df["avg_accuracy"].fillna(float("nan"))
                                            plot_df["avg_accuracy_label"] = plot_df["avg_accuracy"].apply(
                                                lambda v: _format_pct(v) if np.isfinite(v) else "n/a"
                                            )
                                            plot_df["overall_weighted_margin_label"] = plot_df["overall_weighted_margin"].apply(
                                                lambda v: f"{v:.4f}" if np.isfinite(v) else "n/a"
                                            )
                                            tooltip = [
                                                alt.Tooltip("track_strategy:N", title="Track"),
                                                alt.Tooltip("metric_label:N", title="Metric"),
                                                alt.Tooltip("bin_size:Q", title="Bin size", format=",.0f"),
                                                alt.Tooltip(
                                                    "bucket_low:Q", title="Mutations (low)", format=",.0f"
                                                ),
                                                alt.Tooltip(
                                                    "bucket_high:Q",
                                                    title="Mutations (high)",
                                                    format=",.0f",
                                                ),
                                                alt.Tooltip("accuracy_label:N", title="Bucket accuracy"),
                                                alt.Tooltip("avg_accuracy_label:N", title="Overall accuracy"),
                                                alt.Tooltip("weighted_margin_label:N", title="Bucket weighted avg margin"),
                                                alt.Tooltip(
                                                    "overall_weighted_margin_label:N",
                                                    title="Overall weighted avg margin",
                                                ),
                                                alt.Tooltip("n_samples:Q", title="Samples", format=",.0f"),
                                            ]
                                            row_height = 80
                                            if top_n_choice == 3:
                                                row_height = 95
                                            chart_height = max(240, row_height * len(config_domain))
                                            if len(config_domain) <= 2:
                                                chart_height = max(chart_height, 280)
                                            chart = (
                                                alt.Chart(plot_df)
                                                .mark_rect()
                                                .encode(
                                                    x=alt.X(
                                                        "bucket_low:Q",
                                                        title="Mutation Burden",
                                                        scale=alt.Scale(type="log", domain=[100, max_burden]),
                                                        axis=alt.Axis(
                                                            format="~s",
                                                            values=[
                                                                100,
                                                                1000,
                                                                10000,
                                                                100000,
                                                                1000000,
                                                            ],
                                                        ),
                                                    ),
                                                    x2="bucket_high:Q",
                                                    y=alt.Y(
                                                        "config_label:N",
                                                        title=None,
                                                        sort=config_domain,
                                                        axis=alt.Axis(
                                                            labelLimit=0,
                                                            labelOverlap=False,
                                                            labelSeparation=0,
                                                            labelExpr='split(datum.label, "\\n")',
                                                        ),
                                                    ),
                                                    color=alt.Color(
                                                        "accuracy:Q",
                                                        title="Accuracy",
                                                        scale=alt.Scale(domain=[0, 1], scheme="tealblues"),
                                                        legend=alt.Legend(
                                                            format=".0%",
                                                            orient="bottom",
                                                            direction="horizontal",
                                                            offset=12,
                                                            titlePadding=8,
                                                            labelPadding=6,
                                                            values=[0, 0.5, 1],
                                                        ),
                                                    ),
                                                    tooltip=tooltip,
                                                )
                                                .properties(
                                                    height=chart_height + 30,
                                                    padding={"top": 50, "left": 70, "right": 30, "bottom": 30},
                                                )
                                            )
                                            st.altair_chart(chart, use_container_width=True)

    return {
        "track_choice": track_choice,
        "metric_choice_label": metric_choice_label,
        "bin_choice": bin_choice,
        "tumour_choice": tumour_choice,
        "metric_label_to_key": metric_label_to_key,
    }
