"""Unified Streamlit app for thesis-facing result exploration.

The app intentionally reads only from `outputs/thesis`. It presents the
manuscript figures in order and adds lightweight interactive tables for
supporting outputs that were not all shown in the written Results section.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d

alt.data_transformers.disable_max_rows()


ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = ROOT / "outputs" / "thesis"

SECTION_DIRS = {
    "Figure 1": THESIS_ROOT / "01_pan_celltype_benchmark",
    "Figure 2": THESIS_ROOT / "02_foxa2_epigenome_orientation",
    "Figure 3": THESIS_ROOT / "03_hepatocyte_clinical_associations",
    "Figure 4": THESIS_ROOT / "04_differential_expression",
    "Figure 5": THESIS_ROOT / "05_null_bootstrap_validation",
}

FOXA2_PLUS = "#1f77b4"
FOXA2_MINUS = "#ff7f0e"


def assert_thesis_path(path: Path) -> Path:
    """Return `path` if it is inside `outputs/thesis`, otherwise fail."""
    resolved = path.resolve()
    try:
        resolved.relative_to(THESIS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Non-thesis dependency blocked: {resolved}") from exc
    return resolved


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load a thesis-local CSV."""
    resolved = assert_thesis_path(Path(path))
    return pd.read_csv(resolved)


def thesis_path(*parts: str) -> Path:
    return assert_thesis_path(THESIS_ROOT.joinpath(*parts))


def load_section_csv(section: str, *parts: str) -> pd.DataFrame:
    return load_csv(str(assert_thesis_path(SECTION_DIRS[section].joinpath(*parts))))


def figure_path(section: str, filename: str) -> Path:
    return assert_thesis_path(SECTION_DIRS[section] / "figures" / filename)


def data_path(section: str, filename: str) -> Path:
    return assert_thesis_path(SECTION_DIRS[section] / "data" / filename)


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def p_value(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_with_caption(path: Path, caption: str, width: int = 760) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.warning(f"Missing figure: {path.relative_to(THESIS_ROOT)}")


def dataframe_download(df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        "Download table",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def render_table(df: pd.DataFrame, filename: str, height: int = 260) -> None:
    st.dataframe(df, use_container_width=True, height=height)
    dataframe_download(df, filename)


def line_list(items: Iterable[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def display_state(value: str) -> str:
    labels = {
        "foxa2_normal_pos": "FOXA2+",
        "foxa2_abnormal_zero": "FOXA2-",
        "FOXA2+": "FOXA2+",
        "FOXA2-": "FOXA2-",
    }
    return labels.get(str(value), str(value))


def static_panel_expander(section: str, panels: list[tuple[str, str]]) -> None:
    with st.expander("Static thesis panels"):
        cols = st.columns(min(3, len(panels)))
        for idx, (filename, caption) in enumerate(panels):
            with cols[idx % len(cols)]:
                image_with_caption(figure_path(section, filename), caption, width=300)


def cleaned_label(text: str) -> str:
    return str(text).replace("_", " ").replace("spearman linear resid", "spearman_linear_resid")


def add_neg_log10_p(df: pd.DataFrame, p_col: str = "padj") -> pd.DataFrame:
    out = df.copy()
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    floor = out[p_col][out[p_col] > 0].min()
    if pd.isna(floor):
        floor = 1e-300
    out["neg_log10_fdr"] = -out[p_col].clip(lower=floor).map(math.log10)
    return out


def compute_kaplan_meier(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_label, group in labels.groupby("predicted_label"):
        state = display_state(raw_label)
        time = group["days_to_death"].fillna(group["days_to_last_followup"]).fillna(0) / 365.25
        event = group["vital_status"].str.upper().eq("DECEASED")
        km = pd.DataFrame({"time_years": time, "event": event}).sort_values("time_years")
        survival = 1.0
        rows.append(
            {
                "state": state,
                "time_years": 0.0,
                "survival_probability": survival,
                "n_at_risk": len(km),
                "n_events": 0,
            }
        )
        for event_time, block in km.groupby("time_years", sort=True):
            at_risk = int((km["time_years"] >= event_time).sum())
            events = int(block["event"].sum())
            if events > 0 and at_risk > 0:
                survival *= 1 - events / at_risk
                rows.append(
                    {
                        "state": state,
                        "time_years": float(event_time),
                        "survival_probability": survival,
                        "n_at_risk": at_risk,
                        "n_events": events,
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_track_mutations() -> pd.DataFrame:
    return load_section_csv(
        "Figure 2",
        "data",
        "lihc_mutations_for_track_visualisation.csv",
    )


def prepare_atac_bins(atac: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    if bin_size == 500_000:
        out = atac.copy()
    else:
        out = atac.copy()
        out["start"] = (out["start"] // bin_size) * bin_size
        out["end"] = out["start"] + bin_size
        out = (
            out.groupby(["chrom", "start", "end"], as_index=False)[
                ["foxa2_normal_like_atac", "foxa2_abnormal_like_atac"]
            ]
            .sum()
            .sort_values(["chrom", "start"])
        )
    out["midpoint"] = (out["start"] + out["end"]) / 2
    out["midpoint_mb"] = out["midpoint"] / 1e6
    out["delta_abnormal_minus_normal"] = (
        out["foxa2_abnormal_like_atac"] - out["foxa2_normal_like_atac"]
    )
    out["absolute_delta"] = out["delta_abnormal_minus_normal"].abs()
    out["bin_label"] = out["chrom"] + ":" + out["start"].astype(str) + "-" + out["end"].astype(str)
    return out


def chromosome_sort_key(chrom: str) -> tuple[int, str]:
    value = str(chrom).replace("chr", "")
    if value.isdigit():
        return int(value), ""
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return order.get(value, 99), value


@st.cache_data(show_spinner=False)
def load_pan_benchmark_mutations() -> pd.DataFrame:
    return load_section_csv(
        "Figure 1",
        "data",
        "pan_benchmark_sample_mutations.csv.gz",
    )


@st.cache_data(show_spinner=False)
def load_pan_chrom_sizes() -> pd.DataFrame:
    sizes = load_section_csv("Figure 1", "data", "hg19_chrom_sizes.csv")
    return sizes.sort_values("chrom", key=lambda s: s.map(chromosome_sort_key))


@st.cache_data(show_spinner=False)
def load_pan_chromatin_tracks() -> pd.DataFrame:
    return load_section_csv(
        "Figure 1",
        "data",
        "pan_benchmark_chromatin_tracks.csv.gz",
    )


def compute_pan_track_for_chrom(
    mutations: pd.DataFrame,
    chrom: str,
    chrom_length: int,
    bin_size: int,
    strategy: str,
    sigma_bins: float,
    decay_bp: float,
    max_distance_bp: int,
) -> pd.DataFrame:
    starts = np.arange(0, chrom_length, bin_size, dtype=int)
    ends = np.minimum(starts + bin_size, chrom_length)
    centres = (starts + ends) / 2
    chrom_mut = np.sort(
        mutations.loc[mutations["chrom"].eq(chrom), "start"].to_numpy(dtype=float)
    )
    left = np.searchsorted(chrom_mut, starts, side="left")
    right = np.searchsorted(chrom_mut, ends, side="left")
    counts = (right - left).astype(float)
    if strategy == "counts_raw":
        signal = counts
    elif strategy == "counts_gauss":
        signal = gaussian_filter1d(counts, sigma=sigma_bins, mode="constant")
    else:
        signal = np.zeros(len(centres), dtype=float)
        for idx, centre in enumerate(centres):
            lo = np.searchsorted(chrom_mut, centre - max_distance_bp, side="left")
            hi = np.searchsorted(chrom_mut, centre + max_distance_bp, side="right")
            nearby = chrom_mut[lo:hi]
            if len(nearby) > 0:
                signal[idx] = np.exp(-np.abs(nearby - centre) / decay_bp).sum()
    return pd.DataFrame(
        {
            "chrom": chrom,
            "start": starts,
            "end": ends,
            "midpoint": centres,
            "midpoint_mb": centres / 1e6,
            "mutation_count": counts,
            "mutation_signal": signal,
        }
    )


@st.cache_data(show_spinner=False)
def compute_pan_track(
    sample: str,
    bin_size: int,
    strategy: str,
    sigma_bins: float,
    decay_bp: float,
    max_distance_bp: int,
) -> pd.DataFrame:
    mutations = load_pan_benchmark_mutations()
    sample_mutations = mutations[mutations["sample_id"].eq(sample)].copy()
    chrom_sizes = load_pan_chrom_sizes()
    tracks = [
        compute_pan_track_for_chrom(
            sample_mutations,
            str(row.chrom),
            int(row.length),
            bin_size,
            strategy,
            sigma_bins,
            decay_bp,
            max_distance_bp,
        )
        for row in chrom_sizes.itertuples(index=False)
    ]
    return pd.concat(tracks, ignore_index=True)


@st.cache_data(show_spinner=False)
def compute_binned_mutation_track(
    sample: str,
    bin_size: int,
    strategy: str,
    decay_bp: float,
    max_distance_bp: int,
    gaussian_sigma_bins: float,
) -> pd.DataFrame:
    atac = prepare_atac_bins(
        load_section_csv("Figure 2", "data", "foxa2_atac_500kb_bins.csv"),
        bin_size,
    )
    mutations = load_track_mutations()
    sample_mut = mutations[mutations["sample"].eq(sample)].copy()
    rows = []
    for chrom, bins in atac.groupby("chrom", sort=False):
        chrom_mut = np.sort(sample_mut.loc[sample_mut["chrom"].eq(chrom), "start"].to_numpy(dtype=float))
        centres = bins["midpoint"].to_numpy(dtype=float)
        starts = bins["start"].to_numpy(dtype=float)
        ends = bins["end"].to_numpy(dtype=float)
        left = np.searchsorted(chrom_mut, starts, side="left")
        right = np.searchsorted(chrom_mut, ends, side="left")
        counts = (right - left).astype(float)
        if strategy == "counts_raw":
            signal = counts
        elif strategy == "counts_gauss":
            signal = gaussian_filter1d(counts, sigma=gaussian_sigma_bins, mode="constant")
        else:
            signal = np.zeros(len(centres), dtype=float)
            for idx, centre in enumerate(centres):
                left = np.searchsorted(chrom_mut, centre - max_distance_bp, side="left")
                right = np.searchsorted(chrom_mut, centre + max_distance_bp, side="right")
                nearby = chrom_mut[left:right]
                if len(nearby) > 0:
                    signal[idx] = np.exp(-np.abs(nearby - centre) / decay_bp).sum()
        tmp = bins.copy()
        tmp["mutation_signal"] = signal
        tmp["sample"] = sample
        tmp["track_strategy"] = strategy
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def correlation_summary(track_df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for state, col in [
        ("FOXA2+", "foxa2_normal_like_atac"),
        ("FOXA2-", "foxa2_abnormal_like_atac"),
    ]:
        sub = track_df[["mutation_signal", col]].replace([np.inf, -np.inf], np.nan).dropna()
        correlation = track_correlation(sub, col, method)
        rows.append(
            {
                "reference": state,
                "correlation": correlation,
                "alignment_score": -correlation,
                "n_bins": len(sub),
            }
        )
    return pd.DataFrame(rows)


def track_correlation(track_df: pd.DataFrame, atac_col: str, method: str) -> float:
    sub = track_df[["mutation_signal", atac_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub["mutation_signal"].nunique() < 2 or sub[atac_col].nunique() < 2:
        return np.nan
    return sub["mutation_signal"].corr(sub[atac_col], method=method.lower())


def default_aligned_window(
    track_df: pd.DataFrame,
    best_cell_state: str,
    method: str,
    window_mb: float = 10.0,
) -> tuple[str, float, float]:
    target = display_state(best_cell_state)
    target_col = "foxa2_normal_like_atac" if target == "FOXA2+" else "foxa2_abnormal_like_atac"
    other_col = "foxa2_abnormal_like_atac" if target == "FOXA2+" else "foxa2_normal_like_atac"
    rows = []
    for chrom, chrom_track in track_df.groupby("chrom", sort=False):
        chrom_track = chrom_track.sort_values("midpoint")
        for start in chrom_track["start"].to_numpy()[::5]:
            end = start + window_mb * 1e6
            sub = chrom_track[chrom_track["midpoint"].between(start, end, inclusive="both")]
            if len(sub) < 8:
                continue
            target_corr = track_correlation(sub, target_col, method)
            other_corr = track_correlation(sub, other_col, method)
            rows.append(
                {
                    "chrom": chrom,
                    "start_mb": float(start) / 1e6,
                    "end_mb": float(end) / 1e6,
                    "target_corr": target_corr,
                    "margin": other_corr - target_corr,
                }
            )
    windows = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    windows = windows[windows["target_corr"] < 0].sort_values(
        ["margin", "target_corr"],
        ascending=[False, True],
    )
    if windows.empty:
        chrom = "chr19" if "chr19" in track_df["chrom"].unique() else track_df["chrom"].iloc[0]
        return chrom, 19.0, 29.0
    row = windows.iloc[0]
    return str(row["chrom"]), float(row["start_mb"]), float(row["end_mb"])


def add_genome_coordinates(track_df: pd.DataFrame) -> pd.DataFrame:
    chroms = sorted(track_df["chrom"].unique(), key=chromosome_sort_key)
    offsets: dict[str, float] = {}
    offset = 0.0
    for chrom in chroms:
        offsets[chrom] = offset
        offset += float(track_df.loc[track_df["chrom"].eq(chrom), "end"].max())
    out = track_df.copy()
    out["genome_midpoint_mb"] = (out["midpoint"] + out["chrom"].map(offsets)) / 1e6
    return out


def null_range_chart(df: pd.DataFrame, title: str, x_max: float) -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y(
            "label:N",
            sort=list(df["label"]),
            title=None,
            axis=alt.Axis(labelLimit=360, labelPadding=12),
        )
    )
    ranges = base.mark_rule(color="#bdbdbd", strokeWidth=7).encode(
        x=alt.X(
            "null_min:Q",
            title="Number of significant tests/results",
            scale=alt.Scale(domain=[0, x_max]),
        ),
        x2="null_max:Q",
        tooltip=["label", "null_min", "null_max"],
    )
    means = base.mark_circle(color="#6f6f6f", size=90).encode(
        x=alt.X("null_mean:Q", scale=alt.Scale(domain=[0, x_max])),
        tooltip=["label", alt.Tooltip("null_mean:Q", format=".2f")],
    )
    observed = base.mark_circle(color="#1f77b4", size=90).encode(
        x=alt.X("proper_count:Q", scale=alt.Scale(domain=[0, x_max])),
        tooltip=["label", "proper_count", "p_null_ge_proper"],
    )
    return (ranges + means + observed).properties(title=title, height=165)


def set_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1040px;
            padding-top: 2rem;
            padding-left: 3.5rem;
            padding-right: 3.5rem;
        }
        .metric-card {
            border: 1px solid #d9dee7;
            border-radius: 7px;
            padding: 0.75rem 0.85rem;
            min-height: 110px;
            background: #ffffff;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #53606f;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .metric-value {
            font-size: 1.65rem;
            line-height: 1.15;
            font-weight: 760;
            color: #17202a;
            margin-top: 0.2rem;
        }
        .metric-caption {
            font-size: 0.86rem;
            color: #687385;
            margin-top: 0.35rem;
        }
        .section-note {
            border-left: 4px solid #6f7f92;
            padding: 0.2rem 0 0.2rem 0.8rem;
            color: #2f3a45;
            background: #f7f8fa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_selected_locus_track() -> None:
    st.header("Selected Locus Track")
    st.markdown(
        """
        Interactive view of the selected FOXA2 locus used in Figure 2d, using
        precomputed ATAC, mutation-signal, and mutation-position tables from
        `outputs/thesis`.
        """
    )

    profile = load_section_csv(
        "Figure 2",
        "data",
        "selected_locus_foxa2_atac_and_sample_exp_decay_profile.csv",
    )
    mutations = load_section_csv("Figure 2", "data", "selected_locus_mutation_positions.csv")
    locus_summary = load_section_csv("Figure 2", "data", "selected_locus_illustrative_sample_score_summary.csv")
    summary = locus_summary.iloc[0]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Sample", str(summary["sample"]), "Illustrative TCGA-LIHC tumour")
    with cols[1]:
        metric_card("Region", str(summary["region"]), "Shown in Figure 2d")
    with cols[2]:
        metric_card("Region mutations", str(int(summary["region_mutation_count"])), "Vertical mutation marks")
    with cols[3]:
        metric_card("Score gap", f"{float(summary['genome_wide_score_gap']):.3f}", "FOXA2+ minus FOXA2-")

    st.subheader("Interactive locus tracks")
    left, right = st.columns([0.72, 0.28])
    with right:
        st.markdown("**Signals**")
        show_atac = st.checkbox("ATAC references", value=True)
        show_mutation_signal = st.checkbox("exp-decay mutation signal", value=True)
        show_mutation_marks = st.checkbox("mutation positions", value=True)
        position_range = st.slider(
            "chr19 position (Mb)",
            min_value=float(profile["midpoint_mb"].min()),
            max_value=float(profile["midpoint_mb"].max()),
            value=(float(profile["midpoint_mb"].min()), float(profile["midpoint_mb"].max())),
            step=0.05,
        )
    with left:
        sub = profile[
            profile["midpoint_mb"].between(position_range[0], position_range[1], inclusive="both")
        ].copy()
        mut_sub = mutations[
            mutations["position_mb"].between(position_range[0], position_range[1], inclusive="both")
        ].copy()

        charts = []
        if show_atac:
            atac_long = sub.melt(
                id_vars=["midpoint_mb", "bin_label"],
                value_vars=["FOXA2+ ATAC", "FOXA2- ATAC"],
                var_name="track",
                value_name="signal",
            )
            atac_chart = (
                alt.Chart(atac_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("midpoint_mb:Q", title="chr19 position (Mb)"),
                    y=alt.Y("signal:Q", title="ATAC signal"),
                    color=alt.Color(
                        "track:N",
                        scale=alt.Scale(
                            domain=["FOXA2+ ATAC", "FOXA2- ATAC"],
                            range=[FOXA2_PLUS, FOXA2_MINUS],
                        ),
                        title="Reference",
                    ),
                    tooltip=["bin_label", "track", alt.Tooltip("signal:Q", format=".3f")],
                )
                .properties(height=280)
                .interactive()
            )
            charts.append(atac_chart)

        if show_mutation_signal:
            mut_signal = (
                alt.Chart(sub)
                .mark_line(point=True, color="#6f4e7c")
                .encode(
                    x=alt.X("midpoint_mb:Q", title="chr19 position (Mb)"),
                    y=alt.Y("sample_exp_decay_mutation_signal:Q", title="Mutation signal"),
                    tooltip=[
                        "bin_label",
                        alt.Tooltip("sample_exp_decay_mutation_signal:Q", format=".3f"),
                        "n_mutations_used_for_decay_window",
                    ],
                )
                .properties(height=220)
                .interactive()
            )
            charts.append(mut_signal)

        if show_mutation_marks:
            marks = (
                alt.Chart(mut_sub)
                .mark_tick(color="#7d6a86", thickness=1.5, size=36)
                .encode(
                    x=alt.X("position_mb:Q", title="chr19 position (Mb)"),
                    y=alt.value(18),
                    tooltip=["sample", alt.Tooltip("position_mb:Q", format=".3f"), "ref", "alt"],
                )
                .properties(height=70)
            )
            charts.append(marks)

        if charts:
            chart = alt.vconcat(*charts).resolve_scale(x="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Select at least one signal to display.")

    st.subheader("Original thesis panel")
    image_with_caption(
        figure_path("Figure 2", "04_selected_locus_foxa2_atac_with_sample_exp_decay_mutations.png"),
        "Static manuscript panel generated by the Figure 2 notebook.",
    )

    with st.expander("Show track data tables"):
        st.markdown("**Binned ATAC and mutation signal**")
        render_table(profile, "track_visualisation_selected_locus_profile.csv", height=260)
        st.markdown("**Mutation positions**")
        render_table(mutations, "track_visualisation_selected_locus_mutations.csv", height=220)


def render_track_visualisation() -> None:
    st.header("Track Visualisation")
    st.markdown(
        """
        Interactive explorer for the pan-cell-type benchmark rows used in Figure 1.
        This section uses the benchmark output table directly, including
        `counts_raw`, `counts_gauss` and `exp_decay` configurations. It is
        separate from the LIHC FOXA2 reference-alignment analyses in Figures 2-5.
        """
    )

    benchmark = load_section_csv("Figure 1", "pan_celltype_benchmark_results.csv")
    bin_cols = {
        "counts_raw": "counts_raw_bin",
        "counts_gauss": "counts_gauss_bin",
        "exp_decay": "exp_decay_bin",
    }
    scoring = {
        "raw": {
            "score_prefix": "pearson_r_raw_",
            "best_col": "best_celltype_raw",
            "value_col": "best_celltype_raw_value",
            "gap_col": "best_minus_second_raw",
            "correct_col": "is_correct_raw",
        },
        "linear_resid": {
            "score_prefix": "pearson_r_linear_resid_",
            "best_col": "best_celltype_linear_resid",
            "value_col": "best_celltype_linear_resid_value",
            "gap_col": "best_minus_second_linear_resid",
            "correct_col": "is_correct_linear_resid",
        },
        "spearman_raw": {
            "score_prefix": "spearman_r_raw_",
            "best_col": "best_celltype_spearman_raw",
            "value_col": "best_celltype_spearman_raw_value",
            "gap_col": "best_minus_second_spearman_raw",
            "correct_col": "is_correct_spearman_raw",
        },
        "spearman_linear_resid": {
            "score_prefix": "spearman_r_linear_resid_",
            "best_col": "best_celltype_spearman_linear_resid",
            "value_col": "best_celltype_spearman_linear_resid_value",
            "gap_col": "best_minus_second_spearman_linear_resid",
            "correct_col": "is_correct_spearman_linear_resid",
        },
        "pearson_local_score": {
            "score_prefix": "pearson_local_score_global_",
            "best_col": "best_celltype_pearson_local_score",
            "value_col": "best_celltype_pearson_local_score_value",
            "gap_col": "best_minus_second_pearson_local_score",
            "correct_col": "is_correct_pearson_local_score",
        },
        "spearman_local_score": {
            "score_prefix": "spearman_local_score_global_",
            "best_col": "best_celltype_spearman_local_score",
            "value_col": "best_celltype_spearman_local_score_value",
            "gap_col": "best_minus_second_spearman_local_score",
            "correct_col": "is_correct_spearman_local_score",
        },
        "rf_resid": {
            "score_prefix": "pearson_r_rf_resid_",
            "best_col": "best_celltype_rf_resid",
            "value_col": "best_celltype_rf_resid_value",
            "gap_col": "best_minus_second_rf_resid",
            "correct_col": "is_correct_rf_resid",
        },
    }

    sample_meta = load_section_csv(
        "Figure 1",
        "data",
        "track_visualisation_representative_samples.csv",
    )
    sample_labels = {
        row.sample_id: f"{row.sample_id} | {row.selected_tumour_types}"
        for row in sample_meta.itertuples(index=False)
    }

    controls = st.columns([0.34, 0.18, 0.18, 0.30])
    default_sample = (
        sample_meta[sample_meta["default_is_correct"].astype(bool)]
        .sort_values("default_score_gap", ascending=False)
        .iloc[0]["sample_id"]
    )
    sample_options = sample_meta["sample_id"].tolist()
    with controls[0]:
        sample = st.selectbox(
            "Benchmark sample",
            sample_options,
            index=sample_options.index(default_sample) if default_sample in sample_options else 0,
            format_func=lambda value: sample_labels[value],
        )
    with controls[1]:
        strategy = st.selectbox("Mutation track", ["exp_decay", "counts_raw", "counts_gauss"])
    bin_col = bin_cols[strategy]
    available_bins = (
        benchmark[
            benchmark["sample_id"].eq(sample)
            & benchmark["track_strategy"].eq(strategy)
        ][bin_col]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    with controls[2]:
        bin_size = st.selectbox(
            "Bin size",
            available_bins,
            format_func=lambda value: f"{value / 1e6:.1f} Mb",
        )
    with controls[3]:
        scoring_key = st.selectbox(
            "Scoring metric",
            list(scoring),
            index=list(scoring).index("spearman_linear_resid"),
        )

    selected = benchmark[
        benchmark["sample_id"].eq(sample)
        & benchmark["track_strategy"].eq(strategy)
        & benchmark[bin_col].eq(float(bin_size))
    ].copy()
    if selected.empty:
        st.warning("No benchmark row is available for this sample and configuration.")
        return

    row = selected.iloc[0]
    score_def = scoring[scoring_key]
    prediction = str(row[score_def["best_col"]])
    correct_celltype = str(row["correct_celltypes"])
    is_correct = bool(row[score_def["correct_col"]])

    cards = st.columns(5)
    with cards[0]:
        metric_card("Benchmark tumour type", str(row["selected_tumour_types"]), "Pan-cell-type dataset")
    with cards[1]:
        metric_card("Correct cell type", correct_celltype, "Benchmark label")
    with cards[2]:
        metric_card("Predicted cell type", prediction, scoring_key)
    with cards[3]:
        metric_card("Score gap", f"{float(row[score_def['gap_col']]):.3f}", "Best minus second")
    with cards[4]:
        metric_card(
            "Mutation burden",
            f"{int(row['mutations_post_downsample']):,}",
            "Post-downsample mutations",
        )

    if is_correct:
        st.success("Selected configuration assigns this sample to the correct benchmark cell type.")
    else:
        st.warning("Selected configuration does not assign this sample to the correct benchmark cell type.")

    sigma_bins = float(row["counts_gauss_sigma_bins"]) if strategy == "counts_gauss" else 1.0
    decay_bp = float(row["exp_decay_decay_bp"]) if strategy == "exp_decay" else 200_000.0
    max_distance_bp = (
        int(row["exp_decay_max_distance_bp"])
        if strategy == "exp_decay"
        else 1_000_000
    )
    sample_mutations = load_pan_benchmark_mutations()
    sample_mutations = sample_mutations[sample_mutations["sample_id"].eq(sample)].copy()
    track = compute_pan_track(
        sample,
        int(bin_size),
        strategy,
        sigma_bins,
        decay_bp,
        max_distance_bp,
    )

    bin_cards = st.columns(3)
    with bin_cards[0]:
        metric_card("Selected bin size", f"{bin_size / 1e6:.1f} Mb", "Applies to scores and tracks")
    with bin_cards[1]:
        metric_card("Genome bins", f"{len(track):,}", "Mutation track bins")
    with bin_cards[2]:
        chromatin_bin_n = int(
            load_pan_chromatin_tracks()["bin_size"].eq(int(bin_size)).sum()
        )
        metric_card("Chromatin bins", f"{chromatin_bin_n:,}", "All benchmark chromatin tracks")

    st.subheader("Mutation and chromatin tracks")
    chromatin = load_pan_chromatin_tracks()
    chromatin = chromatin[chromatin["bin_size"].eq(int(bin_size))].copy()
    shown_celltypes = [correct_celltype]
    if prediction not in shown_celltypes:
        shown_celltypes.append(prediction)
    extra_celltypes = sorted(
        set(chromatin["celltype"].dropna().astype(str)) - set(shown_celltypes),
        key=str,
    )
    selected_extra_celltypes = st.multiselect(
        "Additional chromatin tracks",
        extra_celltypes,
        default=[],
        help="Correct and selected chromatin tracks are always shown.",
    )
    shown_celltypes.extend(selected_extra_celltypes)

    track_cols = st.columns([0.22, 0.26, 0.52])
    chrom_order = load_pan_chrom_sizes()["chrom"].tolist()
    with track_cols[0]:
        view_mode = st.selectbox("Track view", ["Chromosome window", "Whole genome"])
    with track_cols[1]:
        if view_mode == "Chromosome window":
            mutation_counts = sample_mutations["chrom"].value_counts()
            default_chrom = str(mutation_counts.index[0]) if len(mutation_counts) else "chr1"
            chrom = st.selectbox(
                "Chromosome",
                chrom_order,
                index=chrom_order.index(default_chrom) if default_chrom in chrom_order else 0,
            )
        else:
            chrom = chrom_order[0]
    with track_cols[2]:
        if strategy == "counts_gauss":
            st.caption(f"Gaussian sigma is fixed to the benchmark value: {sigma_bins:g} bins.")
        elif strategy == "counts_raw":
            st.caption("counts_raw uses raw mutation counts per bin.")

    if view_mode == "Whole genome":
        plot_track = add_genome_coordinates(track)
        x_field = "genome_midpoint_mb"
        x_title = "Genome position (concatenated Mb)"
        mutation_marks = pd.DataFrame()
    else:
        chrom_track = track[track["chrom"].eq(chrom)].copy()
        chrom_length_mb = float(chrom_track["end"].max()) / 1e6
        default_end = min(10.0, chrom_length_mb)
        window = st.slider(
            f"{chrom} window (Mb)",
            min_value=0.0,
            max_value=chrom_length_mb,
            value=(0.0, default_end),
            step=0.5,
        )
        plot_track = chrom_track[
            chrom_track["midpoint_mb"].between(window[0], window[1], inclusive="both")
        ].copy()
        mutation_marks = sample_mutations[
            sample_mutations["chrom"].eq(chrom)
            & sample_mutations["start"].between(window[0] * 1e6, window[1] * 1e6, inclusive="both")
        ].copy()
        mutation_marks["position_mb"] = mutation_marks["start"] / 1e6
        x_field = "midpoint_mb"
        x_title = f"{chrom} position (Mb)"

    track_chart = (
        alt.Chart(plot_track)
        .mark_line(color="#6f4e7c", strokeWidth=2)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y("mutation_signal:Q", title="Mutation-track signal"),
            tooltip=[
                "chrom",
                alt.Tooltip("start:Q", format=","),
                alt.Tooltip("end:Q", format=","),
                alt.Tooltip("mutation_count:Q", format=".0f"),
                alt.Tooltip("mutation_signal:Q", format=".3f"),
            ],
        )
        .properties(height=260)
        .interactive()
    )
    if view_mode == "Chromosome window" and len(mutation_marks) > 0:
        marks_chart = (
            alt.Chart(mutation_marks)
            .mark_tick(color="#7d6a86", thickness=1, size=36)
            .encode(
                x=alt.X("position_mb:Q", title=x_title),
                y=alt.value(18),
                tooltip=[
                    "sample_id",
                    alt.Tooltip("position_mb:Q", format=".3f"),
                    "ref",
                    "alt",
                    "tumour_type",
                ],
            )
            .properties(height=70)
        )
        st.altair_chart(
            alt.vconcat(track_chart, marks_chart).resolve_scale(x="shared"),
            use_container_width=True,
        )
    else:
        st.altair_chart(track_chart, use_container_width=True)

    chromatin = chromatin[chromatin["celltype"].isin(shown_celltypes)].copy()
    if view_mode == "Whole genome":
        plot_chromatin = add_genome_coordinates(chromatin)
        chromatin_x_field = "genome_midpoint_mb"
    else:
        plot_chromatin = chromatin[
            chromatin["chrom"].eq(chrom)
            & chromatin["midpoint_mb"].between(window[0], window[1], inclusive="both")
        ].copy()
        chromatin_x_field = "midpoint_mb"
    plot_chromatin["track_label"] = plot_chromatin["celltype"]
    same_track = prediction == correct_celltype
    if same_track:
        plot_chromatin.loc[
            plot_chromatin["celltype"].eq(correct_celltype),
            "track_label",
        ] = f"{correct_celltype} (correct and selected)"
    else:
        plot_chromatin.loc[
            plot_chromatin["celltype"].eq(correct_celltype),
            "track_label",
        ] = f"{correct_celltype} (correct)"
        plot_chromatin.loc[
            plot_chromatin["celltype"].eq(prediction),
            "track_label",
        ] = f"{prediction} (selected)"
    selected_labels = plot_chromatin["track_label"].drop_duplicates().tolist()
    chromatin_palette = [
        "#2ca25f" if "correct and selected" in label else
        "#ff7f0e" if "(correct)" in label else
        "#1f77b4" if "(selected)" in label else
        "#9aa4b2"
        for label in selected_labels
    ]
    chromatin_chart = (
        alt.Chart(plot_chromatin)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{chromatin_x_field}:Q", title=x_title),
            y=alt.Y("chromatin_signal:Q", title="DNase chromatin signal"),
            color=alt.Color(
                "track_label:N",
                scale=alt.Scale(domain=selected_labels, range=chromatin_palette),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                "track_label",
                "celltype_name",
                "chrom",
                alt.Tooltip("start:Q", format=","),
                alt.Tooltip("end:Q", format=","),
                alt.Tooltip("chromatin_signal:Q", format=".4f"),
            ],
        )
        .properties(height=230)
        .interactive()
    )
    st.altair_chart(chromatin_chart, use_container_width=True)

    score_prefix = score_def["score_prefix"]
    score_cols = [
        col for col in benchmark.columns
        if col.startswith(score_prefix) and col.endswith("_mean_weighted")
    ]
    score_rows = []
    for col in score_cols:
        celltype = col.removeprefix(score_prefix).removesuffix("_mean_weighted")
        score_rows.append(
            {
                "celltype": celltype,
                "score": float(row[col]),
                "role": (
                    "Predicted and correct"
                    if celltype == prediction and celltype == correct_celltype
                    else "Predicted"
                    if celltype == prediction
                    else "Correct"
                    if celltype == correct_celltype
                    else "Other"
                ),
            }
        )
    score_df = pd.DataFrame(score_rows).sort_values("score")

    st.subheader("Cell-type scores for the selected benchmark row")
    st.caption(
        "Scores are shown exactly as stored in the pan-cell-type benchmark table. "
        "For these correlation-style scoring systems, lower and more negative "
        "values indicate stronger mutation-to-chromatin alignment."
    )
    score_chart = (
        alt.Chart(score_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "celltype:N",
                title="Cell type",
                sort=score_df["celltype"].tolist(),
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X("score:Q", title="Benchmark score"),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=["Predicted and correct", "Predicted", "Correct", "Other"],
                    range=["#2ca25f", "#1f77b4", "#ff7f0e", "#b7bec8"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["celltype", "role", alt.Tooltip("score:Q", format=".4f")],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(score_chart, use_container_width=True)

    st.subheader("Configuration performance across benchmark samples")
    config_rows = benchmark[
        benchmark["track_strategy"].eq(strategy)
        & benchmark[bin_col].eq(float(bin_size))
    ].copy()
    accuracy = (
        config_rows.groupby("selected_tumour_types", as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            top1_accuracy=(score_def["correct_col"], "mean"),
        )
        .sort_values("top1_accuracy", ascending=False)
    )
    accuracy_chart = (
        alt.Chart(accuracy)
        .mark_bar(color="#6f7f92")
        .encode(
            x=alt.X("selected_tumour_types:N", title="Cancer type", sort="-y"),
            y=alt.Y("top1_accuracy:Q", title="Top-1 accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "selected_tumour_types",
                "n_samples",
                alt.Tooltip("top1_accuracy:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(accuracy_chart, use_container_width=True)

    with st.expander("Benchmark rows and score table"):
        render_table(
            selected[
                [
                    "sample_id",
                    "selected_tumour_types",
                    "correct_celltypes",
                    "track_strategy",
                    bin_col,
                    score_def["best_col"],
                    score_def["value_col"],
                    score_def["gap_col"],
                    score_def["correct_col"],
                ]
            ],
            "track_visualisation_selected_pan_benchmark_row.csv",
            height=120,
        )
        render_table(score_df, "track_visualisation_selected_pan_celltype_scores.csv", height=360)


def render_overview() -> None:
    perf = load_section_csv("Figure 1", "data", "overall_performance_metrics.csv")
    by_cancer = load_section_csv("Figure 1", "data", "top1_accuracy_by_cancer_type_best_setup.csv")
    state_counts = load_section_csv("Figure 3", "data", "state_distribution_best_setup.csv")
    labels = load_section_csv("Figure 4", "data", "source_inputs", "sample_labels_used.csv")
    null_summary = load_section_csv("Figure 5", "data", "null_bootstrap_proper_vs_null_summary.csv")

    best = perf.sort_values("top1_accuracy", ascending=False).iloc[0]
    n_benchmark = int(by_cancer["n_samples"].sum())
    n_lihc = int(state_counts["n_samples"].sum())
    n_expr = int(labels["sample"].nunique())
    n_null = int(null_summary["null_n"].max())

    st.header("Thesis Results Dashboard")
    st.markdown(
        """
        This app presents the curated thesis outputs in Results order and adds
        interactive tables for supporting analyses. It reads only from
        `outputs/thesis`.
        """
    )
    cols = st.columns(4)
    with cols[0]:
        metric_card(
            "Best benchmark accuracy",
            pct(best["top1_accuracy"]),
            f"{best['track_strategy']}, {best['bin_bp'] / 1e6:.1f} Mb, {best['scoring_system']}",
        )
    with cols[1]:
        metric_card("Benchmark samples", f"{n_benchmark}", "Cancer types with n >= 5")
    with cols[2]:
        metric_card("Clinical LIHC samples", f"{n_lihc}", "Complete clinical annotation subset")
    with cols[3]:
        metric_card("Expression samples", f"{n_expr}", f"Null replicates: {n_null}")

    st.subheader("Result sequence")
    line_list(
        [
            "Figure 1 establishes the benchmark-selected mutation-to-chromatin scoring setup.",
            "Figure 2 defines the TCGA-LIHC cohort and FOXA2 hepatocyte references.",
            "Figure 3 tests clinical associations using binary labels and continuous FOXA2 scores.",
            "Figure 4 links FOXA2 reference alignment to expression and pathway patterns.",
            "Figure 5 compares observed signals with mutation-randomised null benchmarks.",
        ]
    )


def render_figure_1() -> None:
    st.header("Figure 1. Pan-cell-type benchmark")
    perf = load_section_csv("Figure 1", "data", "overall_performance_metrics.csv")
    cancer = load_section_csv("Figure 1", "data", "top1_accuracy_by_cancer_type_best_setup.csv")
    gaps = load_section_csv("Figure 1", "data", "score_gap_vs_mutation_burden_selected_setup.csv")
    all_results = load_section_csv("Figure 1", "pan_celltype_benchmark_results.csv")

    best = perf.sort_values("top1_accuracy", ascending=False).iloc[0]
    cols = st.columns(3)
    with cols[0]:
        metric_card("Best top-1 accuracy", pct(best["top1_accuracy"]), "Selected benchmark configuration")
    with cols[1]:
        metric_card("Samples retained", str(int(cancer["n_samples"].sum())), "Cancer types with at least 5 samples")
    with cols[2]:
        metric_card("Best setup", "exp_decay", "0.5 Mb, spearman_linear_resid")

    st.subheader("Panel a: benchmark configurations")
    metric_order = ["raw", "linear_resid", "spearman_raw", "spearman_linear_resid"]
    selected_metrics = st.multiselect("Similarity metrics", metric_order, default=metric_order)
    perf_plot = perf[perf["scoring_system"].isin(selected_metrics)].copy()
    perf_plot["bin_mb"] = perf_plot["bin_bp"] / 1e6
    perf_plot["track_bin"] = (
        perf_plot["track_strategy"] + " | bin=" + perf_plot["bin_mb"].map(lambda x: f"{x:.1f}M")
    )
    chart = (
        alt.Chart(perf_plot)
        .mark_bar()
        .encode(
            x=alt.X("scoring_system:N", title="Similarity metric", sort=selected_metrics),
            xOffset="track_bin:N",
            y=alt.Y("top1_accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("track_bin:N", title="Track strategy and bin size"),
            tooltip=[
                "track_strategy",
                alt.Tooltip("bin_mb:Q", title="bin Mb", format=".1f"),
                "scoring_system",
                alt.Tooltip("top1_accuracy:Q", format=".3f"),
                "n_samples",
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Panel b: selected setup by cancer type")
    cancer_plot = cancer.sort_values("top1_accuracy", ascending=False).copy()
    cancer_plot["x_label"] = cancer_plot["cancer_type"] + "\nn=" + cancer_plot["n_samples"].astype(str)
    cancer_chart = (
        alt.Chart(cancer_plot)
        .mark_bar(color=FOXA2_PLUS)
        .encode(
            x=alt.X("x_label:N", title="Cancer type", sort=list(cancer_plot["x_label"])),
            y=alt.Y("top1_accuracy:Q", title="Top-1 accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["cancer_type", "n_samples", "n_correct", alt.Tooltip("top1_accuracy:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(cancer_chart, use_container_width=True)

    st.subheader("Panel c: confidence versus mutation burden")
    assignment_filter = st.multiselect(
        "Assignment outcome",
        sorted(gaps["assignment"].unique()),
        default=sorted(gaps["assignment"].unique()),
    )
    gaps_plot = gaps[gaps["assignment"].isin(assignment_filter)]
    gap_chart = (
        alt.Chart(gaps_plot)
        .mark_circle(size=50, opacity=0.72)
        .encode(
            x=alt.X("log10_mutation_burden:Q", title="log10(mutation burden + 1)"),
            y=alt.Y("score_gap:Q", title="Score gap"),
            color=alt.Color(
                "assignment:N",
                scale=alt.Scale(domain=["Correct", "Incorrect"], range=["#66c2a5", "#fc8d62"]),
            ),
            tooltip=["sample_id", "assignment", "mutations_post_downsample", "score_gap"],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(gap_chart, use_container_width=True)

    static_panel_expander(
        "Figure 1",
        [
            ("01_overall_benchmark_performance.png", "Panel a"),
            ("02_performance_by_cancer_type_best_setup.png", "Panel b"),
            ("03_score_gap_vs_mutation_burden_best_setup.png", "Panel c"),
        ],
    )

    with st.expander("Explore omitted benchmark rows"):
        top_n = st.slider("Number of ranked configurations", 5, min(30, len(perf)), 12)
        render_table(
            perf.sort_values("top1_accuracy", ascending=False).head(top_n),
            "figure1_configuration_ranking.csv",
        )
        render_table(all_results, "figure1_full_benchmark_results.csv", height=320)
        render_table(cancer, "figure1_cancer_type_accuracy.csv")


def render_figure_2() -> None:
    st.header("Figure 2. TCGA-LIHC orientation and FOXA2 references")
    lihc = load_section_csv("Figure 2", "data", "primary_lihc_sample_table.csv")
    clinical = load_section_csv("Figure 2", "data", "lihc_clinical_variable_plot_data.csv")
    umap = load_section_csv("Figure 2", "data", "gse281574_foxa2_umap_cells.csv")
    locus_summary = load_section_csv("Figure 2", "data", "selected_locus_illustrative_sample_score_summary.csv")
    locus_profile = load_section_csv(
        "Figure 2",
        "data",
        "selected_locus_foxa2_atac_and_sample_exp_decay_profile.csv",
    )
    mutations = load_section_csv("Figure 2", "data", "selected_locus_mutation_positions.csv")

    cols = st.columns(4)
    with cols[0]:
        metric_card("LIHC samples", f"{len(lihc)}", "Clinically complete subset")
    with cols[1]:
        metric_card("Median SNVs", f"{int(lihc['snv_count'].median()):,}", "Per tumour sample")
    with cols[2]:
        metric_card("Illustrative sample", str(locus_summary.iloc[0]["sample"]), "Selected chr19 locus")
    with cols[3]:
        metric_card("Locus mutations", str(int(locus_summary.iloc[0]["region_mutation_count"])), "Shown in panel d")

    st.subheader("Panel a: mutation burden")
    lihc_plot = lihc.copy()
    lihc_plot["log2_snv_count"] = lihc_plot["snv_count"].map(math.log2)
    median_log2 = math.log2(lihc["snv_count"].median())
    burden = (
        alt.Chart(lihc_plot)
        .mark_bar(color=FOXA2_PLUS, opacity=0.72)
        .encode(
            x=alt.X("log2_snv_count:Q", bin=alt.Bin(maxbins=22), title="SNV count per tumour sample (log2 scale)"),
            y=alt.Y("count():Q", title="Number of samples"),
            tooltip=[alt.Tooltip("count():Q", title="samples")],
        )
        .properties(height=320)
    )
    median_rule = alt.Chart(pd.DataFrame({"x": [median_log2]})).mark_rule(
        color="black", strokeDash=[5, 4]
    ).encode(x="x:Q")
    st.altair_chart(burden + median_rule, use_container_width=True)

    st.subheader("Panel b: clinical composition")
    level_colours = {"No": "#8da0cb", "Yes": "#66c2a5", "Not overweight": "#b3b3b3", "Overweight": "#c8b400"}
    clinical_chart = (
        alt.Chart(clinical)
        .mark_bar()
        .encode(
            y=alt.Y("variable:N", title=None, sort=list(clinical["variable"].drop_duplicates())),
            x=alt.X("percentage:Q", title="Percentage of LIHC samples"),
            color=alt.Color("level:N", scale=alt.Scale(domain=list(level_colours), range=list(level_colours.values()))),
            tooltip=["variable", "level", "n", "total", alt.Tooltip("percentage:Q", format=".1f")],
        )
        .properties(height=300)
    )
    st.altair_chart(clinical_chart, use_container_width=True)

    st.subheader("Panel c: FOXA2 hepatocyte reference gates")
    umap_mode = st.radio("UMAP colouring", ["Cell types", "FOXA2 reference hepatocytes"], horizontal=True)
    if umap_mode == "Cell types":
        data = umap
        colour = alt.Color("CellTypeSR:N", title="Cell type")
    else:
        data = umap.copy()
        data["reference_group"] = data["reference_group"].replace(
            {"FOXA2+ reference cells": "FOXA2+", "FOXA2- reference cells": "FOXA2-", "Other cells": "Other cells"}
        )
        colour = alt.Color(
            "reference_group:N",
            title="Reference group",
            scale=alt.Scale(domain=["FOXA2+", "FOXA2-", "Other cells"], range=[FOXA2_PLUS, FOXA2_MINUS, "#d9d9d9"]),
        )
    umap_chart = (
        alt.Chart(data)
        .mark_circle(size=8, opacity=0.65)
        .encode(
            x=alt.X("umap_1:Q", title="WNN UMAP 1"),
            y=alt.Y("umap_2:Q", title="WNN UMAP 2"),
            color=colour,
            tooltip=["cell_id", "CellTypeSR", "Condition", "FOXA2", "reference_group"],
        )
        .properties(height=480)
        .interactive()
    )
    st.altair_chart(umap_chart, use_container_width=True)

    st.subheader("Panel d: selected locus track")
    render_selected_locus_track()

    static_panel_expander(
        "Figure 2",
        [
            ("01_lihc_mutation_burden.png", "Panel a"),
            ("02_lihc_clinical_feature_summary.png", "Panel b"),
            ("03_foxa2_umap_reference_cells.png", "Panel c"),
            ("04_selected_locus_foxa2_atac_with_sample_exp_decay_mutations.png", "Panel d"),
        ],
    )

    with st.expander("Supporting data"):
        st.subheader("Clinical feature plot data")
        render_table(clinical, "figure2_clinical_feature_data.csv")
        st.subheader("Selected locus ATAC and mutation profile")
        render_table(locus_profile, "figure2_selected_locus_profile.csv")
        st.subheader("Selected locus mutation positions")
        render_table(mutations, "figure2_selected_locus_mutations.csv")


def render_figure_3() -> None:
    st.header("Figure 3. Clinical associations")
    states = load_section_csv("Figure 3", "data", "state_distribution_best_setup.csv")
    labels = load_section_csv("Figure 3", "data", "label_based_sensitivity_summary_best_setup.csv")
    plus = load_section_csv("Figure 3", "data", "clinical_score_association_foxa2_plus_summary_best_setup.csv")
    minus = load_section_csv("Figure 3", "data", "clinical_score_association_foxa2_minus_summary_best_setup.csv")
    significant = load_section_csv("Figure 3", "data", "significant_clinical_results_both_tracks_best_setup.csv")
    survival = load_section_csv("Figure 3", "data", "overall_survival_by_state_summary_best_setup.csv")
    rankings = load_section_csv("Figure 3", "data", "source_validation_score_rankings.csv")
    clinical_labels = load_section_csv("Figure 3", "data", "source_clinical_labels.csv")

    state_map = states.set_index("assigned_state")["n_samples"].to_dict()
    cols = st.columns(4)
    with cols[0]:
        metric_card("FOXA2+ labels", str(int(state_map.get("FOXA2+", 0))), "Binary reference alignment")
    with cols[1]:
        metric_card("FOXA2- labels", str(int(state_map.get("FOXA2-", 0))), "Binary reference alignment")
    with cols[2]:
        metric_card("Significant score tests", str(len(significant)), "Mutation-burden-adjusted")
    with cols[3]:
        metric_card("Survival", p_value(survival["logrank_p_value"].iloc[0]), "Log-rank comparison")

    st.subheader("Panel a: binary state labels")
    states_plot = states.copy()
    states_plot["assigned_state"] = states_plot["assigned_state"].map(display_state)
    state_chart = (
        alt.Chart(states_plot)
        .mark_bar()
        .encode(
            x=alt.X("assigned_state:N", title=None, sort=["FOXA2+", "FOXA2-"]),
            y=alt.Y("n_samples:Q", title="Number of samples"),
            color=alt.Color(
                "assigned_state:N",
                scale=alt.Scale(domain=["FOXA2+", "FOXA2-"], range=[FOXA2_PLUS, FOXA2_MINUS]),
                legend=None,
            ),
            tooltip=["assigned_state", "n_samples", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(height=300)
    )
    st.altair_chart(state_chart, use_container_width=True)

    st.subheader("Panel b: label-based sensitivity")
    label_plot = labels.copy()
    label_plot["p_label"] = label_plot["p_value"].map(p_value)
    label_chart = (
        alt.Chart(label_plot)
        .mark_bar(color="#7f8fa6")
        .encode(
            x=alt.X(
                "display_name:N",
                title="Clinical variable",
                sort=None,
                axis=alt.Axis(labelAngle=0, labelLimit=130),
            ),
            y=alt.Y("p_value:Q", title="Group-association p-value"),
            tooltip=["display_name", "test_type", "n_total", "p_label"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(label_chart, use_container_width=True)

    st.subheader("Panel c: continuous score associations")
    score_table = pd.concat(
        [plus.assign(track="FOXA2+ reference score"), minus.assign(track="FOXA2- reference score")],
        ignore_index=True,
    )
    viral = score_table[score_table["clinical_variable"].isin(["hbv_status", "hcv_status"])].copy()
    viral["comparison"] = viral["clinical_variable"].map({"hbv_status": "HBV status", "hcv_status": "HCV status"})
    viral["p_label"] = viral["p_value_mannwhitney_adjusted"].map(p_value)
    viral_chart = (
        alt.Chart(viral)
        .mark_bar()
        .encode(
            x=alt.X(
                "comparison:N",
                title="Clinical variable",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("track:N"),
            y=alt.Y(
                "median_group1_minus_group0_adjusted:Q",
                title="Adjusted median difference (Yes - No)",
            ),
            color=alt.Color(
                "track:N",
                scale=alt.Scale(range=[FOXA2_PLUS, FOXA2_MINUS]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["track", "comparison", "p_label", "n_group0", "n_group1"],
        )
        .properties(height=340)
    )
    st.altair_chart(viral_chart, use_container_width=True)

    st.subheader("Panel d: overall survival")
    km = compute_kaplan_meier(clinical_labels)
    km_chart = (
        alt.Chart(km)
        .mark_line(interpolate="step-after", strokeWidth=2.5)
        .encode(
            x=alt.X("time_years:Q", title="Follow-up time (years)", scale=alt.Scale(domain=[0, 10])),
            y=alt.Y(
                "survival_probability:Q",
                title="Overall survival probability",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=["FOXA2+", "FOXA2-"], range=[FOXA2_PLUS, FOXA2_MINUS]),
            ),
            tooltip=[
                "state",
                alt.Tooltip("time_years:Q", format=".2f"),
                "n_at_risk",
                "n_events",
                alt.Tooltip("survival_probability:Q", format=".3f"),
            ],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(km_chart, use_container_width=True)
    st.caption(f"Log-rank {p_value(survival['logrank_p_value'].iloc[0])}.")

    static_panel_expander(
        "Figure 3",
        [
            ("01_state_distribution_selected_setup.png", "Panel a"),
            ("02_label_sensitivity_state_composition.png", "Panel b"),
            ("03_clinical_boxplots_adjusted_scores_by_viral_status.png", "Panel c"),
            ("04_overall_survival_by_foxa2_state.png", "Panel d"),
        ],
    )

    with st.expander("Explore clinical data"):
        st.subheader("Binary-label tests")
        render_table(labels, "figure3_label_based_tests.csv")
        st.subheader("Continuous score tests")
        render_table(score_table.sort_values("p_value_mannwhitney_adjusted"), "figure3_score_tests.csv")
        st.subheader("Sample score rankings")
        config = st.selectbox("Configuration", sorted(rankings["config_id"].dropna().unique()))
        metric = st.selectbox("Scoring system", sorted(rankings["scoring_system"].dropna().unique()))
        filtered = rankings[(rankings["config_id"] == config) & (rankings["scoring_system"] == metric)]
        render_table(filtered, "figure3_score_rankings_filtered.csv", height=320)


def render_figure_4() -> None:
    st.header("Figure 4. Differential expression and pathway patterns")
    pathways = load_section_csv("Figure 4", "data", "fgsea_selected_pathways_for_lead_plot.csv")
    genes = load_section_csv("Figure 4", "data", "source_inputs", "limma_voom_binary_results.csv")
    gene_summary = load_section_csv(
        "Figure 4",
        "data",
        "significant_and_representative_gene_expression_summary.csv",
    )
    labels = load_section_csv("Figure 4", "data", "source_inputs", "sample_labels_used.csv")
    supporting_dir = SECTION_DIRS["Figure 4"] / "data" / "supporting_results"
    supporting_files = sorted(supporting_dir.glob("*.csv"))

    n_plus = int((labels["group"] == "foxa2_normal_pos").sum())
    n_minus = int((labels["group"] == "foxa2_abnormal_zero").sum())
    n_sig = int((pd.to_numeric(genes["padj"], errors="coerce") <= 0.05).sum())

    cols = st.columns(4)
    with cols[0]:
        metric_card("Expression cohort", f"{len(labels)}", "Tumours with RNA-seq labels")
    with cols[1]:
        metric_card("FOXA2+ aligned", f"{n_plus}", "Expression cohort")
    with cols[2]:
        metric_card("FOXA2- aligned", f"{n_minus}", "Expression cohort")
    with cols[3]:
        metric_card("limma-voom genes", f"{n_sig}", "FDR <= 0.05")

    st.subheader("Panel a: Hallmark pathway enrichment")
    path_plot = pathways.copy()
    path_chart = (
        alt.Chart(path_plot)
        .mark_bar()
        .encode(
            y=alt.Y("pathway_label:N", title="Hallmark pathway", sort=list(path_plot["pathway_label"])),
            x=alt.X("NES:Q", title="Normalised enrichment score (NES)"),
            color=alt.condition("datum.NES < 0", alt.value(FOXA2_PLUS), alt.value(FOXA2_MINUS)),
            tooltip=[
                "pathway_label",
                "direction",
                alt.Tooltip("NES:Q", format=".3f"),
                alt.Tooltip("padj:Q", format=".3g"),
                "leadingEdge",
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(path_chart, use_container_width=True)

    st.subheader("Panel b: limma-voom volcano")
    gene_plot = add_neg_log10_p(genes)
    gene_plot["significance"] = "Not FDR significant"
    gene_plot.loc[(gene_plot["padj"] <= 0.05) & (gene_plot["logFC"] > 0), "significance"] = "FOXA2- higher"
    gene_plot.loc[(gene_plot["padj"] <= 0.05) & (gene_plot["logFC"] < 0), "significance"] = "FOXA2+ higher"
    show_sig_only = st.checkbox("Show significant genes only in volcano", value=False)
    volcano_data = gene_plot[gene_plot["significance"] != "Not FDR significant"] if show_sig_only else gene_plot
    volcano = (
        alt.Chart(volcano_data)
        .mark_circle(size=22, opacity=0.62)
        .encode(
            x=alt.X("logFC:Q", title="log2 fold-change"),
            y=alt.Y("neg_log10_fdr:Q", title="-log10(FDR)"),
            color=alt.Color(
                "significance:N",
                scale=alt.Scale(
                    domain=["FOXA2+ higher", "FOXA2- higher", "Not FDR significant"],
                    range=[FOXA2_PLUS, FOXA2_MINUS, "#cfcfcf"],
                ),
            ),
            tooltip=["gene", alt.Tooltip("logFC:Q", format=".3f"), alt.Tooltip("padj:Q", format=".3g")],
        )
        .properties(height=430)
        .interactive()
    )
    st.altair_chart(volcano, use_container_width=True)

    st.subheader("Panel c: representative genes")
    summary_plot = gene_summary.copy()
    summary_plot["group_display"] = summary_plot["group_display"].map(display_state)
    default_genes = ["FOXA2", "TWIST2", "MYO10", "PLK1", "AURKA"]
    significant_genes = (
        summary_plot.loc[summary_plot["is_significant"].astype(bool), "symbol"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    gene_options = list(dict.fromkeys(default_genes + significant_genes))
    selected_genes = st.multiselect(
        "Genes",
        gene_options,
        default=default_genes,
        help="Includes the manuscript genes plus all limma-voom FDR-significant genes.",
    )
    summary_plot = summary_plot[summary_plot["symbol"].isin(selected_genes)]
    gene_bar = (
        alt.Chart(summary_plot)
        .mark_bar()
        .encode(
            x=alt.X("symbol:N", title="Gene", sort=selected_genes),
            xOffset=alt.XOffset("group_display:N", sort=["FOXA2+", "FOXA2-"]),
            y=alt.Y("median_expression:Q", title="Median expression"),
            color=alt.Color(
                "group_display:N",
                scale=alt.Scale(domain=["FOXA2+", "FOXA2-"], range=[FOXA2_PLUS, FOXA2_MINUS]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                "symbol",
                "group_display",
                "n_samples",
                alt.Tooltip("median_expression:Q", format=".3f"),
                alt.Tooltip("padj:Q", format=".3g"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(gene_bar, use_container_width=True)

    static_panel_expander(
        "Figure 4",
        [
            ("01_pathway_nes_plot.png", "Panel a"),
            ("02_volcano_limma_binary.png", "Panel b"),
            ("03_representative_gene_expression_panel.png", "Panel c"),
        ],
    )

    with st.expander("Explore expression data and omitted follow-up tables"):
        st.subheader("Selected Hallmark pathways")
        render_table(
            pathways[["pathway_label", "direction", "NES", "padj", "size", "leadingEdge"]],
            "figure4_selected_pathways.csv",
            height=320,
        )
        st.subheader("limma-voom genes")
        sig_only = st.toggle("Significant genes only", value=True)
        gene_table = genes.copy()
        gene_table["padj"] = pd.to_numeric(gene_table["padj"], errors="coerce")
        if sig_only:
            gene_table = gene_table[gene_table["padj"] <= 0.05].copy()
        render_table(gene_table.sort_values("padj"), "figure4_limma_voom_genes.csv", height=320)
        st.subheader("Representative gene expression summary")
        render_table(gene_summary, "figure4_representative_gene_summary.csv")
        if supporting_files:
            st.subheader("Supporting follow-up result tables")
            selected = st.selectbox("Supporting result", [p.name for p in supporting_files])
            extra = load_csv(str(supporting_dir / selected))
            render_table(extra, selected, height=360)


def render_figure_5() -> None:
    st.header("Figure 5. Mutation-randomised null benchmark")
    summary = load_section_csv("Figure 5", "data", "null_bootstrap_proper_vs_null_summary.csv")

    adjusted = summary[summary["metric"].eq("validation_group_tests_mb_adjusted_n_p_le_0.05")].iloc[0]
    limma = summary[summary["metric"].eq("limma_significant_genes")].iloc[0]
    fgsea = summary[summary["metric"].eq("fgsea_stat_significant_pathways")].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Null replicates", str(int(summary["null_n"].max())), "Mutation randomisations")
    with cols[1]:
        metric_card("Adjusted score tests", f"{int(adjusted['proper_count'])}", "Null max: " + str(int(adjusted["null_max"])))
    with cols[2]:
        metric_card("limma-voom genes", f"{int(limma['proper_count'])}", "Null max: " + str(int(limma["null_max"])))
    with cols[3]:
        metric_card("FGSEA-stat pathways", f"{int(fgsea['proper_count'])}", "Null max: " + str(int(fgsea["null_max"])))

    labels = {
        "validation_n_p_le_0.05": "Label-based tests",
        "validation_group_tests_mb_adjusted_n_p_le_0.05": "Adjusted score tests",
        "deseq_significant_genes": "DESeq2 genes",
        "limma_significant_genes": "limma-voom genes",
        "fgsea_stat_significant_pathways": "FGSEA-stat pathways",
        "fgsea_fc_significant_pathways": "FGSEA logFC-ranked pathways",
    }
    plot_df = summary[summary["metric"].isin(labels)].copy()
    plot_df["label"] = plot_df["metric"].map(labels)

    clinical_df = plot_df[
        plot_df["metric"].isin(["validation_n_p_le_0.05", "validation_group_tests_mb_adjusted_n_p_le_0.05"])
    ]
    gene_df = plot_df[plot_df["metric"].isin(["deseq_significant_genes", "limma_significant_genes"])]
    pathway_df = plot_df[
        plot_df["metric"].isin(["fgsea_stat_significant_pathways", "fgsea_fc_significant_pathways"])
    ]

    st.subheader("Panel a: clinical association tests")
    st.altair_chart(null_range_chart(clinical_df, "Clinical association tests", 5), use_container_width=True)
    st.subheader("Panel b: genes")
    st.altair_chart(null_range_chart(gene_df, "Genes", 5), use_container_width=True)
    st.subheader("Panel c: pathways")
    st.altair_chart(null_range_chart(pathway_df, "Pathways", 32), use_container_width=True)

    static_panel_expander(
        "Figure 5",
        [
            ("01_null_benchmark_clinical_tests.png", "Panel a"),
            ("02_null_benchmark_genes.png", "Panel b"),
            ("03_null_benchmark_pathways.png", "Panel c"),
        ],
    )

    with st.expander("Null summary table"):
        render_table(summary, "figure5_null_summary.csv", height=360)


def render_data_catalogue() -> None:
    st.header("Thesis Data Catalogue")
    st.markdown(
        "Every table listed here is read from `outputs/thesis`; no experiment or raw-data files are loaded."
    )
    csv_paths = sorted(p for p in THESIS_ROOT.glob("[0-9][0-9]_*/**/*.csv") if "archive" not in p.parts)
    rows = []
    for path in csv_paths:
        df = load_csv(str(path))
        rows.append(
            {
                "file": str(path.relative_to(THESIS_ROOT)),
                "rows": len(df),
                "columns": len(df.columns),
            }
        )
    catalogue = pd.DataFrame(rows)
    render_table(catalogue, "thesis_data_catalogue.csv", height=360)

    selected = st.selectbox("Preview table", catalogue["file"].tolist())
    preview = load_csv(str(thesis_path(selected)))
    st.dataframe(preview, use_container_width=True, height=420)


def main() -> None:
    st.set_page_config(
        page_title="Results App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_style()

    if not THESIS_ROOT.exists():
        st.error(f"Missing thesis output directory: {THESIS_ROOT}")
        st.stop()

    with st.sidebar:
        st.title("Results App")
        st.caption("Curated outputs only")
        page = st.radio(
            "Section",
            [
                "Overview",
                "Track visualisation",
                "Figure 1",
                "Figure 2",
                "Figure 3",
                "Figure 4",
                "Figure 5",
                "Data catalogue",
            ],
            key="page",
            label_visibility="collapsed",
        )

    if page == "Overview":
        render_overview()
    elif page == "Figure 1":
        render_figure_1()
    elif page == "Figure 2":
        render_figure_2()
    elif page == "Figure 3":
        render_figure_3()
    elif page == "Figure 4":
        render_figure_4()
    elif page == "Figure 5":
        render_figure_5()
    elif page == "Track visualisation":
        render_track_visualisation()
    else:
        render_data_catalogue()


if __name__ == "__main__":
    main()
