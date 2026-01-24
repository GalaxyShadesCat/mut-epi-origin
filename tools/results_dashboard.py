import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:  # pragma: no cover - optional
    alt = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dnase_map import DnaseCellTypeMap
ASSETS_DIR = Path(__file__).resolve().with_name("results_dashboard_assets")
CONFIG_PATH = ASSETS_DIR / "results_dashboard_config.json"
DEFAULT_RESULTS_PATH = "outputs/experiments/simple_check/results.csv"
EXPERIMENTS_DIR = ROOT / "outputs" / "experiments"

METRICS = {
    "raw": {
        "best_cell": "best_celltype_raw",
        "best_value": "best_celltype_raw_value",
        "best_margin": "best_minus_second_raw",
        "is_correct": "is_correct_raw",
    },
    "linear_resid": {
        "best_cell": "best_celltype_linear_resid",
        "best_value": "best_celltype_linear_resid_value",
        "best_margin": "best_minus_second_linear_resid",
        "is_correct": "is_correct_linear_resid",
    },
    "spearman_raw": {
        "best_cell": "best_celltype_spearman_raw",
        "best_value": "best_celltype_spearman_raw_value",
        "best_margin": "best_minus_second_spearman_raw",
        "is_correct": "is_correct_spearman_raw",
    },
    "spearman_linear_resid": {
        "best_cell": "best_celltype_spearman_linear_resid",
        "best_value": "best_celltype_spearman_linear_resid_value",
        "best_margin": "best_minus_second_spearman_linear_resid",
        "is_correct": "is_correct_spearman_linear_resid",
    },
    "pearson_local_score": {
        "best_cell": "best_celltype_pearson_local_score",
        "best_value": "best_celltype_pearson_local_score_value",
        "best_margin": "best_minus_second_pearson_local_score",
        "is_correct": "is_correct_pearson_local_score",
    },
    "spearman_local_score": {
        "best_cell": "best_celltype_spearman_local_score",
        "best_value": "best_celltype_spearman_local_score_value",
        "best_margin": "best_minus_second_spearman_local_score",
        "is_correct": "is_correct_spearman_local_score",
    },
    "rf": {
        "best_cell": "best_celltype_rf_resid",
        "best_value": "best_celltype_rf_resid_value",
        "best_margin": "best_minus_second_rf_resid",
        "is_correct": "is_correct_rf_resid",
    },
}

METRIC_LABELS = {
    "raw": "Pearson r",
    "linear_resid": "Pearson r (linear covariate)",
    "spearman_raw": "Spearman r",
    "spearman_linear_resid": "Spearman r (linear covariate)",
    "pearson_local_score": "Pearson local score (linear covariate)",
    "spearman_local_score": "Spearman local score (linear covariate)",
    "rf": "RF (non-linear covariate)",
}


@st.cache_data(show_spinner=False)
def _load_dnase_celltype_map() -> Optional[DnaseCellTypeMap]:
    try:
        return DnaseCellTypeMap.from_project_root(ROOT)
    except (OSError, ValueError, KeyError):
        return None


def _format_celltype_label(
    celltypes: List[str],
    celltype_map: Optional[DnaseCellTypeMap],
) -> str:
    if not celltypes:
        return "n/a"
    labels = []
    for celltype in celltypes:
        label = celltype
        if celltype_map is not None:
            try:
                entry = celltype_map.resolve(celltype)
                alias = entry.aliases[0] if entry.aliases else entry.name
                label = f"{alias} ({entry.key})"
            except KeyError:
                label = celltype
        labels.append(label)
    return ", ".join(labels)


def _style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
        :root {
            --bg: #f4f0e9;
            --bg-2: #fff8ee;
            --ink: #1c1b1a;
            --muted: #5e5a55;
            --accent: #0f766e;
            --accent-2: #e76f51;
            --card: #fffdf8;
            --border: #e6ded2;
        }
        .stApp {
            background: radial-gradient(circle at 20% 10%, #f9efe0 0%, #f4f0e9 50%, #eef6f3 100%);
            color: var(--ink);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .block-container {
            max-width: 80vw;
            padding-top: 1.5rem;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }
        .stat-card {
            background: linear-gradient(180deg, rgba(255, 253, 248, 0.9), rgba(255, 248, 238, 0.85));
            border: 1px solid rgba(230, 222, 210, 0.8);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            min-height: 140px;
            box-shadow: 0 12px 24px rgba(30, 30, 30, 0.08);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .stat-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .stat-value {
            font-size: 30px;
            font-weight: 700;
            color: var(--accent);
        }
        .stat-meta {
            font-size: 13px;
            color: var(--muted);
            line-height: 1.4;
        }
        .section-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            min-height: 140px;
            box-shadow: 0 10px 20px rgba(30, 30, 30, 0.06);
        }
        .section-kicker {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 12px;
            color: var(--muted);
        }
        .bin-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 28px;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 12px;
            color: var(--ink);
        }
        .scroll-row {
            display: flex;
            gap: 16px;
            overflow-x: auto;
            padding-bottom: 8px;
        }
        .scroll-card {
            min-width: 320px;
            max-width: 360px;
            background: rgba(255, 253, 248, 0.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 10px 20px rgba(30, 30, 30, 0.06);
        }
        .scroll-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--ink);
        }
        .scroll-bin {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-top: 10px;
            margin-bottom: 6px;
        }
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 600;
            margin-top: 6px;
            margin-bottom: 8px;
            color: var(--ink);
        }
        .section-subtitle {
            font-size: 15px;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 10px;
        }
        .section-metric {
            font-size: 14px;
            color: var(--muted);
            margin-bottom: 6px;
        }
        .section-highlight {
            font-size: 18px;
            font-weight: 700;
            color: var(--accent);
        }
        .muted {
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_config(config: dict) -> None:
    try:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:
        return


def _list_experiment_results() -> dict[str, Path]:
    if not EXPERIMENTS_DIR.exists():
        return {}
    results = {}
    for entry in sorted(EXPERIMENTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / "results.csv"
        if candidate.exists():
            try:
                rel = candidate.relative_to(ROOT)
            except ValueError:
                rel = candidate
            results[entry.name] = rel
    return results


def _persist_config() -> None:
    _save_config(
        {
            "path_input": st.session_state.get("path_input", DEFAULT_RESULTS_PATH),
            "experiment_name": st.session_state.get("experiment_name"),
        }
    )


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_results_bytes(contents: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(contents))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _required_columns() -> List[str]:
    required = {
        "track_strategy",
        "n_bins_total",
        "sample_id",
        "correct_celltype_canon",
        "mutations_post_downsample",
        "selected_tumour_types",
    }
    for spec in METRICS.values():
        required.update([spec["best_value"], spec["best_margin"], spec["is_correct"]])
    return sorted(required)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    values = series.astype(str).str.strip().str.lower()
    return values.map({"true": True, "1": True, "false": False, "0": False}).astype("boolean")


def _normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "correct_celltype_canon" in d.columns:
        d["correct_celltype_canon"] = (
            d["correct_celltype_canon"].astype(str).str.strip().str.lower()
        )
    for spec in METRICS.values():
        col = spec["is_correct"]
        if col in d.columns:
            d[col] = _coerce_bool(d[col])
    return d


def _standout_weights(margins: pd.Series, best_values: pd.Series) -> pd.Series:
    margin_vals = pd.to_numeric(margins, errors="coerce")
    best_vals = pd.to_numeric(best_values, errors="coerce")
    second_vals = best_vals - margin_vals
    eps = 1e-6
    weights = margin_vals / (second_vals.abs() + eps)
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _resolve_bin_sizes(df: pd.DataFrame) -> pd.Series:
    if "track_strategy" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    d = df.copy()
    d["track_strategy"] = d["track_strategy"].astype(str).str.strip()
    bin_size = pd.Series(np.nan, index=df.index, dtype=float)
    for track in d["track_strategy"].dropna().unique():
        col = f"{track}_bin"
        if col not in d.columns:
            continue
        mask = d["track_strategy"] == track
        bin_size.loc[mask] = pd.to_numeric(d.loc[mask, col], errors="coerce")
    return bin_size


def track_metric_rankings_by_bin(
    df: pd.DataFrame,
    weight_basis: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if "track_strategy" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["bin_size"] = _resolve_bin_sizes(d)
    d = d[d["bin_size"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for metric, spec in METRICS.items():
        is_correct_col = spec["is_correct"]
        best_margin_col = spec["best_margin"]
        best_value_col = spec["best_value"]
        subset = d[
            ["bin_size", "track_strategy", is_correct_col, best_margin_col, best_value_col]
        ].copy()
        subset["is_correct"] = _coerce_bool(subset[is_correct_col]).fillna(False)
        subset["margin"] = pd.to_numeric(subset[best_margin_col], errors="coerce")
        if weight_basis == "metric_abs":
            weights = _standout_weights(subset["margin"], subset[best_value_col])
        elif weight_basis == "n_bins_total":
            weights = pd.to_numeric(d.get("n_bins_total", np.nan), errors="coerce")
        else:
            weights = pd.Series(1.0, index=d.index)
        subset["weight"] = weights.fillna(0.0).to_numpy()
        for (bin_size, track_strategy), sub in subset.groupby(["bin_size", "track_strategy"], dropna=False):
            n_runs = int(len(sub))
            n_wins = int(sub["is_correct"].sum())
            win_rate = float(n_wins / n_runs) if n_runs else float("nan")
            margins = sub.loc[sub["is_correct"], "margin"].dropna()
            if margins.empty:
                continue
            w = sub.loc[margins.index, "weight"]
            if w.sum() > 0:
                avg_margin = float((margins * w).sum() / w.sum())
            else:
                avg_margin = float(margins.mean())
            rows.append(
                {
                    "bin_size": float(bin_size),
                    "track_strategy": str(track_strategy),
                    "metric": metric,
                    "avg_margin": avg_margin,
                    "n_runs": n_runs,
                    "win_rate": win_rate,
                }
            )
    if not rows:
        return pd.DataFrame()

    drows = pd.DataFrame(rows)
    summaries = []
    for bin_size, sub in drows.groupby("bin_size"):
        ranked = sub.sort_values(["win_rate", "avg_margin"], ascending=[False, False]).reset_index(
            drop=True
        )
        best_margin = float(ranked.iloc[0]["avg_margin"])
        second_margin = float(ranked.iloc[1]["avg_margin"]) if len(ranked) > 1 else float("nan")
        beat_second = best_margin - second_margin if np.isfinite(second_margin) else float("nan")
        ranked["rank"] = ranked.index + 1
        ranked["beat_second"] = np.where(
            ranked["rank"] == 1,
            beat_second,
            float("nan"),
        )
        ranked = ranked.head(max(top_n, 1))
        ranked["bin_size"] = float(bin_size)
        summaries.append(ranked)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True).sort_values(["bin_size", "rank"])


def track_metric_rankings_by_tumour(
    df: pd.DataFrame,
    weight_basis: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if (
        "selected_tumour_types" not in df.columns
        or "track_strategy" not in df.columns
        or "correct_celltype_canon" not in df.columns
    ):
        return pd.DataFrame()
    d = df.copy()
    d["bin_size"] = _resolve_bin_sizes(d)
    d = d[d["bin_size"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    if "mutations_post_downsample" in d.columns:
        d["mutation_burden"] = pd.to_numeric(d["mutations_post_downsample"], errors="coerce")
    else:
        d["mutation_burden"] = pd.to_numeric(d.get("n_mutations_total", np.nan), errors="coerce")
    d["correct_celltype"] = d["correct_celltype_canon"].astype(str).str.strip().str.lower()
    d["tumour_type"] = (
        d["selected_tumour_types"]
        .fillna("")
        .astype(str)
        .str.split(",")
    )
    d = d.explode("tumour_type")
    d["tumour_type"] = d["tumour_type"].astype(str).str.strip()
    d = d[d["tumour_type"] != ""].copy()
    if d.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for metric, spec in METRICS.items():
        is_correct_col = spec["is_correct"]
        best_margin_col = spec["best_margin"]
        best_value_col = spec["best_value"]
        subset = d[
            [
                "correct_celltype",
                "tumour_type",
                "bin_size",
                "track_strategy",
                "mutation_burden",
                is_correct_col,
                best_margin_col,
                best_value_col,
            ]
        ].copy()
        subset["is_correct"] = _coerce_bool(subset[is_correct_col]).fillna(False)
        subset["margin"] = pd.to_numeric(subset[best_margin_col], errors="coerce")
        if weight_basis == "metric_abs":
            weights = _standout_weights(subset["margin"], subset[best_value_col])
        elif weight_basis == "n_bins_total":
            weights = pd.to_numeric(d.get("n_bins_total", np.nan), errors="coerce")
        else:
            weights = pd.Series(1.0, index=d.index)
        subset["weight"] = weights.fillna(0.0).to_numpy()
        for (correct_celltype, tumour_type, track_strategy, bin_size), sub in subset.groupby(
            ["correct_celltype", "tumour_type", "track_strategy", "bin_size"], dropna=False
        ):
            n_runs = int(len(sub))
            n_wins = int(sub["is_correct"].sum())
            win_rate = float(n_wins / n_runs) if n_runs else float("nan")
            margins = sub["margin"].dropna()
            if margins.empty:
                continue
            avg_burden = float(sub["mutation_burden"].dropna().mean())
            w = sub.loc[margins.index, "weight"]
            if w.sum() > 0:
                avg_margin = float((margins * w).sum() / w.sum())
            else:
                avg_margin = float(margins.mean())
            rows.append(
                {
                    "correct_celltype": str(correct_celltype),
                    "tumour_type": str(tumour_type),
                    "bin_size": float(bin_size),
                    "track_strategy": str(track_strategy),
                    "metric": metric,
                    "avg_margin": avg_margin,
                    "n_runs": n_runs,
                    "win_rate": win_rate,
                    "avg_burden": avg_burden,
                }
            )
    if not rows:
        return pd.DataFrame()

    drows = pd.DataFrame(rows)
    summaries = []
    for (correct_celltype, tumour_type), sub in drows.groupby(
        ["correct_celltype", "tumour_type"]
    ):
        ranked = sub.sort_values(["win_rate", "avg_margin"], ascending=[False, False]).reset_index(
            drop=True
        )
        ranked["rank"] = ranked.index + 1
        ranked = ranked.head(max(top_n, 1))
        ranked["correct_celltype"] = correct_celltype
        ranked["tumour_type"] = tumour_type
        summaries.append(ranked)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["correct_celltype", "tumour_type", "rank"]
    )


def metric_margin_summary(df: pd.DataFrame, weight_basis: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    n_runs = int(len(df))
    for metric, spec in METRICS.items():
        is_correct_col = spec["is_correct"]
        best_margin = spec["best_margin"]
        best_value = spec["best_value"]
        if weight_basis == "metric_abs":
            weights = _standout_weights(df[best_margin], df[best_value])
        elif weight_basis == "n_bins_total":
            weights = pd.to_numeric(df.get("n_bins_total", np.nan), errors="coerce")
        else:
            weights = pd.Series(1.0, index=df.index)
        weights = weights.fillna(0.0)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=df.index)
        is_correct = _coerce_bool(df[is_correct_col]).fillna(False)
        margins = pd.to_numeric(df[best_margin], errors="coerce")
        valid = is_correct & margins.notna()
        if not valid.any():
            avg_weighted = float("nan")
            avg_unweighted = float("nan")
            n_correct = 0
        else:
            w = weights[valid]
            avg_weighted = float((margins[valid] * w).sum() / w.sum()) if w.sum() else float("nan")
            avg_unweighted = float(margins[valid].mean())
            n_correct = int(is_correct.sum())
        accuracy = float(n_correct / n_runs) if n_runs else float("nan")
        rows.append(
            {
                "metric": metric,
                "n_runs": n_runs,
                "n_correct": n_correct,
                "accuracy": accuracy,
                "avg_margin_weighted": avg_weighted,
                "avg_margin_unweighted": avg_unweighted,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Results dashboard")
_style()
st.title("Mutation vs DNase results explorer")
st.caption("Compare metrics, margins, and accuracy for each correct cell type.")

config = _load_config()
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
    path_input = str(experiments[experiment_name])
    st.session_state["path_input"] = path_input

    weight_basis = "metric_abs"

path = Path(path_input)
if not path.is_absolute():
    path = ROOT / path
if not path.exists():
    st.error(f"File not found: {path}")
    st.stop()
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
_save_config(
    {
        "path_input": loaded_path,
    }
)

total_runs = len(df)
sample_modes = (
    ", ".join(sorted(df["sample_mode"].dropna().astype(str).unique()))
    if "sample_mode" in df.columns
    else "n/a"
)
bin_sizes = sorted(_resolve_bin_sizes(df).dropna().unique().tolist())
bin_label = ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in bin_sizes) or "none"
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Number of runs</div>
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
            <div class='stat-meta'>Mode: {sample_modes}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    strategies = sorted(df["track_strategy"].dropna().unique().tolist())
    strategy_list = ", ".join(strategies) if strategies else "none"
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Track strategies</div>
            <div class='stat-value'>{len(strategies)}</div>
            <div class='stat-meta'>{strategy_list}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_d:
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-title'>Bin sizes</div>
            <div class='stat-value'>{len(bin_sizes)}</div>
            <div class='stat-meta'>{bin_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                "Accuracy: 100%"
                if np.isfinite(win_rate) and abs(win_rate - 1.0) < 1e-12
                else f"Accuracy: {win_rate * 100:.1f}%"
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
    celltype_map = _load_dnase_celltype_map()
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
                "Accuracy: 100%"
                if np.isfinite(win_rate) and abs(win_rate - 1.0) < 1e-12
                else f"Accuracy: {win_rate * 100:.1f}%"
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
