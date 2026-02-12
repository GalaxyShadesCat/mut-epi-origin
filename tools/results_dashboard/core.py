"""Core data, mapping, and ranking utilities for the results dashboard.

This module keeps non-UI logic separate so the Streamlit page code remains
focused on layout and interaction.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st

from scripts.dnase_map import DEFAULT_MAP_PATH, DnaseCellTypeMap

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(__file__).resolve().with_name("assets")
CONFIG_PATH = ASSETS_DIR / "config.json"
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


def _resolve_celltype_map_source(
    experiment_dir: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[str], str]:
    if experiment_dir is not None:
        candidates = [
            ("atac", experiment_dir / "celltype_atac_map.json"),
            ("dnase", experiment_dir / "celltype_dnase_map.json"),
        ]
        for kind, candidate in candidates:
            if candidate.exists():
                return candidate, kind, "experiment"
    default_path = ROOT / DEFAULT_MAP_PATH
    if default_path.exists():
        return default_path, "dnase", "project"
    return None, None, "missing"


def _resolve_atac_map_path(experiment_dir: Optional[Path]) -> Optional[Path]:
    if experiment_dir is not None:
        candidate = experiment_dir / "celltype_atac_map.json"
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def _load_dnase_celltype_map(experiment_dir: Optional[Path] = None) -> Optional[DnaseCellTypeMap]:
    path, kind, scope = _resolve_celltype_map_source(experiment_dir)
    if path is None or kind is None:
        return None
    try:
        if kind == "atac":
            return DnaseCellTypeMap.from_json(path, track_key="atac_path")
        if scope == "project":
            return DnaseCellTypeMap.from_json(path, project_root=ROOT)
        return DnaseCellTypeMap.from_json(path)
    except (OSError, ValueError, KeyError):
        return None


@st.cache_data(show_spinner=False)
def _load_atac_celltype_map(experiment_dir: Optional[Path] = None) -> Optional[DnaseCellTypeMap]:
    path = _resolve_atac_map_path(experiment_dir)
    if path is None:
        return None
    try:
        return DnaseCellTypeMap.from_json(path, track_key="atac_path")
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
                label = alias
            except KeyError:
                label = celltype
        labels.append(label)
    return ", ".join(labels)


def _norm_celltype(value: str) -> str:
    return str(value).strip().lower()


def _infer_state_from_name(name: str) -> str:
    match = re.search(r"\(([^)]+)\)", str(name))
    if not match:
        return ""
    return match.group(1).strip()


def _hepatocyte_state_map(celltype_map: DnaseCellTypeMap) -> Dict[str, str]:
    state_by_celltype: Dict[str, str] = {}
    for entry in celltype_map.entries():
        key_lower = _norm_celltype(entry.key)
        name_lower = _norm_celltype(entry.name)
        if "hepatocyte" not in key_lower and "hepatocyte" not in name_lower:
            continue
        state_value = getattr(entry, "state", "")
        state = state_value.strip() if state_value else _infer_state_from_name(entry.name)
        if not state:
            continue
        aliases = list(entry.aliases) if entry.aliases else []
        for alias in [entry.key, entry.name, *aliases]:
            norm = _norm_celltype(alias)
            if norm:
                state_by_celltype[norm] = state
    return state_by_celltype


def _canonical_state(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper == "AH":
        return "AH"
    if upper == "AC":
        return "AC"
    if upper == "NORMAL":
        return "Normal"
    return raw


def _state_label(state: str) -> str:
    canonical = _canonical_state(state)
    if canonical == "AH":
        return "Alcoholic hepatitis (AH)"
    if canonical == "AC":
        return "Alcohol-associated cirrhosis (AC)"
    if canonical == "Normal":
        return "Normal"
    return canonical


def _norm_tumour_label(value: str) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


def _format_path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _collect_tumours_from_results(df: pd.DataFrame) -> Set[str]:
    if "selected_tumour_types" not in df.columns:
        return set()
    tumours: Set[str] = set()
    for raw in df["selected_tumour_types"].dropna().astype(str):
        for tumour in raw.split(","):
            norm = _norm_tumour_label(tumour)
            if norm:
                tumours.add(norm)
    return tumours


def _build_tumour_options(
        df: pd.DataFrame,
        celltype_map: Optional[DnaseCellTypeMap],
) -> tuple[List[str], Dict[str, str]]:
    tumour_to_celltypes: Dict[str, Set[str]] = {}
    if celltype_map is not None:
        for entry in celltype_map.entries():
            for tumour in entry.tumour_types:
                norm = _norm_tumour_label(tumour)
                if not norm:
                    continue
                tumour_to_celltypes.setdefault(norm, set()).add(entry.key)

    tumours_in_data = _collect_tumours_from_results(df)
    if tumours_in_data:
        tumour_to_celltypes = {
            tumour: tumour_to_celltypes.get(tumour, set()) for tumour in sorted(tumours_in_data)
        }
    elif not tumour_to_celltypes:
        return ["All"], {"All": "All"}

    options = ["All"]
    label_to_tumour = {"All": "All"}
    for tumour in sorted(tumour_to_celltypes):
        celltypes = sorted(tumour_to_celltypes[tumour])
        if celltypes:
            if celltype_map is not None:
                celltype_label = _format_celltype_label(celltypes, celltype_map)
            else:
                celltype_label = ", ".join(celltypes)
            label = f"{tumour}: {celltype_label}"
        else:
            label = tumour
        options.append(label)
        label_to_tumour[label] = tumour
    return options, label_to_tumour


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


def _format_pct(value: float, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "n/a"
    scaled = round(value * 100, decimals)
    if abs(scaled - round(scaled)) < 1e-9:
        return f"{int(round(scaled))}%"
    return f"{scaled:.{decimals}f}%"


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


def _resolve_mutation_burden(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "mutations_post_downsample" in df.columns:
        return (
            pd.to_numeric(df["mutations_post_downsample"], errors="coerce"),
            "mutations_post_downsample",
        )
    if "n_mutations_total" in df.columns:
        return (
            pd.to_numeric(df["n_mutations_total"], errors="coerce"),
            "n_mutations_total",
        )
    return pd.Series([np.nan] * len(df), index=df.index), "n/a"


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


def track_metric_rankings_by_state(
        df: pd.DataFrame,
        state_by_celltype: Dict[str, str],
        weight_basis: str,
        top_n: int = 5,
) -> pd.DataFrame:
    if (
            "track_strategy" not in df.columns
            or "correct_celltype_canon" not in df.columns
    ):
        return pd.DataFrame()
    if not state_by_celltype:
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
    d["state"] = d["correct_celltype"].map(state_by_celltype)
    d = d[d["state"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for metric, spec in METRICS.items():
        is_correct_col = spec["is_correct"]
        best_margin_col = spec["best_margin"]
        best_value_col = spec["best_value"]
        subset = d[
            [
                "state",
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
        for (state, track_strategy, bin_size), sub in subset.groupby(
                ["state", "track_strategy", "bin_size"], dropna=False
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
                    "state": str(state),
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
    for state, sub in drows.groupby("state"):
        ranked = sub.sort_values(["win_rate", "avg_margin"], ascending=[False, False]).reset_index(
            drop=True
        )
        ranked["rank"] = ranked.index + 1
        ranked = ranked.head(max(top_n, 1))
        ranked["state"] = state
        summaries.append(ranked)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["state", "rank"]
    )


def track_metric_rankings_by_state_prediction(
        df: pd.DataFrame,
        state_by_celltype: Dict[str, str],
        weight_basis: str,
        top_n: int = 5,
        metrics_to_use: Optional[List[str]] = None,
) -> pd.DataFrame:
    if "track_strategy" not in df.columns:
        return pd.DataFrame()
    if not state_by_celltype:
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

    rows: List[Dict[str, object]] = []
    metric_items = METRICS.items()
    if metrics_to_use:
        metric_items = [(m, METRICS[m]) for m in metrics_to_use if m in METRICS]
    for metric, spec in metric_items:
        best_cell_col = spec["best_cell"]
        best_margin_col = spec["best_margin"]
        best_value_col = spec["best_value"]
        if best_cell_col not in d.columns:
            continue
        subset = d[
            [
                "bin_size",
                "track_strategy",
                "mutation_burden",
                best_cell_col,
                best_margin_col,
                best_value_col,
            ]
        ].copy()
        subset["pred_celltype"] = (
            subset[best_cell_col].astype(str).str.strip().str.lower()
        )
        subset["state"] = subset["pred_celltype"].map(state_by_celltype).map(_canonical_state)
        subset = subset[subset["state"].notna()].copy()
        if subset.empty:
            continue
        subset["margin"] = pd.to_numeric(subset[best_margin_col], errors="coerce")
        if weight_basis == "metric_abs":
            weights = _standout_weights(subset["margin"], subset[best_value_col])
        elif weight_basis == "n_bins_total":
            weights = pd.to_numeric(d.get("n_bins_total", np.nan), errors="coerce")
        else:
            weights = pd.Series(1.0, index=d.index)
        subset["weight"] = weights.fillna(0.0).to_numpy()
        for (state, track_strategy, bin_size), sub in subset.groupby(
                ["state", "track_strategy", "bin_size"], dropna=False
        ):
            n_runs = int(len(sub))
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
                    "state": str(state),
                    "bin_size": float(bin_size),
                    "track_strategy": str(track_strategy),
                    "metric": metric,
                    "avg_margin": avg_margin,
                    "n_runs": n_runs,
                    "avg_burden": avg_burden,
                }
            )
    if not rows:
        return pd.DataFrame()

    drows = pd.DataFrame(rows)
    summaries = []
    for state, sub in drows.groupby("state"):
        ranked = sub.sort_values(["n_runs", "avg_margin"], ascending=[False, False]).reset_index(
            drop=True
        )
        ranked["rank"] = ranked.index + 1
        ranked = ranked.head(max(top_n, 1))
        ranked["state"] = state
        summaries.append(ranked)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["state", "rank"]
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
