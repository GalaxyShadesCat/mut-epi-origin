import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dnase_map import DEFAULT_MAP_PATH, DnaseCellTypeMap

ASSETS_DIR = Path(__file__).resolve().with_name("assets") / "results_dashboard"
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
            state_track_choice = track_choice
            state_metric_choice_label = metric_choice_label
            state_bin_choice = bin_choice
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
            if state_df.empty:
                st.info("Hepatocyte state summary unavailable: no runs after track filtering.")
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
                metrics_to_use = None
                if state_metric_choice is not None:
                    metrics_to_use = [state_metric_choice]
                state_rankings = track_metric_rankings_by_state_prediction(
                    rankings_source,
                    state_by_celltype,
                    weight_basis,
                    top_n=5,
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
