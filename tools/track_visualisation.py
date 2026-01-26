from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pyBigWig

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().with_name("assets") / "track_visualisation"
CONFIG_PATH = ASSETS_DIR / "track_visualisation_config.json"
CONFIGS_PATH = ASSETS_DIR / "track_visualisation_configs.json"
RESULTS_DIR = ASSETS_DIR / "saved_results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contigs import canonical_primary_list, canonicalise_contig, ContigResolver
from scripts.genome_bins import build_bins, load_fai
from scripts.tracks import (
    mutations_to_bin_counts,
    track_counts_gauss,
    track_inv_dist_gauss,
    track_exp_decay,
    track_exp_decay_adaptive,
)
from scripts.covariates import (
    gc_fraction_per_bin,
    cpg_frequency_per_bin,
    trinuc_frequency_per_bin,
    bigwig_mean_per_bin,
)
from scripts.grid_search.metrics import rf_residualise
from scripts.scores import compute_local_scores

TRACK_DESCRIPTIONS = {
    "counts_raw": "Raw mutation counts per bin (step track).",
    "counts_gauss": "Binned counts smoothed with a Gaussian kernel (in bin space).",
    "inv_dist_gauss": "Inverse distance to nearest mutation at bin centres, then Gaussian-smoothed.",
    "exp_decay": "Sum of exponential decays from all mutations (bp space).",
    "exp_decay_adaptive": "Exponential decay with bandwidth set by k-nearest mutations (bp space).",
}


def sigma_to_bins(sigma: float, bin_size: int, units: str) -> float:
    if units == "bins":
        return float(sigma)
    if units == "bp":
        return float(sigma) / float(bin_size)
    raise ValueError(f"Unsupported sigma_units: {units}. Use 'bins' or 'bp'.")


def pearsonr_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask] - np.mean(x[mask])
    yy = y[mask] - np.mean(y[mask])
    denom = np.sqrt((xx ** 2).sum()) * np.sqrt((yy ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((xx * yy).sum() / denom)


def spearmanr_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yy = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return pearsonr_nan(xx, yy)


def linear_residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return y.copy()
    resid = np.full(len(y), np.nan, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 5:
        return resid
    yy = y[mask]
    XX = X[mask]
    A = np.column_stack([np.ones(len(yy)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid[mask] = yy - A @ beta
    return resid


@st.cache_data(show_spinner=False)
def load_mutations_bed(path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["chrom", "start"],
        dtype={"chrom": str, "start": int},
    )
    df["canonical"] = df["chrom"].map(canonicalise_contig)
    df = df.dropna(subset=["canonical"])
    grouped = {}
    for chrom, sub in df.groupby("canonical", sort=False):
        grouped[str(chrom)] = np.sort(sub["start"].to_numpy(dtype=int))
    return grouped


@st.cache_data(show_spinner=False)
def load_fai_lengths(path: str) -> dict[str, int]:
    fai = load_fai(path)
    contigs = fai["chrom"].astype(str).tolist()
    resolver = ContigResolver(fasta_contigs=contigs, bigwig_contigs=None)
    length_map = dict(zip(fai["chrom"].astype(str), fai["length"].astype(int)))

    canonical_lengths: dict[str, int] = {}
    for canonical in canonical_primary_list():
        if resolver.has_canonical_in_fasta(canonical):
            resolved = resolver.resolve_for_fasta(canonical)
            canonical_lengths[canonical] = int(length_map[resolved])
    return canonical_lengths


@st.cache_data(show_spinner=False)
def build_covariate_matrix(
    covariates: tuple[str, ...],
    fasta_path: str,
    chrom: str,
    chrom_length: int,
    bin_size: int,
    timing_bigwig: str | None,
) -> np.ndarray:
    edges, _ = build_bins(chrom_length, bin_size)
    cols: dict[str, np.ndarray] = {}
    if "gc" in covariates:
        cols["gc_fraction"] = gc_fraction_per_bin(fasta_path, chrom, edges)
    if "cpg" in covariates:
        cols["cpg_per_bp"] = cpg_frequency_per_bin(fasta_path, chrom, edges)
    if "timing" in covariates:
        if not timing_bigwig:
            raise ValueError("timing covariate requested but timing bigWig is missing.")
        cols["timing_mean"] = bigwig_mean_per_bin(timing_bigwig, chrom, edges)
    if "trinuc" in covariates:
        tri = trinuc_frequency_per_bin(fasta_path, chrom, edges)
        for k, v in tri.items():
            cols[f"tri_{k}"] = v
    if not cols:
        return np.zeros((len(edges) - 1, 0), dtype=float)
    cov_df = pd.DataFrame(cols)
    return cov_df.to_numpy(dtype=float)


@st.cache_data(show_spinner=False)
def load_dnase_means(
    bigwig_path: str,
    chrom: str,
    chrom_length: int,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges, centres = build_bins(chrom_length, bin_size)
    with pyBigWig.open(str(bigwig_path)) as bw:
        resolver = ContigResolver(fasta_contigs=None, bigwig_contigs=list(bw.chroms().keys()))
        resolved = resolver.resolve_for_bigwig(chrom)
        means = bw.stats(resolved, 0, chrom_length, nBins=len(centres), type="mean")
    means_arr = np.array([float(x) if x is not None else np.nan for x in means], dtype=float)
    return centres, means_arr


def compute_track(
    *,
    strategy: str,
    mut_positions: np.ndarray,
    chrom_length: int,
    bin_size: int,
    sigma_units: str,
    sigma: float | None,
    max_distance_bp: int | None,
    eps: float | None,
    decay_bp: float | None,
    exp_max_distance_bp: int | None,
    adaptive_k: int | None,
    adaptive_min_bandwidth_bp: float | None,
    adaptive_max_distance_bp: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    edges, centres = build_bins(chrom_length, bin_size)

    if strategy == "counts_raw":
        track = mutations_to_bin_counts(mut_positions, edges)
    elif strategy == "counts_gauss":
        if sigma is None:
            raise ValueError("counts_gauss requires sigma")
        sigma_bins = sigma_to_bins(float(sigma), int(bin_size), sigma_units)
        track = track_counts_gauss(mut_positions, edges, sigma_bins)
    elif strategy == "inv_dist_gauss":
        if sigma is None or max_distance_bp is None or eps is None:
            raise ValueError("inv_dist_gauss requires sigma, max_distance_bp, and eps")
        sigma_bins = sigma_to_bins(float(sigma), int(bin_size), sigma_units)
        track = track_inv_dist_gauss(
            mut_positions,
            centres,
            sigma_bins=sigma_bins,
            max_distance_bp=int(max_distance_bp),
            eps=float(eps),
        )
    elif strategy == "exp_decay":
        if decay_bp is None or exp_max_distance_bp is None:
            raise ValueError("exp_decay requires decay_bp and exp_max_distance_bp")
        track = track_exp_decay(
            mut_positions,
            centres,
            decay_bp=float(decay_bp),
            max_distance_bp=int(exp_max_distance_bp),
        )
    elif strategy == "exp_decay_adaptive":
        if adaptive_k is None or adaptive_min_bandwidth_bp is None or adaptive_max_distance_bp is None:
            raise ValueError("exp_decay_adaptive requires adaptive_k and adaptive_*_bp parameters")
        track = track_exp_decay_adaptive(
            mut_positions,
            centres,
            k_nearest=int(adaptive_k),
            min_bandwidth_bp=float(adaptive_min_bandwidth_bp),
            max_distance_bp=int(adaptive_max_distance_bp),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return centres, track


def init_state() -> None:
    defaults = {
        "bed_path": str(ROOT / "data/processed/mela_EXTERN-MELA-20140514-012.bed"),
        "fai_path": str(ROOT / "data/raw/reference/GRCh37.fa.fai"),
        "fasta_path": str(ROOT / "data/raw/reference/GRCh37.fa"),
        "chrom": "chr1",
        "dnase_tracks": [
            {
                "name": "mela",
                "path": str(ROOT / "data/raw/DNase-seq/mela_ENCFF285GEW.bigWig"),
                "overlay": True,
            },
            {
                "name": "fibr",
                "path": str(ROOT / "data/raw/DNase-seq/fibr_ENCFF355OPU.bigWig"),
                "overlay": False,
            },
            {
                "name": "kera",
                "path": str(ROOT / "data/raw/DNase-seq/kera_ENCFF597YXQ.bigWig"),
                "overlay": False,
            },
        ],
        "timing_bigwig": str(ROOT / "data/raw/timing/repliSeq_SknshWaveSignalRep1.bigWig"),
        "covariates": [],
        "window_all": (0, 20_000_000),
        "window_all_slider": (0, 20_000_000),
        "bin_size_counts_raw": 1_000_000,
        "bin_size_counts_gauss": 1_000_000,
        "sigma_units_counts": "bins",
        "sigma_counts": 1.0,
        "bin_size_inv": 1_000_000,
        "sigma_units_inv": "bins",
        "sigma_inv": 0.5,
        "max_distance_bp": 1_000_000,
        "eps": 1.0,
        "bin_size_exp": 1_000_000,
        "decay_bp": 200_000,
        "exp_max_distance_bp": 1_000_000,
        "bin_size_adaptive": 1_000_000,
        "adaptive_k": 5,
        "adaptive_min_bandwidth_bp": 50_000,
        "adaptive_max_distance_bp": 1_000_000,
        "pearson_local_score_w": 1,
        "pearson_score_smoothing": "none",
        "pearson_score_smooth_param": 1.0,
        "pearson_score_transform": "none",
        "pearson_score_zscore": False,
        "pearson_score_weight_shape": 0.7,
        "pearson_score_weight_slope": 0.3,
        "spearman_local_score_w": 5,
        "spearman_score_smoothing": "none",
        "spearman_score_smooth_param": 1.0,
        "spearman_score_transform": "none",
        "spearman_score_zscore": False,
        "spearman_score_weight_shape": 0.7,
        "spearman_score_weight_slope": 0.3,
        "highlight_mode": "pearson_local_score",
    }
    config = load_config()
    if "pearson_local_score_w" not in config and "anti_corr_w" in config:
        config = dict(config)
        config["pearson_local_score_w"] = config["anti_corr_w"]
    for key, value in defaults.items():
        st.session_state.setdefault(key, config.get(key, value))


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_config(config: dict) -> None:
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:
        return


def load_named_configs() -> dict[str, dict]:
    if not CONFIGS_PATH.exists():
        return {}
    try:
        with CONFIGS_PATH.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_named_configs(configs: dict[str, dict]) -> None:
    try:
        with CONFIGS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(configs, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:
        return


def slugify_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    safe = safe.strip("_")
    return safe or "config"


def unique_results_path(configs: dict[str, dict], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify_name(name)
    existing_paths = {str(entry.get("results_path", "")) for entry in configs.values()}
    candidate = RESULTS_DIR / f"{slug}.npz"
    if str(candidate) not in existing_paths and not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = RESULTS_DIR / f"{slug}_{idx}.npz"
        if str(candidate) not in existing_paths and not candidate.exists():
            return candidate
        idx += 1


def params_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def params_match(current: dict, saved: dict) -> bool:
    return params_signature(current) == params_signature(saved)


def apply_config_params(params: dict) -> None:
    if "pearson_local_score_w" not in params and "anti_corr_w" in params:
        params = dict(params)
        params["pearson_local_score_w"] = params["anti_corr_w"]
    for key, value in params.items():
        st.session_state[key] = value


def save_results_bundle(
    results_path: Path,
    *,
    results: dict,
    dnase_tracks: list[dict[str, object]],
) -> None:
    data: dict[str, np.ndarray] = {
        "centres_raw": results["centres_raw"],
        "track_raw": results["track_raw"],
        "centres_gauss": results["centres_gauss"],
        "track_gauss": results["track_gauss"],
        "centres_inv": results["centres_inv"],
        "track_inv": results["track_inv"],
        "centres_exp": results["centres_exp"],
        "track_exp": results["track_exp"],
        "centres_adaptive": results["centres_adaptive"],
        "track_adaptive": results["track_adaptive"],
        "X_raw": results.get("X_raw", np.zeros((0, 0), dtype=float)),
        "X_gauss": results.get("X_gauss", np.zeros((0, 0), dtype=float)),
        "X_inv": results.get("X_inv", np.zeros((0, 0), dtype=float)),
        "X_exp": results.get("X_exp", np.zeros((0, 0), dtype=float)),
        "X_adaptive": results.get("X_adaptive", np.zeros((0, 0), dtype=float)),
    }
    for idx, dnase in enumerate(dnase_tracks):
        data[f"dnase_{idx}_raw"] = dnase["raw"]
        data[f"dnase_{idx}_gauss"] = dnase["gauss"]
        data[f"dnase_{idx}_inv"] = dnase["inv"]
        data[f"dnase_{idx}_exp"] = dnase["exp"]
        data[f"dnase_{idx}_adaptive"] = dnase["adaptive"]
    np.savez_compressed(results_path, **data)


def load_results_bundle(entry: dict) -> dict | None:
    results_path = Path(entry.get("results_path", ""))
    if not results_path.exists():
        return None
    with np.load(results_path) as data:
        keys = set(data.files)
        results = {
            "centres_raw": data["centres_raw"],
            "track_raw": data["track_raw"],
            "centres_gauss": data["centres_gauss"],
            "track_gauss": data["track_gauss"],
            "centres_inv": data["centres_inv"],
            "track_inv": data["track_inv"],
            "centres_exp": data["centres_exp"],
            "track_exp": data["track_exp"],
            "centres_adaptive": data["centres_adaptive"],
            "track_adaptive": data["track_adaptive"],
            "X_raw": data["X_raw"] if "X_raw" in keys else np.zeros((0, 0), dtype=float),
            "X_gauss": data["X_gauss"] if "X_gauss" in keys else np.zeros((0, 0), dtype=float),
            "X_inv": data["X_inv"] if "X_inv" in keys else np.zeros((0, 0), dtype=float),
            "X_exp": data["X_exp"] if "X_exp" in keys else np.zeros((0, 0), dtype=float),
            "X_adaptive": data["X_adaptive"] if "X_adaptive" in keys else np.zeros((0, 0), dtype=float),
        }
        dnase_tracks = entry.get("params", {}).get("dnase_tracks", [])
        dnase_tracks_data = []
        for idx, meta in enumerate(dnase_tracks):
            dnase_tracks_data.append(
                {
                    "name": meta.get("name", f"dnase_{idx + 1}"),
                    "path": meta.get("path", ""),
                    "overlay": bool(meta.get("overlay", False)),
                    "raw": data[f"dnase_{idx}_raw"],
                    "gauss": data[f"dnase_{idx}_gauss"],
                    "inv": data[f"dnase_{idx}_inv"],
                    "exp": data[f"dnase_{idx}_exp"],
                    "adaptive": data[f"dnase_{idx}_adaptive"],
                }
            )
        results["dnase_tracks_data"] = dnase_tracks_data
    return results


def save_named_config(
    configs: dict[str, dict],
    *,
    name: str,
    params: dict,
    results: dict,
) -> dict[str, dict]:
    entry = configs.get(name, {})
    results_path_str = entry.get("results_path", "")
    results_path = Path(results_path_str) if results_path_str else unique_results_path(configs, name)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_bundle(results_path, results=results, dnase_tracks=results["dnase_tracks_data"])
    configs[name] = {
        "params": params,
        "results_path": str(results_path),
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return configs


def main() -> None:
    st.set_page_config(page_title="Track Visualisation", layout="wide")
    st.title("Track Visualisation")
    st.caption("Interactive view of mutation tracks across strategies.")
    init_state()

    with st.sidebar:
        st.header("Configs")
        configs = load_named_configs()
        config_names = sorted(configs.keys())
        def handle_config_select() -> None:
            selected = st.session_state.get("config_select", "None")
            if selected == "None":
                apply_config_params(load_config())
                st.session_state["last_loaded_config"] = None
                return
            if st.session_state.get("last_loaded_config") == selected:
                return
            entry = configs.get(selected)
            if entry:
                apply_config_params(entry.get("params", {}))
                loaded_results = load_results_bundle(entry)
                if loaded_results is not None:
                    st.session_state["saved_results"] = loaded_results
                    st.session_state["saved_results_name"] = selected
                    st.session_state["saved_results_params"] = entry.get("params", {})
                st.session_state["last_loaded_config"] = selected

        selected_config = st.selectbox(
            "Saved configs",
            options=["None"] + config_names,
            key="config_select",
            on_change=handle_config_select,
        )
        config_name_input = st.text_input(
            "Config name",
            key="config_name_input",
            value=selected_config if selected_config != "None" else "",
        )
        save_col, update_col = st.columns(2)
        save_clicked = save_col.button("Save new")
        update_clicked = update_col.button("Update selected", disabled=selected_config == "None")
        if save_clicked:
            name = (config_name_input or "").strip()
            if not name:
                st.warning("Provide a config name before saving.")
            elif name in configs:
                st.warning(f"Config '{name}' already exists. Use Update selected.")
            else:
                st.session_state["pending_config_action"] = {"action": "save_new", "name": name}
        if update_clicked and selected_config != "None":
            name = (config_name_input or "").strip()
            if not name:
                st.warning("Provide a config name before updating.")
            elif name != selected_config and name in configs:
                st.warning(f"Config '{name}' already exists.")
            else:
                st.session_state["pending_config_action"] = {
                    "action": "update",
                    "name": selected_config,
                    "rename_to": name,
                }

        st.header("Inputs")
        bed_path = st.text_input(
            "Mutation BED path",
            key="bed_path",
            help="Path to the BED-like file containing mutation positions (chrom, start).",
        )
        fai_path = st.text_input(
            "FASTA .fai path",
            key="fai_path",
            help="FASTA index used to get chromosome lengths.",
        )
        fasta_path = st.text_input(
            "FASTA path",
            key="fasta_path",
            help="FASTA reference for covariate calculations (GC/CpG/trinuc).",
        )
        st.markdown("DNase-seq tracks")
        dnase_df = pd.DataFrame(st.session_state["dnase_tracks"])
        dnase_df = st.data_editor(
            dnase_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name", help="Label shown in plots and metrics."),
                "path": st.column_config.TextColumn("bigWig path", help="DNase-seq bigWig file."),
                "overlay": st.column_config.CheckboxColumn("Overlay", help="Show on the track plot."),
            },
            key="dnase_tracks_editor",
        )
        st.session_state["dnase_tracks"] = dnase_df.to_dict("records")
        timing_bigwig = st.text_input(
            "Timing bigWig path",
            key="timing_bigwig",
            help="Replication timing bigWig used when 'timing' covariate is selected.",
        )
        covariates = st.multiselect(
            "Covariates for linear covariate adjustment",
            options=["gc", "cpg", "timing", "trinuc"],
            key="covariates",
            help="These are regressed out of both tracks before computing linear adjusted correlations.",
        )

        st.header("Track parameters")

        st.subheader("Correlation display")
        highlight_mode = st.session_state.get("highlight_mode", "pearson_local_score")
        if highlight_mode in {"anti_score", "adj_pearson_r", "local_score"}:
            if highlight_mode == "adj_pearson_r":
                highlight_mode = "pearson_r_linear"
            else:
                highlight_mode = "pearson_local_score"
            st.session_state["highlight_mode"] = highlight_mode
        highlight_options = [
            "pearson_r",
            "pearson_r_linear",
            "spearman_r",
            "spearman_r_linear",
            "pearson_local_score",
            "spearman_local_score",
            "rf_non_linear",
        ]
        highlight_mode = st.selectbox(
            "Highlight best (most negative)",
            options=highlight_options,
            key="highlight_mode",
            format_func=lambda v: {
                "pearson_r": "Pearson r",
                "pearson_r_linear": "Pearson r (linear covariate)",
                "spearman_r": "Spearman r",
                "spearman_r_linear": "Spearman r (linear covariate)",
                "pearson_local_score": "Pearson local score (linear covariate)",
                "spearman_local_score": "Spearman local score (linear covariate)",
                "rf_non_linear": "RF (non-linear covariate)",
            }[v],
            help="Choose which correlation metric to use for highlighting the strongest negative cell type.",
        )
        st.markdown("Pearson local score settings")
        pearson_local_score_w = st.number_input(
            "Pearson local score window half-size (bins)",
            min_value=1,
            max_value=500,
            step=1,
            key="pearson_local_score_w",
            help="Bin half-window size used for Pearson local score computation.",
        )
        pearson_score_smoothing = st.selectbox(
            "Pearson local score smoothing",
            options=["none", "moving_average", "gaussian"],
            key="pearson_score_smoothing",
            help="Optional smoothing applied to both tracks before Pearson local scoring.",
        )
        pearson_smooth_min = 1.0 if pearson_score_smoothing == "moving_average" else 0.05
        pearson_smooth_step = 1.0 if pearson_score_smoothing == "moving_average" else 0.05
        pearson_score_smooth_param = st.number_input(
            "Pearson local score smoothing parameter",
            min_value=pearson_smooth_min,
            step=pearson_smooth_step,
            key="pearson_score_smooth_param",
            help="Window size (moving_average) or sigma (gaussian); ignored when smoothing is none.",
        )
        pearson_score_transform = st.selectbox(
            "Pearson local score transform",
            options=["none", "log1p"],
            key="pearson_score_transform",
            help="Optional transform applied before Pearson local scoring (log1p requires non-negative values).",
        )
        pearson_score_zscore = st.checkbox(
            "Pearson local score z-score",
            key="pearson_score_zscore",
            help="Z-score both tracks before Pearson local scoring.",
        )
        pearson_score_weight_shape = st.number_input(
            "Pearson local score weight: shape",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="pearson_score_weight_shape",
            help="Weight on shape (raw correlation) component.",
        )
        pearson_score_weight_slope = st.number_input(
            "Pearson local score weight: slope",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="pearson_score_weight_slope",
            help="Weight on slope (first-difference correlation) component.",
        )

        st.markdown("Spearman local score settings")
        spearman_local_score_w = st.number_input(
            "Spearman local score window half-size (bins)",
            min_value=1,
            max_value=500,
            step=1,
            key="spearman_local_score_w",
            help="Bin half-window size used for Spearman local score computation.",
        )
        spearman_score_smoothing = st.selectbox(
            "Spearman local score smoothing",
            options=["none", "moving_average", "gaussian"],
            key="spearman_score_smoothing",
            help="Optional smoothing applied to both tracks before Spearman local scoring.",
        )
        spearman_smooth_min = 1.0 if spearman_score_smoothing == "moving_average" else 0.05
        spearman_smooth_step = 1.0 if spearman_score_smoothing == "moving_average" else 0.05
        spearman_score_smooth_param = st.number_input(
            "Spearman local score smoothing parameter",
            min_value=spearman_smooth_min,
            step=spearman_smooth_step,
            key="spearman_score_smooth_param",
            help="Window size (moving_average) or sigma (gaussian); ignored when smoothing is none.",
        )
        spearman_score_transform = st.selectbox(
            "Spearman local score transform",
            options=["none", "log1p"],
            key="spearman_score_transform",
            help="Optional transform applied before Spearman local scoring (log1p requires non-negative values).",
        )
        spearman_score_zscore = st.checkbox(
            "Spearman local score z-score",
            key="spearman_score_zscore",
            help="Z-score both tracks before Spearman local scoring.",
        )
        spearman_score_weight_shape = st.number_input(
            "Spearman local score weight: shape",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="spearman_score_weight_shape",
            help="Weight on shape (raw correlation) component.",
        )
        spearman_score_weight_slope = st.number_input(
            "Spearman local score weight: slope",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="spearman_score_weight_slope",
            help="Weight on slope (first-difference correlation) component.",
        )

        st.subheader("counts_raw", help=TRACK_DESCRIPTIONS["counts_raw"])
        bin_size_counts_raw = st.number_input(
            "counts_raw bin size (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="bin_size_counts_raw",
            help="Bin size used for raw mutation counts; smaller bins give finer detail.",
        )

        st.subheader("counts_gauss", help=TRACK_DESCRIPTIONS["counts_gauss"])
        bin_size_counts_gauss = st.number_input(
            "counts_gauss bin size (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="bin_size_counts_gauss",
            help="Bin size used before Gaussian smoothing of mutation counts.",
        )
        sigma_units_counts = st.selectbox(
            "counts_gauss sigma units",
            options=["bins", "bp"],
            index=0,
            key="sigma_units_counts",
            help="Interpret sigma as bin units or base pairs.",
        )
        sigma_counts = st.number_input(
            "counts_gauss sigma",
            min_value=0.05,
            max_value=50.0,
            step=0.05,
            key="sigma_counts",
            help="Gaussian smoothing width; larger values smooth more.",
        )

        st.subheader("inv_dist_gauss", help=TRACK_DESCRIPTIONS["inv_dist_gauss"])
        bin_size_inv = st.number_input(
            "inv_dist_gauss bin size (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="bin_size_inv",
            help="Bin size used for inverse-distance computation.",
        )
        sigma_units_inv = st.selectbox(
            "inv_dist_gauss sigma units",
            options=["bins", "bp"],
            index=0,
            key="sigma_units_inv",
            help="Interpret sigma as bin units or base pairs.",
        )
        sigma_inv = st.number_input(
            "inv_dist_gauss sigma",
            min_value=0.05,
            max_value=50.0,
            step=0.05,
            key="sigma_inv",
            help="Gaussian smoothing width applied to inverse-distance track.",
        )
        max_distance_bp = st.number_input(
            "inv_dist_gauss max distance (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=10_000,
            key="max_distance_bp",
            help="Distances to nearest mutation are capped at this value.",
        )
        eps = st.number_input(
            "inv_dist_gauss epsilon",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            key="eps",
            help="Small constant added to distance to avoid division by zero.",
        )

        st.subheader("exp_decay", help=TRACK_DESCRIPTIONS["exp_decay"])
        bin_size_exp = st.number_input(
            "exp_decay bin size (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="bin_size_exp",
            help="Bin size used for exponential decay field.",
        )
        decay_bp = st.number_input(
            "exp_decay length (bp)",
            min_value=1_000,
            max_value=10_000_000,
            step=5_000,
            key="decay_bp",
            help="Decay length in bp; larger values smooth more broadly.",
        )
        exp_max_distance_bp = st.number_input(
            "exp_decay max distance (bp)",
            min_value=10_000,
            max_value=10_000_000,
            step=10_000,
            key="exp_max_distance_bp",
            help="Ignore mutations further than this distance from each bin centre.",
        )

        st.subheader("exp_decay_adaptive", help=TRACK_DESCRIPTIONS["exp_decay_adaptive"])
        bin_size_adaptive = st.number_input(
            "exp_decay_adaptive bin size (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="bin_size_adaptive",
            help="Bin size used for adaptive decay field.",
        )
        adaptive_k = st.number_input(
            "exp_decay_adaptive k-nearest",
            min_value=1,
            max_value=50,
            step=1,
            key="adaptive_k",
            help="k-th nearest mutation distance defines local bandwidth.",
        )
        adaptive_min_bandwidth_bp = st.number_input(
            "exp_decay_adaptive min bandwidth (bp)",
            min_value=1_000,
            max_value=5_000_000,
            step=5_000,
            key="adaptive_min_bandwidth_bp",
            help="Lower bound on the adaptive bandwidth in bp.",
        )
        adaptive_max_distance_bp = st.number_input(
            "exp_decay_adaptive max distance (bp)",
            min_value=10_000,
            max_value=10_000_000,
            step=10_000,
            key="adaptive_max_distance_bp",
            help="Ignore mutations further than this distance from each bin centre.",
        )

    if not Path(bed_path).exists():
        st.error(f"Mutation BED not found: {bed_path}")
        return
    if not Path(fai_path).exists():
        st.error(f"FAI not found: {fai_path}")
        return
    if covariates:
        if not fasta_path or not Path(fasta_path).exists():
            st.error("FASTA path is required for GC/CpG/trinuc covariates.")
            return
        if "timing" in covariates and (not timing_bigwig or not Path(timing_bigwig).exists()):
            st.error("Timing bigWig path is required when 'timing' covariate is selected.")
            return
    dnase_tracks: list[dict[str, object]] = []
    for idx, row in enumerate(st.session_state.get("dnase_tracks", []), start=1):
        name = str(row.get("name") or f"dnase_{idx}").strip()
        path = str(row.get("path") or "").strip()
        overlay = bool(row.get("overlay", False))
        if not path:
            continue
        if not Path(path).exists():
            st.warning(f"DNase bigWig not found (skipping): {path}")
            continue
        dnase_tracks.append({"name": name, "path": path, "overlay": overlay})

    with st.spinner("Loading inputs..."):
        mut_by_chrom = load_mutations_bed(bed_path)
        chrom_lengths = load_fai_lengths(fai_path)

    available = [c for c in canonical_primary_list() if c in chrom_lengths]
    if not available:
        st.error("No canonical chromosomes found in the FAI.")
        return

    chrom_options = available + ["all"]
    if st.session_state.get("chrom") not in chrom_options:
        st.session_state["chrom"] = "chr1" if "chr1" in chrom_options else chrom_options[0]
    prev_chrom = st.session_state.get("prev_chrom", st.session_state.get("chrom"))
    chrom = st.selectbox(
        "Chromosome",
        options=chrom_options,
        key="chrom",
        format_func=lambda x: "all chromosomes" if x == "all" else x,
    )
    if chrom == "all":
        chroms_to_use = available
        chrom_length = int(sum(chrom_lengths[c] for c in chroms_to_use))
        mut_positions = None
    else:
        chroms_to_use = [chrom]
        chrom_length = chrom_lengths[chrom]
        mut_positions = mut_by_chrom.get(chrom, np.array([], dtype=int))
    if chrom == "all" and prev_chrom != "all":
        st.session_state["window_all"] = (0, int(chrom_length))
        st.session_state["window_all_slider"] = (0, int(chrom_length))
    st.session_state["prev_chrom"] = chrom

    def bin_count(chrom_len: int, bin_size: int) -> int:
        return int((int(chrom_len) + int(bin_size) - 1) // int(bin_size))

    if chrom == "all":
        max_bins_raw = sum(bin_count(chrom_lengths[c], int(bin_size_counts_raw)) for c in chroms_to_use)
        max_bins_gauss = sum(bin_count(chrom_lengths[c], int(bin_size_counts_gauss)) for c in chroms_to_use)
        max_bins_inv = sum(bin_count(chrom_lengths[c], int(bin_size_inv)) for c in chroms_to_use)
        max_bins_exp = sum(bin_count(chrom_lengths[c], int(bin_size_exp)) for c in chroms_to_use)
        max_bins_adaptive = sum(bin_count(chrom_lengths[c], int(bin_size_adaptive)) for c in chroms_to_use)
    else:
        max_bins_raw = bin_count(int(chrom_length), int(bin_size_counts_raw))
        max_bins_gauss = bin_count(int(chrom_length), int(bin_size_counts_gauss))
        max_bins_inv = bin_count(int(chrom_length), int(bin_size_inv))
        max_bins_exp = bin_count(int(chrom_length), int(bin_size_exp))
        max_bins_adaptive = bin_count(int(chrom_length), int(bin_size_adaptive))
    if max(max_bins_raw, max_bins_gauss, max_bins_inv, max_bins_exp, max_bins_adaptive) > 2_000_000:
        st.warning(
            "One of the selected bin sizes creates more than 2,000,000 bins. "
            "Increase bin sizes for faster updates."
        )
        return

    window_key = "window_all"
    slider_key = "window_all_slider"
    if window_key not in st.session_state:
        st.session_state[window_key] = (0, int(chrom_length))
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[window_key]
    start, end = st.session_state[window_key]
    start = max(0, min(int(start), int(chrom_length)))
    end = max(0, min(int(end), int(chrom_length)))
    if end <= start:
        start, end = 0, int(chrom_length)
    st.session_state[window_key] = (start, end)
    st.session_state[slider_key] = (start, end)
    st.session_state["window_start_input"] = int(start)
    st.session_state["window_end_input"] = int(end)
    slider_step = int(min(bin_size_counts_raw, bin_size_counts_gauss, bin_size_inv, bin_size_exp, bin_size_adaptive))

    def _clamp_window(start_val: int, end_val: int) -> tuple[int, int]:
        def _snap(val: int, *, allow_max: bool) -> int:
            if slider_step <= 0:
                return int(val)
            if allow_max and int(val) >= int(chrom_length):
                return int(chrom_length)
            if allow_max and int(val) >= int(chrom_length) - slider_step:
                return int(chrom_length)
            return int((int(val) // slider_step) * slider_step)

        start_val = max(0, min(int(start_val), int(chrom_length)))
        end_val = max(0, min(int(end_val), int(chrom_length)))
        start_val = _snap(start_val, allow_max=False)
        end_val = _snap(end_val, allow_max=True)
        if end_val <= start_val:
            end_val = min(int(chrom_length), start_val + slider_step)
            if end_val <= start_val:
                start_val = max(0, end_val - slider_step)
        return int(start_val), int(end_val)

    def sync_window_from_inputs() -> None:
        start_val = st.session_state.get("window_start_input", 0)
        end_val = st.session_state.get("window_end_input", chrom_length)
        start_val, end_val = _clamp_window(start_val, end_val)
        st.session_state[window_key] = (start_val, end_val)
        st.session_state[slider_key] = (start_val, end_val)
        st.session_state["window_start_input"] = int(start_val)
        st.session_state["window_end_input"] = int(end_val)

    def sync_window_from_slider() -> None:
        start_val, end_val = st.session_state[slider_key]
        start_val, end_val = _clamp_window(start_val, end_val)
        st.session_state[window_key] = (start_val, end_val)
        st.session_state["window_start_input"] = int(start_val)
        st.session_state["window_end_input"] = int(end_val)

    window_start_input = st.number_input(
        "Window start (bp)",
        min_value=0,
        max_value=int(chrom_length),
        step=slider_step,
        key="window_start_input",
        help="Manually set the start of the window.",
        on_change=sync_window_from_inputs,
    )
    window_end_input = st.number_input(
        "Window end (bp)",
        min_value=0,
        max_value=int(chrom_length),
        step=slider_step,
        key="window_end_input",
        help="Manually set the end of the window.",
        on_change=sync_window_from_inputs,
    )

    window = st.slider(
        "Window (bp)",
        min_value=0,
        max_value=int(chrom_length),
        step=slider_step,
        key=slider_key,
        help="Zoom into a specific genomic interval for all tracks.",
        on_change=sync_window_from_slider,
    )
    window_start, window_end = st.session_state[window_key]

    current_params = {
        "bed_path": bed_path,
        "fai_path": fai_path,
        "fasta_path": fasta_path,
        "chrom": chrom,
        "dnase_tracks": st.session_state.get("dnase_tracks", []),
        "timing_bigwig": timing_bigwig,
        "covariates": list(covariates),
        "window_all": list(st.session_state.get(window_key, (0, int(chrom_length)))),
        "window_all_slider": list(st.session_state.get(slider_key, (0, int(chrom_length)))),
        "bin_size_counts_raw": int(bin_size_counts_raw),
        "bin_size_counts_gauss": int(bin_size_counts_gauss),
        "sigma_units_counts": sigma_units_counts,
        "sigma_counts": float(sigma_counts),
        "bin_size_inv": int(bin_size_inv),
        "sigma_units_inv": sigma_units_inv,
        "sigma_inv": float(sigma_inv),
        "max_distance_bp": int(max_distance_bp),
        "eps": float(eps),
        "bin_size_exp": int(bin_size_exp),
        "decay_bp": int(decay_bp),
        "exp_max_distance_bp": int(exp_max_distance_bp),
        "bin_size_adaptive": int(bin_size_adaptive),
        "adaptive_k": int(adaptive_k),
        "adaptive_min_bandwidth_bp": int(adaptive_min_bandwidth_bp),
        "adaptive_max_distance_bp": int(adaptive_max_distance_bp),
        "pearson_local_score_w": int(pearson_local_score_w),
        "pearson_score_smoothing": pearson_score_smoothing,
        "pearson_score_smooth_param": float(pearson_score_smooth_param),
        "pearson_score_transform": pearson_score_transform,
        "pearson_score_zscore": bool(pearson_score_zscore),
        "pearson_score_weight_shape": float(pearson_score_weight_shape),
        "pearson_score_weight_slope": float(pearson_score_weight_slope),
        "spearman_local_score_w": int(spearman_local_score_w),
        "spearman_score_smoothing": spearman_score_smoothing,
        "spearman_score_smooth_param": float(spearman_score_smooth_param),
        "spearman_score_transform": spearman_score_transform,
        "spearman_score_zscore": bool(spearman_score_zscore),
        "spearman_score_weight_shape": float(spearman_score_weight_shape),
        "spearman_score_weight_slope": float(spearman_score_weight_slope),
        "highlight_mode": highlight_mode,
    }
    save_config(current_params)
    if window_end <= window_start:
        st.error("Window end must be greater than start.")
        return

    def compute_track_multi(
        *,
        strategy: str,
        bin_size: int,
        sigma_units: str,
        sigma: float | None,
        max_distance_bp: int | None,
        eps: float | None,
        decay_bp: float | None,
        exp_max_distance_bp: int | None,
        adaptive_k: int | None,
        adaptive_min_bandwidth_bp: float | None,
        adaptive_max_distance_bp: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        centres_parts = []
        track_parts = []
        offset = 0
        for chrom_name in chroms_to_use:
            chrom_len = chrom_lengths[chrom_name]
            mut_pos = mut_by_chrom.get(chrom_name, np.array([], dtype=int))
            centres, track = compute_track(
                strategy=strategy,
                mut_positions=mut_pos,
                chrom_length=int(chrom_len),
                bin_size=int(bin_size),
                sigma_units=sigma_units,
                sigma=sigma,
                max_distance_bp=max_distance_bp,
                eps=eps,
                decay_bp=decay_bp,
                exp_max_distance_bp=exp_max_distance_bp,
                adaptive_k=adaptive_k,
                adaptive_min_bandwidth_bp=adaptive_min_bandwidth_bp,
                adaptive_max_distance_bp=adaptive_max_distance_bp,
            )
            centres_parts.append(centres + offset)
            track_parts.append(track)
            offset += int(chrom_len)
        return np.concatenate(centres_parts), np.concatenate(track_parts)

    def build_covariate_matrix_multi(covariates: tuple[str, ...], bin_size: int) -> np.ndarray:
        if not covariates:
            total_bins = int(sum(chrom_lengths[c] // int(bin_size) + 1 for c in chroms_to_use))
            return np.zeros((total_bins, 0), dtype=float)
        parts = []
        for chrom_name in chroms_to_use:
            chrom_len = chrom_lengths[chrom_name]
            X_part = build_covariate_matrix(
                covariates, fasta_path, chrom_name, int(chrom_len), int(bin_size), timing_bigwig or None
            )
            parts.append(X_part)
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=float)

    def load_dnase_means_multi(bigwig_path: str, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
        centres_parts = []
        dnase_parts = []
        offset = 0
        for chrom_name in chroms_to_use:
            chrom_len = chrom_lengths[chrom_name]
            centres, means = load_dnase_means(bigwig_path, chrom_name, int(chrom_len), int(bin_size))
            centres_parts.append(centres + offset)
            dnase_parts.append(means)
            offset += int(chrom_len)
        return np.concatenate(centres_parts), np.concatenate(dnase_parts)

    saved_results = st.session_state.get("saved_results")
    saved_params = st.session_state.get("saved_results_params")
    use_saved = saved_results is not None and saved_params is not None and params_match(current_params, saved_params)
    if use_saved:
        centres_raw = saved_results["centres_raw"]
        track_raw = saved_results["track_raw"]
        centres_gauss = saved_results["centres_gauss"]
        track_gauss = saved_results["track_gauss"]
        centres_inv = saved_results["centres_inv"]
        track_inv = saved_results["track_inv"]
        centres_exp = saved_results["centres_exp"]
        track_exp = saved_results["track_exp"]
        centres_adaptive = saved_results["centres_adaptive"]
        track_adaptive = saved_results["track_adaptive"]
        X_raw = saved_results["X_raw"]
        X_gauss = saved_results["X_gauss"]
        X_inv = saved_results["X_inv"]
        X_exp = saved_results["X_exp"]
        X_adaptive = saved_results["X_adaptive"]
        dnase_tracks_data = saved_results["dnase_tracks_data"]
        st.caption("Loaded saved results for this config.")
    else:
        with st.spinner("Computing tracks..."):
            centres_raw, track_raw = compute_track_multi(
                strategy="counts_raw",
                bin_size=int(bin_size_counts_raw),
                sigma_units="bins",
                sigma=None,
                max_distance_bp=None,
                eps=None,
                decay_bp=None,
                exp_max_distance_bp=None,
                adaptive_k=None,
                adaptive_min_bandwidth_bp=None,
                adaptive_max_distance_bp=None,
            )
            centres_gauss, track_gauss = compute_track_multi(
                strategy="counts_gauss",
                bin_size=int(bin_size_counts_gauss),
                sigma_units=str(sigma_units_counts),
                sigma=float(sigma_counts),
                max_distance_bp=None,
                eps=None,
                decay_bp=None,
                exp_max_distance_bp=None,
                adaptive_k=None,
                adaptive_min_bandwidth_bp=None,
                adaptive_max_distance_bp=None,
            )
            centres_inv, track_inv = compute_track_multi(
                strategy="inv_dist_gauss",
                bin_size=int(bin_size_inv),
                sigma_units=str(sigma_units_inv),
                sigma=float(sigma_inv),
                max_distance_bp=int(max_distance_bp),
                eps=float(eps),
                decay_bp=None,
                exp_max_distance_bp=None,
                adaptive_k=None,
                adaptive_min_bandwidth_bp=None,
                adaptive_max_distance_bp=None,
            )
            centres_exp, track_exp = compute_track_multi(
                strategy="exp_decay",
                bin_size=int(bin_size_exp),
                sigma_units="bins",
                sigma=None,
                max_distance_bp=None,
                eps=None,
                decay_bp=float(decay_bp),
                exp_max_distance_bp=int(exp_max_distance_bp),
                adaptive_k=None,
                adaptive_min_bandwidth_bp=None,
                adaptive_max_distance_bp=None,
            )
            centres_adaptive, track_adaptive = compute_track_multi(
                strategy="exp_decay_adaptive",
                bin_size=int(bin_size_adaptive),
                sigma_units="bins",
                sigma=None,
                max_distance_bp=None,
                eps=None,
                decay_bp=None,
                exp_max_distance_bp=None,
                adaptive_k=int(adaptive_k),
                adaptive_min_bandwidth_bp=float(adaptive_min_bandwidth_bp),
                adaptive_max_distance_bp=int(adaptive_max_distance_bp),
            )
            covariate_set = tuple(covariates)
            if covariate_set:
                X_raw = build_covariate_matrix_multi(covariate_set, int(bin_size_counts_raw))
                X_gauss = build_covariate_matrix_multi(covariate_set, int(bin_size_counts_gauss))
                X_inv = build_covariate_matrix_multi(covariate_set, int(bin_size_inv))
                X_exp = build_covariate_matrix_multi(covariate_set, int(bin_size_exp))
                X_adaptive = build_covariate_matrix_multi(covariate_set, int(bin_size_adaptive))
            else:
                X_raw = np.zeros((len(centres_raw), 0), dtype=float)
                X_gauss = np.zeros((len(centres_gauss), 0), dtype=float)
                X_inv = np.zeros((len(centres_inv), 0), dtype=float)
                X_exp = np.zeros((len(centres_exp), 0), dtype=float)
                X_adaptive = np.zeros((len(centres_adaptive), 0), dtype=float)
            dnase_tracks_data = []
            for dnase in dnase_tracks:
                dnase_centres_raw, dnase_raw = load_dnase_means_multi(dnase["path"], int(bin_size_counts_raw))
                dnase_centres_gauss, dnase_gauss = load_dnase_means_multi(
                    dnase["path"], int(bin_size_counts_gauss)
                )
                dnase_centres_inv, dnase_inv = load_dnase_means_multi(dnase["path"], int(bin_size_inv))
                dnase_centres_exp, dnase_exp = load_dnase_means_multi(dnase["path"], int(bin_size_exp))
                dnase_centres_adaptive, dnase_adaptive = load_dnase_means_multi(
                    dnase["path"], int(bin_size_adaptive)
                )
                dnase_tracks_data.append(
                    {
                        "name": dnase["name"],
                        "path": dnase["path"],
                        "overlay": dnase["overlay"],
                        "raw": dnase_raw,
                        "gauss": dnase_gauss,
                        "inv": dnase_inv,
                        "exp": dnase_exp,
                        "adaptive": dnase_adaptive,
                    }
                )
        saved_results = {
            "centres_raw": centres_raw,
            "track_raw": track_raw,
            "centres_gauss": centres_gauss,
            "track_gauss": track_gauss,
            "centres_inv": centres_inv,
            "track_inv": track_inv,
            "centres_exp": centres_exp,
            "track_exp": track_exp,
            "centres_adaptive": centres_adaptive,
            "track_adaptive": track_adaptive,
            "X_raw": X_raw,
            "X_gauss": X_gauss,
            "X_inv": X_inv,
            "X_exp": X_exp,
            "X_adaptive": X_adaptive,
            "dnase_tracks_data": dnase_tracks_data,
        }
    st.session_state["last_results"] = saved_results
    st.session_state["last_params"] = current_params

    mask_raw = (centres_raw >= window_start) & (centres_raw <= window_end)
    mask_gauss = (centres_gauss >= window_start) & (centres_gauss <= window_end)
    mask_inv = (centres_inv >= window_start) & (centres_inv <= window_end)
    mask_exp = (centres_exp >= window_start) & (centres_exp <= window_end)
    mask_adaptive = (centres_adaptive >= window_start) & (centres_adaptive <= window_end)
    centres_raw_win = centres_raw[mask_raw]
    centres_gauss_win = centres_gauss[mask_gauss]
    centres_inv_win = centres_inv[mask_inv]
    centres_exp_win = centres_exp[mask_exp]
    centres_adaptive_win = centres_adaptive[mask_adaptive]
    track_raw_win = track_raw[mask_raw]
    track_gauss_win = track_gauss[mask_gauss]
    track_inv_win = track_inv[mask_inv]
    track_exp_win = track_exp[mask_exp]
    track_adaptive_win = track_adaptive[mask_adaptive]
    dnase_tracks_win = []
    for dnase in dnase_tracks_data:
        dnase_tracks_win.append(
            {
                "name": dnase["name"],
                "overlay": dnase["overlay"],
                "raw": dnase["raw"][mask_raw],
                "gauss": dnase["gauss"][mask_gauss],
                "inv": dnase["inv"][mask_inv],
                "exp": dnase["exp"][mask_exp],
                "adaptive": dnase["adaptive"][mask_adaptive],
                "raw_full": dnase["raw"],
                "gauss_full": dnase["gauss"],
                "inv_full": dnase["inv"],
                "exp_full": dnase["exp"],
                "adaptive_full": dnase["adaptive"],
            }
        )

    x_raw_mb = centres_raw_win / 1e6
    x_gauss_mb = centres_gauss_win / 1e6
    x_inv_mb = centres_inv_win / 1e6
    x_exp_mb = centres_exp_win / 1e6
    x_adaptive_mb = centres_adaptive_win / 1e6

    st.subheader("Track output")
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax[0].step(x_raw_mb, track_raw_win, color="#2c7fb8", linewidth=1.1, where="mid")
    ax[0].set_ylabel("counts_raw")
    ax[1].plot(x_gauss_mb, track_gauss_win, color="#d95f0e", linewidth=1.1)
    ax[1].set_ylabel("counts_gauss")
    ax[2].plot(x_inv_mb, track_inv_win, color="#1b9e77", linewidth=1.1)
    ax[2].set_ylabel("inv_dist_gauss")
    ax[3].plot(x_exp_mb, track_exp_win, color="#7570b3", linewidth=1.1)
    ax[3].set_ylabel("exp_decay")
    ax[4].plot(x_adaptive_mb, track_adaptive_win, color="#e7298a", linewidth=1.1)
    ax[4].set_ylabel("exp_decay_adaptive")
    overlay_tracks = [d for d in dnase_tracks_win if d["overlay"]]
    if overlay_tracks:
        colors = ["#666666", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        ax0 = ax[0].twinx()
        ax1 = ax[1].twinx()
        ax2 = ax[2].twinx()
        ax3 = ax[3].twinx()
        ax4 = ax[4].twinx()
        for idx, dnase in enumerate(overlay_tracks):
            color = colors[idx % len(colors)]
            ax0.plot(x_raw_mb, dnase["raw"], color=color, linewidth=0.9, alpha=0.7, label=dnase["name"])
            ax1.plot(x_gauss_mb, dnase["gauss"], color=color, linewidth=0.9, alpha=0.7, label=dnase["name"])
            ax2.plot(x_inv_mb, dnase["inv"], color=color, linewidth=0.9, alpha=0.7, label=dnase["name"])
            ax3.plot(x_exp_mb, dnase["exp"], color=color, linewidth=0.9, alpha=0.7, label=dnase["name"])
            ax4.plot(x_adaptive_mb, dnase["adaptive"], color=color, linewidth=0.9, alpha=0.7, label=dnase["name"])
        ax0.set_ylabel("DNase-seq")
        ax1.set_ylabel("DNase-seq")
        ax2.set_ylabel("DNase-seq")
        ax3.set_ylabel("DNase-seq")
        ax4.set_ylabel("DNase-seq")
        handles, labels = ax4.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(4, len(labels)),
                fontsize=10,
                title="DNase overlays",
                title_fontsize=10,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
            )
    x_label = f"Position on {chrom} (Mb)" if chrom != "all" else "Cumulative genome position (Mb)"
    ax[4].set_xlabel(x_label)
    ax[0].set_title("")
    fig.subplots_adjust(bottom=0.12)
    st.pyplot(fig, clear_figure=True)

    def fmt_metric(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3f}"

    pearson_local_score_bins = int(max(1, int(pearson_local_score_w)))
    spearman_local_score_bins = int(max(1, int(spearman_local_score_w)))
    if pearson_score_weight_shape + pearson_score_weight_slope <= 0:
        st.error("Pearson local score weights must sum to a positive value.")
        return
    if spearman_score_weight_shape + spearman_score_weight_slope <= 0:
        st.error("Spearman local score weights must sum to a positive value.")
        return
    pearson_score_weights = (float(pearson_score_weight_shape), float(pearson_score_weight_slope))
    spearman_score_weights = (float(spearman_score_weight_shape), float(spearman_score_weight_slope))

    def _smooth_param_value(method: str, raw_value: float) -> float | int | None:
        if method == "none":
            return None
        if method == "moving_average":
            return int(max(1, round(raw_value)))
        return float(raw_value)

    pearson_score_smooth_param_val = _smooth_param_value(
        pearson_score_smoothing, pearson_score_smooth_param
    )
    spearman_score_smooth_param_val = _smooth_param_value(
        spearman_score_smoothing, spearman_score_smooth_param
    )
    resid_raw = linear_residualise(track_raw, X_raw) if covariates else None
    resid_gauss = linear_residualise(track_gauss, X_gauss) if covariates else None
    resid_inv = linear_residualise(track_inv, X_inv) if covariates else None
    resid_exp = linear_residualise(track_exp, X_exp) if covariates else None
    resid_adaptive = linear_residualise(track_adaptive, X_adaptive) if covariates else None

    def _pearson_local_score(track_a: np.ndarray, track_b: np.ndarray) -> float:
        return compute_local_scores(
            track_a,
            track_b,
            w=pearson_local_score_bins,
            corr_type="pearson",
            smoothing=pearson_score_smoothing,
            smooth_param=pearson_score_smooth_param_val,
            transform=pearson_score_transform,
            zscore=pearson_score_zscore,
            weights=pearson_score_weights,
        ).global_score

    def _spearman_local_score(track_a: np.ndarray, track_b: np.ndarray) -> float:
        return compute_local_scores(
            track_a,
            track_b,
            w=spearman_local_score_bins,
            corr_type="spearman",
            smoothing=spearman_score_smoothing,
            smooth_param=spearman_score_smooth_param_val,
            transform=spearman_score_transform,
            zscore=spearman_score_zscore,
            weights=spearman_score_weights,
        ).global_score

    st.subheader("DNase correlations")
    dnase_corr_rows = []
    for dnase in dnase_tracks_win:
        corr_raw = pearsonr_nan(track_raw_win, dnase["raw"])
        corr_gauss = pearsonr_nan(track_gauss_win, dnase["gauss"])
        corr_inv = pearsonr_nan(track_inv_win, dnase["inv"])
        corr_exp = pearsonr_nan(track_exp_win, dnase["exp"])
        corr_adaptive = pearsonr_nan(track_adaptive_win, dnase["adaptive"])
        spearman_raw = spearmanr_nan(track_raw_win, dnase["raw"])
        spearman_gauss = spearmanr_nan(track_gauss_win, dnase["gauss"])
        spearman_inv = spearmanr_nan(track_inv_win, dnase["inv"])
        spearman_exp = spearmanr_nan(track_exp_win, dnase["exp"])
        spearman_adaptive = spearmanr_nan(track_adaptive_win, dnase["adaptive"])
        if covariates:
            resid_dnase_raw = linear_residualise(dnase["raw_full"], X_raw)
            resid_dnase_gauss = linear_residualise(dnase["gauss_full"], X_gauss)
            resid_dnase_inv = linear_residualise(dnase["inv_full"], X_inv)
            resid_dnase_exp = linear_residualise(dnase["exp_full"], X_exp)
            resid_dnase_adaptive = linear_residualise(dnase["adaptive_full"], X_adaptive)
            adj_raw = pearsonr_nan(resid_raw[mask_raw], resid_dnase_raw[mask_raw])
            adj_gauss = pearsonr_nan(resid_gauss[mask_gauss], resid_dnase_gauss[mask_gauss])
            adj_inv = pearsonr_nan(resid_inv[mask_inv], resid_dnase_inv[mask_inv])
            adj_exp = pearsonr_nan(resid_exp[mask_exp], resid_dnase_exp[mask_exp])
            adj_adaptive = pearsonr_nan(resid_adaptive[mask_adaptive], resid_dnase_adaptive[mask_adaptive])
            spearman_adj_raw = spearmanr_nan(resid_raw[mask_raw], resid_dnase_raw[mask_raw])
            spearman_adj_gauss = spearmanr_nan(resid_gauss[mask_gauss], resid_dnase_gauss[mask_gauss])
            spearman_adj_inv = spearmanr_nan(resid_inv[mask_inv], resid_dnase_inv[mask_inv])
            spearman_adj_exp = spearmanr_nan(resid_exp[mask_exp], resid_dnase_exp[mask_exp])
            spearman_adj_adaptive = spearmanr_nan(
                resid_adaptive[mask_adaptive], resid_dnase_adaptive[mask_adaptive]
            )
            rf_dnase_raw = rf_residualise(dnase["raw_full"], X_raw, seed=123)
            rf_dnase_gauss = rf_residualise(dnase["gauss_full"], X_gauss, seed=123)
            rf_dnase_inv = rf_residualise(dnase["inv_full"], X_inv, seed=123)
            rf_dnase_exp = rf_residualise(dnase["exp_full"], X_exp, seed=123)
            rf_dnase_adaptive = rf_residualise(dnase["adaptive_full"], X_adaptive, seed=123)
            rf_raw = pearsonr_nan(track_raw_win, rf_dnase_raw[mask_raw])
            rf_gauss = pearsonr_nan(track_gauss_win, rf_dnase_gauss[mask_gauss])
            rf_inv = pearsonr_nan(track_inv_win, rf_dnase_inv[mask_inv])
            rf_exp = pearsonr_nan(track_exp_win, rf_dnase_exp[mask_exp])
            rf_adaptive = pearsonr_nan(track_adaptive_win, rf_dnase_adaptive[mask_adaptive])
            pearson_local_raw = _pearson_local_score(resid_raw[mask_raw], resid_dnase_raw[mask_raw])
            pearson_local_gauss = _pearson_local_score(
                resid_gauss[mask_gauss], resid_dnase_gauss[mask_gauss]
            )
            pearson_local_inv = _pearson_local_score(
                resid_inv[mask_inv], resid_dnase_inv[mask_inv]
            )
            pearson_local_exp = _pearson_local_score(
                resid_exp[mask_exp], resid_dnase_exp[mask_exp]
            )
            pearson_local_adaptive = _pearson_local_score(
                resid_adaptive[mask_adaptive], resid_dnase_adaptive[mask_adaptive]
            )
            spearman_local_raw = _spearman_local_score(resid_raw[mask_raw], resid_dnase_raw[mask_raw])
            spearman_local_gauss = _spearman_local_score(
                resid_gauss[mask_gauss], resid_dnase_gauss[mask_gauss]
            )
            spearman_local_inv = _spearman_local_score(
                resid_inv[mask_inv], resid_dnase_inv[mask_inv]
            )
            spearman_local_exp = _spearman_local_score(
                resid_exp[mask_exp], resid_dnase_exp[mask_exp]
            )
            spearman_local_adaptive = _spearman_local_score(
                resid_adaptive[mask_adaptive], resid_dnase_adaptive[mask_adaptive]
            )
        else:
            adj_raw = adj_gauss = adj_inv = float("nan")
            adj_exp = adj_adaptive = float("nan")
            spearman_adj_raw = spearman_adj_gauss = spearman_adj_inv = float("nan")
            spearman_adj_exp = spearman_adj_adaptive = float("nan")
            rf_raw = rf_gauss = rf_inv = float("nan")
            rf_exp = rf_adaptive = float("nan")
            pearson_local_raw = _pearson_local_score(track_raw_win, dnase["raw"])
            pearson_local_gauss = _pearson_local_score(track_gauss_win, dnase["gauss"])
            pearson_local_inv = _pearson_local_score(track_inv_win, dnase["inv"])
            pearson_local_exp = _pearson_local_score(track_exp_win, dnase["exp"])
            pearson_local_adaptive = _pearson_local_score(track_adaptive_win, dnase["adaptive"])
            spearman_local_raw = _spearman_local_score(track_raw_win, dnase["raw"])
            spearman_local_gauss = _spearman_local_score(track_gauss_win, dnase["gauss"])
            spearman_local_inv = _spearman_local_score(track_inv_win, dnase["inv"])
            spearman_local_exp = _spearman_local_score(track_exp_win, dnase["exp"])
            spearman_local_adaptive = _spearman_local_score(track_adaptive_win, dnase["adaptive"])

        dnase_corr_rows.append(
            {
                "dnase_track": dnase["name"],
                "overlay": dnase["overlay"],
                "counts_raw_r": corr_raw,
                "counts_raw_adj_r": adj_raw,
                "counts_raw_spearman": spearman_raw,
                "counts_raw_spearman_adj": spearman_adj_raw,
                "counts_raw_pearson_local": pearson_local_raw,
                "counts_raw_spearman_local": spearman_local_raw,
                "counts_raw_rf": rf_raw,
                "counts_gauss_r": corr_gauss,
                "counts_gauss_adj_r": adj_gauss,
                "counts_gauss_spearman": spearman_gauss,
                "counts_gauss_spearman_adj": spearman_adj_gauss,
                "counts_gauss_pearson_local": pearson_local_gauss,
                "counts_gauss_spearman_local": spearman_local_gauss,
                "counts_gauss_rf": rf_gauss,
                "inv_dist_gauss_r": corr_inv,
                "inv_dist_gauss_adj_r": adj_inv,
                "inv_dist_gauss_spearman": spearman_inv,
                "inv_dist_gauss_spearman_adj": spearman_adj_inv,
                "inv_dist_gauss_pearson_local": pearson_local_inv,
                "inv_dist_gauss_spearman_local": spearman_local_inv,
                "inv_dist_gauss_rf": rf_inv,
                "exp_decay_r": corr_exp,
                "exp_decay_adj_r": adj_exp,
                "exp_decay_spearman": spearman_exp,
                "exp_decay_spearman_adj": spearman_adj_exp,
                "exp_decay_pearson_local": pearson_local_exp,
                "exp_decay_spearman_local": spearman_local_exp,
                "exp_decay_rf": rf_exp,
                "exp_decay_adaptive_r": corr_adaptive,
                "exp_decay_adaptive_adj_r": adj_adaptive,
                "exp_decay_adaptive_spearman": spearman_adaptive,
                "exp_decay_adaptive_spearman_adj": spearman_adj_adaptive,
                "exp_decay_adaptive_pearson_local": pearson_local_adaptive,
                "exp_decay_adaptive_spearman_local": spearman_local_adaptive,
                "exp_decay_adaptive_rf": rf_adaptive,
            }
        )

    if dnase_corr_rows:
        corr_df = pd.DataFrame(dnase_corr_rows)
        st.markdown(
            """
            <style>
            .track-card {
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 12px 14px;
                background: #fbfbfb;
                margin-bottom: 12px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }
            .track-card.best {
                border-color: #f4b400;
                box-shadow: 0 2px 6px rgba(244, 180, 0, 0.35);
                background: #fff7db;
            }
            .track-card h4 {
                margin: 0 0 6px 0;
                font-size: 14px;
                color: #333333;
            }
            .track-card .metric {
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #555555;
                margin: 2px 0;
            }
            .track-card .value {
                font-weight: 600;
                color: #1f1f1f;
            }
            .track-section {
                border-left: 4px solid #dadada;
                padding-left: 8px;
                margin: 8px 0 12px 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        strategy_specs = [
            (
                "counts_raw",
                "counts_raw_r",
                "counts_raw_adj_r",
                "counts_raw_spearman",
                "counts_raw_spearman_adj",
                "counts_raw_pearson_local",
                "counts_raw_spearman_local",
                "counts_raw_rf",
            ),
            (
                "counts_gauss",
                "counts_gauss_r",
                "counts_gauss_adj_r",
                "counts_gauss_spearman",
                "counts_gauss_spearman_adj",
                "counts_gauss_pearson_local",
                "counts_gauss_spearman_local",
                "counts_gauss_rf",
            ),
            (
                "inv_dist_gauss",
                "inv_dist_gauss_r",
                "inv_dist_gauss_adj_r",
                "inv_dist_gauss_spearman",
                "inv_dist_gauss_spearman_adj",
                "inv_dist_gauss_pearson_local",
                "inv_dist_gauss_spearman_local",
                "inv_dist_gauss_rf",
            ),
            (
                "exp_decay",
                "exp_decay_r",
                "exp_decay_adj_r",
                "exp_decay_spearman",
                "exp_decay_spearman_adj",
                "exp_decay_pearson_local",
                "exp_decay_spearman_local",
                "exp_decay_rf",
            ),
            (
                "exp_decay_adaptive",
                "exp_decay_adaptive_r",
                "exp_decay_adaptive_adj_r",
                "exp_decay_adaptive_spearman",
                "exp_decay_adaptive_spearman_adj",
                "exp_decay_adaptive_pearson_local",
                "exp_decay_adaptive_spearman_local",
                "exp_decay_adaptive_rf",
            ),
        ]

        highlight_mode = st.session_state.get("highlight_mode", "local_score")
        if highlight_mode == "anti_score":
            highlight_mode = "local_score"
            st.session_state["highlight_mode"] = highlight_mode
        for (
            label,
            r_col,
            adj_col,
            s_col,
            s_adj_col,
            pearson_local_col,
            spearman_local_col,
            rf_col,
        ) in strategy_specs:
            st.markdown(f"<div class='track-section'><strong>{label}</strong></div>", unsafe_allow_html=True)
            sub = corr_df[
                [
                    "dnase_track",
                    "overlay",
                    r_col,
                    adj_col,
                    s_col,
                    s_adj_col,
                    pearson_local_col,
                    spearman_local_col,
                    rf_col,
                ]
            ].copy()
            sub = sub.rename(
                columns={
                    r_col: "pearson_r",
                    adj_col: "pearson_r_linear",
                    s_col: "spearman_r",
                    s_adj_col: "spearman_r_linear",
                    pearson_local_col: "pearson_local_score",
                    spearman_local_col: "spearman_local_score",
                    rf_col: "rf_non_linear",
                }
            )
            best_idx = (
                sub[highlight_mode].idxmin()
                if sub[highlight_mode].notna().any()
                else None
            )

            cards_per_row = 3
            for start in range(0, len(sub), cards_per_row):
                cols = st.columns(cards_per_row)
                for col, (_, row) in zip(cols, sub.iloc[start:start + cards_per_row].iterrows()):
                    with col:
                        best_class = " best" if best_idx is not None and row.name == best_idx else ""
                        st.markdown(
                            f"""
                            <div class="track-card{best_class}">
                                <h4>{row['dnase_track']}</h4>
                                <div class="metric"><span>Pearson r</span><span class="value">{fmt_metric(row['pearson_r'])}</span></div>
                                <div class="metric"><span>Pearson r (linear covariate)</span><span class="value">{fmt_metric(row['pearson_r_linear'])}</span></div>
                                <div class="metric"><span>Spearman r</span><span class="value">{fmt_metric(row['spearman_r'])}</span></div>
                                <div class="metric"><span>Spearman r (linear covariate)</span><span class="value">{fmt_metric(row['spearman_r_linear'])}</span></div>
                                <div class="metric"><span>Pearson local score (linear covariate)</span><span class="value">{fmt_metric(row['pearson_local_score'])}</span></div>
                                <div class="metric"><span>Spearman local score (linear covariate)</span><span class="value">{fmt_metric(row['spearman_local_score'])}</span></div>
                                <div class="metric"><span>RF (non-linear covariate)</span><span class="value">{fmt_metric(row['rf_non_linear'])}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.subheader("Summary")
    if chrom == "all":
        mut_total = int(sum(len(mut_by_chrom.get(c, [])) for c in chroms_to_use))
        mut_in_window = 0
        offset = 0
        for chrom_name in chroms_to_use:
            chrom_len = chrom_lengths[chrom_name]
            start_rel = max(0, window_start - offset)
            end_rel = min(int(chrom_len), window_end - offset)
            if end_rel > start_rel:
                mut_pos = mut_by_chrom.get(chrom_name, np.array([], dtype=int))
                left_idx = np.searchsorted(mut_pos, start_rel, side="left")
                right_idx = np.searchsorted(mut_pos, end_rel, side="right")
                mut_in_window += int(max(0, right_idx - left_idx))
            offset += int(chrom_len)
    else:
        mut_total = int(mut_positions.size)
        left_idx = np.searchsorted(mut_positions, window_start, side="left")
        right_idx = np.searchsorted(mut_positions, window_end, side="right")
        mut_in_window = int(max(0, right_idx - left_idx))
    st.write(
        {
            "chromosome": chrom,
            "window_bp": [int(window_start), int(window_end)],
            "mutations_in_chrom": mut_total,
            "mutations_in_window": mut_in_window,
        }
    )
    stats = pd.DataFrame(
        [
            {
                "strategy": "counts_raw",
                "bin_size_bp": int(bin_size_counts_raw),
                "sigma_units": None,
                "sigma": None,
                "max_distance_bp": None,
                "bins_in_window": int(centres_raw_win.size),
                "nonzero_bins": int(np.count_nonzero(track_raw_win)),
                "mean": float(np.nanmean(track_raw_win)) if track_raw_win.size else float("nan"),
                "median": float(np.nanmedian(track_raw_win)) if track_raw_win.size else float("nan"),
                "max": float(np.nanmax(track_raw_win)) if track_raw_win.size else float("nan"),
            },
            {
                "strategy": "counts_gauss",
                "bin_size_bp": int(bin_size_counts_gauss),
                "sigma_units": sigma_units_counts,
                "sigma": float(sigma_counts),
                "max_distance_bp": None,
                "bins_in_window": int(centres_gauss_win.size),
                "nonzero_bins": int(np.count_nonzero(track_gauss_win)),
                "mean": float(np.nanmean(track_gauss_win)) if track_gauss_win.size else float("nan"),
                "median": float(np.nanmedian(track_gauss_win)) if track_gauss_win.size else float("nan"),
                "max": float(np.nanmax(track_gauss_win)) if track_gauss_win.size else float("nan"),
            },
            {
                "strategy": "inv_dist_gauss",
                "bin_size_bp": int(bin_size_inv),
                "sigma_units": sigma_units_inv,
                "sigma": float(sigma_inv),
                "max_distance_bp": int(max_distance_bp),
                "bins_in_window": int(centres_inv_win.size),
                "nonzero_bins": int(np.count_nonzero(track_inv_win)),
                "mean": float(np.nanmean(track_inv_win)) if track_inv_win.size else float("nan"),
                "median": float(np.nanmedian(track_inv_win)) if track_inv_win.size else float("nan"),
                "max": float(np.nanmax(track_inv_win)) if track_inv_win.size else float("nan"),
            },
            {
                "strategy": "exp_decay",
                "bin_size_bp": int(bin_size_exp),
                "sigma_units": None,
                "sigma": float(decay_bp),
                "max_distance_bp": int(exp_max_distance_bp),
                "bins_in_window": int(centres_exp_win.size),
                "nonzero_bins": int(np.count_nonzero(track_exp_win)),
                "mean": float(np.nanmean(track_exp_win)) if track_exp_win.size else float("nan"),
                "median": float(np.nanmedian(track_exp_win)) if track_exp_win.size else float("nan"),
                "max": float(np.nanmax(track_exp_win)) if track_exp_win.size else float("nan"),
            },
            {
                "strategy": "exp_decay_adaptive",
                "bin_size_bp": int(bin_size_adaptive),
                "sigma_units": None,
                "sigma": float(adaptive_min_bandwidth_bp),
                "max_distance_bp": int(adaptive_max_distance_bp),
                "bins_in_window": int(centres_adaptive_win.size),
                "nonzero_bins": int(np.count_nonzero(track_adaptive_win)),
                "mean": float(np.nanmean(track_adaptive_win)) if track_adaptive_win.size else float("nan"),
                "median": float(np.nanmedian(track_adaptive_win)) if track_adaptive_win.size else float("nan"),
                "max": float(np.nanmax(track_adaptive_win)) if track_adaptive_win.size else float("nan"),
            },
        ]
    )
    st.dataframe(stats, use_container_width=True)

    pending = st.session_state.pop("pending_config_action", None)
    if pending:
        action = pending.get("action")
        name = pending.get("name")
        rename_to = pending.get("rename_to")
        results = st.session_state.get("last_results")
        params = st.session_state.get("last_params")
        if not results or not params or not name:
            st.warning("Unable to save config: results are not ready yet.")
        else:
            configs = load_named_configs()
            target_name = rename_to or name
            if action == "save_new" and name in configs:
                st.warning(f"Config '{name}' already exists.")
            else:
                if rename_to and name in configs and rename_to != name:
                    configs.pop(name, None)
                configs = save_named_config(configs, name=target_name, params=params, results=results)
                save_named_configs(configs)
                st.session_state["config_select"] = target_name
                st.session_state["config_name_input"] = target_name
                st.success(f"Saved config '{target_name}'.")


if __name__ == "__main__":
    main()
