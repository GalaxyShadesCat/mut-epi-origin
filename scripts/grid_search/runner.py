"""Grid search runner for mutation vs accessibility experiments."""

from __future__ import annotations

import json
import logging
import shlex
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from scripts.bigwig_utils import get_bigwig_chrom_lengths
from scripts.contigs import ContigResolver
from scripts.dnase_map import DEFAULT_MAP_PATH, DnaseCellTypeMap
from scripts.genome_bins import load_fai, iter_chroms
from scripts.grid_search.config import (
    _normalize_downsample_values,
    _prefixed_track_params,
    expand_grid_values,
    sigma_to_bins,
)
from scripts.grid_search.features import build_shared_inputs, mutations_to_positions_by_chrom
from scripts.grid_search.io import _load_dnase_map_path
from scripts.grid_search.metrics import (
    aggregate_dict_column,
    best_and_margin,
    linear_residualise,
    pearsonr_nan,
    rf_feature_analysis,
    rf_residualise,
    spearmanr_nan,
)
from scripts.grid_search.results import _append_results_row, _build_results_columns, compute_derived_fields
from scripts.grid_search.sampling import (
    _downsample_mutations_df,
    _eligible_sample_keys,
    _prepare_non_overlapping_plan,
    _select_non_overlapping_samples,
    _unique_nonempty,
)
from scripts.io_utils import ensure_dir, save_df, save_json
from scripts.logging_utils import (
    log_kv,
    log_section,
    progress_line,
    setup_rich_logging,
    summarise_run,
    timed,
)
from scripts.scores import compute_local_scores
from scripts.stats_utils import weighted_mean, zscore_nan
from scripts.targets import dnase_mean_per_bin


_SKLEARN_PARALLEL_WARNING = (
    r".*sklearn\.utils\.parallel\.delayed.*sklearn\.utils\.parallel\.Parallel.*"
)


def suppress_sklearn_parallel_warning() -> None:
    """Silence sklearn/joblib warning about delayed/Parallel pairing."""
    warnings.filterwarnings(
        "ignore",
        message=_SKLEARN_PARALLEL_WARNING,
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )


def _join_csv(values: Sequence[Any]) -> str:
    return ",".join(str(v) for v in values)


def _format_k_samples(k_samples: Sequence[Optional[int]]) -> str:
    tokens: List[str] = []
    for val in k_samples:
        if val is None:
            tokens.append("all")
        else:
            tokens.append(str(int(val)))
    return ",".join(tokens)


def _format_covariate_sets(covariate_sets: Sequence[Sequence[str]]) -> str:
    groups = []
    for covs in covariate_sets:
        groups.append("+".join(str(c) for c in covs))
    return ";".join(groups)


def _format_downsample(downsample_counts: Sequence[Optional[int]]) -> str:
    if not downsample_counts:
        return "none"
    if all(val is None for val in downsample_counts):
        return "none"
    return _join_csv(int(val) for val in downsample_counts if val is not None)


def _format_inv_dist_pairs(pairs: Optional[Sequence[Sequence[float | int]]]) -> Optional[str]:
    if not pairs:
        return None
    tokens = []
    for sigma, md in pairs:
        tokens.append(f"{sigma}:{md}")
    return ",".join(tokens)


def _build_cli_command(grid_params: Dict[str, Any], out_dir: Path) -> str:
    project_root = Path(__file__).resolve().parents[2]

    def _relpath(path_value: str | Path) -> str:
        path = Path(path_value).resolve()
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            return str(path)
        return rel.as_posix()

    def _quote(value: str) -> str:
        return shlex.quote(value)

    def _add_arg(items: List[Tuple[str, Optional[str]]], flag: str, value: Optional[str] = None) -> None:
        items.append((flag, value))

    args: List[Tuple[str, Optional[str]]] = []

    mut_path = grid_params.get("mut_path")
    if isinstance(mut_path, list):
        mut_path = _join_csv(mut_path)
    _add_arg(args, "--mut-path", str(mut_path))
    _add_arg(args, "--fai-path", str(grid_params.get("fai_path")))
    _add_arg(args, "--fasta-path", str(grid_params.get("fasta_path")))
    atac_map_path = grid_params.get("atac_map_path")
    dnase_map_path = grid_params.get("dnase_map_path")
    if atac_map_path:
        _add_arg(args, "--atac-map-path", str(atac_map_path))
    elif dnase_map_path:
        _add_arg(args, "--dnase-map-path", str(dnase_map_path))
    else:
        _add_arg(args, "--dnase-map-json", json.dumps(grid_params.get("dnase_bigwigs", {})))

    timing_bw = grid_params.get("timing_bigwig")
    if timing_bw:
        _add_arg(args, "--timing-bw", str(timing_bw))

    tumour_filter = grid_params.get("tumour_filter")
    if tumour_filter:
        _add_arg(args, "--tumour-filter", _join_csv(tumour_filter))

    _add_arg(args, "--out-dir", _relpath(out_dir))
    _add_arg(args, "--base-seed", str(grid_params.get("base_seed")))
    _add_arg(args, "--n-resamples", str(grid_params.get("n_resamples")))
    _add_arg(args, "--k-samples", _format_k_samples(grid_params.get("k_samples", [])))
    per_sample_count = grid_params.get("per_sample_count")
    if per_sample_count is not None:
        _add_arg(args, "--per-sample-count", str(per_sample_count))
    _add_arg(args, "--downsample", _format_downsample(grid_params.get("downsample_counts", [])))
    _add_arg(args, "--track-strategies", _join_csv(grid_params.get("track_strategies", [])))
    _add_arg(args, "--covariate-sets", _format_covariate_sets(grid_params.get("covariate_sets", [])))

    _add_arg(args, "--counts-raw-bins", _join_csv(grid_params.get("counts_raw_bins", [])))
    _add_arg(args, "--counts-gauss-bins", _join_csv(grid_params.get("counts_gauss_bins", [])))
    _add_arg(args, "--inv-dist-gauss-bins", _join_csv(grid_params.get("inv_dist_gauss_bins", [])))
    _add_arg(args, "--exp-decay-bins", _join_csv(grid_params.get("exp_decay_bins", [])))
    _add_arg(args, "--exp-decay-adaptive-bins", _join_csv(grid_params.get("exp_decay_adaptive_bins", [])))

    _add_arg(args, "--counts-gauss-sigma-grid", _join_csv(grid_params.get("counts_gauss_sigma_grid", [])))
    _add_arg(args, "--counts-gauss-sigma-units", str(grid_params.get("counts_gauss_sigma_units")))
    _add_arg(args, "--inv-dist-gauss-sigma-grid", _join_csv(grid_params.get("inv_dist_gauss_sigma_grid", [])))
    _add_arg(
        args,
        "--inv-dist-gauss-max-distance-bp-grid",
        _join_csv(grid_params.get("inv_dist_gauss_max_distance_bp_grid", [])),
    )
    _add_arg(args, "--inv-dist-gauss-sigma-units", str(grid_params.get("inv_dist_gauss_sigma_units")))

    pairs = _format_inv_dist_pairs(grid_params.get("inv_dist_gauss_pairs"))
    if pairs:
        _add_arg(args, "--inv-dist-gauss-pairs", pairs)

    _add_arg(args, "--exp-decay-decay-bp-grid", _join_csv(grid_params.get("exp_decay_decay_bp_grid", [])))
    _add_arg(args, "--exp-decay-max-distance-bp-grid", _join_csv(grid_params.get("exp_decay_max_distance_bp_grid", [])))
    _add_arg(args, "--exp-decay-adaptive-k-grid", _join_csv(grid_params.get("exp_decay_adaptive_k_grid", [])))
    _add_arg(
        args,
        "--exp-decay-adaptive-min-bandwidth-bp-grid",
        _join_csv(grid_params.get("exp_decay_adaptive_min_bandwidth_bp_grid", [])),
    )
    _add_arg(
        args,
        "--exp-decay-adaptive-max-distance-bp-grid",
        _join_csv(grid_params.get("exp_decay_adaptive_max_distance_bp_grid", [])),
    )

    _add_arg(args, "--pearson-score-window-bins", str(grid_params.get("pearson_score_window_bins")))
    _add_arg(args, "--pearson-score-smoothing", str(grid_params.get("pearson_score_smoothing")))
    if grid_params.get("pearson_score_smooth_param") is not None:
        _add_arg(args, "--pearson-score-smooth-param", str(grid_params.get("pearson_score_smooth_param")))
    _add_arg(args, "--pearson-score-transform", str(grid_params.get("pearson_score_transform")))
    if grid_params.get("pearson_score_zscore"):
        _add_arg(args, "--pearson-score-zscore")
    pearson_score_weights = grid_params.get("pearson_score_weights")
    if pearson_score_weights:
        _add_arg(args, "--pearson-score-weights", _join_csv(pearson_score_weights))

    _add_arg(args, "--spearman-score-window-bins", str(grid_params.get("spearman_score_window_bins")))
    _add_arg(args, "--spearman-score-smoothing", str(grid_params.get("spearman_score_smoothing")))
    if grid_params.get("spearman_score_smooth_param") is not None:
        _add_arg(args, "--spearman-score-smooth-param", str(grid_params.get("spearman_score_smooth_param")))
    _add_arg(args, "--spearman-score-transform", str(grid_params.get("spearman_score_transform")))
    if grid_params.get("spearman_score_zscore"):
        _add_arg(args, "--spearman-score-zscore")
    spearman_score_weights = grid_params.get("spearman_score_weights")
    if spearman_score_weights:
        _add_arg(args, "--spearman-score-weights", _join_csv(spearman_score_weights))

    if grid_params.get("include_trinuc"):
        _add_arg(args, "--include-trinuc")

    chroms = grid_params.get("chroms")
    if chroms:
        _add_arg(args, "--chroms", _join_csv(chroms))

    if grid_params.get("save_per_bin"):
        _add_arg(args, "--save-per-bin")

    if not grid_params.get("standardise_tracks", True):
        _add_arg(args, "--no-standardise-tracks")
    _add_arg(args, "--standardise-scope", str(grid_params.get("standardise_scope")))

    lines = ["python -m scripts.grid_search.cli \\"]
    for flag, value in args:
        if value is None:
            lines.append(f"  {flag} \\")
        else:
            lines.append(f"  {flag} {_quote(str(value))} \\")
    if lines:
        lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


def run_one_config(
        *,
        muts_df: pd.DataFrame,
        chrom: str,
        chrom_length: int,
        bin_size: int,
        track_strategy: str,
        # track params
        counts_sigma_bins: float,
        inv_sigma_bins: float,
        max_distance_bp: int,
        exp_decay_bp: float,
        exp_max_distance_bp: int,
        adaptive_k: int,
        adaptive_min_bandwidth_bp: float,
        adaptive_max_distance_bp: int,
        # target
        dnase_bigwig: str | Path,
        celltype: str,
        # covariates
        covariates: List[str],
        fasta_path: str | Path,
        timing_bigwig: Optional[str | Path],
        include_trinuc: bool,
        # random seeds
        rf_seed: int,
        # local score settings
        pearson_score_window_bins: int,
        pearson_score_smoothing: str,
        pearson_score_smooth_param: float | int | None,
        pearson_score_transform: str,
        pearson_score_zscore: bool,
        pearson_score_weights: Tuple[float, float],
        spearman_score_window_bins: int,
        spearman_score_smoothing: str,
        spearman_score_smooth_param: float | int | None,
        spearman_score_transform: str,
        spearman_score_zscore: bool,
        spearman_score_weights: Tuple[float, float],
        # standardisation
        standardise_tracks: bool = True,
        standardise_scope: str = "per_chrom",
        accessibility_prefix: str = "dnase",
        # shared inputs
        shared: Optional[Dict[str, Any]] = None,
        pos_by_chrom: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    if shared is None:
        if pos_by_chrom is None:
            pos_by_chrom = mutations_to_positions_by_chrom(muts_df)
        shared = build_shared_inputs(
            pos_by_chrom=pos_by_chrom,
            chrom=chrom,
            chrom_length=chrom_length,
            bin_size=bin_size,
            track_strategy=track_strategy,
            counts_sigma_bins=counts_sigma_bins,
            inv_sigma_bins=inv_sigma_bins,
            max_distance_bp=max_distance_bp,
            exp_decay_bp=exp_decay_bp,
            exp_max_distance_bp=exp_max_distance_bp,
            adaptive_k=adaptive_k,
            adaptive_min_bandwidth_bp=adaptive_min_bandwidth_bp,
            adaptive_max_distance_bp=adaptive_max_distance_bp,
            covariates=covariates,
            fasta_path=fasta_path,
            timing_bigwig=timing_bigwig,
            include_trinuc=include_trinuc,
        )

    edges = shared["edges"]
    mut_track = shared["mut_track"]
    X = shared["X"]

    dnase_track = dnase_mean_per_bin(dnase_bigwig, chrom, edges).astype(float)

    if standardise_tracks:
        if standardise_scope != "per_chrom":
            raise ValueError(f"Unsupported standardise_scope: {standardise_scope}")
        mut_track_corr = zscore_nan(mut_track)
        dnase_track_corr = zscore_nan(dnase_track)
    else:
        mut_track_corr = mut_track
        dnase_track_corr = dnase_track

    mut_resid = linear_residualise(mut_track_corr, X) if X.shape[1] else mut_track_corr.copy()
    dnase_resid_lin = linear_residualise(dnase_track_corr, X) if X.shape[1] else dnase_track_corr.copy()

    r_raw = pearsonr_nan(mut_track_corr, dnase_track_corr)
    r_lin = pearsonr_nan(mut_resid, dnase_resid_lin)
    s_raw = spearmanr_nan(mut_track_corr, dnase_track_corr)
    s_lin = spearmanr_nan(mut_resid, dnase_resid_lin)

    r_rf = float("nan")
    if X.shape[1]:
        dnase_resid_rf = rf_residualise(dnase_track_corr, X, seed=rf_seed)
        r_rf = pearsonr_nan(mut_track_corr, dnase_resid_rf)

    score_mut = mut_resid if X.shape[1] else mut_track_corr
    score_dnase = dnase_resid_lin if X.shape[1] else dnase_track_corr

    score_res_pearson = compute_local_scores(
        score_mut,
        score_dnase,
        w=int(pearson_score_window_bins),
        corr_type="pearson",
        smoothing=str(pearson_score_smoothing),
        smooth_param=pearson_score_smooth_param,
        transform=str(pearson_score_transform),
        zscore=bool(pearson_score_zscore),
        weights=pearson_score_weights,
    )
    score_res_spearman = compute_local_scores(
        score_mut,
        score_dnase,
        w=int(spearman_score_window_bins),
        corr_type="spearman",
        smoothing=str(spearman_score_smoothing),
        smooth_param=spearman_score_smooth_param,
        transform=str(spearman_score_transform),
        zscore=bool(spearman_score_zscore),
        weights=spearman_score_weights,
    )

    mask_valid = np.isfinite(mut_track_corr) & np.isfinite(dnase_track_corr)
    summary = {
        "celltype": celltype,
        "pearson_r_raw": float(r_raw),
        "pearson_r_linear_resid": float(r_lin),
        "pearson_r_rf_resid": float(r_rf),
        "spearman_r_raw": float(s_raw),
        "spearman_r_linear_resid": float(s_lin),
        f"n_bins_valid_mut_and_{accessibility_prefix}": int(mask_valid.sum()),
        "pearson_local_score_global": float(score_res_pearson.global_score),
        "pearson_local_score_negative_corr_fraction": float(
            score_res_pearson.negative_corr_fraction
        ),
        "spearman_local_score_global": float(score_res_spearman.global_score),
        "spearman_local_score_negative_corr_fraction": float(
            score_res_spearman.negative_corr_fraction
        ),
    }

    return summary, dnase_track


# -------------------------
# Grid runner for notebook use
# -------------------------

def run_grid_experiment(
        *,
        mut_path: str | Path | Sequence[str | Path],
        fai_path: str | Path,
        fasta_path: str | Path,
        dnase_bigwigs: Optional[Dict[str, str | Path]] = None,
        dnase_map_path: str | Path | None = None,
        atac_map_path: str | Path | None = None,
        celltype_map: Optional[DnaseCellTypeMap] = None,
        timing_bigwig: Optional[str | Path],
        k_samples: Optional[List[int | None]] = None,
        n_resamples: Optional[int] = None,
        base_seed: int,
        track_strategies: List[str],
        covariate_sets: List[List[str]],
        include_trinuc: bool = False,
        chroms: Optional[List[str]] = None,
        standardise_tracks: bool = True,
        standardise_scope: str = "per_chrom",
        verbose: bool = False,
        resume: bool = False,
        per_sample_count: Optional[int] = None,
        # local score settings
        pearson_score_window_bins: int = 1,
        pearson_score_smoothing: str = "none",
        pearson_score_smooth_param: float | int | None = None,
        pearson_score_transform: str = "none",
        pearson_score_zscore: bool = False,
        pearson_score_weights: Tuple[float, float] = (0.7, 0.3),
        spearman_score_window_bins: int = 5,
        spearman_score_smoothing: str = "none",
        spearman_score_smooth_param: float | int | None = None,
        spearman_score_transform: str = "none",
        spearman_score_zscore: bool = False,
        spearman_score_weights: Tuple[float, float] = (0.7, 0.3),
        # track params (derived from per-track grids)
        # per-track hyperparameter grids
        counts_raw_bins: Sequence[int] | int | str = (1_000_000,),
        counts_gauss_bins: Sequence[int] | int | str = (1_000_000,),
        inv_dist_gauss_bins: Sequence[int] | int | str = (1_000_000,),
        exp_decay_bins: Sequence[int] | int | str = (1_000_000,),
        exp_decay_adaptive_bins: Sequence[int] | int | str = (1_000_000,),
        counts_gauss_sigma_grid: Sequence[float] | float | str = (1.0,),
        counts_gauss_sigma_units: str = "bins",
        inv_dist_gauss_sigma_grid: Sequence[float] | float | str = (0.5,),
        inv_dist_gauss_max_distance_bp_grid: Sequence[int] | int | str = (1_000_000,),
        inv_dist_gauss_pairs: Optional[Sequence[Tuple[float, int]]] = None,
        inv_dist_gauss_sigma_units: str = "bins",
        exp_decay_decay_bp_grid: Sequence[int] | int | str = (200_000,),
        exp_decay_max_distance_bp_grid: Sequence[int] | int | str = (1_000_000,),
        exp_decay_adaptive_k_grid: Sequence[int] | int | str = (5,),
        exp_decay_adaptive_min_bandwidth_bp_grid: Sequence[int] | int | str = (50_000,),
        exp_decay_adaptive_max_distance_bp_grid: Sequence[int] | int | str = (1_000_000,),
        downsample_counts: Optional[Sequence[int] | int | str] = None,
        # IO
        out_dir: str | Path = "outputs/experiments/run",
        save_per_bin: bool = True,
        chunksize: int = 250_000,
        tumour_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    suppress_sklearn_parallel_warning()

    def _nanmean_safe(values: np.ndarray) -> float:
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    project_root = Path(__file__).resolve().parents[2]

    def _resolve_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return project_root / path

    def _relpath(path_value: str | Path) -> str:
        path = Path(path_value).resolve()
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            return str(path)
        return rel.as_posix()

    def _load_sample_plan(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Sample plan JSON is invalid; regenerating: %s", path)
            return None
        ordered_keys = payload.get("ordered_keys")
        if not isinstance(ordered_keys, list):
            logger.warning("Sample plan missing ordered_keys; regenerating: %s", path)
            return None
        paths = payload.get("paths")
        if paths is None:
            plan_paths = None
        elif isinstance(paths, list):
            plan_paths = [Path(p) for p in paths]
        else:
            logger.warning("Sample plan paths invalid; regenerating: %s", path)
            return None
        return {"paths": plan_paths, "ordered_keys": ordered_keys}

    def _save_sample_plan(path: Path, plan: Dict[str, Any], allowed_keys: Optional[Set[str]]) -> None:
        payload = {
            "ordered_keys": plan["ordered_keys"],
            "paths": None if plan["paths"] is None else [str(p) for p in plan["paths"]],
            "seed": int(base_seed),
            "chunksize": int(chunksize),
            "allowed_keys": None if allowed_keys is None else sorted(allowed_keys),
        }
        save_json(payload, path)

    out_dir_path = Path(out_dir)
    if not out_dir_path.is_absolute():
        out_dir_path = project_root / out_dir_path

    resume_params: Optional[Dict[str, Any]] = None
    if resume:
        if not out_dir_path.exists():
            raise FileNotFoundError(f"Resume requested but out_dir does not exist: {out_dir_path}")
        grid_params_path = out_dir_path / "grid_search_params.json"
        if not grid_params_path.exists():
            raise FileNotFoundError(
                f"Resume requested but grid_search_params.json not found: {grid_params_path}"
            )
        resume_params = json.loads(grid_params_path.read_text(encoding="utf-8"))

        def _resolve_from_grid(path_value: str | Path | None) -> Optional[Path]:
            if path_value is None:
                return None
            return _resolve_path(path_value)

        mut_path = resume_params.get("mut_path")
        if isinstance(mut_path, list):
            mut_path = [_resolve_from_grid(p) for p in mut_path]
        else:
            mut_path = _resolve_from_grid(mut_path)
        fai_path = _resolve_path(resume_params["fai_path"])
        fasta_path = _resolve_path(resume_params["fasta_path"])
        dnase_map_path = resume_params.get("dnase_map_path", dnase_map_path)
        atac_map_path = resume_params.get("atac_map_path", atac_map_path)
        timing_bigwig = _resolve_from_grid(resume_params.get("timing_bigwig"))
        dnase_bigwigs = {
            key: _resolve_path(path) for key, path in resume_params.get("dnase_bigwigs", {}).items()
        }
        celltype_map = None
        k_samples = resume_params.get("k_samples", k_samples)
        n_resamples = resume_params.get("n_resamples", n_resamples)
        base_seed = resume_params.get("base_seed", base_seed)
        track_strategies = resume_params.get("track_strategies", track_strategies)
        covariate_sets = resume_params.get("covariate_sets", covariate_sets)
        include_trinuc = resume_params.get("include_trinuc", include_trinuc)
        chroms = resume_params.get("chroms", chroms)
        standardise_tracks = resume_params.get("standardise_tracks", standardise_tracks)
        standardise_scope = resume_params.get("standardise_scope", standardise_scope)
        pearson_score_window_bins = resume_params.get(
            "pearson_score_window_bins", pearson_score_window_bins
        )
        pearson_score_smoothing = resume_params.get("pearson_score_smoothing", pearson_score_smoothing)
        pearson_score_smooth_param = resume_params.get(
            "pearson_score_smooth_param", pearson_score_smooth_param
        )
        pearson_score_transform = resume_params.get("pearson_score_transform", pearson_score_transform)
        pearson_score_zscore = resume_params.get("pearson_score_zscore", pearson_score_zscore)
        pearson_score_weights = tuple(
            resume_params.get("pearson_score_weights", pearson_score_weights)
        )
        spearman_score_window_bins = resume_params.get(
            "spearman_score_window_bins", spearman_score_window_bins
        )
        spearman_score_smoothing = resume_params.get(
            "spearman_score_smoothing", spearman_score_smoothing
        )
        spearman_score_smooth_param = resume_params.get(
            "spearman_score_smooth_param", spearman_score_smooth_param
        )
        spearman_score_transform = resume_params.get(
            "spearman_score_transform", spearman_score_transform
        )
        spearman_score_zscore = resume_params.get("spearman_score_zscore", spearman_score_zscore)
        spearman_score_weights = tuple(
            resume_params.get("spearman_score_weights", spearman_score_weights)
        )
        counts_raw_bins = resume_params.get("counts_raw_bins", counts_raw_bins)
        counts_gauss_bins = resume_params.get("counts_gauss_bins", counts_gauss_bins)
        inv_dist_gauss_bins = resume_params.get("inv_dist_gauss_bins", inv_dist_gauss_bins)
        exp_decay_bins = resume_params.get("exp_decay_bins", exp_decay_bins)
        exp_decay_adaptive_bins = resume_params.get("exp_decay_adaptive_bins", exp_decay_adaptive_bins)
        counts_gauss_sigma_grid = resume_params.get("counts_gauss_sigma_grid", counts_gauss_sigma_grid)
        counts_gauss_sigma_units = resume_params.get("counts_gauss_sigma_units", counts_gauss_sigma_units)
        inv_dist_gauss_sigma_grid = resume_params.get("inv_dist_gauss_sigma_grid", inv_dist_gauss_sigma_grid)
        inv_dist_gauss_max_distance_bp_grid = resume_params.get(
            "inv_dist_gauss_max_distance_bp_grid",
            inv_dist_gauss_max_distance_bp_grid,
        )
        inv_dist_gauss_pairs = resume_params.get("inv_dist_gauss_pairs", inv_dist_gauss_pairs)
        inv_dist_gauss_sigma_units = resume_params.get("inv_dist_gauss_sigma_units", inv_dist_gauss_sigma_units)
        exp_decay_decay_bp_grid = resume_params.get("exp_decay_decay_bp_grid", exp_decay_decay_bp_grid)
        exp_decay_max_distance_bp_grid = resume_params.get(
            "exp_decay_max_distance_bp_grid",
            exp_decay_max_distance_bp_grid,
        )
        exp_decay_adaptive_k_grid = resume_params.get("exp_decay_adaptive_k_grid", exp_decay_adaptive_k_grid)
        exp_decay_adaptive_min_bandwidth_bp_grid = resume_params.get(
            "exp_decay_adaptive_min_bandwidth_bp_grid",
            exp_decay_adaptive_min_bandwidth_bp_grid,
        )
        exp_decay_adaptive_max_distance_bp_grid = resume_params.get(
            "exp_decay_adaptive_max_distance_bp_grid",
            exp_decay_adaptive_max_distance_bp_grid,
        )
        downsample_counts = resume_params.get("downsample_counts", downsample_counts)
        if isinstance(downsample_counts, (list, tuple)) and any(val is None for val in downsample_counts):
            downsample_counts = "none"
        save_per_bin = resume_params.get("save_per_bin", save_per_bin)
        chunksize = resume_params.get("chunksize", chunksize)
        tumour_filter = resume_params.get("tumour_filter", tumour_filter)
        per_sample_count = resume_params.get("per_sample_count", per_sample_count)

    accessibility_prefix = "atac" if atac_map_path is not None else "dnase"
    logger = setup_rich_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        logger_name=f"mut_vs_{accessibility_prefix}",
        force=True,
    )

    if per_sample_count is None and (k_samples is None or n_resamples is None):
        raise ValueError("k_samples and n_resamples are required unless per_sample_count is set.")
    if per_sample_count is not None:
        if k_samples is None:
            k_samples = [1]
        if n_resamples is None:
            n_resamples = 1

    if isinstance(mut_path, (list, tuple)):
        mut_path = [_resolve_path(p) for p in mut_path]
    else:
        mut_path = _resolve_path(mut_path)
    fai_path = _resolve_path(fai_path)
    fasta_path = _resolve_path(fasta_path)
    if timing_bigwig is not None:
        timing_bigwig = _resolve_path(timing_bigwig)

    if standardise_tracks and standardise_scope != "per_chrom":
        raise ValueError(f"Unsupported standardise_scope: {standardise_scope}")

    if dnase_map_path is not None and atac_map_path is not None:
        raise ValueError("Provide only one of dnase_map_path or atac_map_path.")

    if dnase_bigwigs is None and dnase_map_path is None and atac_map_path is None:
        dnase_map_path = DEFAULT_MAP_PATH

    accessibility_label = "ATAC-seq" if atac_map_path is not None else "DNase-seq"
    accessibility_map_path = atac_map_path if atac_map_path is not None else dnase_map_path

    if accessibility_map_path is not None:
        accessibility_map_path = _resolve_path(accessibility_map_path)
        if atac_map_path is not None:
            atac_map_path = accessibility_map_path
        else:
            dnase_map_path = accessibility_map_path
    track_key = "atac_path" if atac_map_path is not None else "dnase_path"
    if dnase_bigwigs is None:
        if accessibility_map_path is None:
            raise ValueError(f"{accessibility_label} bigwigs must be provided when no map path is set.")
        dnase_bigwigs, celltype_map = _load_dnase_map_path(
            Path(accessibility_map_path),
            track_key=track_key,
            label=accessibility_label,
        )
    elif celltype_map is None and accessibility_map_path is not None:
        _, celltype_map = _load_dnase_map_path(
            Path(accessibility_map_path),
            track_key=track_key,
            label=accessibility_label,
        )
    if not dnase_bigwigs:
        raise ValueError(
            f"{accessibility_label} bigwigs must be a non-empty mapping of celltype->bigWig path."
        )

    if tumour_filter is None and celltype_map is not None:
        mapped_filter = celltype_map.tumour_filter()
        if mapped_filter:
            tumour_filter = mapped_filter

    counts_raw_bins = expand_grid_values(counts_raw_bins, name="counts_raw_bins", cast=int)
    counts_gauss_bins = expand_grid_values(counts_gauss_bins, name="counts_gauss_bins", cast=int)
    inv_dist_gauss_bins = expand_grid_values(inv_dist_gauss_bins, name="inv_dist_gauss_bins", cast=int)
    exp_decay_bins = expand_grid_values(exp_decay_bins, name="exp_decay_bins", cast=int)
    exp_decay_adaptive_bins = expand_grid_values(
        exp_decay_adaptive_bins, name="exp_decay_adaptive_bins", cast=int
    )
    counts_gauss_sigma_grid = expand_grid_values(
        counts_gauss_sigma_grid, name="counts_gauss_sigma_grid", cast=float
    )
    inv_dist_gauss_sigma_grid = expand_grid_values(
        inv_dist_gauss_sigma_grid, name="inv_dist_gauss_sigma_grid", cast=float
    )
    inv_dist_gauss_max_distance_bp_grid = expand_grid_values(
        inv_dist_gauss_max_distance_bp_grid, name="inv_dist_gauss_max_distance_bp_grid", cast=int
    )
    exp_decay_decay_bp_grid = expand_grid_values(
        exp_decay_decay_bp_grid, name="exp_decay_decay_bp_grid", cast=int
    )
    exp_decay_max_distance_bp_grid = expand_grid_values(
        exp_decay_max_distance_bp_grid, name="exp_decay_max_distance_bp_grid", cast=int
    )
    exp_decay_adaptive_k_grid = expand_grid_values(
        exp_decay_adaptive_k_grid, name="exp_decay_adaptive_k_grid", cast=int
    )
    exp_decay_adaptive_min_bandwidth_bp_grid = expand_grid_values(
        exp_decay_adaptive_min_bandwidth_bp_grid,
        name="exp_decay_adaptive_min_bandwidth_bp_grid",
        cast=int,
    )
    exp_decay_adaptive_max_distance_bp_grid = expand_grid_values(
        exp_decay_adaptive_max_distance_bp_grid,
        name="exp_decay_adaptive_max_distance_bp_grid",
        cast=int,
    )
    downsample_grid = _normalize_downsample_values(downsample_counts)

    bins_by_strategy = {
        "counts_raw": counts_raw_bins,
        "counts_gauss": counts_gauss_bins,
        "inv_dist_gauss": inv_dist_gauss_bins,
        "exp_decay": exp_decay_bins,
        "exp_decay_adaptive": exp_decay_adaptive_bins,
    }
    for strategy in track_strategies:
        if not bins_by_strategy.get(strategy):
            raise ValueError(f"{strategy} bins are empty after expansion.")
    if "counts_gauss" in track_strategies and not counts_gauss_sigma_grid:
        raise ValueError("counts_gauss_sigma_grid is empty after expansion.")
    if "inv_dist_gauss" in track_strategies:
        if not inv_dist_gauss_sigma_grid:
            raise ValueError("inv_dist_gauss_sigma_grid is empty after expansion.")
        if not inv_dist_gauss_max_distance_bp_grid:
            raise ValueError("inv_dist_gauss_max_distance_bp_grid is empty after expansion.")
    if "exp_decay" in track_strategies:
        if not exp_decay_decay_bp_grid:
            raise ValueError("exp_decay_decay_bp_grid is empty after expansion.")
        if not exp_decay_max_distance_bp_grid:
            raise ValueError("exp_decay_max_distance_bp_grid is empty after expansion.")
    if "exp_decay_adaptive" in track_strategies:
        if not exp_decay_adaptive_k_grid:
            raise ValueError("exp_decay_adaptive_k_grid is empty after expansion.")
        if not exp_decay_adaptive_min_bandwidth_bp_grid:
            raise ValueError("exp_decay_adaptive_min_bandwidth_bp_grid is empty after expansion.")
        if not exp_decay_adaptive_max_distance_bp_grid:
            raise ValueError("exp_decay_adaptive_max_distance_bp_grid is empty after expansion.")
    if not downsample_grid:
        raise ValueError("downsample grid is empty after expansion.")
    for val in downsample_grid:
        if val is not None and val <= 0:
            raise ValueError("downsample values must be positive integers.")
    if int(pearson_score_window_bins) < 1:
        raise ValueError("pearson_score_window_bins must be >= 1.")
    if int(spearman_score_window_bins) < 1:
        raise ValueError("spearman_score_window_bins must be >= 1.")
    if len(pearson_score_weights) != 2:
        raise ValueError("pearson_score_weights must be a length-2 tuple (shape, slope).")
    if len(spearman_score_weights) != 2:
        raise ValueError("spearman_score_weights must be a length-2 tuple (shape, slope).")

    default_counts_sigma_bins = 1.0
    default_inv_sigma_bins = 0.5
    default_max_distance_bp = 1_000_000
    default_exp_decay_bp = 200_000
    default_exp_max_distance_bp = 1_000_000
    default_adaptive_k = 5
    default_adaptive_min_bandwidth_bp = 50_000
    default_adaptive_max_distance_bp = 1_000_000

    normalised_dnase: Dict[str, Path] = {}
    for key, path in dnase_bigwigs.items():
        celltype = str(key).strip()
        if not celltype:
            raise ValueError(f"{accessibility_label} bigwigs contains an empty cell type key.")
        bw_path = Path(path)
        if not bw_path.exists():
            raise FileNotFoundError(f"{accessibility_label} bigWig not found for '{celltype}': {bw_path}")
        chrom_lengths = get_bigwig_chrom_lengths(bw_path)
        if not chrom_lengths:
            raise ValueError(f"{accessibility_label} bigWig has no contigs: {bw_path}")
        resolver = ContigResolver(fasta_contigs=None, bigwig_contigs=list(chrom_lengths.keys()))
        if not resolver.has_canonical_in_bigwig("chr1"):
            raise ValueError(
                f"{accessibility_label} bigWig missing canonical chr1 (or alias) for '{celltype}'. "
                f"Example contigs: {', '.join(sorted(chrom_lengths.keys())[:10]) or 'none'}."
            )
        normalised_dnase[celltype] = bw_path
    dnase_bigwigs = normalised_dnase

    def _next_available_dir(path: Path) -> Path:
        if not path.exists():
            return path
        base = str(path)
        idx = 1
        while True:
            candidate = Path(f"{base}_{idx}")
            if not candidate.exists():
                return candidate
            idx += 1

    if resume:
        out_dir = ensure_dir(out_dir_path)
    else:
        out_dir_path = _next_available_dir(out_dir_path)
        out_dir = ensure_dir(out_dir_path)
    runs_dir = ensure_dir(out_dir / "runs")
    results_path = out_dir / "results.csv"
    results_columns = _build_results_columns(
        list(dnase_bigwigs.keys()),
        track_strategies,
        accessibility_prefix=accessibility_prefix,
    )
    if not results_path.exists():
        pd.DataFrame(columns=results_columns).to_csv(results_path, index=False)
    completed_run_ids: Set[str] = set()
    existing_results_df: Optional[pd.DataFrame] = None
    if results_path.exists() and results_path.stat().st_size > 0:
        try:
            existing_results_df = pd.read_csv(results_path)
        except pd.errors.EmptyDataError:
            existing_results_df = pd.DataFrame(columns=results_columns)
        if existing_results_df is not None:
            if list(existing_results_df.columns):
                results_columns = list(existing_results_df.columns)
            if "run_id" in existing_results_df.columns:
                completed_run_ids = set(existing_results_df["run_id"].astype(str))
    completed_sample_counts: Dict[str, int] = {}
    if completed_run_ids:
        for run_id in completed_run_ids:
            for part in str(run_id).split("__"):
                if part.startswith("sample="):
                    tag = part[len("sample="):]
                    if tag:
                        completed_sample_counts[tag] = completed_sample_counts.get(tag, 0) + 1

    # README.txt output disabled by request.

    grid_params_path = out_dir / "grid_search_params.json"
    if resume_params is None:
        mut_path_out: str | list[str]
        if isinstance(mut_path, (list, tuple)):
            mut_path_out = [_relpath(p) for p in mut_path]
        else:
            mut_path_out = _relpath(mut_path)
        grid_params = {
            "mut_path": mut_path_out,
            "fai_path": _relpath(fai_path),
            "fasta_path": _relpath(fasta_path),
            "dnase_map_path": None if dnase_map_path is None else _relpath(dnase_map_path),
            "atac_map_path": None if atac_map_path is None else _relpath(atac_map_path),
            "dnase_bigwigs": {key: _relpath(path) for key, path in dnase_bigwigs.items()},
            "timing_bigwig": None if timing_bigwig is None else _relpath(timing_bigwig),
            "k_samples": list(k_samples),
            "n_resamples": int(n_resamples),
            "base_seed": int(base_seed),
            "track_strategies": list(track_strategies),
            "covariate_sets": covariate_sets,
            "include_trinuc": bool(include_trinuc),
            "chroms": None if chroms is None else list(chroms),
            "standardise_tracks": bool(standardise_tracks),
            "standardise_scope": str(standardise_scope),
            "pearson_score_window_bins": int(pearson_score_window_bins),
            "pearson_score_smoothing": str(pearson_score_smoothing),
            "pearson_score_smooth_param": None
            if pearson_score_smooth_param is None
            else float(pearson_score_smooth_param),
            "pearson_score_transform": str(pearson_score_transform),
            "pearson_score_zscore": bool(pearson_score_zscore),
            "pearson_score_weights": [
                float(pearson_score_weights[0]),
                float(pearson_score_weights[1]),
            ],
            "spearman_score_window_bins": int(spearman_score_window_bins),
            "spearman_score_smoothing": str(spearman_score_smoothing),
            "spearman_score_smooth_param": None
            if spearman_score_smooth_param is None
            else float(spearman_score_smooth_param),
            "spearman_score_transform": str(spearman_score_transform),
            "spearman_score_zscore": bool(spearman_score_zscore),
            "spearman_score_weights": [
                float(spearman_score_weights[0]),
                float(spearman_score_weights[1]),
            ],
            "counts_raw_bins": list(counts_raw_bins),
            "counts_gauss_bins": list(counts_gauss_bins),
            "inv_dist_gauss_bins": list(inv_dist_gauss_bins),
            "exp_decay_bins": list(exp_decay_bins),
            "exp_decay_adaptive_bins": list(exp_decay_adaptive_bins),
            "counts_gauss_sigma_grid": list(counts_gauss_sigma_grid),
            "counts_gauss_sigma_units": str(counts_gauss_sigma_units),
            "inv_dist_gauss_sigma_grid": list(inv_dist_gauss_sigma_grid),
            "inv_dist_gauss_max_distance_bp_grid": list(inv_dist_gauss_max_distance_bp_grid),
            "inv_dist_gauss_pairs": None if inv_dist_gauss_pairs is None else [
                [float(sigma), int(max_dist)] for sigma, max_dist in inv_dist_gauss_pairs
            ],
            "inv_dist_gauss_sigma_units": str(inv_dist_gauss_sigma_units),
            "exp_decay_decay_bp_grid": list(exp_decay_decay_bp_grid),
            "exp_decay_max_distance_bp_grid": list(exp_decay_max_distance_bp_grid),
            "exp_decay_adaptive_k_grid": list(exp_decay_adaptive_k_grid),
            "exp_decay_adaptive_min_bandwidth_bp_grid": list(exp_decay_adaptive_min_bandwidth_bp_grid),
            "exp_decay_adaptive_max_distance_bp_grid": list(exp_decay_adaptive_max_distance_bp_grid),
            "downsample_counts": list(downsample_grid),
            "save_per_bin": bool(save_per_bin),
            "chunksize": int(chunksize),
            "tumour_filter": None if tumour_filter is None else list(tumour_filter),
            "per_sample_count": None if per_sample_count is None else int(per_sample_count),
        }
        save_json(grid_params, grid_params_path)
        command_init_txt = out_dir / "command_init.txt"
        command_init_txt.write_text(_build_cli_command(grid_params, out_dir), encoding="utf-8")
        command_resume_txt = out_dir / "command_resume.txt"
        command_resume_txt.write_text(
            f"python -m scripts.grid_search.cli resume-experiment {_relpath(out_dir)}\n",
            encoding="utf-8",
        )

    fai = load_fai(fai_path)
    chrom_infos = list(iter_chroms(fai, chroms=chroms))
    track_combo_count = 0
    for strategy in track_strategies:
        if strategy == "counts_raw":
            combos = 1
        elif strategy == "counts_gauss":
            combos = len(counts_gauss_sigma_grid)
        elif strategy == "inv_dist_gauss":
            if inv_dist_gauss_pairs:
                combos = len(inv_dist_gauss_pairs)
            else:
                combos = len(inv_dist_gauss_sigma_grid) * len(inv_dist_gauss_max_distance_bp_grid)
        elif strategy == "exp_decay":
            combos = len(exp_decay_decay_bp_grid) * len(exp_decay_max_distance_bp_grid)
        elif strategy == "exp_decay_adaptive":
            combos = (
                    len(exp_decay_adaptive_k_grid)
                    * len(exp_decay_adaptive_min_bandwidth_bp_grid)
                    * len(exp_decay_adaptive_max_distance_bp_grid)
            )
        else:
            raise ValueError(f"Unknown track_strategy: {strategy}")
        track_combo_count += combos * len(bins_by_strategy[strategy])

    rows: List[Dict[str, Any]] = []
    correct_counts = {"true": 0, "false": 0, "none": 0}
    rf_top_feature_counts: Dict[str, int] = {}
    run_counter = len(completed_run_ids)

    max_downsample = max(
        (val for val in downsample_grid if val is not None),
        default=None,
    )
    eligible_keys: Optional[Set[str]] = None
    if max_downsample is not None:
        with timed(logger, f"Counting mutations per sample (min={max_downsample})"):
            eligible_keys = _eligible_sample_keys(
                mut_path=mut_path,
                min_mutations=max_downsample,
                chunksize=chunksize,
                tumour_filter=tumour_filter,
            )
        log_kv(logger, "eligible_samples_min_mutations", str(len(eligible_keys)))
        if not eligible_keys:
            raise ValueError(
                f"No samples found with >= {max_downsample} mutations for downsampling."
            )

    sample_plan_path = out_dir / "sample_plan.json"
    cached_plan: Optional[Dict[str, Any]] = None
    if resume:
        cached_plan = _load_sample_plan(sample_plan_path)

    def _get_sample_plan() -> Dict[str, Any]:
        nonlocal cached_plan
        if cached_plan is not None:
            return cached_plan
        plan = _prepare_non_overlapping_plan(
            mut_path=mut_path,
            k=None,
            repeats=1,
            seed=int(base_seed),
            chunksize=chunksize,
            allowed_keys=eligible_keys,
        )
        _save_sample_plan(sample_plan_path, plan, eligible_keys)
        cached_plan = plan
        return plan

    per_sample_plan: Optional[Dict[str, Any]] = None
    per_sample_keys: Optional[List[str]] = None
    if per_sample_count is not None:
        per_sample_plan = _get_sample_plan()
        per_sample_keys = list(per_sample_plan["ordered_keys"])
        if per_sample_count < 0:
            raise ValueError("per_sample_count must be >= 0.")
        if per_sample_count:
            per_sample_keys = per_sample_keys[: int(per_sample_count)]
        if not per_sample_keys:
            raise ValueError("per-sample mode found no samples to run.")

    if per_sample_keys is not None:
        per_sample_expected_runs = (
                len(per_sample_keys)
                * int(track_combo_count)
                * len(downsample_grid)
                * len(covariate_sets)
        )
        expected_runs_per_sample = int(track_combo_count) * len(downsample_grid) * len(covariate_sets)
        total_runs = per_sample_expected_runs
    else:
        total_runs = (
                len(k_samples)
                * int(n_resamples)
                * int(track_combo_count)
                * len(downsample_grid)
                * len(covariate_sets)
        )
        expected_runs_per_sample = None

    log_section(logger, "Session start")
    log_section(logger, "Inputs")
    if isinstance(mut_path, (list, tuple)):
        mut_path_display = ", ".join(_relpath(p) for p in mut_path)
    else:
        mut_path_display = _relpath(mut_path)
    log_kv(logger, "mutations_bed", mut_path_display)
    log_kv(logger, "fasta", str(fasta_path))
    log_kv(logger, "fai", str(fai_path))
    log_kv(
        logger,
        f"{accessibility_label} bigwigs",
        ", ".join(dnase_bigwigs.keys()) if dnase_bigwigs else "none",
    )
    log_kv(logger, "timing_bigwig", str(timing_bigwig) if timing_bigwig else "none")
    log_kv(logger, "tumour_filter", ",".join(tumour_filter) if tumour_filter else "none")

    log_section(logger, "Grid")
    log_kv(logger, "chroms", str(len(chrom_infos)))
    log_kv(logger, "configs", str(total_runs))

    def _sanitize_tag(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in value)
        cleaned = cleaned.strip("-")
        return cleaned or "sample"

    def _iter_sample_runs() -> Iterable[tuple[Dict[str, Any], int, List[str], pd.DataFrame, Optional[int], Optional[int]]]:
        if per_sample_keys is not None:
            for sample_index, _ in enumerate(per_sample_keys):
                sample_tag = _sanitize_tag(per_sample_keys[sample_index])
                if (
                    resume
                    and expected_runs_per_sample
                    and completed_sample_counts.get(sample_tag, 0) >= expected_runs_per_sample
                ):
                    logger.info(
                        "Skipping completed per-sample idx=%d sample=%s",
                        sample_index,
                        sample_tag,
                    )
                    continue
                with timed(logger, f"Sample selection (per-sample idx={sample_index})"):
                    chosen_samples, muts_df, slice_start, slice_end = _select_non_overlapping_samples(
                        per_sample_plan,
                        mut_path,
                        rep=sample_index,
                        k=1,
                        chunksize=chunksize,
                        tumour_filter=tumour_filter,
                    )
                log_kv(logger, "mutations_loaded", f"{len(muts_df):,}")

                sample_id = _unique_nonempty(chosen_samples)
                sample_tag = _sanitize_tag(sample_id[0]) if sample_id else f"idx{sample_index}"
                seed_samples = int(base_seed)
                sample_meta = {
                    "sample_size_k": 1,
                    "repeat": 0,
                    "seed_samples": int(seed_samples),
                    "n_selected_samples": int(len(chosen_samples)),
                    "sample_slice_start": int(slice_start),
                    "sample_slice_end": int(slice_end),
                    "sample_index": int(sample_index),
                    "sample_id": sample_id[0] if sample_id else None,
                    "sample_tag": sample_tag,
                    "sample_mode": "per_sample",
                }
                yield sample_meta, seed_samples, chosen_samples, muts_df, slice_start, slice_end
        else:
            for k in k_samples:
                non_overlap_plan: Optional[Dict[str, Any]] = None
                effective_repeats = 1 if (k is None and n_resamples > 1) else n_resamples
                if eligible_keys is not None or (effective_repeats > 1 and k is not None):
                    non_overlap_plan = _get_sample_plan()
                    if k is not None and int(effective_repeats) * int(k) > len(non_overlap_plan["ordered_keys"]):
                        raise ValueError(
                            "Non-overlapping repeats require repeats * k <= total samples "
                            f"({effective_repeats} * {k} > {len(non_overlap_plan['ordered_keys'])})."
                        )
                if k is None and n_resamples > 1:
                    logger.warning(
                        "k_samples includes 'all'; overriding n_resamples=%d to 1 for full-cohort runs.",
                        n_resamples,
                    )
                for rep in range(int(effective_repeats)):
                    slice_start = None
                    slice_end = None
                    if non_overlap_plan is not None:
                        seed_samples = base_seed
                        with timed(
                                logger,
                                f"Sample selection (k={k}, rep={rep}, seed={seed_samples}, non-overlap)",
                        ):
                            chosen_samples, muts_df, slice_start, slice_end = _select_non_overlapping_samples(
                                non_overlap_plan,
                                mut_path,
                                rep=rep,
                                k=int(k),
                                chunksize=chunksize,
                                tumour_filter=tumour_filter,
                            )
                    else:
                        seed_samples = base_seed + rep
                        plan = _get_sample_plan()
                        with timed(logger, f"Sample selection (k={k}, rep={rep}, seed={seed_samples})"):
                            chosen_samples, muts_df, slice_start, slice_end = _select_non_overlapping_samples(
                                plan,
                                mut_path,
                                rep=rep,
                                k=int(k) if k is not None else None,
                                chunksize=chunksize,
                                tumour_filter=tumour_filter,
                            )
                    log_kv(logger, "mutations_loaded", f"{len(muts_df):,}")

                    sample_meta = {
                        "sample_size_k": None if k is None else int(k),
                        "repeat": int(rep),
                        "seed_samples": int(seed_samples),
                        "n_selected_samples": int(len(chosen_samples)),
                        "sample_slice_start": int(slice_start) if slice_start is not None else None,
                        "sample_slice_end": int(slice_end) if slice_end is not None else None,
                    }
                    yield sample_meta, seed_samples, chosen_samples, muts_df, slice_start, slice_end
    for sample_meta, seed_samples, chosen_samples, muts_df, slice_start, slice_end in _iter_sample_runs():
        rep = int(sample_meta.get("repeat", 0))
        if max_downsample is not None and len(muts_df) < max_downsample:
            raise ValueError(
                "Selected samples have too few mutations for downsampling "
                f"(have {len(muts_df)}, need >= {max_downsample})."
            )

        downsample_variants: List[Tuple[Optional[int], pd.DataFrame, Dict[str, np.ndarray]]] = []
        for downsample_target in downsample_grid:
            if downsample_target is None:
                muts_df_run = muts_df
            else:
                seed_downsample = int(seed_samples) + int(downsample_target)
                muts_df_run = _downsample_mutations_df(
                    muts_df,
                    target_n=int(downsample_target),
                    seed=seed_downsample,
                )
            pos_by_chrom = mutations_to_positions_by_chrom(muts_df_run)
            downsample_variants.append((downsample_target, muts_df_run, pos_by_chrom))

            # model seed (for RF), independent but deterministic
            rf_seed = 10_000 + base_seed + rep

            for downsample_target, muts_df_run, pos_by_chrom in downsample_variants:
                log_kv(
                    logger,
                    "downsample_target",
                    "none" if downsample_target is None else str(downsample_target),
                )
                for track_strategy in track_strategies:
                    for bin_size in bins_by_strategy[track_strategy]:
                        if track_strategy == "counts_raw":
                            track_param_combos = [("raw", {})]
                        elif track_strategy == "counts_gauss":
                            track_param_combos = []
                            for sigma in counts_gauss_sigma_grid:
                                track_param_combos.append((
                                    f"cs={sigma}",
                                    {"counts_sigma": float(sigma)},
                                ))
                        elif track_strategy == "inv_dist_gauss":
                            track_param_combos = []
                            if inv_dist_gauss_pairs:
                                for idx, pair in enumerate(inv_dist_gauss_pairs):
                                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                                        raise ValueError(
                                            "inv_dist_gauss_pairs must be a sequence of (inv_sigma, max_distance_bp) pairs; "
                                            f"invalid entry at index {idx}: {pair!r}"
                                        )
                                    sigma, md = pair
                                    try:
                                        sigma_val = float(sigma)
                                    except (TypeError, ValueError) as exc:
                                        raise ValueError(
                                            "inv_dist_gauss_pairs inv_sigma must be float-like; "
                                            f"invalid entry at index {idx}: {sigma!r}"
                                        ) from exc
                                    try:
                                        md_float = float(md)
                                    except (TypeError, ValueError) as exc:
                                        raise ValueError(
                                            "inv_dist_gauss_pairs max_distance_bp must be int-like; "
                                            f"invalid entry at index {idx}: {md!r}"
                                        ) from exc
                                    if not md_float.is_integer():
                                        raise ValueError(
                                            "inv_dist_gauss_pairs max_distance_bp must be int-like; "
                                            f"invalid entry at index {idx}: {md!r}"
                                        )
                                    md_val = int(md_float)
                                    track_param_combos.append((
                                        f"is={sigma_val}_md={md_val}",
                                        {"inv_sigma": sigma_val, "max_distance_bp": md_val},
                                    ))
                            else:
                                for sigma in inv_dist_gauss_sigma_grid:
                                    for md in inv_dist_gauss_max_distance_bp_grid:
                                        track_param_combos.append((
                                            f"is={sigma}_md={md}",
                                            {"inv_sigma": float(sigma), "max_distance_bp": int(md)},
                                        ))
                        elif track_strategy == "exp_decay":
                            track_param_combos = []
                            for decay_bp in exp_decay_decay_bp_grid:
                                for md in exp_decay_max_distance_bp_grid:
                                    track_param_combos.append((
                                        f"decay={decay_bp}_md={md}",
                                        {"exp_decay_bp": float(decay_bp), "exp_max_distance_bp": int(md)},
                                    ))
                        elif track_strategy == "exp_decay_adaptive":
                            track_param_combos = []
                            for k_near in exp_decay_adaptive_k_grid:
                                for min_bw in exp_decay_adaptive_min_bandwidth_bp_grid:
                                    for md in exp_decay_adaptive_max_distance_bp_grid:
                                        track_param_combos.append((
                                            f"k={k_near}_minbw={min_bw}_md={md}",
                                            {
                                                "adaptive_k": int(k_near),
                                                "adaptive_min_bandwidth_bp": float(min_bw),
                                                "adaptive_max_distance_bp": int(md),
                                            },
                                        ))
                        else:
                            raise ValueError(f"Unknown track_strategy: {track_strategy}")

                        for param_tag, param_dict in track_param_combos:
                            for covs in covariate_sets:
                                counts_sigma_bins_run = float(default_counts_sigma_bins)
                                inv_sigma_bins_run = float(default_inv_sigma_bins)
                                max_distance_bp_run = int(default_max_distance_bp)
                                exp_decay_bp_run = float(default_exp_decay_bp)
                                exp_max_distance_bp_run = int(default_exp_max_distance_bp)
                                adaptive_k_run = int(default_adaptive_k)
                                adaptive_min_bandwidth_bp_run = float(default_adaptive_min_bandwidth_bp)
                                adaptive_max_distance_bp_run = int(default_adaptive_max_distance_bp)

                                if track_strategy == "counts_gauss":
                                    counts_sigma_bins_run = sigma_to_bins(
                                        param_dict["counts_sigma"],
                                        bin_size,
                                        counts_gauss_sigma_units,
                                    )

                                if track_strategy == "inv_dist_gauss":
                                    inv_sigma_bins_run = sigma_to_bins(
                                        param_dict["inv_sigma"],
                                        bin_size,
                                        inv_dist_gauss_sigma_units,
                                    )
                                    max_distance_bp_run = int(param_dict["max_distance_bp"])
                                if track_strategy == "exp_decay":
                                    exp_decay_bp_run = float(param_dict["exp_decay_bp"])
                                    exp_max_distance_bp_run = int(param_dict["exp_max_distance_bp"])
                                if track_strategy == "exp_decay_adaptive":
                                    adaptive_k_run = int(param_dict["adaptive_k"])
                                    adaptive_min_bandwidth_bp_run = float(param_dict["adaptive_min_bandwidth_bp"])
                                    adaptive_max_distance_bp_run = int(param_dict["adaptive_max_distance_bp"])

                                sigma_units_for_track = ""
                                if track_strategy == "counts_gauss":
                                    sigma_units_for_track = counts_gauss_sigma_units
                                elif track_strategy == "inv_dist_gauss":
                                    sigma_units_for_track = inv_dist_gauss_sigma_units
                                prefixed_params = _prefixed_track_params(
                                    track_strategy=track_strategy,
                                    bin_size=int(bin_size),
                                    counts_sigma_bins_run=float(counts_sigma_bins_run),
                                    inv_sigma_bins_run=float(inv_sigma_bins_run),
                                    max_distance_bp_run=int(max_distance_bp_run),
                                    exp_decay_bp_run=float(exp_decay_bp_run),
                                    exp_max_distance_bp_run=int(exp_max_distance_bp_run),
                                    adaptive_k_run=int(adaptive_k_run),
                                    adaptive_min_bandwidth_bp_run=float(adaptive_min_bandwidth_bp_run),
                                    adaptive_max_distance_bp_run=int(adaptive_max_distance_bp_run),
                                    sigma_units=str(sigma_units_for_track),
                                )

                                logger.info(
                                    "config bin=%d track=%s covs=%s",
                                    int(bin_size),
                                    track_strategy,
                                    "-".join(covs) if covs else "none",
                                )
                                ds_tag = "ds=none" if downsample_target is None else f"ds={downsample_target}"
                                bin_tag = f"{track_strategy}_bin={bin_size}"
                                sample_tag = sample_meta.get("sample_tag")
                                sample_tag_part = f"sample={sample_tag}" if sample_tag else None
                                if track_strategy == "counts_raw":
                                    parts = [
                                        f"k={sample_meta['sample_size_k']}",
                                        f"rep={rep}",
                                        f"seed={seed_samples}",
                                    ]
                                    if sample_tag_part:
                                        parts.append(sample_tag_part)
                                    parts.append(ds_tag)
                                    parts.append(bin_tag)
                                    parts.append(f"track={track_strategy}")
                                    parts.append(f"covs={'-'.join(covs) if covs else 'none'}")
                                    parts.append(f"tri={int(include_trinuc)}")
                                    run_id = (
                                        "__".join(parts)
                                    )
                                else:
                                    if track_strategy == "counts_gauss":
                                        unit_tag = counts_gauss_sigma_units
                                    elif track_strategy == "inv_dist_gauss":
                                        unit_tag = inv_dist_gauss_sigma_units
                                    else:
                                        unit_tag = "bp"
                                    parts = [
                                        f"k={sample_meta['sample_size_k']}",
                                        f"rep={rep}",
                                        f"seed={seed_samples}",
                                    ]
                                    if sample_tag_part:
                                        parts.append(sample_tag_part)
                                    parts.append(ds_tag)
                                    parts.append(bin_tag)
                                    parts.append(f"track={track_strategy}")
                                    parts.append(param_tag)
                                    parts.append(f"u={unit_tag}")
                                    parts.append(f"covs={'-'.join(covs) if covs else 'none'}")
                                    parts.append(f"tri={int(include_trinuc)}")
                                    run_id = (
                                        "__".join(parts)
                                    )
                                if run_id in completed_run_ids:
                                    logger.info("Skipping completed run %s", run_id)
                                    continue
                                t0 = time.perf_counter()
                                run_dir = ensure_dir(runs_dir / run_id)
                                celltypes = list(dnase_bigwigs.keys())
                                run_counter += 1
                                log_section(logger, f"Run start [{run_counter}/{total_runs}]")
                                log_kv(logger, "id", run_id)
                                log_kv(
                                    logger,
                                    "standardise",
                                    f"{standardise_scope} ({'on' if standardise_tracks else 'off'})",
                                )
                                log_kv(logger, "celltypes", ", ".join(celltypes))
                                log_kv(logger, "rf_seed", str(rf_seed))
                                run_start = time.perf_counter()

                                # save config for this run
                                cfg = {
                                    "mut_path": str(mut_path),
                                    "fai_path": str(fai_path),
                                    "fasta_path": str(fasta_path),
                                    "dnase_bigwigs": {k: str(v) for k, v in dnase_bigwigs.items()},
                                    "timing_bigwig": None if timing_bigwig is None else str(timing_bigwig),
                                    "sample_meta": sample_meta,
                                    "downsample_target": None if downsample_target is None else int(downsample_target),
                                    "mutations_pre_downsample": int(len(muts_df)),
                                    "mutations_post_downsample": int(len(muts_df_run)),
                                    "track_strategy": track_strategy,
                                    "track_param_tag": param_tag,
                                    "covariates": covs,
                                    "include_trinuc": bool(include_trinuc),
                                    "standardise_tracks": bool(standardise_tracks),
                                    "standardise_scope": str(standardise_scope),
                                    "pearson_score_window_bins": int(pearson_score_window_bins),
                                    "pearson_score_smoothing": str(pearson_score_smoothing),
                                    "pearson_score_smooth_param": None
                                    if pearson_score_smooth_param is None
                                    else float(pearson_score_smooth_param),
                                    "pearson_score_transform": str(pearson_score_transform),
                                    "pearson_score_zscore": bool(pearson_score_zscore),
                                    "pearson_score_weights": [
                                        float(pearson_score_weights[0]),
                                        float(pearson_score_weights[1]),
                                    ],
                                    "spearman_score_window_bins": int(spearman_score_window_bins),
                                    "spearman_score_smoothing": str(spearman_score_smoothing),
                                    "spearman_score_smooth_param": None
                                    if spearman_score_smooth_param is None
                                    else float(spearman_score_smooth_param),
                                    "spearman_score_transform": str(spearman_score_transform),
                                    "spearman_score_zscore": bool(spearman_score_zscore),
                                    "spearman_score_weights": [
                                        float(spearman_score_weights[0]),
                                        float(spearman_score_weights[1]),
                                    ],
                                    **prefixed_params,
                                    "counts_gauss_sigma_grid": list(map(float, counts_gauss_sigma_grid)),
                                    "inv_dist_gauss_sigma_grid": list(map(float, inv_dist_gauss_sigma_grid)),
                                    "inv_dist_gauss_max_distance_bp_grid": list(
                                        map(int, inv_dist_gauss_max_distance_bp_grid)),
                                    "inv_dist_gauss_pairs": list(inv_dist_gauss_pairs) if inv_dist_gauss_pairs else [],
                                    "exp_decay_decay_bp_grid": list(map(float, exp_decay_decay_bp_grid)),
                                    "exp_decay_max_distance_bp_grid": list(map(int, exp_decay_max_distance_bp_grid)),
                                    "exp_decay_adaptive_k_grid": list(map(int, exp_decay_adaptive_k_grid)),
                                    "exp_decay_adaptive_min_bandwidth_bp_grid": list(
                                        map(float, exp_decay_adaptive_min_bandwidth_bp_grid)),
                                    "exp_decay_adaptive_max_distance_bp_grid": list(
                                        map(int, exp_decay_adaptive_max_distance_bp_grid)),
                                    "rf_seed": int(rf_seed),
                                    "chroms": [ci.chrom for ci in chrom_infos],
                                    "tumour_filter": list(tumour_filter) if tumour_filter else [],
                                }
                                save_json(cfg, run_dir / "config.json")

                                # run per-chrom and aggregate metrics
                                chrom_summaries = []
                                per_bin_parts = []
                                for i, ci in enumerate(chrom_infos, start=1):
                                    progress_line(
                                        logger,
                                        i=i,
                                        total=len(chrom_infos),
                                        run_start=run_start,
                                        every=5,
                                    )
                                    shared = build_shared_inputs(
                                        pos_by_chrom=pos_by_chrom,
                                        chrom=ci.chrom,
                                        chrom_length=ci.length,
                                        bin_size=bin_size,
                                        track_strategy=track_strategy,
                                        counts_sigma_bins=counts_sigma_bins_run,
                                        inv_sigma_bins=inv_sigma_bins_run,
                                        max_distance_bp=max_distance_bp_run,
                                        exp_decay_bp=exp_decay_bp_run,
                                        exp_max_distance_bp=exp_max_distance_bp_run,
                                        adaptive_k=adaptive_k_run,
                                        adaptive_min_bandwidth_bp=adaptive_min_bandwidth_bp_run,
                                        adaptive_max_distance_bp=adaptive_max_distance_bp_run,
                                        covariates=covs,
                                        fasta_path=fasta_path,
                                        timing_bigwig=timing_bigwig,
                                        include_trinuc=include_trinuc,
                                    )

                                    edges = shared["edges"]
                                    centres = shared["centres"]
                                    mut_track = shared["mut_track"]
                                    cov_df = shared["cov_df"]

                                    chrom_row: Dict[str, Any] = {
                                        "chrom": ci.chrom,
                                        "track_strategy": track_strategy,
                                        "n_bins": int(len(edges) - 1),
                                        "n_mutations_chr": int(shared["mut_pos"].size),
                                        "covariates": ",".join(covs) if covs else "",
                                        "include_trinuc": bool(include_trinuc),
                                        **prefixed_params,
                                    }

                                    dnase_tracks: Dict[str, np.ndarray] = {}

                                    for celltype in celltypes:
                                        summ, dnase_track = run_one_config(
                                            muts_df=muts_df_run,
                                            chrom=ci.chrom,
                                            chrom_length=ci.length,
                                            bin_size=bin_size,
                                            track_strategy=track_strategy,
                                            counts_sigma_bins=counts_sigma_bins_run,
                                            inv_sigma_bins=inv_sigma_bins_run,
                                            max_distance_bp=max_distance_bp_run,
                                            exp_decay_bp=exp_decay_bp_run,
                                            exp_max_distance_bp=exp_max_distance_bp_run,
                                            adaptive_k=adaptive_k_run,
                                            adaptive_min_bandwidth_bp=adaptive_min_bandwidth_bp_run,
                                            adaptive_max_distance_bp=adaptive_max_distance_bp_run,
                                            dnase_bigwig=dnase_bigwigs[celltype],
                                            celltype=celltype,
                                            covariates=covs,
                                            fasta_path=fasta_path,
                                            timing_bigwig=timing_bigwig,
                                            include_trinuc=include_trinuc,
                                            rf_seed=rf_seed,
                                            pearson_score_window_bins=pearson_score_window_bins,
                                            pearson_score_smoothing=pearson_score_smoothing,
                                            pearson_score_smooth_param=pearson_score_smooth_param,
                                            pearson_score_transform=pearson_score_transform,
                                            pearson_score_zscore=pearson_score_zscore,
                                            pearson_score_weights=pearson_score_weights,
                                            spearman_score_window_bins=spearman_score_window_bins,
                                            spearman_score_smoothing=spearman_score_smoothing,
                                            spearman_score_smooth_param=spearman_score_smooth_param,
                                            spearman_score_transform=spearman_score_transform,
                                            spearman_score_zscore=spearman_score_zscore,
                                            spearman_score_weights=spearman_score_weights,
                                            standardise_tracks=standardise_tracks,
                                            standardise_scope=standardise_scope,
                                            accessibility_prefix=accessibility_prefix,
                                            shared=shared,
                                            pos_by_chrom=pos_by_chrom,
                                        )
                                        dnase_tracks[celltype] = dnase_track
                                        chrom_row[f"pearson_r_raw_{celltype}"] = summ["pearson_r_raw"]
                                        chrom_row[f"pearson_r_linear_resid_{celltype}"] = summ["pearson_r_linear_resid"]
                                        chrom_row[f"pearson_r_rf_resid_{celltype}"] = summ["pearson_r_rf_resid"]
                                        chrom_row[f"spearman_r_raw_{celltype}"] = summ["spearman_r_raw"]
                                        chrom_row[f"spearman_r_linear_resid_{celltype}"] = summ[
                                            "spearman_r_linear_resid"
                                        ]
                                        chrom_row[f"pearson_local_score_global_{celltype}"] = summ[
                                            "pearson_local_score_global"
                                        ]
                                        chrom_row[
                                            f"pearson_local_score_negative_corr_fraction_{celltype}"
                                        ] = summ["pearson_local_score_negative_corr_fraction"]
                                        chrom_row[f"spearman_local_score_global_{celltype}"] = summ[
                                            "spearman_local_score_global"
                                        ]
                                        chrom_row[
                                            f"spearman_local_score_negative_corr_fraction_{celltype}"
                                        ] = summ["spearman_local_score_negative_corr_fraction"]
                                        chrom_row[f"n_bins_valid_mut_and_{accessibility_prefix}_{celltype}"] = summ[
                                            f"n_bins_valid_mut_and_{accessibility_prefix}"
                                        ]

                                    # RF feature analysis: predict mutation track from covariates + accessibility tracks
                                    feature_names = list(cov_df.columns) + [
                                        f"{accessibility_prefix}_{ct}" for ct in celltypes
                                    ]
                                    if feature_names:
                                        dnase_feature_matrix = np.column_stack([dnase_tracks[ct] for ct in celltypes])
                                        X_full = (
                                            np.column_stack([cov_df.to_numpy(dtype=float), dnase_feature_matrix])
                                            if cov_df.shape[1]
                                            else dnase_feature_matrix
                                        )
                                        mut_track_rf = zscore_nan(mut_track) if standardise_tracks else mut_track
                                        perm_imp, sign_corr, impurity_imp, rf_r2, ridge_coef, ridge_r2 = (
                                            rf_feature_analysis(
                                                mut_track_rf,
                                                X_full,
                                                feature_names,
                                                seed=rf_seed,
                                            )
                                        )
                                        mask_rf = np.isfinite(mut_track_rf) & np.all(np.isfinite(X_full), axis=1)
                                        chrom_row["n_bins_valid_rf_features"] = int(mask_rf.sum())
                                        chrom_row["rf_r2"] = float(rf_r2)
                                        chrom_row["ridge_r2"] = float(ridge_r2)
                                        chrom_row["rf_perm_importances_json"] = json.dumps(perm_imp)
                                        chrom_row["rf_feature_sign_corr_json"] = json.dumps(sign_corr)
                                        chrom_row["rf_feature_importances_json"] = json.dumps(impurity_imp)
                                        chrom_row["ridge_coef_json"] = json.dumps(ridge_coef)
                                        acc_perm = {
                                            k: v for k, v in perm_imp.items()
                                            if k.startswith(f"{accessibility_prefix}_")
                                        }
                                        if acc_perm:
                                            top_feature = max(acc_perm.items(), key=lambda kv: kv[1])
                                            chrom_row["rf_top_celltype_feature_perm"] = top_feature[0].replace(
                                                f"{accessibility_prefix}_",
                                                "",
                                            )
                                            chrom_row["rf_top_celltype_importance_perm"] = float(top_feature[1])
                                        else:
                                            chrom_row["rf_top_celltype_feature_perm"] = None
                                            chrom_row["rf_top_celltype_importance_perm"] = float("nan")
                                    else:
                                        chrom_row["n_bins_valid_rf_features"] = 0
                                        chrom_row["rf_r2"] = float("nan")
                                        chrom_row["ridge_r2"] = float("nan")
                                        chrom_row["rf_perm_importances_json"] = json.dumps({})
                                        chrom_row["rf_feature_sign_corr_json"] = json.dumps({})
                                        chrom_row["rf_feature_importances_json"] = json.dumps({})
                                        chrom_row["ridge_coef_json"] = json.dumps({})
                                        chrom_row["rf_top_celltype_feature_perm"] = None
                                        chrom_row["rf_top_celltype_importance_perm"] = float("nan")

                                    if save_per_bin:
                                        per_bin_df = pd.DataFrame(
                                            {
                                                "chrom": ci.chrom,
                                                "bin_start": edges[:-1].astype(int),
                                                "bin_end": edges[1:].astype(int),
                                                "bin_centre": centres.astype(int),
                                                "mut_track": mut_track,
                                            }
                                        )
                                        for celltype, dnase_track in dnase_tracks.items():
                                            per_bin_df[f"{accessibility_prefix}_{celltype}"] = dnase_track
                                        per_bin_df = pd.concat([per_bin_df, cov_df], axis=1)
                                        per_bin_parts.append(per_bin_df)

                                    chrom_summaries.append(chrom_row)

                                chrom_df = pd.DataFrame(chrom_summaries)

                                # aggregate: weighted means across chroms
                                selected_sample_ids = _unique_nonempty(chosen_samples)
                                selected_tumour_types = _unique_nonempty(
                                    muts_df["Tumour"].tolist() if not muts_df.empty else []
                                )
                                correct_celltypes = (
                                    celltype_map.infer_correct_celltypes(selected_tumour_types)
                                    if celltype_map is not None
                                    else []
                                )
                                def _lower_celltype(value: Any) -> Optional[str]:
                                    if value is None:
                                        return None
                                    s = str(value).strip()
                                    if not s:
                                        return None
                                    return s.lower()

                                correct_celltypes_lower = [
                                    str(ct).strip().lower()
                                    for ct in correct_celltypes
                                    if str(ct).strip()
                                ]

                                agg = {
                                    **sample_meta,
                                    "selected_sample_ids": ",".join(selected_sample_ids),
                                    "selected_tumour_types": ",".join(selected_tumour_types),
                                    "correct_celltypes": ",".join(correct_celltypes_lower),
                                    "downsample_target": None if downsample_target is None else int(downsample_target),
                                    "mutations_pre_downsample": int(len(muts_df)),
                                    "mutations_post_downsample": int(len(muts_df_run)),
                                    "track_strategy": track_strategy,
                                    "track_param_tag": param_tag,
                                    "covariates": ",".join(covs) if covs else "",
                                    "include_trinuc": bool(include_trinuc),
                                    "n_mutations_total": int(chrom_df["n_mutations_chr"].sum()),
                                    "n_bins_total": int(chrom_df["n_bins"].sum()),
                                    "run_id": run_id,
                                    **prefixed_params,
                                }

                                raw_vals: Dict[str, float] = {}
                                lin_vals: Dict[str, float] = {}
                                rf_vals: Dict[str, float] = {}
                                spearman_raw_vals: Dict[str, float] = {}
                                spearman_lin_vals: Dict[str, float] = {}
                                pearson_score_vals: Dict[str, float] = {}
                                spearman_score_vals: Dict[str, float] = {}
                                for ct in celltypes:
                                    weights = chrom_df[
                                        f"n_bins_valid_mut_and_{accessibility_prefix}_{ct}"
                                    ].to_numpy(dtype=float)
                                    raw = chrom_df[f"pearson_r_raw_{ct}"].to_numpy(dtype=float)
                                    lin = chrom_df[f"pearson_r_linear_resid_{ct}"].to_numpy(dtype=float)
                                    rf = chrom_df[f"pearson_r_rf_resid_{ct}"].to_numpy(dtype=float)
                                    s_raw = chrom_df[f"spearman_r_raw_{ct}"].to_numpy(dtype=float)
                                    s_lin = chrom_df[f"spearman_r_linear_resid_{ct}"].to_numpy(dtype=float)
                                    pearson_score_global = chrom_df[
                                        f"pearson_local_score_global_{ct}"
                                    ].to_numpy(dtype=float)
                                    pearson_score_neg_frac = chrom_df[
                                        f"pearson_local_score_negative_corr_fraction_{ct}"
                                    ].to_numpy(dtype=float)
                                    spearman_score_global = chrom_df[
                                        f"spearman_local_score_global_{ct}"
                                    ].to_numpy(dtype=float)
                                    spearman_score_neg_frac = chrom_df[
                                        f"spearman_local_score_negative_corr_fraction_{ct}"
                                    ].to_numpy(dtype=float)

                                    agg[f"pearson_r_raw_{ct}_mean_weighted"] = weighted_mean(raw, weights)
                                    agg[f"pearson_r_linear_resid_{ct}_mean_weighted"] = weighted_mean(lin, weights)
                                    agg[f"pearson_r_rf_resid_{ct}_mean_weighted"] = weighted_mean(rf, weights)
                                    agg[f"spearman_r_raw_{ct}_mean_weighted"] = weighted_mean(s_raw, weights)
                                    agg[f"spearman_r_linear_resid_{ct}_mean_weighted"] = weighted_mean(
                                        s_lin, weights
                                    )
                                    agg[f"pearson_r_raw_{ct}_mean_unweighted"] = float(np.nanmean(raw))
                                    agg[f"pearson_r_linear_resid_{ct}_mean_unweighted"] = float(np.nanmean(lin))
                                    agg[f"pearson_r_rf_resid_{ct}_mean_unweighted"] = float(np.nanmean(rf))
                                    agg[f"spearman_r_raw_{ct}_mean_unweighted"] = float(np.nanmean(s_raw))
                                    agg[f"spearman_r_linear_resid_{ct}_mean_unweighted"] = float(
                                        np.nanmean(s_lin)
                                    )
                                    agg[
                                        f"pearson_local_score_global_{ct}_mean_weighted"
                                    ] = weighted_mean(pearson_score_global, weights)
                                    agg[
                                        f"pearson_local_score_global_{ct}_mean_unweighted"
                                    ] = _nanmean_safe(pearson_score_global)
                                    agg[
                                        f"pearson_local_score_negative_corr_fraction_{ct}_mean_weighted"
                                    ] = weighted_mean(pearson_score_neg_frac, weights)
                                    agg[
                                        f"pearson_local_score_negative_corr_fraction_{ct}_mean_unweighted"
                                    ] = _nanmean_safe(pearson_score_neg_frac)
                                    agg[
                                        f"spearman_local_score_global_{ct}_mean_weighted"
                                    ] = weighted_mean(spearman_score_global, weights)
                                    agg[
                                        f"spearman_local_score_global_{ct}_mean_unweighted"
                                    ] = _nanmean_safe(spearman_score_global)
                                    agg[
                                        f"spearman_local_score_negative_corr_fraction_{ct}_mean_weighted"
                                    ] = weighted_mean(spearman_score_neg_frac, weights)
                                    agg[
                                        f"spearman_local_score_negative_corr_fraction_{ct}_mean_unweighted"
                                    ] = _nanmean_safe(spearman_score_neg_frac)

                                    raw_vals[ct] = agg[f"pearson_r_raw_{ct}_mean_weighted"]
                                    lin_vals[ct] = agg[f"pearson_r_linear_resid_{ct}_mean_weighted"]
                                    rf_vals[ct] = agg[f"pearson_r_rf_resid_{ct}_mean_weighted"]
                                    spearman_raw_vals[ct] = agg[f"spearman_r_raw_{ct}_mean_weighted"]
                                    spearman_lin_vals[ct] = agg[f"spearman_r_linear_resid_{ct}_mean_weighted"]
                                    pearson_score_vals[ct] = agg[
                                        f"pearson_local_score_global_{ct}_mean_weighted"
                                    ]
                                    spearman_score_vals[ct] = agg[
                                        f"spearman_local_score_global_{ct}_mean_weighted"
                                    ]

                                best_raw, best_raw_val, best_raw_margin = best_and_margin(raw_vals)
                                best_lin, best_lin_val, best_lin_margin = best_and_margin(lin_vals)
                                best_rf, best_rf_val, best_rf_margin = best_and_margin(rf_vals)
                                best_s_raw, best_s_raw_val, best_s_raw_margin = best_and_margin(
                                    spearman_raw_vals
                                )
                                best_s_lin, best_s_lin_val, best_s_lin_margin = best_and_margin(
                                    spearman_lin_vals
                                )
                                best_p_score, best_p_score_val, best_p_score_margin = best_and_margin(
                                    pearson_score_vals
                                )
                                best_s_score, best_s_score_val, best_s_score_margin = best_and_margin(
                                    spearman_score_vals
                                )

                                agg["best_celltype_raw"] = _lower_celltype(best_raw)
                                agg["best_celltype_raw_value"] = float(best_raw_val)
                                agg["best_minus_second_raw"] = float(best_raw_margin)
                                agg["best_celltype_linear_resid"] = _lower_celltype(best_lin)
                                agg["best_celltype_linear_resid_value"] = float(best_lin_val)
                                agg["best_minus_second_linear_resid"] = float(best_lin_margin)
                                agg["best_celltype_rf_resid"] = _lower_celltype(best_rf)
                                agg["best_celltype_rf_resid_value"] = float(best_rf_val)
                                agg["best_minus_second_rf_resid"] = float(best_rf_margin)
                                agg["best_celltype_spearman_raw"] = _lower_celltype(best_s_raw)
                                agg["best_celltype_spearman_raw_value"] = float(best_s_raw_val)
                                agg["best_minus_second_spearman_raw"] = float(best_s_raw_margin)
                                agg["best_celltype_spearman_linear_resid"] = _lower_celltype(best_s_lin)
                                agg["best_celltype_spearman_linear_resid_value"] = float(best_s_lin_val)
                                agg["best_minus_second_spearman_linear_resid"] = float(best_s_lin_margin)
                                agg["best_celltype_pearson_local_score"] = _lower_celltype(best_p_score)
                                agg["best_celltype_pearson_local_score_value"] = float(best_p_score_val)
                                agg["best_minus_second_pearson_local_score"] = float(best_p_score_margin)
                                agg["best_celltype_spearman_local_score"] = _lower_celltype(best_s_score)
                                agg["best_celltype_spearman_local_score_value"] = float(best_s_score_val)
                                agg["best_minus_second_spearman_local_score"] = float(best_s_score_margin)

                                agg["rf_perm_importances_mean_json"] = json.dumps(
                                    aggregate_dict_column(chrom_df, "rf_perm_importances_json",
                                                          "n_bins_valid_rf_features")
                                )
                                agg["rf_feature_sign_corr_mean_json"] = json.dumps(
                                    aggregate_dict_column(chrom_df, "rf_feature_sign_corr_json",
                                                          "n_bins_valid_rf_features")
                                )
                                agg["ridge_coef_mean_json"] = json.dumps(
                                    aggregate_dict_column(chrom_df, "ridge_coef_json", "n_bins_valid_rf_features")
                                )
                                agg["rf_r2_mean_weighted"] = weighted_mean(
                                    chrom_df["rf_r2"].to_numpy(dtype=float),
                                    chrom_df["n_bins_valid_rf_features"].to_numpy(dtype=float),
                                )
                                agg["ridge_r2_mean_weighted"] = weighted_mean(
                                    chrom_df["ridge_r2"].to_numpy(dtype=float),
                                    chrom_df["n_bins_valid_rf_features"].to_numpy(dtype=float),
                                )

                                rf_perm_means = json.loads(agg["rf_perm_importances_mean_json"])
                                acc_perm = {
                                    k: v for k, v in rf_perm_means.items()
                                    if k.startswith(f"{accessibility_prefix}_")
                                }
                                if acc_perm:
                                    top_feature = max(acc_perm.items(), key=lambda kv: kv[1])
                                    agg["rf_top_celltype_feature_perm"] = top_feature[0].replace(
                                        f"{accessibility_prefix}_",
                                        "",
                                    )
                                    agg["rf_top_celltype_importance_perm"] = float(top_feature[1])
                                else:
                                    agg["rf_top_celltype_feature_perm"] = None
                                    agg["rf_top_celltype_importance_perm"] = float("nan")

                                agg.update(
                                    compute_derived_fields(
                                        agg,
                                        accessibility_prefix=accessibility_prefix,
                                    )
                                )
                                is_correct = agg.get("is_correct_pearson_local_score")
                                if is_correct is True:
                                    correct_counts["true"] += 1
                                elif is_correct is False:
                                    correct_counts["false"] += 1
                                else:
                                    correct_counts["none"] += 1
                                top_feat = agg.get("rf_top_feature_perm")
                                if top_feat:
                                    rf_top_feature_counts[top_feat] = rf_top_feature_counts.get(top_feat, 0) + 1

                                rows.append(agg)

                                out_paths = {
                                    "config": _relpath(run_dir / "config.json"),
                                    "chrom_sum": _relpath(run_dir / "chrom_summary.csv"),
                                    "per_bin": _relpath(run_dir / "per_bin.csv") if save_per_bin else "skipped",
                                    "results": _relpath(out_dir / "results.csv"),
                                }

                                summarise_run(
                                    logger,
                                    n_bins_total=int(agg["n_bins_total"]),
                                    n_mutations_total=int(agg["n_mutations_total"]),
                                    correct_celltypes=str(agg.get("correct_celltypes") or ""),
                                    metric_summaries=[
                                        (
                                            "Pearson r",
                                            agg.get("best_celltype_raw"),
                                            float(agg.get("best_celltype_raw_value", float("nan"))),
                                        ),
                                        (
                                            "Pearson r (linear covariate)"
                                            if covs
                                            else "Pearson r (no covariates)",
                                            agg.get("best_celltype_linear_resid"),
                                            float(agg.get("best_celltype_linear_resid_value", float("nan"))),
                                        ),
                                        (
                                            "Spearman r",
                                            agg.get("best_celltype_spearman_raw"),
                                            float(agg.get("best_celltype_spearman_raw_value", float("nan"))),
                                        ),
                                        (
                                            "Spearman r (linear covariate)"
                                            if covs
                                            else "Spearman r (no covariates)",
                                            agg.get("best_celltype_spearman_linear_resid"),
                                            float(
                                                agg.get(
                                                    "best_celltype_spearman_linear_resid_value", float("nan")
                                                )
                                            ),
                                        ),
                                        (
                                            "Local score (pearson, linear covariate)"
                                            if covs
                                            else "Local score (pearson, no covariates)",
                                            agg.get("best_celltype_pearson_local_score"),
                                            float(
                                                agg.get(
                                                    "best_celltype_pearson_local_score_value", float("nan")
                                                )
                                            ),
                                        ),
                                        (
                                            "Local score (spearman, linear covariate)"
                                            if covs
                                            else "Local score (spearman, no covariates)",
                                            agg.get("best_celltype_spearman_local_score"),
                                            float(
                                                agg.get(
                                                    "best_celltype_spearman_local_score_value", float("nan")
                                                )
                                            ),
                                        ),
                                        (
                                            "RF (non-linear covariate)" if covs else "RF (no covariates)",
                                            agg.get("best_celltype_rf_resid"),
                                            float(agg.get("best_celltype_rf_resid_value", float("nan"))),
                                        ),
                                    ],
                                    out_paths=out_paths,
                                )

                                # save run summaries
                                save_df(chrom_df, run_dir / "chrom_summary.csv")

                                if save_per_bin and per_bin_parts:
                                    per_bin_df = pd.concat(per_bin_parts, ignore_index=True)
                                    save_df(per_bin_df, run_dir / "per_bin.csv")

                                results_columns = _append_results_row(
                                    results_path,
                                    agg,
                                    results_columns,
                                )
                                completed_run_ids.add(str(run_id))
                                elapsed = time.perf_counter() - t0
                                logger.info("Run end %s (%s)", run_id, f"{elapsed:.1f}s")

    results = pd.DataFrame(rows)
    if existing_results_df is not None:
        frames = [df for df in (existing_results_df, results) if df is not None and not df.empty]
        if not frames:
            results = results
        elif len(frames) == 1:
            results = frames[0].reset_index(drop=True)
        else:
            results = pd.concat(frames, ignore_index=True)
    if verbose and rows:
        logger.info(
            "is_correct_pearson_local_score counts: true=%d false=%d none=%d",
            correct_counts["true"],
            correct_counts["false"],
            correct_counts["none"],
        )
        if rf_top_feature_counts:
            top_feat = max(rf_top_feature_counts.items(), key=lambda kv: (kv[1], kv[0]))
            logger.info("rf_top_feature_perm most frequent: %s (%d)", top_feat[0], top_feat[1])
    return results


def resume_experiment(out_dir: str | Path) -> pd.DataFrame:
    out_dir_path = Path(out_dir)
    grid_params_path = out_dir_path / "grid_search_params.json"
    if not grid_params_path.exists():
        raise FileNotFoundError(f"grid_search_params.json not found: {grid_params_path}")
    params = json.loads(grid_params_path.read_text(encoding="utf-8"))
    mut_path = params.get("mut_path")
    if mut_path is None:
        raise ValueError("grid_search_params.json missing mut_path")
    fai_path = params.get("fai_path")
    fasta_path = params.get("fasta_path")
    if fai_path is None or fasta_path is None:
        raise ValueError("grid_search_params.json missing fai_path or fasta_path")
    base_seed = params.get("base_seed")
    if base_seed is None:
        raise ValueError("grid_search_params.json missing base_seed")
    track_strategies = params.get("track_strategies")
    if track_strategies is None:
        raise ValueError("grid_search_params.json missing track_strategies")
    covariate_sets = params.get("covariate_sets")
    if covariate_sets is None:
        raise ValueError("grid_search_params.json missing covariate_sets")
    downsample_counts = params.get("downsample_counts", ())
    if isinstance(downsample_counts, (list, tuple)) and any(val is None for val in downsample_counts):
        downsample_counts = "none"
    return run_grid_experiment(
        mut_path=mut_path,
        fai_path=fai_path,
        fasta_path=fasta_path,
        dnase_map_path=params.get("dnase_map_path"),
        atac_map_path=params.get("atac_map_path"),
        dnase_bigwigs=params.get("dnase_bigwigs"),
        timing_bigwig=params.get("timing_bigwig"),
        k_samples=params.get("k_samples"),
        n_resamples=params.get("n_resamples"),
        base_seed=base_seed,
        track_strategies=track_strategies,
        covariate_sets=covariate_sets,
        include_trinuc=params.get("include_trinuc", False),
        chroms=params.get("chroms"),
        standardise_tracks=params.get("standardise_tracks", True),
        standardise_scope=params.get("standardise_scope", "per_chrom"),
        pearson_score_window_bins=params.get("pearson_score_window_bins", 1),
        pearson_score_smoothing=params.get("pearson_score_smoothing", "none"),
        pearson_score_smooth_param=params.get("pearson_score_smooth_param"),
        pearson_score_transform=params.get("pearson_score_transform", "none"),
        pearson_score_zscore=params.get("pearson_score_zscore", False),
        pearson_score_weights=tuple(params.get("pearson_score_weights", (0.7, 0.3))),
        spearman_score_window_bins=params.get("spearman_score_window_bins", 1),
        spearman_score_smoothing=params.get("spearman_score_smoothing", "none"),
        spearman_score_smooth_param=params.get("spearman_score_smooth_param"),
        spearman_score_transform=params.get("spearman_score_transform", "none"),
        spearman_score_zscore=params.get("spearman_score_zscore", False),
        spearman_score_weights=tuple(params.get("spearman_score_weights", (0.7, 0.3))),
        counts_raw_bins=params.get("counts_raw_bins", ()),
        counts_gauss_bins=params.get("counts_gauss_bins", ()),
        inv_dist_gauss_bins=params.get("inv_dist_gauss_bins", ()),
        exp_decay_bins=params.get("exp_decay_bins", ()),
        exp_decay_adaptive_bins=params.get("exp_decay_adaptive_bins", ()),
        counts_gauss_sigma_grid=params.get("counts_gauss_sigma_grid", ()),
        counts_gauss_sigma_units=params.get("counts_gauss_sigma_units", "bins"),
        inv_dist_gauss_sigma_grid=params.get("inv_dist_gauss_sigma_grid", ()),
        inv_dist_gauss_max_distance_bp_grid=params.get("inv_dist_gauss_max_distance_bp_grid", ()),
        inv_dist_gauss_pairs=params.get("inv_dist_gauss_pairs"),
        inv_dist_gauss_sigma_units=params.get("inv_dist_gauss_sigma_units", "bins"),
        exp_decay_decay_bp_grid=params.get("exp_decay_decay_bp_grid", ()),
        exp_decay_max_distance_bp_grid=params.get("exp_decay_max_distance_bp_grid", ()),
        exp_decay_adaptive_k_grid=params.get("exp_decay_adaptive_k_grid", ()),
        exp_decay_adaptive_min_bandwidth_bp_grid=params.get("exp_decay_adaptive_min_bandwidth_bp_grid", ()),
        exp_decay_adaptive_max_distance_bp_grid=params.get("exp_decay_adaptive_max_distance_bp_grid", ()),
        downsample_counts=downsample_counts,
        save_per_bin=params.get("save_per_bin", True),
        chunksize=params.get("chunksize", 250_000),
        tumour_filter=params.get("tumour_filter"),
        per_sample_count=params.get("per_sample_count"),
        out_dir=out_dir_path,
        resume=True,
    )
