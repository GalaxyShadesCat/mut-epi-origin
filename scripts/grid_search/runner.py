"""Grid search runner for mutation vs DNase experiments."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
from scripts.sample_selector import compile_k_samples
from scripts.scores import compute_local_scores
from scripts.stats_utils import weighted_mean, zscore_nan
from scripts.targets import dnase_mean_per_bin


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
        score_window_bins: int,
        score_corr_type: str,
        score_smoothing: str,
        score_smooth_param: float | int | None,
        score_transform: str,
        score_zscore: bool,
        score_weights: Tuple[float, float],
        # standardisation
        standardise_tracks: bool = True,
        standardise_scope: str = "per_chrom",
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

    r_rf = float("nan")
    if X.shape[1]:
        dnase_resid_rf = rf_residualise(dnase_track_corr, X, seed=rf_seed)
        r_rf = pearsonr_nan(mut_track_corr, dnase_resid_rf)

    score_res = compute_local_scores(
        mut_track_corr,
        dnase_track_corr,
        w=int(score_window_bins),
        corr_type=str(score_corr_type),
        smoothing=str(score_smoothing),
        smooth_param=score_smooth_param,
        transform=str(score_transform),
        zscore=bool(score_zscore),
        weights=score_weights,
    )

    mask_valid = np.isfinite(mut_track_corr) & np.isfinite(dnase_track_corr)
    summary = {
        "celltype": celltype,
        "pearson_r_raw": float(r_raw),
        "pearson_r_linear_resid": float(r_lin),
        "pearson_r_rf_resid": float(r_rf),
        "n_bins_valid_mut_and_dnase": int(mask_valid.sum()),
        "local_score_global": float(score_res.global_score),
        "local_score_negative_corr_fraction": float(score_res.negative_corr_fraction),
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
        dnase_map_path: str | Path = DEFAULT_MAP_PATH,
        celltype_map: Optional[DnaseCellTypeMap] = None,
        timing_bigwig: Optional[str | Path],
        sample_sizes: List[int | None],
        repeats: int,
        base_seed: int,
        track_strategies: List[str],
        covariate_sets: List[List[str]],
        include_trinuc: bool = False,
        chroms: Optional[List[str]] = None,
        standardise_tracks: bool = True,
        standardise_scope: str = "per_chrom",
        verbose: bool = False,
        # local score settings
        score_window_bins: int = 1,
        score_corr_type: str = "pearson",
        score_smoothing: str = "none",
        score_smooth_param: float | int | None = None,
        score_transform: str = "none",
        score_zscore: bool = False,
        score_weights: Tuple[float, float] = (0.7, 0.3),
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
    logger = setup_rich_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        logger_name="mut_vs_dnase",
        force=True,
    )

    if standardise_tracks and standardise_scope != "per_chrom":
        raise ValueError(f"Unsupported standardise_scope: {standardise_scope}")

    if dnase_bigwigs is None:
        dnase_bigwigs, celltype_map = _load_dnase_map_path(Path(dnase_map_path))
    if not dnase_bigwigs:
        raise ValueError("dnase_bigwigs must be a non-empty mapping of celltype->bigWig path.")

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
    if int(score_window_bins) < 1:
        raise ValueError("score_window_bins must be >= 1.")
    if len(score_weights) != 2:
        raise ValueError("score_weights must be a length-2 tuple (shape, slope).")

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
            raise ValueError("dnase_bigwigs contains an empty cell type key.")
        bw_path = Path(path)
        if not bw_path.exists():
            raise FileNotFoundError(f"DNase bigWig not found for '{celltype}': {bw_path}")
        chrom_lengths = get_bigwig_chrom_lengths(bw_path)
        if not chrom_lengths:
            raise ValueError(f"DNase bigWig has no contigs: {bw_path}")
        resolver = ContigResolver(fasta_contigs=None, bigwig_contigs=list(chrom_lengths.keys()))
        if not resolver.has_canonical_in_bigwig("chr1"):
            raise ValueError(
                f"DNase bigWig missing canonical chr1 (or alias) for '{celltype}'. "
                f"Example contigs: {', '.join(sorted(chrom_lengths.keys())[:10]) or 'none'}."
            )
        normalised_dnase[celltype] = bw_path
    dnase_bigwigs = normalised_dnase

    out_dir_path = Path(out_dir)
    if not out_dir_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        out_dir_path = project_root / out_dir_path
    out_dir = ensure_dir(out_dir_path)
    command_txt = out_dir / "command.txt"
    command_txt.write_text(
        "\n".join(
            [
                "run_grid_experiment invoked via Python API",
                f"timestamp: {datetime.utcnow().isoformat(timespec='seconds')}Z",
                f"cwd: {Path.cwd()}",
                f"out_dir: {out_dir}",
                "params: see grid_search_params.json",
            ]
        ),
        encoding="utf-8",
    )
    runs_dir = ensure_dir(out_dir / "runs")
    results_path = out_dir / "results.csv"
    results_columns = _build_results_columns(list(dnase_bigwigs.keys()), track_strategies)
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

    readme_path = out_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Outputs\n"
            "-------\n"
            "results.csv: one row per run configuration with aggregated metrics.\n"
            "grid_search_params.json: parameters used to build this grid search.\n"
            "runs/<run_id>/config.json: configuration used for that run.\n"
            "runs/<run_id>/chrom_summary.csv: per-chromosome metrics (per cell type).\n"
            "runs/<run_id>/per_bin.csv: per-bin tracks (optional; includes covariates + dnase_*).\n"
            "\n"
            "Key fields\n"
            "----------\n"
            "best_celltype_*: winners per metric (most negative correlation).\n"
            "best_celltype_*_value: correlation value for that best cell type.\n"
            "best_minus_second_*: margin between best and second-best (larger is clearer).\n"
            "rf_perm_importances_mean_json: mean permutation importances across chroms.\n"
            "rf_feature_sign_corr_mean_json: mean feature sign correlations across chroms.\n"
            "ridge_coef_mean_json: mean ridge coefficients across chroms.\n",
            encoding="utf-8",
        )

    grid_params_path = out_dir / "grid_search_params.json"
    mut_path_out: str | list[str]
    if isinstance(mut_path, (list, tuple)):
        mut_path_out = [str(p) for p in mut_path]
    else:
        mut_path_out = str(mut_path)
    grid_params = {
        "mut_path": mut_path_out,
        "fai_path": str(fai_path),
        "fasta_path": str(fasta_path),
        "dnase_bigwigs": {key: str(path) for key, path in dnase_bigwigs.items()},
        "timing_bigwig": None if timing_bigwig is None else str(timing_bigwig),
        "sample_sizes": list(sample_sizes),
        "repeats": int(repeats),
        "base_seed": int(base_seed),
        "track_strategies": list(track_strategies),
        "covariate_sets": covariate_sets,
        "include_trinuc": bool(include_trinuc),
        "chroms": None if chroms is None else list(chroms),
        "standardise_tracks": bool(standardise_tracks),
        "standardise_scope": str(standardise_scope),
        "score_window_bins": int(score_window_bins),
        "score_corr_type": str(score_corr_type),
        "score_smoothing": str(score_smoothing),
        "score_smooth_param": None if score_smooth_param is None else float(score_smooth_param),
        "score_transform": str(score_transform),
        "score_zscore": bool(score_zscore),
        "score_weights": [float(score_weights[0]), float(score_weights[1])],
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
    }
    save_json(grid_params, grid_params_path)

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

    total_runs = (
            len(sample_sizes)
            * int(repeats)
            * int(track_combo_count)
            * len(downsample_grid)
            * len(covariate_sets)
    )

    log_section(logger, "Session start")
    log_section(logger, "Inputs")
    log_kv(logger, "mutations_bed", str(mut_path))
    log_kv(logger, "fasta", str(fasta_path))
    log_kv(logger, "fai", str(fai_path))
    log_kv(logger, "dnase_bigwigs", ", ".join(dnase_bigwigs.keys()) if dnase_bigwigs else "none")
    log_kv(logger, "timing_bigwig", str(timing_bigwig) if timing_bigwig else "none")
    log_kv(logger, "tumour_filter", ",".join(tumour_filter) if tumour_filter else "none")

    log_section(logger, "Grid")
    log_kv(logger, "chroms", str(len(chrom_infos)))
    log_kv(logger, "configs", str(total_runs))

    rows: List[Dict[str, Any]] = []
    correct_counts = {"true": 0, "false": 0, "none": 0}
    rf_top_feature_counts: Dict[str, int] = {}

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

    for k in sample_sizes:
        non_overlap_plan: Optional[Dict[str, Any]] = None
        effective_repeats = 1 if (k is None and repeats > 1) else repeats
        if eligible_keys is not None or (effective_repeats > 1 and k is not None):
            non_overlap_plan = _prepare_non_overlapping_plan(
                mut_path=mut_path,
                k=None if k is None else int(k),
                repeats=int(effective_repeats),
                seed=int(base_seed),
                chunksize=chunksize,
                allowed_keys=eligible_keys,
            )
        if k is None and repeats > 1:
            logger.warning(
                "sample_sizes includes 'all'; overriding repeats=%d to 1 for full-cohort runs.",
                repeats,
            )
        for rep in range(effective_repeats):
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
                log_kv(logger, "sample_slice", f"{slice_start}:{slice_end}")
            else:
                seed_samples = base_seed + rep
                with timed(logger, f"Sample selection (k={k}, rep={rep}, seed={seed_samples})"):
                    chosen_samples, muts_df = compile_k_samples(
                        mut_path,
                        k=k,
                        seed=seed_samples,
                        chunksize=chunksize,
                        tumour_filter=tumour_filter,
                    )
            log_kv(logger, "selected_samples", str(len(chosen_samples)))
            log_kv(logger, "mutations_loaded", f"{len(muts_df):,}")
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

            # record sample selection metadata
            sample_meta = {
                "sample_size_k": None if k is None else int(k),
                "repeat": int(rep),
                "seed_samples": int(seed_samples),
                "n_selected_samples": int(len(chosen_samples)),
                "sample_slice_start": None if non_overlap_plan is None else int(slice_start),
                "sample_slice_end": None if non_overlap_plan is None else int(slice_end),
            }

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
                                if track_strategy == "counts_raw":
                                    run_id = (
                                        f"k={sample_meta['sample_size_k']}_rep={rep}_seed={seed_samples}"
                                        f"_{ds_tag}"
                                        f"_{bin_tag}_track={track_strategy}_covs={'-'.join(covs) if covs else 'none'}"
                                        f"_tri={int(include_trinuc)}"
                                    )
                                else:
                                    if track_strategy == "counts_gauss":
                                        unit_tag = counts_gauss_sigma_units
                                    elif track_strategy == "inv_dist_gauss":
                                        unit_tag = inv_dist_gauss_sigma_units
                                    else:
                                        unit_tag = "bp"
                                    run_id = (
                                        f"k={sample_meta['sample_size_k']}_rep={rep}_seed={seed_samples}"
                                        f"_{ds_tag}"
                                        f"_{bin_tag}_track={track_strategy}_{param_tag}_u={unit_tag}"
                                        f"_covs={'-'.join(covs) if covs else 'none'}"
                                        f"_tri={int(include_trinuc)}"
                                    )
                                if run_id in completed_run_ids:
                                    logger.info("Skipping completed run %s", run_id)
                                    continue
                                t0 = time.perf_counter()
                                run_dir = ensure_dir(runs_dir / run_id)
                                celltypes = list(dnase_bigwigs.keys())
                                log_section(logger, f"Run start  [{len(rows) + 1}/{total_runs}]")
                                log_kv(logger, "id", run_id.replace("_", " "))
                                log_kv(
                                    logger,
                                    "standardise",
                                    f"{standardise_scope} ({'on' if standardise_tracks else 'off'})",
                                )
                                log_kv(logger, "rf_seed", str(rf_seed))
                                log_kv(logger, "out_dir", str(run_dir))
                                log_kv(logger, "celltypes", ", ".join(celltypes))
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
                                    "score_window_bins": int(score_window_bins),
                                    "score_corr_type": str(score_corr_type),
                                    "score_smoothing": str(score_smoothing),
                                    "score_smooth_param": None if score_smooth_param is None else float(
                                        score_smooth_param),
                                    "score_transform": str(score_transform),
                                    "score_zscore": bool(score_zscore),
                                    "score_weights": [float(score_weights[0]), float(score_weights[1])],
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
                                            score_window_bins=score_window_bins,
                                            score_corr_type=score_corr_type,
                                            score_smoothing=score_smoothing,
                                            score_smooth_param=score_smooth_param,
                                            score_transform=score_transform,
                                            score_zscore=score_zscore,
                                            score_weights=score_weights,
                                            standardise_tracks=standardise_tracks,
                                            standardise_scope=standardise_scope,
                                            shared=shared,
                                            pos_by_chrom=pos_by_chrom,
                                        )
                                        dnase_tracks[celltype] = dnase_track
                                        chrom_row[f"pearson_r_raw_{celltype}"] = summ["pearson_r_raw"]
                                        chrom_row[f"pearson_r_linear_resid_{celltype}"] = summ["pearson_r_linear_resid"]
                                        chrom_row[f"pearson_r_rf_resid_{celltype}"] = summ["pearson_r_rf_resid"]
                                        chrom_row[f"local_score_global_{celltype}"] = summ["local_score_global"]
                                        chrom_row[
                                            f"local_score_negative_corr_fraction_{celltype}"
                                        ] = summ["local_score_negative_corr_fraction"]
                                        chrom_row[f"n_bins_valid_mut_and_dnase_{celltype}"] = summ[
                                            "n_bins_valid_mut_and_dnase"
                                        ]

                                    # RF feature analysis: predict mutation track from covariates + DNase tracks
                                    feature_names = list(cov_df.columns) + [f"dnase_{ct}" for ct in celltypes]
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
                                        dnase_perm = {k: v for k, v in perm_imp.items() if k.startswith("dnase_")}
                                        if dnase_perm:
                                            top_feature = max(dnase_perm.items(), key=lambda kv: kv[1])
                                            chrom_row["rf_top_celltype_feature_perm"] = top_feature[0].replace("dnase_",
                                                                                                               "")
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
                                            per_bin_df[f"dnase_{celltype}"] = dnase_track
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
                                score_vals: Dict[str, float] = {}
                                for ct in celltypes:
                                    weights = chrom_df[f"n_bins_valid_mut_and_dnase_{ct}"].to_numpy(dtype=float)
                                    raw = chrom_df[f"pearson_r_raw_{ct}"].to_numpy(dtype=float)
                                    lin = chrom_df[f"pearson_r_linear_resid_{ct}"].to_numpy(dtype=float)
                                    rf = chrom_df[f"pearson_r_rf_resid_{ct}"].to_numpy(dtype=float)
                                    score_global = chrom_df[f"local_score_global_{ct}"].to_numpy(dtype=float)
                                    score_neg_frac = chrom_df[
                                        f"local_score_negative_corr_fraction_{ct}"
                                    ].to_numpy(dtype=float)

                                    agg[f"pearson_r_raw_{ct}_mean_weighted"] = weighted_mean(raw, weights)
                                    agg[f"pearson_r_linear_resid_{ct}_mean_weighted"] = weighted_mean(lin, weights)
                                    agg[f"pearson_r_rf_resid_{ct}_mean_weighted"] = weighted_mean(rf, weights)
                                    agg[f"pearson_r_raw_{ct}_mean_unweighted"] = float(np.nanmean(raw))
                                    agg[f"pearson_r_linear_resid_{ct}_mean_unweighted"] = float(np.nanmean(lin))
                                    agg[f"pearson_r_rf_resid_{ct}_mean_unweighted"] = float(np.nanmean(rf))
                                    agg[f"local_score_global_{ct}_mean_weighted"] = weighted_mean(score_global, weights)
                                    agg[f"local_score_global_{ct}_mean_unweighted"] = float(np.nanmean(score_global))
                                    agg[
                                        f"local_score_negative_corr_fraction_{ct}_mean_weighted"
                                    ] = weighted_mean(score_neg_frac, weights)
                                    agg[
                                        f"local_score_negative_corr_fraction_{ct}_mean_unweighted"
                                    ] = float(np.nanmean(score_neg_frac))

                                    raw_vals[ct] = agg[f"pearson_r_raw_{ct}_mean_weighted"]
                                    lin_vals[ct] = agg[f"pearson_r_linear_resid_{ct}_mean_weighted"]
                                    rf_vals[ct] = agg[f"pearson_r_rf_resid_{ct}_mean_weighted"]
                                    score_vals[ct] = agg[f"local_score_global_{ct}_mean_weighted"]

                                best_raw, best_raw_val, best_raw_margin = best_and_margin(raw_vals)
                                best_lin, best_lin_val, best_lin_margin = best_and_margin(lin_vals)
                                best_rf, best_rf_val, best_rf_margin = best_and_margin(rf_vals)
                                best_score, best_score_val, best_score_margin = best_and_margin(score_vals)

                                agg["best_celltype_raw"] = _lower_celltype(best_raw)
                                agg["best_celltype_raw_value"] = float(best_raw_val)
                                agg["best_minus_second_raw"] = float(best_raw_margin)
                                agg["best_celltype_linear_resid"] = _lower_celltype(best_lin)
                                agg["best_celltype_linear_resid_value"] = float(best_lin_val)
                                agg["best_minus_second_linear_resid"] = float(best_lin_margin)
                                agg["best_celltype_rf_resid"] = _lower_celltype(best_rf)
                                agg["best_celltype_rf_resid_value"] = float(best_rf_val)
                                agg["best_minus_second_rf_resid"] = float(best_rf_margin)
                                agg["best_celltype_local_score"] = _lower_celltype(best_score)
                                agg["best_celltype_local_score_value"] = float(best_score_val)
                                agg["best_minus_second_local_score"] = float(best_score_margin)

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
                                dnase_perm = {k: v for k, v in rf_perm_means.items() if k.startswith("dnase_")}
                                if dnase_perm:
                                    top_feature = max(dnase_perm.items(), key=lambda kv: kv[1])
                                    agg["rf_top_celltype_feature_perm"] = top_feature[0].replace("dnase_", "")
                                    agg["rf_top_celltype_importance_perm"] = float(top_feature[1])
                                else:
                                    agg["rf_top_celltype_feature_perm"] = None
                                    agg["rf_top_celltype_importance_perm"] = float("nan")

                                agg.update(compute_derived_fields(agg))
                                is_correct = agg.get("is_correct_local_score")
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
                                    "config": str(run_dir / "config.json"),
                                    "chrom_sum": str(run_dir / "chrom_summary.csv"),
                                    "per_bin": str(run_dir / "per_bin.csv") if save_per_bin else "skipped",
                                    "results": str(out_dir / "results.csv"),
                                }

                                summarise_run(
                                    logger,
                                    n_bins_total=int(agg["n_bins_total"]),
                                    n_mutations_total=int(agg["n_mutations_total"]),
                                    best_celltype=agg.get("best_celltype_rf_resid"),
                                    best_value=float(agg.get("best_celltype_rf_resid_value", float("nan"))),
                                    margin=float(agg.get("best_minus_second_rf_resid", float("nan"))),
                                    rf_r2=float(agg.get("rf_r2_mean_weighted", float("nan"))),
                                    ridge_r2=float(agg.get("ridge_r2_mean_weighted", float("nan"))),
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
                                logger.info("saved run outputs %s", str(run_dir))
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
            "is_correct_local_score counts: true=%d false=%d none=%d",
            correct_counts["true"],
            correct_counts["false"],
            correct_counts["none"],
        )
        if rf_top_feature_counts:
            top_feat = max(rf_top_feature_counts.items(), key=lambda kv: (kv[1], kv[0]))
            logger.info("rf_top_feature_perm most frequent: %s (%d)", top_feat[0], top_feat[1])
    return results
