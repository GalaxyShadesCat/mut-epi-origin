"""
pipeline_mut_vs_accessibility.py

Run a grid of experiments to evaluate negative association between mutation tracks
and DNase accessibility across multiple cell types, adjusting for non-epigenomic
covariates and replication timing.

Core outputs
------------
- A tidy results table: one row per run configuration
- Optional per-run per-bin table for detailed inspection in notebooks

Track strategies (your three tracks)
------------------------------------
- counts_raw
- counts_gauss
- inv_dist_gauss

Evaluation metrics
------------------
- pearson_r_raw: corr(mutation_track, dnase_target)
- pearson_r_linear_resid: corr(resid_mut, resid_dnase) after linear residualisation on covariates
- pearson_r_rf_resid: corr(mutation_track, resid_dnase) where resid_dnase is from RF(covariates)

Notes
-----
- DNase is NOT treated as a covariate for correlation comparisons. It is the target.
- Replication timing is treated as a covariate (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd

from scripts.bigwig_utils import get_bigwig_chrom_lengths
from scripts.contigs import ContigResolver
from scripts.covariates import (
    gc_fraction_per_bin,
    cpg_frequency_per_bin,
    trinuc_frequency_per_bin,
    bigwig_mean_per_bin,
)
from scripts.genome_bins import load_fai, build_bins, iter_chroms
from scripts.io_utils import ensure_dir, save_json, save_df
from scripts.logging_utils import (
    setup_rich_logging,
    log_section,
    log_kv,
    timed,
    progress_line,
    summarise_run,
)
from scripts.sample_subset_compiler import compile_k_samples
from scripts.stats_utils import zscore_nan, weighted_mean
from scripts.targets import dnase_mean_per_bin
from scripts.tracks import TRACK_REGISTRY

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

try:
    from sklearn.inspection import permutation_importance
except ImportError:
    permutation_importance = None

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None


# -------------------------
# Stats helpers
# -------------------------
def pearsonr_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    # Avoid importing scipy; do it directly
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = np.sqrt((xx ** 2).sum()) * np.sqrt((yy ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((xx * yy).sum() / denom)


def linear_residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualise y on X using the least squares with intercept.
    Rows with NaNs in y or X are ignored; residuals returned with NaN where dropped.
    """
    n = len(y)
    resid = np.full(n, np.nan, dtype=float)
    if X.size == 0:
        return y.copy()

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 5:
        return resid

    yy = y[mask]
    XX = X[mask]

    # add intercept
    A = np.column_stack([np.ones(len(yy)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    yhat = A @ beta
    resid[mask] = yy - yhat
    return resid


def rf_residualise(y: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    """
    Fit RandomForestRegressor: y ~ X, return residuals.
    """
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn is required for RF residualisation. Install with: pip install scikit-learn")

    n = len(y)
    resid = np.full(n, np.nan, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return resid

    yy = y[mask]
    XX = X[mask]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    rf.fit(XX, yy)
    yhat = rf.predict(XX)
    resid[mask] = yy - yhat
    return resid


def standardise_matrix(X: np.ndarray) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=float)
    for i in range(X.shape[1]):
        out[:, i] = zscore_nan(X[:, i])
    return out


def rf_feature_analysis(
        y: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        seed: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, Dict[str, float], float]:
    """
    Fit RF for y ~ X and return (perm_importances, sign_corr, impurity_importances, rf_r2, ridge_coef, ridge_r2).
    """
    if RandomForestRegressor is None or permutation_importance is None:
        raise ImportError("scikit-learn is required for RF models. Install with: pip install scikit-learn")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names must align with X columns")

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return {}, {}, {}, float("nan"), {}, float("nan")

    yy = y[mask]
    XX = X[mask]

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    rf.fit(XX, yy)
    rf_r2 = float(rf.score(XX, yy))
    impurity_importances = {name: float(val) for name, val in zip(feature_names, rf.feature_importances_)}

    perm = permutation_importance(rf, XX, yy, n_repeats=10, random_state=seed, n_jobs=-1)
    perm_importances = {name: float(val) for name, val in zip(feature_names, perm.importances_mean)}

    sign_corr = {name: float(pearsonr_nan(XX[:, i], yy)) for i, name in enumerate(feature_names)}

    ridge_coef: Dict[str, float] = {}
    ridge_r2 = float("nan")
    if Ridge is not None:
        Xz = standardise_matrix(XX)
        yz = zscore_nan(yy)
        mask2 = np.isfinite(yz) & np.all(np.isfinite(Xz), axis=1)
        if mask2.sum() >= 20:
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xz[mask2], yz[mask2])
            ridge_coef = {name: float(val) for name, val in zip(feature_names, ridge.coef_)}
            ridge_r2 = float(ridge.score(Xz[mask2], yz[mask2]))

    return perm_importances, sign_corr, impurity_importances, rf_r2, ridge_coef, ridge_r2


def best_and_margin(values: Dict[str, float]) -> Tuple[Optional[str], float, float]:
    valid = [(k, v) for k, v in values.items() if np.isfinite(v)]
    if not valid:
        return None, float("nan"), float("nan")
    valid.sort(key=lambda kv: kv[1])
    best_k, best_v = valid[0]
    if len(valid) < 2:
        return best_k, float(best_v), float("nan")
    second_v = valid[1][1]
    margin = float(second_v - best_v)
    return best_k, float(best_v), margin


def aggregate_dict_column(df: pd.DataFrame, col: str, weight_col: str) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for raw, w in zip(df[col].fillna("{}"), df[weight_col].fillna(0)):
        if not np.isfinite(w) or w <= 0:
            continue
        try:
            d = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            d = {}
        for k, v in d.items():
            if not np.isfinite(v):
                continue
            sums[k] = sums.get(k, 0.0) + float(v) * float(w)
            weights[k] = weights.get(k, 0.0) + float(w)
    return {k: sums[k] / weights[k] for k in sums if weights.get(k, 0.0) > 0}


# -------------------------
# Feature building
# -------------------------
def build_covariate_matrix(
        covariates: List[str],
        fasta_path: str | Path,
        chrom: str,
        bin_edges: np.ndarray,
        timing_bigwig: Optional[str | Path] = None,
        include_trinuc: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns (cov_df, X_matrix).
    covariates: subset of {"gc", "cpg", "timing"} plus optional "trinuc".
    """
    cols: Dict[str, np.ndarray] = {}

    if "gc" in covariates:
        cols["gc_fraction"] = gc_fraction_per_bin(fasta_path, chrom, bin_edges)
    if "cpg" in covariates:
        cols["cpg_per_bp"] = cpg_frequency_per_bin(fasta_path, chrom, bin_edges)
    if "timing" in covariates:
        if timing_bigwig is None:
            raise ValueError("timing covariate requested but timing_bigwig is None")
        cols["timing_mean"] = bigwig_mean_per_bin(timing_bigwig, chrom, bin_edges)

    if include_trinuc or ("trinuc" in covariates):
        tri = trinuc_frequency_per_bin(fasta_path, chrom, bin_edges)
        for k, v in tri.items():
            cols[f"tri_{k}"] = v

    cov_df = pd.DataFrame(cols)
    X = cov_df.to_numpy(dtype=float) if len(cov_df.columns) else np.zeros((len(bin_edges) - 1, 0), dtype=float)
    return cov_df, X


def mutations_to_positions_by_chrom(muts_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    muts_df expects columns Chromosome, Start as strings.
    Return sorted int arrays of Start positions per chrom.
    """
    out: Dict[str, np.ndarray] = {}
    if muts_df.empty:
        return out

    for chrom, sub in muts_df.groupby("Chromosome"):
        pos = sub["Start"].astype(int).to_numpy()
        pos.sort()
        out[str(chrom)] = pos
    return out


def build_shared_inputs(
        *,
        pos_by_chrom: Dict[str, np.ndarray],
        chrom: str,
        chrom_length: int,
        bin_size: int,
        track_strategy: str,
        counts_sigma_bins: float,
        inv_sigma_bins: float,
        max_distance_bp: int,
        covariates: List[str],
        fasta_path: str | Path,
        timing_bigwig: Optional[str | Path],
        include_trinuc: bool,
) -> Dict[str, Any]:
    edges, centres = build_bins(chrom_length, bin_size)
    mut_pos = pos_by_chrom.get(chrom, np.array([], dtype=int))

    if track_strategy not in TRACK_REGISTRY:
        raise ValueError(f"Unknown track_strategy: {track_strategy}. Choose from: {list(TRACK_REGISTRY.keys())}")

    if track_strategy == "counts_raw":
        mut_track = TRACK_REGISTRY[track_strategy](mut_pos, edges)
    elif track_strategy == "counts_gauss":
        mut_track = TRACK_REGISTRY[track_strategy](mut_pos, edges, counts_sigma_bins)
    elif track_strategy == "inv_dist_gauss":
        mut_track = TRACK_REGISTRY[track_strategy](mut_pos, centres, inv_sigma_bins, max_distance_bp)
    else:
        raise ValueError("Unreachable: track_strategy registry mismatch")

    mut_track = mut_track.astype(float)

    cov_df, X = build_covariate_matrix(
        covariates=covariates,
        fasta_path=fasta_path,
        chrom=chrom,
        bin_edges=edges,
        timing_bigwig=timing_bigwig,
        include_trinuc=include_trinuc,
    )

    return {
        "edges": edges,
        "centres": centres,
        "mut_pos": mut_pos,
        "mut_track": mut_track,
        "cov_df": cov_df,
        "X": X,
    }


# -------------------------
# One run
# -------------------------
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

    mask_valid = np.isfinite(mut_track_corr) & np.isfinite(dnase_track_corr)
    summary = {
        "celltype": celltype,
        "pearson_r_raw": float(r_raw),
        "pearson_r_linear_resid": float(r_lin),
        "pearson_r_rf_resid": float(r_rf),
        "n_bins_valid_mut_and_dnase": int(mask_valid.sum()),
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
        dnase_bigwigs: Dict[str, str | Path],
        timing_bigwig: Optional[str | Path],
        sample_sizes: List[int | None],
        repeats: int,
        base_seed: int,
        bin_sizes: List[int],
        track_strategies: List[str],
        covariate_sets: List[List[str]],
        include_trinuc: bool = False,
        chroms: Optional[List[str]] = None,
        standardise_tracks: bool = True,
        standardise_scope: str = "per_chrom",
        verbose: bool = False,
        # track params
        counts_sigma_bins: float = 1.0,
        inv_sigma_bins: float = 0.5,
        max_distance_bp: int = 1_000_000,
        # IO
        out_dir: str | Path = "outputs/experiments/run",
        save_per_bin: bool = True,
        chunksize: int = 250_000,
        tumour_whitelist: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    logger = setup_rich_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        logger_name="mut_vs_dnase",
        force=True,
    )

    if standardise_tracks and standardise_scope != "per_chrom":
        raise ValueError(f"Unsupported standardise_scope: {standardise_scope}")

    if not dnase_bigwigs:
        raise ValueError("dnase_bigwigs must be a non-empty mapping of celltype->bigWig path.")

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

    out_dir = ensure_dir(out_dir)
    runs_dir = ensure_dir(out_dir / "runs")
    readme_path = out_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Outputs\n"
            "-------\n"
            "results.csv: one row per run configuration with aggregated metrics.\n"
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

    fai = load_fai(fai_path)
    chrom_infos = list(iter_chroms(fai, chroms=chroms))
    total_runs = (
            len(sample_sizes)
            * int(repeats)
            * len(bin_sizes)
            * len(track_strategies)
            * len(covariate_sets)
    )

    log_section(logger, "Session start")
    log_section(logger, "Inputs")
    log_kv(logger, "mutations_bed", str(mut_path))
    log_kv(logger, "fasta", str(fasta_path))
    log_kv(logger, "fai", str(fai_path))
    log_kv(logger, "dnase_bigwigs", ", ".join(dnase_bigwigs.keys()) if dnase_bigwigs else "none")
    log_kv(logger, "timing_bigwig", str(timing_bigwig) if timing_bigwig else "none")
    log_kv(logger, "tumour_whitelist", ",".join(tumour_whitelist) if tumour_whitelist else "none")

    log_section(logger, "Grid")
    log_kv(logger, "chroms", str(len(chrom_infos)))
    log_kv(logger, "configs", str(total_runs))

    rows: List[Dict[str, Any]] = []

    for k in sample_sizes:
        for rep in range(repeats):
            seed_samples = base_seed + rep
            with timed(logger, f"Sample selection (k={k}, rep={rep}, seed={seed_samples})"):
                chosen_samples, muts_df = compile_k_samples(
                    mut_path, k=k, seed=seed_samples, chunksize=chunksize, tumour_whitelist=tumour_whitelist
                )
            log_kv(logger, "selected_samples", str(len(chosen_samples)))
            log_kv(logger, "mutations_loaded", f"{len(muts_df):,}")
            pos_by_chrom = mutations_to_positions_by_chrom(muts_df)

            # model seed (for RF), independent but deterministic
            rf_seed = 10_000 + base_seed + rep

            # record sample selection metadata
            sample_meta = {
                "sample_size_k": None if k is None else int(k),
                "repeat": int(rep),
                "seed_samples": int(seed_samples),
                "n_selected_samples": int(len(chosen_samples)),
            }

            for bin_size in bin_sizes:
                for track_strategy in track_strategies:
                    for covs in covariate_sets:
                        logger.info(
                            "config bin=%d track=%s covs=%s",
                            int(bin_size),
                            track_strategy,
                            "-".join(covs) if covs else "none",
                        )
                        run_id = (
                            f"k={sample_meta['sample_size_k']}_rep={rep}_seed={seed_samples}"
                            f"_bin={bin_size}_track={track_strategy}_covs={'-'.join(covs) if covs else 'none'}"
                            f"_tri={int(include_trinuc)}"
                        )
                        t0 = time.perf_counter()
                        run_dir = ensure_dir(runs_dir / run_id)
                        celltypes = list(dnase_bigwigs.keys())
                        log_section(logger, f"Run start  [{len(rows) + 1}/{total_runs}]")
                        log_kv(logger, "id", run_id.replace("_", " "))
                        log_kv(logger, "standardise", f"{standardise_scope} ({'on' if standardise_tracks else 'off'})")
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
                            "bin_size": int(bin_size),
                            "track_strategy": track_strategy,
                            "covariates": covs,
                            "include_trinuc": bool(include_trinuc),
                            "standardise_tracks": bool(standardise_tracks),
                            "standardise_scope": str(standardise_scope),
                            "counts_sigma_bins": float(counts_sigma_bins),
                            "inv_sigma_bins": float(inv_sigma_bins),
                            "max_distance_bp": int(max_distance_bp),
                            "rf_seed": int(rf_seed),
                            "chroms": [ci.chrom for ci in chrom_infos],
                            "tumour_whitelist": list(tumour_whitelist) if tumour_whitelist else [],
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
                                counts_sigma_bins=counts_sigma_bins,
                                inv_sigma_bins=inv_sigma_bins,
                                max_distance_bp=max_distance_bp,
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
                                "bin_size": int(bin_size),
                                "track_strategy": track_strategy,
                                "n_bins": int(len(edges) - 1),
                                "n_mutations_chr": int(shared["mut_pos"].size),
                                "covariates": ",".join(covs) if covs else "",
                                "include_trinuc": bool(include_trinuc),
                            }

                            dnase_tracks: Dict[str, np.ndarray] = {}

                            for celltype in celltypes:
                                summ, dnase_track = run_one_config(
                                    muts_df=muts_df,
                                    chrom=ci.chrom,
                                    chrom_length=ci.length,
                                    bin_size=bin_size,
                                    track_strategy=track_strategy,
                                    counts_sigma_bins=counts_sigma_bins,
                                    inv_sigma_bins=inv_sigma_bins,
                                    max_distance_bp=max_distance_bp,
                                    dnase_bigwig=dnase_bigwigs[celltype],
                                    celltype=celltype,
                                    covariates=covs,
                                    fasta_path=fasta_path,
                                    timing_bigwig=timing_bigwig,
                                    include_trinuc=include_trinuc,
                                    rf_seed=rf_seed,
                                    standardise_tracks=standardise_tracks,
                                    standardise_scope=standardise_scope,
                                    shared=shared,
                                    pos_by_chrom=pos_by_chrom,
                                )
                                dnase_tracks[celltype] = dnase_track
                                chrom_row[f"pearson_r_raw_{celltype}"] = summ["pearson_r_raw"]
                                chrom_row[f"pearson_r_linear_resid_{celltype}"] = summ["pearson_r_linear_resid"]
                                chrom_row[f"pearson_r_rf_resid_{celltype}"] = summ["pearson_r_rf_resid"]
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
                                perm_imp, sign_corr, impurity_imp, rf_r2, ridge_coef, ridge_r2 = rf_feature_analysis(
                                    mut_track_rf,
                                    X_full,
                                    feature_names,
                                    seed=rf_seed,
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
                                    chrom_row["rf_top_celltype_feature_perm"] = top_feature[0].replace("dnase_", "")
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
                        agg = {
                            **sample_meta,
                            "bin_size": int(bin_size),
                            "track_strategy": track_strategy,
                            "covariates": ",".join(covs) if covs else "",
                            "include_trinuc": bool(include_trinuc),
                            "n_mutations_total": int(chrom_df["n_mutations_chr"].sum()),
                            "n_bins_total": int(chrom_df["n_bins"].sum()),
                            "run_id": run_id,
                        }

                        raw_vals: Dict[str, float] = {}
                        lin_vals: Dict[str, float] = {}
                        rf_vals: Dict[str, float] = {}
                        for ct in celltypes:
                            weights = chrom_df[f"n_bins_valid_mut_and_dnase_{ct}"].to_numpy(dtype=float)
                            raw = chrom_df[f"pearson_r_raw_{ct}"].to_numpy(dtype=float)
                            lin = chrom_df[f"pearson_r_linear_resid_{ct}"].to_numpy(dtype=float)
                            rf = chrom_df[f"pearson_r_rf_resid_{ct}"].to_numpy(dtype=float)

                            agg[f"pearson_r_raw_{ct}_mean_weighted"] = weighted_mean(raw, weights)
                            agg[f"pearson_r_linear_resid_{ct}_mean_weighted"] = weighted_mean(lin, weights)
                            agg[f"pearson_r_rf_resid_{ct}_mean_weighted"] = weighted_mean(rf, weights)
                            agg[f"pearson_r_raw_{ct}_mean_unweighted"] = float(np.nanmean(raw))
                            agg[f"pearson_r_linear_resid_{ct}_mean_unweighted"] = float(np.nanmean(lin))
                            agg[f"pearson_r_rf_resid_{ct}_mean_unweighted"] = float(np.nanmean(rf))

                            raw_vals[ct] = agg[f"pearson_r_raw_{ct}_mean_weighted"]
                            lin_vals[ct] = agg[f"pearson_r_linear_resid_{ct}_mean_weighted"]
                            rf_vals[ct] = agg[f"pearson_r_rf_resid_{ct}_mean_weighted"]

                        best_raw, best_raw_val, best_raw_margin = best_and_margin(raw_vals)
                        best_lin, best_lin_val, best_lin_margin = best_and_margin(lin_vals)
                        best_rf, best_rf_val, best_rf_margin = best_and_margin(rf_vals)

                        agg["best_celltype_raw"] = best_raw
                        agg["best_celltype_raw_value"] = float(best_raw_val)
                        agg["best_minus_second_raw"] = float(best_raw_margin)
                        agg["best_celltype_linear_resid"] = best_lin
                        agg["best_celltype_linear_resid_value"] = float(best_lin_val)
                        agg["best_minus_second_linear_resid"] = float(best_lin_margin)
                        agg["best_celltype_rf_resid"] = best_rf
                        agg["best_celltype_rf_resid_value"] = float(best_rf_val)
                        agg["best_minus_second_rf_resid"] = float(best_rf_margin)

                        agg["rf_perm_importances_mean_json"] = json.dumps(
                            aggregate_dict_column(chrom_df, "rf_perm_importances_json", "n_bins_valid_rf_features")
                        )
                        agg["rf_feature_sign_corr_mean_json"] = json.dumps(
                            aggregate_dict_column(chrom_df, "rf_feature_sign_corr_json", "n_bins_valid_rf_features")
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

                        logger.info("saved run outputs %s", str(run_dir))
                        elapsed = time.perf_counter() - t0
                        logger.info("Run end %s (%s)", run_id, f"{elapsed:.1f}s")

    results = pd.DataFrame(rows)
    # save_df(results, out_dir / "results.parquet")
    save_df(results, out_dir / "results.csv")
    return results


# -------------------------
# CLI
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Grid experiments: mutations vs DNase accessibility.")
    parser.add_argument("--mut-path", type=str, required=True, help="Path to UV_mutations.bed")
    parser.add_argument("--fai-path", type=str, required=True, help="Path to hg19.fa.fai")
    parser.add_argument("--fasta-path", type=str, required=True, help="Path to hg19.fa (indexed)")
    parser.add_argument(
        "--dnase-map-json",
        type=str,
        default=None,
        help="JSON dict of celltype->bigWig path. Example: '{\"mela\":\"/path/mela.bw\"}'",
    )
    parser.add_argument(
        "--dnase-map-path",
        type=str,
        default=None,
        help="Path to JSON file with dict of celltype->bigWig path",
    )
    parser.add_argument("--timing-bw", type=str, default=None, help="Path to RepliSeq bigWig (optional)")
    parser.add_argument(
        "--tumour-whitelist",
        type=str,
        default=None,
        help="Comma list of tumour codes to keep (e.g., 'SKCM,MELA'); case-insensitive",
    )

    parser.add_argument("--out-dir", type=str, default="outputs/experiments/run1", help="Output directory")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)

    parser.add_argument("--sample-sizes", type=str, default="1,5,10,20,all", help="Comma list; use 'all' for None")
    parser.add_argument("--bin-sizes", type=str, default="10000,50000,100000", help="Comma list of bin sizes")
    parser.add_argument("--track-strategies", type=str, default="counts_raw,counts_gauss,inv_dist_gauss")

    parser.add_argument(
        "--covariate-sets",
        type=str,
        default="gc+cpg,gc+cpg+timing",
        help="Semicolon-separated sets; each set is + separated. Example: gc+cpg;gc+cpg+timing",
    )
    parser.add_argument("--include-trinuc", action="store_true", help="Include small trinuc feature set")
    parser.add_argument("--chroms", type=str, default=None, help="Comma list of chroms or omit for all in fai")
    parser.add_argument("--save-per-bin", action="store_true", help="Save per-bin tables for inspection")
    parser.add_argument("--no-standardise-tracks", action="store_true", help="Disable per-chrom track standardisation")
    parser.add_argument("--standardise-scope", type=str, default="per_chrom", help="Standardisation scope")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()
    log_level = logging.DEBUG if args.debug else (logging.DEBUG if args.verbose else logging.INFO)
    setup_rich_logging(level=log_level, logger_name="mut_vs_dnase", force=True)

    sample_sizes = []
    for tok in args.sample_sizes.split(","):
        tok = tok.strip().lower()
        if tok == "all":
            sample_sizes.append(None)
        else:
            sample_sizes.append(int(tok))

    bin_sizes = [int(x.strip()) for x in args.bin_sizes.split(",") if x.strip()]
    track_strategies = [x.strip() for x in args.track_strategies.split(",") if x.strip()]

    covariate_sets: List[List[str]] = []
    for group in args.covariate_sets.split(";"):
        group = group.strip()
        if not group:
            continue
        covariate_sets.append([c.strip() for c in group.split("+") if c.strip()])

    chroms = None
    if args.chroms:
        chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]

    tumour_whitelist = None
    if args.tumour_whitelist:
        tumour_whitelist = [t.strip() for t in args.tumour_whitelist.split(",") if t.strip()]

    dnase_bigwigs: Dict[str, str | Path]
    if args.dnase_map_json:
        dnase_bigwigs = json.loads(args.dnase_map_json)
    elif args.dnase_map_path:
        with open(args.dnase_map_path, "r", encoding="utf-8") as f:
            dnase_bigwigs = json.load(f)
    else:
        raise ValueError("Provide --dnase-map-json or --dnase-map-path.")

    run_grid_experiment(
        mut_path=args.mut_path,
        fai_path=args.fai_path,
        fasta_path=args.fasta_path,
        dnase_bigwigs=dnase_bigwigs,
        timing_bigwig=args.timing_bw if args.timing_bw else None,
        sample_sizes=sample_sizes,
        repeats=args.repeats,
        base_seed=args.base_seed,
        bin_sizes=bin_sizes,
        track_strategies=track_strategies,
        covariate_sets=covariate_sets,
        include_trinuc=bool(args.include_trinuc),
        chroms=chroms,
        standardise_tracks=not args.no_standardise_tracks,
        standardise_scope=args.standardise_scope,
        verbose=bool(args.verbose or args.debug),
        out_dir=args.out_dir,
        save_per_bin=bool(args.save_per_bin),
        tumour_whitelist=tumour_whitelist,
    )


if __name__ == "__main__":
    main()
