"""Feature and covariate construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.covariates import (
    gc_fraction_per_bin,
    cpg_frequency_per_bin,
    trinuc_frequency_per_bin,
    bigwig_mean_per_bin,
)
from scripts.genome_bins import build_bins
from scripts.tracks import TRACK_REGISTRY


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
    exp_decay_bp: float,
    exp_max_distance_bp: int,
    adaptive_k: int,
    adaptive_min_bandwidth_bp: float,
    adaptive_max_distance_bp: int,
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
    elif track_strategy == "exp_decay":
        mut_track = TRACK_REGISTRY[track_strategy](mut_pos, centres, exp_decay_bp, exp_max_distance_bp)
    elif track_strategy == "exp_decay_adaptive":
        mut_track = TRACK_REGISTRY[track_strategy](
            mut_pos,
            centres,
            adaptive_k,
            adaptive_min_bandwidth_bp,
            adaptive_max_distance_bp,
        )
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
