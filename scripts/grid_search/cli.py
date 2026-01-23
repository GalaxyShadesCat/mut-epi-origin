"""CLI entrypoint for grid search."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.dnase_map import DEFAULT_MAP_PATH, DnaseCellTypeMap
from scripts.io_utils import ensure_dir
from scripts.logging_utils import setup_rich_logging
from scripts.grid_search.config import expand_grid_values
from scripts.grid_search.io import _load_dnase_map_path
from scripts.grid_search.runner import run_grid_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid experiments: mutations vs DNase accessibility.")
    parser.add_argument(
        "--mut-path",
        type=str,
        required=True,
        help="Path to mutations BED (comma-separated for multiple inputs)",
    )
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
        default=str(DEFAULT_MAP_PATH),
        help="Path to JSON file with dict of celltype->bigWig path or celltype mapping list",
    )
    parser.add_argument("--timing-bw", type=str, default=None, help="Path to RepliSeq bigWig (optional)")
    parser.add_argument(
        "--tumour-filter",
        type=str,
        default=None,
        help="Comma list of tumour codes to keep (e.g., 'SKCM,MELA'); case-insensitive",
    )

    parser.add_argument("--out-dir", type=str, default="outputs/experiments/run1", help="Output directory")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)

    parser.add_argument("--sample-sizes", type=str, default="1,5,10,20,all", help="Comma list; use 'all' for None")
    parser.add_argument(
        "--downsample",
        type=str,
        default="none",
        help="Downsample mutation rows to N (comma list or range spec). Use 'none' to disable.",
    )
    parser.add_argument(
        "--counts-raw-bins",
        type=str,
        default="10000,50000,100000",
        help="Comma list or range spec [start,end,step] for counts_raw bins",
    )
    parser.add_argument(
        "--counts-gauss-bins",
        type=str,
        default="10000,50000,100000",
        help="Comma list or range spec [start,end,step] for counts_gauss bins",
    )
    parser.add_argument(
        "--inv-dist-gauss-bins",
        type=str,
        default="10000,50000,100000",
        help="Comma list or range spec [start,end,step] for inv_dist_gauss bins",
    )
    parser.add_argument(
        "--exp-decay-bins",
        type=str,
        default="10000,50000,100000",
        help="Comma list or range spec [start,end,step] for exp_decay bins",
    )
    parser.add_argument(
        "--exp-decay-adaptive-bins",
        type=str,
        default="10000,50000,100000",
        help="Comma list or range spec [start,end,step] for exp_decay_adaptive bins",
    )
    parser.add_argument("--track-strategies", type=str, default="counts_raw,counts_gauss,inv_dist_gauss")
    parser.add_argument(
        "--counts-gauss-sigma-grid",
        type=str,
        default="1.0",
        help="Comma list or range spec [start,end,step] for counts_gauss sigma",
    )
    parser.add_argument(
        "--counts-gauss-sigma-units",
        type=str,
        default="bins",
        choices=["bins", "bp"],
        help="Interpret counts_gauss sigma as 'bins' or 'bp'",
    )
    parser.add_argument(
        "--inv-dist-gauss-sigma-grid",
        type=str,
        default="0.5",
        help="Comma list or range spec [start,end,step] for inv_dist_gauss sigma "
             "(ignored when --inv-dist-gauss-pairs is set)",
    )
    parser.add_argument(
        "--inv-dist-gauss-max-distance-bp-grid",
        type=str,
        default="1000000",
        help="Comma list or range spec [start,end,step] for inv_dist_gauss max_distance_bp "
             "(ignored when --inv-dist-gauss-pairs is set)",
    )
    parser.add_argument(
        "--inv-dist-gauss-pairs",
        type=str,
        default="",
        help="Comma list of inv_sigma:max_distance_bp pairs for inv_dist_gauss "
             "(e.g., '0.25:200000,0.5:500000'); overrides sigma/max-distance grids",
    )
    parser.add_argument(
        "--inv-dist-gauss-sigma-units",
        type=str,
        default="bins",
        choices=["bins", "bp"],
        help="Interpret inv_dist_gauss sigma as 'bins' or 'bp'",
    )
    parser.add_argument(
        "--exp-decay-decay-bp-grid",
        type=str,
        default="200000",
        help="Comma list or range spec [start,end,step] for exp_decay decay lengths (bp)",
    )
    parser.add_argument(
        "--exp-decay-max-distance-bp-grid",
        type=str,
        default="1000000",
        help="Comma list or range spec [start,end,step] for exp_decay max_distance_bp",
    )
    parser.add_argument(
        "--exp-decay-adaptive-k-grid",
        type=str,
        default="5",
        help="Comma list or range spec [start,end,step] for exp_decay_adaptive k-nearest",
    )
    parser.add_argument(
        "--exp-decay-adaptive-min-bandwidth-bp-grid",
        type=str,
        default="50000",
        help="Comma list or range spec [start,end,step] for exp_decay_adaptive min bandwidth (bp)",
    )
    parser.add_argument(
        "--exp-decay-adaptive-max-distance-bp-grid",
        type=str,
        default="1000000",
        help="Comma list or range spec [start,end,step] for exp_decay_adaptive max_distance_bp",
    )

    parser.add_argument("--score-window-bins", type=int, default=1, help="Half-window size (bins) for local score")
    parser.add_argument(
        "--score-corr-type",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method for local score windows",
    )
    parser.add_argument(
        "--score-smoothing",
        type=str,
        default="none",
        choices=["none", "moving_average", "gaussian"],
        help="Smoothing method for local score",
    )
    parser.add_argument(
        "--score-smooth-param",
        type=float,
        default=None,
        help="Smoothing parameter (window size or sigma) for local score",
    )
    parser.add_argument(
        "--score-transform",
        type=str,
        default="none",
        choices=["none", "log1p"],
        help="Transform applied before local score",
    )
    parser.add_argument("--score-zscore", action="store_true", help="Z-score tracks before local score")
    parser.add_argument(
        "--score-weights",
        type=str,
        default="0.7,0.3",
        help="Comma pair of weights for (shape, slope) local score components",
    )

    parser.add_argument(
        "--covariate-sets",
        type=str,
        default="gc+cpg,gc+cpg+timing",
        help="Semicolon-separated sets; each set is + separated. Example: gc+cpg;gc+cpg+timing",
    )
    parser.add_argument("--include-trinuc", action="store_true", help="Include small trinuc feature set")
    parser.add_argument(
        "--chroms",
        type=str,
        default=None,
        help="Comma list of chroms or omit for autosomes only in fai",
    )
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

    track_strategies = [x.strip() for x in args.track_strategies.split(",") if x.strip()]
    counts_raw_bins = expand_grid_values(args.counts_raw_bins, name="counts_raw_bins", cast=int)
    counts_gauss_bins = expand_grid_values(args.counts_gauss_bins, name="counts_gauss_bins", cast=int)
    inv_dist_gauss_bins = expand_grid_values(args.inv_dist_gauss_bins, name="inv_dist_gauss_bins", cast=int)
    exp_decay_bins = expand_grid_values(args.exp_decay_bins, name="exp_decay_bins", cast=int)
    exp_decay_adaptive_bins = expand_grid_values(
        args.exp_decay_adaptive_bins, name="exp_decay_adaptive_bins", cast=int
    )
    counts_gauss_sigma_grid = expand_grid_values(
        args.counts_gauss_sigma_grid, name="counts_gauss_sigma_grid", cast=float
    )
    counts_gauss_sigma_units = args.counts_gauss_sigma_units
    inv_dist_gauss_sigma_grid = expand_grid_values(
        args.inv_dist_gauss_sigma_grid, name="inv_dist_gauss_sigma_grid", cast=float
    )
    inv_dist_gauss_max_distance_bp_grid = expand_grid_values(
        args.inv_dist_gauss_max_distance_bp_grid,
        name="inv_dist_gauss_max_distance_bp_grid",
        cast=int,
    )
    inv_dist_gauss_sigma_units = args.inv_dist_gauss_sigma_units
    exp_decay_decay_bp_grid = expand_grid_values(
        args.exp_decay_decay_bp_grid, name="exp_decay_decay_bp_grid", cast=float
    )
    exp_decay_max_distance_bp_grid = expand_grid_values(
        args.exp_decay_max_distance_bp_grid,
        name="exp_decay_max_distance_bp_grid",
        cast=int,
    )
    exp_decay_adaptive_k_grid = expand_grid_values(
        args.exp_decay_adaptive_k_grid, name="exp_decay_adaptive_k_grid", cast=int
    )
    exp_decay_adaptive_min_bandwidth_bp_grid = expand_grid_values(
        args.exp_decay_adaptive_min_bandwidth_bp_grid,
        name="exp_decay_adaptive_min_bandwidth_bp_grid",
        cast=float,
    )
    exp_decay_adaptive_max_distance_bp_grid = expand_grid_values(
        args.exp_decay_adaptive_max_distance_bp_grid,
        name="exp_decay_adaptive_max_distance_bp_grid",
        cast=int,
    )
    inv_dist_gauss_pairs: List[Tuple[float, int]] = []
    if args.inv_dist_gauss_pairs:
        for idx, token in enumerate([t for t in args.inv_dist_gauss_pairs.split(",") if t.strip()]):
            if ":" not in token:
                raise ValueError(
                    "Invalid --inv-dist-gauss-pairs entry; expected format 'sigma:max_distance_bp' "
                    f"(got {token!r})"
                )
            sigma_str, md_str = token.split(":", 1)
            try:
                sigma_val = float(sigma_str)
            except ValueError as exc:
                raise ValueError(
                    "Invalid --inv-dist-gauss-pairs entry; sigma must be float-like "
                    f"(got {sigma_str!r})"
                ) from exc
            try:
                md_float = float(md_str)
            except ValueError as exc:
                raise ValueError(
                    "Invalid --inv-dist-gauss-pairs entry; max_distance_bp must be int-like "
                    f"(got {md_str!r})"
                ) from exc
            if not md_float.is_integer():
                raise ValueError(
                    "Invalid --inv-dist-gauss-pairs entry; max_distance_bp must be int-like "
                    f"(got {md_str!r})"
                )
            inv_dist_gauss_pairs.append((sigma_val, int(md_float)))

    covariate_sets: List[List[str]] = []
    for group in args.covariate_sets.split(";"):
        group = group.strip()
        if not group:
            continue
        covariate_sets.append([c.strip() for c in group.split("+") if c.strip()])

    chroms = None
    if args.chroms:
        chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]

    score_weights_tokens = [float(x) for x in args.score_weights.split(",") if x.strip()]
    if len(score_weights_tokens) != 2:
        raise ValueError("--score-weights must be a comma pair like '0.7,0.3'.")
    score_weights = (score_weights_tokens[0], score_weights_tokens[1])

    tumour_filter = None
    if args.tumour_filter:
        tumour_filter = [t.strip() for t in args.tumour_filter.split(",") if t.strip()]

    dnase_bigwigs: Dict[str, str | Path]
    celltype_map: Optional[DnaseCellTypeMap] = None
    if args.dnase_map_json:
        dnase_bigwigs = json.loads(args.dnase_map_json)
        if not isinstance(dnase_bigwigs, dict) or not all(
            isinstance(v, str) for v in dnase_bigwigs.values()
        ):
            raise ValueError("--dnase-map-json must be a JSON object mapping celltype->path.")
    else:
        dnase_map_path = Path(args.dnase_map_path)
        if not dnase_map_path.exists():
            raise FileNotFoundError(f"DNase map path not found: {dnase_map_path}")
        dnase_bigwigs, celltype_map = _load_dnase_map_path(dnase_map_path)

    mut_path: str | Path | List[Path]
    mut_path_tokens = [p.strip() for p in args.mut_path.split(",") if p.strip()]
    if len(mut_path_tokens) > 1:
        mut_path = [Path(p) for p in mut_path_tokens]
    else:
        mut_path = mut_path_tokens[0]

    run_grid_experiment(
        mut_path=mut_path,
        fai_path=args.fai_path,
        fasta_path=args.fasta_path,
        dnase_bigwigs=dnase_bigwigs,
        celltype_map=celltype_map,
        timing_bigwig=args.timing_bw if args.timing_bw else None,
        sample_sizes=sample_sizes,
        repeats=args.repeats,
        base_seed=args.base_seed,
        track_strategies=track_strategies,
        covariate_sets=covariate_sets,
        include_trinuc=bool(args.include_trinuc),
        downsample_counts=args.downsample,
        chroms=chroms,
        standardise_tracks=not args.no_standardise_tracks,
        standardise_scope=args.standardise_scope,
        verbose=bool(args.verbose or args.debug),
        score_window_bins=args.score_window_bins,
        score_corr_type=args.score_corr_type,
        score_smoothing=args.score_smoothing,
        score_smooth_param=args.score_smooth_param,
        score_transform=args.score_transform,
        score_zscore=bool(args.score_zscore),
        score_weights=score_weights,
        out_dir=args.out_dir,
        save_per_bin=bool(args.save_per_bin),
        tumour_filter=tumour_filter,
        counts_raw_bins=counts_raw_bins,
        counts_gauss_bins=counts_gauss_bins,
        inv_dist_gauss_bins=inv_dist_gauss_bins,
        exp_decay_bins=exp_decay_bins,
        exp_decay_adaptive_bins=exp_decay_adaptive_bins,
        counts_gauss_sigma_grid=counts_gauss_sigma_grid,
        counts_gauss_sigma_units=counts_gauss_sigma_units,
        inv_dist_gauss_sigma_grid=inv_dist_gauss_sigma_grid,
        inv_dist_gauss_max_distance_bp_grid=inv_dist_gauss_max_distance_bp_grid,
        inv_dist_gauss_pairs=inv_dist_gauss_pairs,
        inv_dist_gauss_sigma_units=inv_dist_gauss_sigma_units,
        exp_decay_decay_bp_grid=exp_decay_decay_bp_grid,
        exp_decay_max_distance_bp_grid=exp_decay_max_distance_bp_grid,
        exp_decay_adaptive_k_grid=exp_decay_adaptive_k_grid,
        exp_decay_adaptive_min_bandwidth_bp_grid=exp_decay_adaptive_min_bandwidth_bp_grid,
        exp_decay_adaptive_max_distance_bp_grid=exp_decay_adaptive_max_distance_bp_grid,
    )

    out_dir_path = Path(args.out_dir)
    if not out_dir_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        out_dir_path = project_root / out_dir_path
    out_dir = ensure_dir(out_dir_path)
    command_txt = out_dir / "command.txt"
    command_txt.write_text(shlex.join([sys.executable, *sys.argv]), encoding="utf-8")


if __name__ == "__main__":
    main()
