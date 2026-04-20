#!/usr/bin/env python3
"""Merge sharded grid-search outputs into one combined experiment directory.

This script merges `results.csv` files from multiple shard output directories
created with `scripts.grid_search.cli --parallel-shard-count/--parallel-shard-index`.
It can also merge per-run artefacts from each shard `runs/` directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dirs",
        type=str,
        required=True,
        help="Comma-separated shard output directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for merged results.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="results.csv",
        help="Results filename in each shard directory (default: results.csv).",
    )
    parser.add_argument(
        "--dedupe-key",
        type=str,
        default="run_id",
        help="Column used to drop duplicate rows after merging (default: run_id).",
    )
    parser.add_argument(
        "--merge-runs",
        action="store_true",
        help="Copy shard run artefacts from shard_dir/runs into out_dir/runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def parse_shard_dirs(raw_value: str) -> list[Path]:
    """Convert a comma-separated shard directory string to Path list."""
    shard_dirs = [Path(token.strip()) for token in raw_value.split(",") if token.strip()]
    if not shard_dirs:
        raise ValueError("No shard directories were provided.")
    return shard_dirs


def load_shard_results(shard_dirs: list[Path], results_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and stack shard result tables, and return per-shard summary."""
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, str | int]] = []

    for shard_dir in shard_dirs:
        results_path = shard_dir / results_name
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        shard_df = pd.read_csv(results_path)
        shard_df["_source_shard_dir"] = str(shard_dir)
        shard_df["_source_results_file"] = str(results_path)
        frames.append(shard_df)

        summary_rows.append(
            {
                "shard_dir": str(shard_dir),
                "results_path": str(results_path),
                "row_count": int(len(shard_df)),
                "column_count": int(len(shard_df.columns)),
            }
        )

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return merged, summary


def merge_run_artefacts(shard_dirs: list[Path], out_dir: Path) -> tuple[int, int]:
    """Copy run artefact directories from shards into merged output."""
    runs_out_dir = out_dir / "runs"
    runs_out_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_existing = 0

    for shard_dir in shard_dirs:
        shard_runs_dir = shard_dir / "runs"
        if not shard_runs_dir.exists():
            continue

        for run_dir in sorted(shard_runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            dest_dir = runs_out_dir / run_dir.name
            if dest_dir.exists():
                skipped_existing += 1
                continue
            shutil.copytree(run_dir, dest_dir)
            copied_count += 1

    return copied_count, skipped_existing


def main() -> None:
    """Execute shard merge."""
    args = parse_args()
    shard_dirs = parse_shard_dirs(args.shard_dirs)

    for shard_dir in shard_dirs:
        if not shard_dir.exists():
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {args.out_dir}. "
            "Use --overwrite to allow writing into it."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_df, shard_summary = load_shard_results(shard_dirs, args.results_name)

    pre_dedupe_rows = int(len(merged_df))
    deduped_rows = pre_dedupe_rows
    dropped_duplicates = 0
    if args.dedupe_key in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=[args.dedupe_key], keep="first")
        deduped_rows = int(len(merged_df))
        dropped_duplicates = pre_dedupe_rows - deduped_rows

    merged_results_path = args.out_dir / args.results_name
    merged_df.to_csv(merged_results_path, index=False)

    shard_summary_path = args.out_dir / "shard_merge_summary.tsv"
    shard_summary.to_csv(shard_summary_path, sep="\t", index=False)

    copied_runs = 0
    skipped_runs = 0
    if args.merge_runs:
        copied_runs, skipped_runs = merge_run_artefacts(shard_dirs, args.out_dir)

    print(f"out_dir\t{args.out_dir}")
    print(f"merged_results\t{merged_results_path}")
    print(f"shard_summary\t{shard_summary_path}")
    print(f"shard_count\t{len(shard_dirs)}")
    print(f"rows_before_dedupe\t{pre_dedupe_rows}")
    print(f"rows_after_dedupe\t{deduped_rows}")
    print(f"dropped_duplicates\t{dropped_duplicates}")
    print(f"dedupe_key\t{args.dedupe_key}")
    print(f"merge_runs\t{'yes' if args.merge_runs else 'no'}")
    if args.merge_runs:
        print(f"runs_copied\t{copied_runs}")
        print(f"runs_skipped_existing\t{skipped_runs}")


if __name__ == "__main__":
    main()
