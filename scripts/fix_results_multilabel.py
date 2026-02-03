"""Backfill is_correct_* fields for multi-label correct_celltypes in results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.grid_search.results import compute_derived_fields


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Update derived correctness fields in a grid-search results.csv, "
            "supporting multi-label correct_celltypes."
        )
    )
    parser.add_argument(
        "results_csv",
        type=Path,
        help="Path to results.csv to update.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing to a new file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <input>.fixed.csv if not in-place.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    path = args.results_csv
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found: {path}")

    df = pd.read_csv(path)
    updates = []
    for _, row in df.iterrows():
        updates.append(compute_derived_fields(row.to_dict()))
    updates_df = pd.DataFrame(updates)
    for col in updates_df.columns:
        df[col] = updates_df[col]

    if args.in_place:
        output = path
    else:
        output = args.output or path.with_suffix(".fixed.csv")
    df.to_csv(output, index=False)
    print(f"Wrote updated results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
