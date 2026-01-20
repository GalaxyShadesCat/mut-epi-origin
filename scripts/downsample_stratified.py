"""
downsample_stratified.py

Stratified downsampling for BED-like mutation files.

This module reads a tab-delimited mutation file (BED-like), groups rows by
chromosome, and downsamples to an exact target size using proportional
allocation per chromosome with deterministic shuffling via a fixed RNG seed.
Comments (lines starting with "#") can be preserved in the output.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _count_top(chrom_counts: Dict[str, int], limit: int = 10) -> List[Tuple[str, int]]:
    return sorted(chrom_counts.items(), key=lambda x: (-x[1], x[0]))[:limit]


def _allocate_counts(
        chrom_counts: Dict[str, int],
        target_n: int,
) -> Dict[str, int]:
    total = sum(chrom_counts.values())
    if total <= 0:
        raise ValueError("Input contains no mutation rows.")

    ideals: Dict[str, float] = {}
    floor_alloc: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for chrom, count in chrom_counts.items():
        ideal = (count / total) * target_n
        floor_val = int(math.floor(ideal))
        ideals[chrom] = ideal
        floor_alloc[chrom] = floor_val
        remainders.append((ideal - floor_val, chrom))

    allocated = sum(floor_alloc.values())
    remaining = target_n - allocated

    # Hamilton's largest remainder method.
    remainders.sort(key=lambda x: (-x[0], x[1]))
    alloc = dict(floor_alloc)
    idx = 0
    while remaining > 0 and remainders:
        _, chrom = remainders[idx]
        if alloc[chrom] < chrom_counts[chrom]:
            alloc[chrom] += 1
            remaining -= 1
        idx = (idx + 1) % len(remainders)

    # Guard against any allocation exceeding available rows.
    remaining = target_n - sum(alloc.values())
    if remaining < 0:
        raise ValueError("Internal allocation error: allocated more than target.")

    if remaining > 0:
        # Redistribute deterministically to chroms with capacity, prioritizing
        # larger fractional remainders.
        rem_lookup = {chrom: rem for rem, chrom in remainders}
        while remaining > 0:
            candidates = [
                (rem_lookup.get(chrom, 0.0), chrom)
                for chrom, count in chrom_counts.items()
                if alloc[chrom] < count
            ]
            if not candidates:
                break
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, chrom = candidates[0]
            alloc[chrom] += 1
            remaining -= 1

    return alloc


def stratified_downsample_bed(
        in_path: str,
        out_path: str,
        target_n: int,
        seed: int = 1,
        chrom_col: int = 0,
        preserve_comments: bool = True,
) -> dict:
    """
    Downsample mutation rows to exactly target_n using stratified sampling by chromosome.

    Returns a summary dict with keys:
      - input_n, output_n, target_n
      - per_chrom_input_counts (dict)
      - per_chrom_output_counts (dict)
      - seed
      - status
    """
    if target_n <= 0:
        raise ValueError("target_n must be > 0")

    comments: List[str] = []
    rows: List[str] = []
    with open(in_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                if preserve_comments:
                    comments.append(line.rstrip("\n"))
                continue
            if not line.strip():
                continue
            rows.append(line.rstrip("\n"))

    input_n = len(rows)
    per_chrom_input_counts: Dict[str, int] = defaultdict(int)
    rows_by_chrom: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        parts = row.split("\t")
        if chrom_col >= len(parts):
            raise ValueError(
                f"chrom_col index {chrom_col} out of range for row: {row}"
            )
        chrom = parts[chrom_col]
        per_chrom_input_counts[chrom] += 1
        rows_by_chrom[chrom].append(row)

    if input_n <= target_n:
        _write_output(out_path, comments, rows, preserve_comments)
        return {
            "input_n": input_n,
            "output_n": input_n,
            "target_n": target_n,
            "per_chrom_input_counts": dict(per_chrom_input_counts),
            "per_chrom_output_counts": dict(per_chrom_input_counts),
            "seed": seed,
            "status": "unchanged_already_below_target",
        }

    alloc = _allocate_counts(per_chrom_input_counts, target_n)
    rng = np.random.default_rng(seed)

    sampled_rows: List[str] = []
    per_chrom_output_counts: Dict[str, int] = {}
    for chrom in sorted(rows_by_chrom.keys()):
        chrom_rows = rows_by_chrom[chrom]
        need = alloc.get(chrom, 0)
        if need < 0:
            raise ValueError("Allocation cannot be negative.")
        if need > len(chrom_rows):
            need = len(chrom_rows)
        per_chrom_output_counts[chrom] = need
        if need == len(chrom_rows):
            sampled_rows.extend(chrom_rows)
            continue
        idxs = rng.choice(list(range(len(chrom_rows))), size=need, replace=False)
        sampled_rows.extend([chrom_rows[i] for i in idxs])

    rng.shuffle(sampled_rows)
    if len(sampled_rows) != target_n:
        raise ValueError(
            f"Sampling produced {len(sampled_rows)} rows, expected {target_n}."
        )

    _write_output(out_path, comments, sampled_rows, preserve_comments)

    return {
        "input_n": input_n,
        "output_n": len(sampled_rows),
        "target_n": target_n,
        "per_chrom_input_counts": dict(per_chrom_input_counts),
        "per_chrom_output_counts": dict(per_chrom_output_counts),
        "seed": seed,
        "status": "downsampled",
    }


def _write_output(
        out_path: str, comments: Iterable[str], rows: Iterable[str], preserve_comments: bool
) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        if preserve_comments:
            for comment in comments:
                handle.write(f"{comment}\n")
        for row in rows:
            handle.write(f"{row}\n")


def _print_summary(summary: dict, limit: int = 10) -> None:
    print(
        "input_n={input_n} output_n={output_n} target_n={target_n} status={status}".format(
            **summary
        )
    )
    print(f"seed={summary['seed']}")
    print("per_chrom_input_counts (top 10):")
    for chrom, count in _count_top(summary["per_chrom_input_counts"], limit=limit):
        print(f"  {chrom}\t{count}")
    print("per_chrom_output_counts (top 10):")
    for chrom, count in _count_top(summary["per_chrom_output_counts"], limit=limit):
        print(f"  {chrom}\t{count}")


def _run_self_tests() -> None:
    data = [
        "# test header",
        "chr1\t0\t1\tA",
        "chr1\t1\t2\tB",
        "chr1\t2\t3\tC",
        "chr1\t3\t4\tD",
        "chr2\t0\t1\tE",
        "chr2\t1\t2\tF",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = f"{tmpdir}/input.bed"
        out_path = f"{tmpdir}/output.bed"
        with open(in_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(data) + "\n")

        summary = stratified_downsample_bed(
            in_path=in_path,
            out_path=out_path,
            target_n=3,
            seed=123,
        )
        assert summary["output_n"] == 3
        # Total 6; chr1 count=4, chr2 count=2; ideal allocations 2 and 1.
        assert summary["per_chrom_output_counts"]["chr1"] == 2
        assert summary["per_chrom_output_counts"]["chr2"] == 1

        summary_again = stratified_downsample_bed(
            in_path=in_path,
            out_path=out_path,
            target_n=3,
            seed=123,
        )
        assert summary == summary_again

        with open(out_path, "r", encoding="utf-8") as handle:
            out_lines = [line.rstrip("\n") for line in handle]

        summary_third = stratified_downsample_bed(
            in_path=in_path,
            out_path=out_path,
            target_n=3,
            seed=123,
        )
        with open(out_path, "r", encoding="utf-8") as handle:
            out_lines_third = [line.rstrip("\n") for line in handle]
        assert out_lines == out_lines_third
        assert summary_third == summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Downsample BED-like mutation rows stratified by chromosome."
    )
    parser.add_argument("--in", dest="in_path", required=False)
    parser.add_argument("--out", dest="out_path", required=False)
    parser.add_argument("--target", dest="target_n", type=int, required=False)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--chrom-col", dest="chrom_col", type=int, default=0)
    parser.add_argument(
        "--no-preserve-comments",
        dest="preserve_comments",
        action="store_false",
    )
    parser.add_argument(
        "--run-tests",
        dest="run_tests",
        action="store_true",
        help="Run self-tests and exit.",
    )

    args = parser.parse_args(argv)

    if args.run_tests:
        _run_self_tests()
        print("Self-tests passed.")
        return 0

    if not args.in_path or not args.out_path or args.target_n is None:
        parser.error("--in, --out, and --target are required unless --run-tests is used")

    summary = stratified_downsample_bed(
        in_path=args.in_path,
        out_path=args.out_path,
        target_n=args.target_n,
        seed=args.seed,
        chrom_col=args.chrom_col,
        preserve_comments=args.preserve_comments,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
