"""
filter_mutations.py

Filter one or more mutation BEDs to keep only tumour types supported by the DNase map.
Multiple inputs are concatenated into a single output file.
Samples with fewer than a minimum number of mutations are excluded.

This script expects known, hardcoded input formats:
- UV_mutations.bed:
  Chrom  Start  End  Sample_ID  Ref  Alt  Cancer_Type  Trinucleotide_Context
- ICGC_WGS_Feb20_mutations.bed:
  Chrom  Start  End  Donor_ID  Ref  Alt  Project  Sample_ID

Input format (tab-separated, no header):
Chrom  Start  End  Donor_ID  Ref  Alt  Project  Sample_ID

Output format (tab-separated, no header):
Chrom  Start  End  Sample_ID  Ref  Alt  Cancer_Type
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dnase_map import DnaseCellTypeMap

FORMAT_SPECS: Dict[str, Dict[str, object]] = {
    "UV_mutations.bed": {
        "columns": [
            "Chromosome",
            "Start",
            "End",
            "Sample_ID",
            "Ref",
            "Alt",
            "Cancer_Type",
            "Trinucleotide_Context",
        ],
        "project_key": "Cancer_Type",
    },
    "ICGC_WGS_Feb20_mutations.bed": {
        "columns": [
            "Chromosome",
            "Start",
            "End",
            "Donor_ID",
            "Ref",
            "Alt",
            "Project",
            "Sample_ID",
        ],
        "project_key": "Project",
    },
}


def _norm_tumour(value: str) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


def _load_allowed_tumours(project_root: Path, dnase_map_path: Path | None) -> Set[str]:
    if dnase_map_path is None:
        cell_map = DnaseCellTypeMap.from_project_root(project_root)
    else:
        cell_map = DnaseCellTypeMap.from_json(dnase_map_path, project_root=project_root)
    allowed = {_norm_tumour(t) for t in cell_map.tumour_filter()}
    return {t for t in allowed if t}


def keep_line(project: str, *, allowed: Set[str]) -> bool:
    return _norm_tumour(project) in allowed


def _resolve_format(path: Path) -> Tuple[List[str], int, int]:
    spec = FORMAT_SPECS.get(path.name)
    if spec is None:
        known = ", ".join(sorted(FORMAT_SPECS))
        raise ValueError(f"Unknown mutation format for {path.name}. Known: {known}")
    columns = spec["columns"]
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Invalid format spec for {path.name}")
    project_key = spec["project_key"]
    if not isinstance(project_key, str) or project_key not in columns:
        raise ValueError(f"Invalid project_key for {path.name}")
    project_idx = columns.index(project_key)
    sample_idx = columns.index("Sample_ID")
    return columns, project_idx, sample_idx


def _normalize_row(parts: List[str], columns: List[str]) -> List[str]:
    col_map = {name: idx for idx, name in enumerate(columns)}
    if "Cancer_Type" in col_map:
        cancer_idx = col_map["Cancer_Type"]
    else:
        cancer_idx = col_map["Project"]
    return [
        parts[col_map["Chromosome"]],
        parts[col_map["Start"]],
        parts[col_map["End"]],
        parts[col_map["Sample_ID"]],
        parts[col_map["Ref"]],
        parts[col_map["Alt"]],
        parts[cancer_idx],
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-bed",
        required=True,
        type=Path,
        nargs="+",
        help="One or more mutation BEDs (tab-separated, no header).",
    )
    ap.add_argument(
        "--out-bed",
        default=None,
        type=Path,
        help="Output path (default: <input>.tumour_filtered.bed or combined.tumour_filtered.bed).",
    )
    ap.add_argument(
        "--dnase-map",
        default=None,
        type=Path,
        help="Optional DNase map JSON; default resolves from project root.",
    )
    ap.add_argument(
        "--min-mutations",
        type=int,
        default=100,
        help="Exclude samples with fewer than this many mutations (default: 100).",
    )
    args = ap.parse_args()

    in_paths: List[Path] = list(args.in_bed)
    for in_path in in_paths:
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

    if args.out_bed is not None:
        out_path: Path = args.out_bed
    elif len(in_paths) == 1:
        in_path = in_paths[0]
        out_path = in_path.with_suffix("").with_name(in_path.stem + ".tumour_filtered.bed")
    else:
        out_path = Path("combined.tumour_filtered.bed")

    allowed = _load_allowed_tumours(PROJECT_ROOT, args.dnase_map)
    if not allowed:
        raise ValueError("No tumour types found in DNase map.")

    if args.min_mutations < 1:
        raise ValueError("--min-mutations must be >= 1")

    n_in = 0
    n_out = 0
    seen_samples_global: Set[str] = set()
    sample_counts: Dict[str, int] = {}

    for in_path in in_paths:
        columns, project_idx, sample_idx = _resolve_format(in_path)
        expected_cols = len(columns)
        seen_samples_file: Set[str] = set()
        with in_path.open("r", encoding="utf-8", errors="replace") as fin:
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != expected_cols:
                    raise ValueError(
                        f"{in_path} line {line_no}: expected {expected_cols} columns "
                        f"({', '.join(columns)}), got {len(parts)}"
                    )
                project = parts[project_idx]
                if not keep_line(project, allowed=allowed):
                    continue
                sample_id = parts[sample_idx].strip()
                if sample_id not in seen_samples_file:
                    if sample_id in seen_samples_global:
                        raise ValueError(
                            f"Sample_ID '{sample_id}' appears in multiple files "
                            f"(duplicate found in {in_path} line {line_no})"
                        )
                    seen_samples_file.add(sample_id)
                    seen_samples_global.add(sample_id)
                sample_counts[sample_id] = sample_counts.get(sample_id, 0) + 1

    with out_path.open("w", encoding="utf-8") as fout:
        for in_path in in_paths:
            columns, project_idx, sample_idx = _resolve_format(in_path)
            expected_cols = len(columns)
            with in_path.open("r", encoding="utf-8", errors="replace") as fin:
                for line_no, line in enumerate(fin, start=1):
                    if not line.strip():
                        continue
                    n_in += 1
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != expected_cols:
                        raise ValueError(
                            f"{in_path} line {line_no}: expected {expected_cols} columns "
                            f"({', '.join(columns)}), got {len(parts)}"
                        )

                    project = parts[project_idx]
                    if keep_line(project, allowed=allowed):
                        sample_id = parts[sample_idx].strip()
                        if sample_counts.get(sample_id, 0) < args.min_mutations:
                            continue
                        fout.write("\t".join(_normalize_row(parts, columns)) + "\n")
                        n_out += 1

    print(f"Input lines:   {n_in:,}")
    print(f"Output lines:  {n_out:,}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
