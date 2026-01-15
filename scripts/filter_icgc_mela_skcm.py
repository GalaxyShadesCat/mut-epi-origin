"""
filter_icgc_mela_skcm.py

Filter ICGC_WGS_Feb20_mutations.bed to keep only melanoma-related projects:
- MELA-* (e.g., MELA-AU)
- SKCM-* (e.g., SKCM-US)

Input format (tab-separated, no header):
Chrom  Start  End  Donor_ID  Ref  Alt  Project  Sample_ID
"""

from __future__ import annotations

import argparse
from pathlib import Path


def keep_line(project: str) -> bool:
    p = project.strip().upper()
    return p == "MELA" or p == "SKCM" or p.startswith("MELA-") or p.startswith("SKCM-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-bed",
        required=True,
        type=Path,
        help="Path to ICGC_WGS_Feb20_mutations.bed",
    )
    ap.add_argument(
        "--out-bed",
        default=None,
        type=Path,
        help="Output path (default: <input>.MELA_SKCM.bed)",
    )
    args = ap.parse_args()

    in_path: Path = args.in_bed
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path: Path = args.out_bed or in_path.with_suffix("").with_name(in_path.stem + ".MELA_SKCM.bed")

    n_in = 0
    n_out = 0
    n_skipped_short = 0

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, out_path.open(
            "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                n_skipped_short += 1
                continue

            project = parts[6]  # 0-based col 6 = Project (e.g., BRCA-UK, MELA-AU, SKCM-US)
            if keep_line(project):
                fout.write(line)
                n_out += 1

    print(f"Input lines:   {n_in:,}")
    print(f"Output lines:  {n_out:,}")
    if n_skipped_short:
        print(f"Skipped (<7 cols): {n_skipped_short:,}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
