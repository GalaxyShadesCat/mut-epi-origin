#!/usr/bin/env python3
"""Build a standardised SNV-only LIHC mutation table from per-sample VCF files.

This script reads a VCF manifest, selects one VCF per tumour sample
(`tumour_sample_submitter_id`) using the largest file size when duplicates exist,
extracts PASS SNVs, converts VCF coordinates to 0-based start coordinates, and
writes a tab-separated mutation table.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import pysam


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "data_paths.json"


def load_wgs_tcga25_root() -> Path:
    """Load WGS_TCGA25 root path from the shared data-path config file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing config file: {CONFIG_PATH}. "
            "Create it with key 'wgs_tcga25_root'."
        )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_root = str(config.get("wgs_tcga25_root", "")).strip()
    if not raw_root:
        raise ValueError(
            f"Config {CONFIG_PATH} must define non-empty 'wgs_tcga25_root'."
        )
    root = Path(raw_root)
    if not root.is_absolute():
        root = BASE_DIR / root
    return root.resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    wgs_tcga25_root = load_wgs_tcga25_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/derived/manifests/lihc_tumour_vcf_candidates.tsv"),
        help="Path to manifest TSV (default: data/derived/manifests/lihc_tumour_vcf_candidates.tsv)",
    )
    parser.add_argument(
        "--vcf-root",
        type=Path,
        default=wgs_tcga25_root / "AtoL" / "VCF",
        help="Root directory containing manifest relative VCF paths (default: <wgs_tcga25_root>/AtoL/VCF)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mutations/lihc_snv_mutation_table.tsv"),
        help="Output TSV path (default: data/raw/mutations/lihc_snv_mutation_table.tsv)",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure simple console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load manifest rows from TSV and validate required columns."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    required_columns = {
        "sample_key",
        "tumour_sample_submitter_id",
        "tumour_sample_id",
        "relative_path",
    }

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header row: {manifest_path}")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Manifest missing required columns: {sorted(missing_columns)}"
            )

        rows = [
            {
                "sample_key": row["sample_key"],
                "tumour_sample_submitter_id": row["tumour_sample_submitter_id"],
                "tumour_sample_id": row["tumour_sample_id"],
                "relative_path": row["relative_path"],
            }
            for row in reader
        ]

    if not rows:
        raise ValueError(f"Manifest has no data rows: {manifest_path}")

    return rows


def choose_vcfs_by_sample(
    manifest_rows: list[dict[str, str]],
    vcf_root: Path,
) -> tuple[dict[str, Path], int]:
    """Choose one payload VCF per sample, selecting the largest when duplicates exist.

    Only real VCF payloads are considered as candidates:
    - *.vcf
    - *.vcf.gz

    Sidecar index files such as *.tbi and *.idx are ignored.
    """
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in manifest_rows:
        sample = row["tumour_sample_submitter_id"].strip()
        if not sample:
            raise ValueError("Found blank tumour_sample_submitter_id in manifest")

        relative_path = row["relative_path"].strip()
        if not relative_path:
            raise ValueError(
                f"Sample {sample} has a blank relative_path in manifest"
            )

        # Only payload VCFs should be treated as candidates for parsing.
        if not (relative_path.endswith(".vcf") or relative_path.endswith(".vcf.gz")):
            continue

        rows_by_sample[sample].append(row)

    selected: dict[str, Path] = {}
    samples_with_multiple_vcfs = 0

    for sample, rows in sorted(rows_by_sample.items()):
        candidates: list[tuple[Path, int]] = []

        for row in rows:
            relative_path = row["relative_path"].strip()
            vcf_path = vcf_root / relative_path

            if not vcf_path.exists():
                raise FileNotFoundError(
                    "VCF path from manifest does not exist for sample "
                    f"{sample}: {vcf_path}"
                )

            file_size = vcf_path.stat().st_size
            candidates.append((vcf_path, file_size))

        if not candidates:
            raise ValueError(
                f"Sample {sample} has no payload VCF candidates after filtering"
            )

        if len(candidates) == 1:
            selected[sample] = candidates[0][0]
            continue

        samples_with_multiple_vcfs += 1
        candidates_sorted = sorted(
            candidates,
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )
        selected_path, selected_size = candidates_sorted[0]
        selected[sample] = selected_path

        LOGGER.info("Sample %s has %d payload VCFs", sample, len(candidates_sorted))
        LOGGER.info(
            "Selected: %s (%s)",
            selected_path.name,
            format_bytes(selected_size),
        )
        for discarded_path, discarded_size in candidates_sorted[1:]:
            LOGGER.info(
                "Discarded: %s (%s)",
                discarded_path.name,
                format_bytes(discarded_size),
            )

    return selected, samples_with_multiple_vcfs


def format_bytes(byte_count: int) -> str:
    """Format byte size with a compact IEC-style unit."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(byte_count)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    return f"{size:.1f}{units[unit_index]}"


def is_pass_record(record: pysam.VariantRecord) -> bool:
    """Return True when FILTER is exactly PASS."""
    filter_keys = tuple(record.filter.keys())
    return len(filter_keys) == 1 and filter_keys[0] == "PASS"


def iter_mutation_rows(
    sample: str,
    vcf_path: Path,
) -> tuple[list[tuple[str, int, int, str, str, str, str]], int, int, int]:
    """Parse a VCF and return SNV rows plus record-level counts."""
    rows: list[tuple[str, int, int, str, str, str, str]] = []
    variants_seen = 0
    pass_variants = 0
    snv_rows_emitted = 0

    with pysam.VariantFile(str(vcf_path)) as variant_file:
        for record in variant_file:
            variants_seen += 1
            if not is_pass_record(record):
                continue

            pass_variants += 1

            ref = record.ref
            if ref is None or len(ref) != 1:
                continue

            alts = record.alts
            if not alts:
                continue

            chrom = record.chrom
            start = record.pos - 1
            end = record.pos

            for alt in alts:
                if alt is None or len(alt) != 1:
                    continue
                rows.append((chrom, start, end, sample, ref, alt, "LIHC"))
                snv_rows_emitted += 1

    return rows, variants_seen, pass_variants, snv_rows_emitted


def write_output(
    output_path: Path,
    rows: list[tuple[str, int, int, str, str, str, str]],
) -> None:
    """Write mutation rows to a TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["chrom", "start", "end", "sample", "ref", "alt", "cohort"])
        writer.writerows(rows)


def deduplicate_rows(
    rows: list[tuple[str, int, int, str, str, str, str]],
) -> list[tuple[str, int, int, str, str, str, str]]:
    """Remove duplicate rows using sample/chrom/start/ref/alt as key."""
    deduplicated: list[tuple[str, int, int, str, str, str, str]] = []
    seen_keys: set[tuple[str, str, int, str, str]] = set()

    for chrom, start, end, sample, ref, alt, cohort in rows:
        key = (sample, chrom, start, ref, alt)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append((chrom, start, end, sample, ref, alt, cohort))

    return deduplicated


def main() -> None:
    """Run the full mutation-table build workflow."""
    args = parse_args()
    configure_logging()

    manifest_rows = load_manifest(args.manifest)
    total_samples_in_manifest = len(
        {row["tumour_sample_submitter_id"].strip() for row in manifest_rows}
    )
    selected_vcfs, samples_with_multiple_vcfs = choose_vcfs_by_sample(
        manifest_rows=manifest_rows,
        vcf_root=args.vcf_root,
    )

    all_rows: list[tuple[str, int, int, str, str, str, str]] = []
    variants_seen = 0
    pass_variants = 0
    snv_rows_emitted = 0

    for sample, vcf_path in sorted(selected_vcfs.items()):
        rows, sample_variants_seen, sample_pass_variants, sample_snv_rows = (
            iter_mutation_rows(sample=sample, vcf_path=vcf_path)
        )
        all_rows.extend(rows)
        variants_seen += sample_variants_seen
        pass_variants += sample_pass_variants
        snv_rows_emitted += sample_snv_rows

    deduplicated_rows = deduplicate_rows(all_rows)
    write_output(args.output, deduplicated_rows)

    LOGGER.info("Total samples in manifest: %d", total_samples_in_manifest)
    LOGGER.info("Samples with multiple VCFs: %d", samples_with_multiple_vcfs)
    LOGGER.info("Samples processed: %d", len(selected_vcfs))
    LOGGER.info("VCFs parsed: %d", len(selected_vcfs))
    LOGGER.info("Variants seen: %d", variants_seen)
    LOGGER.info("PASS variants: %d", pass_variants)
    LOGGER.info("SNV rows emitted: %d", snv_rows_emitted)
    LOGGER.info("Rows after deduplication: %d", len(deduplicated_rows))
    LOGGER.info("Output written: %s", args.output)


if __name__ == "__main__":
    main()
