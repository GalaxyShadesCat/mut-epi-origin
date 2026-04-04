#!/usr/bin/env python3
"""Build a LIHC tumour VCF manifest from metadata and a source VCF directory.

This script is a manifest-only replacement for transfer workflows. It:
1. extracts expected VCF-related filenames per tumour sample from metadata,
2. inventories files under a source directory,
3. matches expected filenames to discovered files,
4. writes a manifest suitable for build_snv_mutation_table.py.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


MISSING_TOKENS = {"", "na", "nan", "none", "null", "n/a"}
ALLOWED_PRIMARY_DIAGNOSES = {
    "hepatocellular carcinoma, nos",
    "hepatocellular carcinoma, clear cell type",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/derived/master_sample_metadata.csv"),
        help="Input metadata CSV path.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/storage5/jwlab/mutationDatabase/WGS_TCGA25/AtoL/VCF"),
        help="Root directory containing VCF files.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/derived/manifests/lihc_tumour_vcf_candidates_all.tsv"),
        help="Output manifest TSV path.",
    )
    parser.add_argument(
        "--lihc-primary-tumour-only",
        action="store_true",
        help=(
            "Restrict to TCGA-LIHC primary tumours with allowed LIHC diagnoses. "
            "By default, every metadata row is considered."
        ),
    )
    parser.add_argument(
        "--required-complete-fields",
        type=str,
        default="",
        help=(
            "Comma-separated metadata fields that must be non-missing in a row "
            "(for example 'alcohol_status,hbv_status')."
        ),
    )
    return parser.parse_args()


def clean_text(value: str | None) -> str | None:
    """Return stripped text or None for blank values."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return text


def normalise(value: str | None) -> str | None:
    """Return cleaned value with missing tokens normalised to None."""
    text = clean_text(value)
    if text is None:
        return None
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def is_allowed_vcf_related(file_name: str) -> bool:
    """Return True for VCF payload and index sidecar files."""
    return (
        file_name.endswith(".vcf")
        or file_name.endswith(".vcf.gz")
        or file_name.endswith(".vcf.tbi")
        or file_name.endswith(".vcf.gz.tbi")
        or file_name.endswith(".idx")
    )


def load_expected_files(
    metadata_csv: Path,
    *,
    lihc_primary_tumour_only: bool,
    required_complete_fields: list[str],
) -> tuple[dict[str, set[str]], dict[str, tuple[str, str]], dict[str, int]]:
    """Load expected file names per sample from metadata."""
    required_columns = {
        "tumour_sample_submitter_id",
        "tumour_sample_id",
        "file_name",
    }
    if lihc_primary_tumour_only:
        required_columns.update({"project_id", "primary_diagnosis", "tumour_sample_type"})
    required_columns.update(required_complete_fields)

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header row: {metadata_csv}")
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"Metadata CSV missing required columns: {missing_columns}")

        sample_to_expected_files: dict[str, set[str]] = defaultdict(set)
        sample_meta: dict[str, tuple[str, str]] = {}
        stats = defaultdict(int)

        for row in reader:
            stats["metadata_rows_total"] += 1

            if lihc_primary_tumour_only:
                if normalise(row.get("project_id")) != "TCGA-LIHC":
                    continue
                if normalise(row.get("tumour_sample_type")) != "Primary Tumor":
                    continue
                primary_diagnosis = normalise(row.get("primary_diagnosis"))
                if primary_diagnosis is None:
                    continue
                if primary_diagnosis.lower() not in ALLOWED_PRIMARY_DIAGNOSES:
                    continue
                stats["lihc_primary_tumour_rows"] += 1

            if any(normalise(row.get(field)) is None for field in required_complete_fields):
                continue
            stats["required_fields_pass_rows"] += 1

            sample_submitter_id = clean_text(row.get("tumour_sample_submitter_id")) or ""
            sample_uuid = clean_text(row.get("tumour_sample_id")) or ""
            sample_key = sample_submitter_id or sample_uuid
            if not sample_key:
                continue

            file_name = clean_text(row.get("file_name"))
            if file_name is None:
                continue
            if not is_allowed_vcf_related(file_name):
                continue

            sample_to_expected_files[sample_key].add(file_name)
            sample_meta[sample_key] = (sample_submitter_id, sample_uuid)

        stats["selected_samples"] = len(sample_to_expected_files)
        stats["expected_unique_file_records"] = sum(
            len(file_names) for file_names in sample_to_expected_files.values()
        )
        return sample_to_expected_files, sample_meta, dict(stats)


def build_inventory(src_dir: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Build basename-to-relative-path inventory from source directory."""
    basename_to_relpaths: dict[str, set[str]] = defaultdict(set)
    all_rel_paths: set[str] = set()

    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if not is_allowed_vcf_related(path.name):
            continue
        rel_path = str(path.relative_to(src_dir))
        basename_to_relpaths[path.name].add(rel_path)
        all_rel_paths.add(rel_path)
    return basename_to_relpaths, all_rel_paths


def match_manifest_rows(
    sample_to_expected_files: dict[str, set[str]],
    sample_meta: dict[str, tuple[str, str]],
    basename_to_relpaths: dict[str, set[str]],
    all_rel_paths: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Match expected names to inventory and return manifest rows."""
    matched_by_sample: dict[str, set[str]] = defaultdict(set)
    matched_expected_names = 0

    for sample_key, expected_names in sample_to_expected_files.items():
        for expected_name in expected_names:
            direct_matches = set(basename_to_relpaths.get(expected_name, set()))
            secondary_matches: set[str] = set()
            if not direct_matches and (
                expected_name.endswith(".tbi") or expected_name.endswith(".idx")
            ):
                secondary_matches = set(basename_to_relpaths.get(expected_name[:-4], set()))

            all_matches = direct_matches | secondary_matches
            if all_matches:
                matched_expected_names += 1

            rel_paths_to_add = set(all_matches)
            for rel_path in all_matches:
                if rel_path.endswith(".vcf") or rel_path.endswith(".vcf.gz"):
                    for sidecar_suffix in (".tbi", ".idx"):
                        sidecar_path = rel_path + sidecar_suffix
                        if sidecar_path in all_rel_paths:
                            rel_paths_to_add.add(sidecar_path)

            for rel_path in rel_paths_to_add:
                matched_by_sample[sample_key].add(rel_path)

    rows: list[dict[str, str]] = []
    for sample_key in sorted(sample_to_expected_files):
        submitter_id, sample_id = sample_meta.get(sample_key, ("", ""))
        for rel_path in sorted(matched_by_sample.get(sample_key, set())):
            rows.append(
                {
                    "sample_key": sample_key,
                    "tumour_sample_submitter_id": submitter_id,
                    "tumour_sample_id": sample_id,
                    "relative_path": rel_path,
                }
            )

    samples_with_matches = sum(
        1 for sample_key in sample_to_expected_files if matched_by_sample.get(sample_key)
    )
    stats = {
        "samples_with_matches": samples_with_matches,
        "samples_without_matches": len(sample_to_expected_files) - samples_with_matches,
        "matched_expected_names": matched_expected_names,
        "matched_manifest_rows": len(rows),
    }
    return rows, stats


def write_manifest(output_manifest: Path, rows: list[dict[str, str]]) -> None:
    """Write manifest rows to TSV."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "tumour_sample_submitter_id",
                "tumour_sample_id",
                "relative_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run manifest generation."""
    args = parse_args()
    required_complete_fields = [
        token.strip() for token in args.required_complete_fields.split(",") if token.strip()
    ]

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")
    if not args.src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.src_dir}")

    sample_to_expected_files, sample_meta, metadata_stats = load_expected_files(
        args.metadata_csv,
        lihc_primary_tumour_only=bool(args.lihc_primary_tumour_only),
        required_complete_fields=required_complete_fields,
    )
    basename_to_relpaths, all_rel_paths = build_inventory(args.src_dir)
    rows, match_stats = match_manifest_rows(
        sample_to_expected_files,
        sample_meta,
        basename_to_relpaths,
        all_rel_paths,
    )
    write_manifest(args.output_manifest, rows)

    print(f"metadata_csv\t{args.metadata_csv}")
    print(f"src_dir\t{args.src_dir}")
    print(f"output_manifest\t{args.output_manifest}")
    print(f"required_complete_fields\t{','.join(required_complete_fields) or '<none>'}")
    print(
        "lihc_primary_tumour_only\t"
        + ("yes" if args.lihc_primary_tumour_only else "no")
    )
    for key in sorted(metadata_stats):
        print(f"{key}\t{metadata_stats[key]}")
    print(f"inventory_allowed_files\t{sum(len(v) for v in basename_to_relpaths.values())}")
    for key in sorted(match_stats):
        print(f"{key}\t{match_stats[key]}")


if __name__ == "__main__":
    main()
