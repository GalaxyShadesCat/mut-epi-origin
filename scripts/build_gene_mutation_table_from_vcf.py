#!/usr/bin/env python3
"""Build a gene-annotated LIHC mutation table from per-sample VCF files.

This script reads a LIHC tumour VCF manifest, selects one VCF per sample,
extracts PASS SNVs, parses Ensembl VEP CSQ annotations, and writes a
tab-separated, MAF-like mutation table with gene names.

The output is intended for downstream gene-level differential mutation analysis.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pysam


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/derived/manifests/lihc_tumour_vcf_candidates.tsv"),
        help="Path to LIHC tumour VCF manifest TSV.",
    )
    parser.add_argument(
        "--vcf-root",
        type=Path,
        default=Path("data/raw/WGS_TCGA25/AtoL/VCF"),
        help="Root directory containing manifest-relative VCF paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mutations/lihc_gene_mutation_table.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--rankings-path",
        type=Path,
        default=None,
        help="Optional validation_score_rankings.csv used to keep only needed samples.",
    )
    parser.add_argument(
        "--rankings-sample-col",
        default="sample",
        help="Sample column name in rankings file.",
    )
    parser.add_argument(
        "--config-id",
        default="",
        help="Optional config_id filter applied to rankings rows.",
    )
    parser.add_argument(
        "--scoring-system",
        default="",
        help="Optional scoring_system filter applied to rankings rows.",
    )
    parser.add_argument(
        "--score-gap-threshold",
        type=float,
        default=0.0,
        help="Minimum score_gap filter applied to rankings rows.",
    )
    parser.add_argument(
        "--id-prefix-length",
        type=int,
        default=15,
        help="Prefix length used to harmonise TCGA sample IDs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on processed samples for quick test runs.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def normalise_tcga_id(sample_id: str, prefix_len: int) -> str:
    """Normalise sample identifiers to harmonised TCGA format."""
    clean = sample_id.strip().upper().replace(".", "-")
    return clean[:prefix_len]


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load and validate manifest rows."""
    required_columns = {
        "sample_key",
        "tumour_sample_submitter_id",
        "tumour_sample_id",
        "relative_path",
    }
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest missing required columns: {missing_text}")
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"Manifest has no data rows: {manifest_path}")
    return rows


def choose_vcfs_by_sample(
    manifest_rows: list[dict[str, str]],
    vcf_root: Path,
) -> tuple[dict[str, Path], int]:
    """Choose one payload VCF per sample, preferring largest file size."""
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        rel_path = row["relative_path"].strip()
        if not (rel_path.endswith(".vcf") or rel_path.endswith(".vcf.gz")):
            continue
        sample = row["tumour_sample_submitter_id"].strip()
        rows_by_sample[sample].append(row)

    selected: dict[str, Path] = {}
    duplicated_samples = 0

    for sample, rows in sorted(rows_by_sample.items()):
        candidates: list[tuple[Path, int]] = []
        for row in rows:
            vcf_path = vcf_root / row["relative_path"].strip()
            if not vcf_path.exists():
                raise FileNotFoundError(f"VCF path does not exist: {vcf_path}")
            candidates.append((vcf_path, vcf_path.stat().st_size))

        if len(candidates) > 1:
            duplicated_samples += 1

        best_path = sorted(
            candidates,
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )[0][0]
        selected[sample] = best_path

    return selected, duplicated_samples


def load_rankings_allowlist(
    args: argparse.Namespace,
) -> set[str]:
    """Load optional sample allowlist from rankings file."""
    if args.rankings_path is None:
        return set()

    rankings = pd.read_csv(args.rankings_path)
    if args.rankings_sample_col not in rankings.columns:
        raise ValueError(
            f"Rankings file missing sample column: {args.rankings_sample_col}"
        )

    if args.config_id:
        if "config_id" not in rankings.columns:
            raise ValueError("Rankings file missing config_id column.")
        rankings = rankings[rankings["config_id"] == args.config_id]
    if args.scoring_system:
        if "scoring_system" not in rankings.columns:
            raise ValueError("Rankings file missing scoring_system column.")
        rankings = rankings[rankings["scoring_system"] == args.scoring_system]
    if "score_gap" in rankings.columns:
        rankings = rankings[rankings["score_gap"] >= args.score_gap_threshold]

    sample_ids = {
        normalise_tcga_id(sample, args.id_prefix_length)
        for sample in rankings[args.rankings_sample_col].dropna().astype(str)
    }
    if not sample_ids:
        raise ValueError("No samples remained after rankings-based filtering.")
    return sample_ids


def parse_csq_fields(header: pysam.VariantHeader) -> list[str]:
    """Parse CSQ field order from VCF header."""
    if "CSQ" not in header.info:
        raise ValueError("VCF header does not contain INFO/CSQ.")
    description = header.info["CSQ"].description
    match = re.search(r"Format:\s*(.+)$", description)
    if not match:
        raise ValueError("Could not parse CSQ format from VCF header.")
    return [field.strip() for field in match.group(1).split("|")]


def consequence_to_variant_classification(consequence: str) -> str:
    """Map VEP consequence terms to MAF-like Variant_Classification labels."""
    consequence_map = {
        "synonymous_variant": "Silent",
        "missense_variant": "Missense_Mutation",
        "stop_gained": "Nonsense_Mutation",
        "stop_lost": "Nonstop_Mutation",
        "start_lost": "Translation_Start_Site",
        "splice_acceptor_variant": "Splice_Site",
        "splice_donor_variant": "Splice_Site",
        "splice_region_variant": "Splice_Region",
        "frameshift_variant": "Frame_Shift",
        "inframe_insertion": "In_Frame_Ins",
        "inframe_deletion": "In_Frame_Del",
        "protein_altering_variant": "Protein_Altering_Variant",
        "5_prime_utr_variant": "5UTR",
        "3_prime_utr_variant": "3UTR",
        "intron_variant": "Intron",
        "upstream_gene_variant": "5Flank",
        "downstream_gene_variant": "3Flank",
    }
    return consequence_map.get(consequence, consequence)


def pick_csq_entry(
    csq_entries: tuple[str, ...],
    csq_fields: list[str],
    alt_allele: str,
) -> dict[str, str]:
    """Pick one CSQ annotation for an ALT allele, preferring PICK=1."""
    field_index = {name: idx for idx, name in enumerate(csq_fields)}
    allele_idx = field_index.get("Allele")
    pick_idx = field_index.get("PICK")

    parsed_entries: list[list[str]] = []
    for entry in csq_entries:
        values = entry.split("|")
        if len(values) < len(csq_fields):
            values.extend([""] * (len(csq_fields) - len(values)))
        parsed_entries.append(values)

    allele_matched = parsed_entries
    if allele_idx is not None:
        filtered = [
            values for values in parsed_entries if values[allele_idx] == alt_allele
        ]
        if filtered:
            allele_matched = filtered

    if pick_idx is not None:
        picked = [values for values in allele_matched if values[pick_idx] == "1"]
        if picked:
            allele_matched = picked

    best = allele_matched[0]
    result: dict[str, str] = {}
    for name, idx in field_index.items():
        result[name] = best[idx]
    return result


def build_rows_for_sample(
    sample_id: str,
    vcf_path: Path,
) -> tuple[list[dict[str, object]], int, int, int]:
    """Extract PASS SNV rows with gene annotations from one sample VCF."""
    rows: list[dict[str, object]] = []
    variants_seen = 0
    pass_variants = 0
    rows_emitted = 0

    with pysam.VariantFile(str(vcf_path)) as vcf:
        csq_fields = parse_csq_fields(vcf.header)
        for record in vcf:
            variants_seen += 1
            filter_keys = tuple(record.filter.keys())
            if not (len(filter_keys) == 1 and filter_keys[0] == "PASS"):
                continue

            pass_variants += 1
            ref = record.ref
            alts = record.alts
            if ref is None or len(ref) != 1 or not alts:
                continue
            if "CSQ" not in record.info:
                continue

            csq_entries = tuple(str(v) for v in record.info["CSQ"])
            for alt in alts:
                if alt is None or len(alt) != 1:
                    continue

                csq = pick_csq_entry(csq_entries, csq_fields, alt)
                gene_symbol = csq.get("SYMBOL", "").strip()
                if not gene_symbol:
                    gene_symbol = csq.get("Gene", "").strip()
                if not gene_symbol:
                    continue

                consequence = csq.get("Consequence", "").split("&")[0].strip()
                if not consequence:
                    consequence = "sequence_variant"
                variant_classification = consequence_to_variant_classification(
                    consequence
                )

                rows.append(
                    {
                        "Tumor_Sample_Barcode": sample_id,
                        "Hugo_Symbol": gene_symbol,
                        "Variant_Classification": variant_classification,
                        "Consequence": consequence,
                        "Chromosome": record.chrom,
                        "Start_Position": record.pos,
                        "End_Position": record.pos,
                        "Reference_Allele": ref,
                        "Tumor_Seq_Allele2": alt,
                        "Source_VCF": str(vcf_path),
                    }
                )
                rows_emitted += 1

    return rows, variants_seen, pass_variants, rows_emitted


def deduplicate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove duplicate variant-gene rows."""
    deduped: list[dict[str, object]] = []
    seen_keys: set[tuple[object, ...]] = set()
    for row in rows:
        key = (
            row["Tumor_Sample_Barcode"],
            row["Hugo_Symbol"],
            row["Chromosome"],
            row["Start_Position"],
            row["Reference_Allele"],
            row["Tumor_Seq_Allele2"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped


def write_output(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Write output TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Variant_Classification",
        "Consequence",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Source_VCF",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the table-building workflow."""
    args = parse_args()
    configure_logging()

    manifest_rows = load_manifest_rows(args.manifest)
    selected_vcfs, duplicated_samples = choose_vcfs_by_sample(
        manifest_rows, args.vcf_root
    )
    LOGGER.info("Selected one VCF for %d samples.", len(selected_vcfs))
    LOGGER.info("Samples with duplicate payload VCFs: %d", duplicated_samples)

    allowlist = load_rankings_allowlist(args)
    selected_items = sorted(selected_vcfs.items())
    if allowlist:
        selected_items = [
            item
            for item in selected_items
            if normalise_tcga_id(item[0], args.id_prefix_length) in allowlist
        ]
        LOGGER.info("Samples after rankings filter: %d", len(selected_items))
    if args.max_samples > 0:
        selected_items = selected_items[: args.max_samples]
        LOGGER.info("Samples after max-samples cap: %d", len(selected_items))
    if not selected_items:
        raise ValueError("No samples available after filtering.")

    all_rows: list[dict[str, object]] = []
    total_seen = 0
    total_pass = 0
    total_rows = 0

    for index, (sample, vcf_path) in enumerate(selected_items, start=1):
        sample_rows, seen, passed, emitted = build_rows_for_sample(sample, vcf_path)
        all_rows.extend(sample_rows)
        total_seen += seen
        total_pass += passed
        total_rows += emitted
        if index % 10 == 0 or index == len(selected_items):
            LOGGER.info(
                "Processed %d/%d samples (rows emitted so far: %d)",
                index,
                len(selected_items),
                total_rows,
            )

    deduped_rows = deduplicate_rows(all_rows)
    write_output(args.output, deduped_rows)

    LOGGER.info("Variants seen: %d", total_seen)
    LOGGER.info("PASS variants: %d", total_pass)
    LOGGER.info("Rows before deduplication: %d", len(all_rows))
    LOGGER.info("Rows after deduplication: %d", len(deduped_rows))
    LOGGER.info("Output written: %s", args.output)


if __name__ == "__main__":
    main()
