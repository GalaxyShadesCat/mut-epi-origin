from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Iterable, Sequence, Set

import pandas as pd
import pysam


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class RefCheckResult:
    reference_fasta: str
    seed: int
    n_tested: int
    n_matched: int
    n_mismatched: int
    n_skipped: int
    mismatch_examples: List[Dict[str, Any]]  # small examples for debugging


@dataclass
class MutationStats:
    file_path: str
    file_format: Literal["tcga_maf_tsv", "uv_bed_like", "unknown"]

    n_rows: int  # total mutations (rows)
    n_unique_samples: int
    n_unique_patients: int

    # These are meaningful for TCGA MAF-style; UV bed-like will set to 0/NA.
    n_unique_genes: int
    n_unique_variants: int

    avg_mutations_per_sample: float
    median_mutations_per_sample: float
    median_mutations_per_patient: float

    # Meaningful for TCGA MAF-style; UV sets to "NA"/0.
    top_variant_classification: str
    top_variant_classification_count: int

    # For both: inferred from allele lengths when possible
    snv_count: int
    indel_count: int

    missing_sample_barcode_rows: int

    # Optional reference check
    ref_check: Optional[RefCheckResult] = None


# -----------------------------
# Helpers
# -----------------------------
def _normalise_tumour_whitelist(tumour_whitelist: Optional[Sequence[str]]) -> Set[str]:
    if not tumour_whitelist:
        return set()
    return {str(x).strip().upper() for x in tumour_whitelist if str(x).strip()}


def _matches_tumour_whitelist(series: pd.Series, allowed: Set[str]) -> pd.Series:
    if not allowed:
        return pd.Series([True] * len(series), index=series.index)

    t_up = series.fillna("").astype(str).str.strip().str.upper()
    exact = t_up.isin(allowed)

    prefix = False
    for code in allowed:
        prefix = prefix | t_up.str.startswith(code + "-")

    return exact | prefix
def patient_id_from_mixed_sample_id(sample_id: str, *, uv_unique_patients: bool = False) -> Optional[str]:
    """
    Derive patient ID from mixed sample IDs.

    If uv_unique_patients=True: treat each sample ID as a unique patient (no underscore collapsing).
    Otherwise:
      1) TCGA barcodes: patient = first 3 dash-separated blocks (TCGA-XX-YYYY)
      2) Underscore IDs with >= 2 underscores: patient = everything except last token (prefix)
         e.g. Zheng_BCC_WT1 -> Zheng_BCC
      3) Otherwise: patient = sample_id
    """
    if not isinstance(sample_id, str):
        return None
    s = sample_id.strip()
    if not s:
        return None

    # TCGA rule
    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return s

    # UV override: do not group underscore IDs
    if uv_unique_patients:
        return s

    # Underscore rule (group replicates/timepoints)
    if "_" in s:
        parts = [p for p in s.split("_") if p]
        if len(parts) >= 3:
            return "_".join(parts[:-1])
        if len(parts) == 2:
            return parts[0]

    return s


def _patient_id_from_tcga_barcode(barcode: str) -> Optional[str]:
    if not isinstance(barcode, str):
        return None
    parts = barcode.split("-")
    if len(parts) < 3:
        return None
    if parts[0] != "TCGA":
        return None
    return "-".join(parts[:3])


def _normalise_chrom(chrom: str, fasta: pysam.FastaFile) -> Optional[str]:
    """
    Map chrom names to FASTA contigs: '1' vs 'chr1', 'MT' vs 'chrM', etc.
    Returns a contig present in FASTA or None.
    """
    if chrom is None:
        return None
    c = str(chrom).strip()
    if not c:
        return None

    contigs = set(fasta.references)

    if c in contigs:
        return c

    if c.startswith("chr"):
        alt = c[3:]
        if alt in contigs:
            return alt
    else:
        alt = "chr" + c
        if alt in contigs:
            return alt

    mito_map = {
        "MT": ["MT", "M", "chrM", "chrMT"],
        "M": ["MT", "M", "chrM", "chrMT"],
        "chrM": ["MT", "M", "chrM", "chrMT"],
        "chrMT": ["MT", "M", "chrM", "chrMT"],
    }
    if c in mito_map:
        for cand in mito_map[c]:
            if cand in contigs:
                return cand

    return None


def _read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.readline().strip("\n")


def detect_file_format(path: str | Path) -> Literal["tcga_maf_tsv", "uv_bed_like", "unknown"]:
    """
    Heuristics (stricter):
      - TCGA/cBioPortal MAF-style TSV: header contains Tumor_Sample_Barcode AND Chromosome AND Start_Position.
      - UV bed-like: first non-empty, non-comment line starts with chr* and has >= 6 columns.
    """
    p = Path(path)
    if not p.exists():
        return "unknown"

    # Read the first "real" line (skip empty/comment/track lines)
    first = ""
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip("\n")
            if not s.strip():
                continue
            if s.startswith("#") or s.lower().startswith("track") or s.lower().startswith("browser"):
                continue
            first = s
            break

    if not first:
        return "unknown"

    # TCGA MAF-style header detection
    if (
        "\t" in first
        and "Tumor_Sample_Barcode" in first
        and "Chromosome" in first
        and "Start_Position" in first
    ):
        return "tcga_maf_tsv"

    # UV bed-like detection (no header assumed, chr* and many columns)
    parts = first.split("\t")
    if parts[0].startswith("chr") and len(parts) >= 6:
        return "uv_bed_like"

    return "unknown"


def _reservoir_sample_rows_tsv_with_header(
    path: Path,
    seed: int,
    k: int,
    usecols: List[str],
    sep: str = "\t",
    chunksize: int = 250_000,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for chunk in pd.read_csv(
        path,
        sep=sep,
        usecols=lambda c: c in usecols,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        for row in chunk.to_dict(orient="records"):
            seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = row
    return reservoir


def _reservoir_sample_rows_tsv_with_header_multi(
    paths: Sequence[Path],
    seed: int,
    k: int,
    usecols: List[str],
    sep: str = "\t",
    chunksize: int = 250_000,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for path in paths:
        for chunk in pd.read_csv(
            path,
            sep=sep,
            usecols=lambda c: c in usecols,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        ):
            for row in chunk.to_dict(orient="records"):
                seen += 1
                if len(reservoir) < k:
                    reservoir.append(row)
                else:
                    j = rng.randint(1, seen)
                    if j <= k:
                        reservoir[j - 1] = row
    return reservoir


def _reservoir_sample_rows_bed_no_header(
    path: Path,
    seed: int,
    k: int,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    UV_mutations.bed appears tab-delimited with NO header.
    Expected columns (0-index):
      0 chrom, 1 start, 2 end, 3 sample, 4 ref, 5 alt, 6 cancer, 7 context
    We sample rows and return dicts with standardised keys.
    """
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    colnames = ["Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Cancer_Type", "Trinuc_Context"]

    allowed = _normalise_tumour_whitelist(tumour_whitelist)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=colnames,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        if allowed:
            mask = _matches_tumour_whitelist(chunk["Cancer_Type"], allowed)
            chunk = chunk.loc[mask]
            if chunk.empty:
                continue
        for row in chunk.to_dict(orient="records"):
            seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = row
    return reservoir


def _reservoir_sample_rows_bed_no_header_multi(
    paths: Sequence[Path],
    seed: int,
    k: int,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    colnames = ["Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Cancer_Type", "Trinuc_Context"]
    allowed = _normalise_tumour_whitelist(tumour_whitelist)

    for path in paths:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=colnames,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        ):
            if allowed:
                mask = _matches_tumour_whitelist(chunk["Cancer_Type"], allowed)
                chunk = chunk.loc[mask]
                if chunk.empty:
                    continue
            for row in chunk.to_dict(orient="records"):
                seen += 1
                if len(reservoir) < k:
                    reservoir.append(row)
                else:
                    j = rng.randint(1, seen)
                    if j <= k:
                        reservoir[j - 1] = row
    return reservoir


# -----------------------------
# Reference checks
# -----------------------------
def check_reference_genome_tcga_maf(
    mutations_path: str | Path,
    reference_fasta: str | Path,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    max_mismatch_examples: int = 5,
) -> RefCheckResult:
    mutations_path = Path(mutations_path)
    reference_fasta = Path(reference_fasta)

    if not mutations_path.exists():
        raise FileNotFoundError(f"Mutations file not found: {mutations_path}")
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    usecols = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Sample_Barcode"]
    sampled = _reservoir_sample_rows_tsv_with_header(
        path=mutations_path,
        seed=seed,
        k=n_rows_to_test,
        usecols=usecols,
        chunksize=chunksize,
    )

    fasta = pysam.FastaFile(str(reference_fasta))

    matched = mismatched = skipped = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for r in sampled:
        chrom = r.get("Chromosome")
        pos = r.get("Start_Position")
        ref = r.get("Reference_Allele")
        sample = r.get("Tumor_Sample_Barcode")

        if ref is None or str(ref).strip() in ("", ".", "NA"):
            skipped += 1
            continue
        ref = str(ref).strip().upper()
        if len(ref) != 1:
            skipped += 1
            continue

        try:
            pos_i = int(str(pos).strip())  # MAF pos is 1-based
        except Exception:
            skipped += 1
            continue
        if pos_i <= 0:
            skipped += 1
            continue

        contig = _normalise_chrom(chrom, fasta)
        if contig is None:
            skipped += 1
            continue

        try:
            base = fasta.fetch(contig, pos_i - 1, pos_i).upper()
        except Exception:
            skipped += 1
            continue

        if base == ref:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < max_mismatch_examples:
                mismatch_examples.append(
                    {
                        "Tumor_Sample_Barcode": sample,
                        "Chromosome": chrom,
                        "Start_Position": pos_i,
                        "Reference_Allele_in_file": ref,
                        "Reference_Allele_in_fasta": base,
                        "Resolved_Contig": contig,
                    }
                )

    fasta.close()

    return RefCheckResult(
        reference_fasta=str(reference_fasta),
        seed=seed,
        n_tested=len(sampled),
        n_matched=matched,
        n_mismatched=mismatched,
        n_skipped=skipped,
        mismatch_examples=mismatch_examples,
    )


def check_reference_genome_tcga_maf_multi(
    mutations_paths: Sequence[str | Path],
    reference_fasta: str | Path,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    max_mismatch_examples: int = 5,
) -> RefCheckResult:
    paths = [Path(p) for p in mutations_paths]
    reference_fasta = Path(reference_fasta)

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Mutations file not found: {p}")
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    usecols = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Sample_Barcode"]
    sampled = _reservoir_sample_rows_tsv_with_header_multi(
        paths=paths,
        seed=seed,
        k=n_rows_to_test,
        usecols=usecols,
        chunksize=chunksize,
    )

    fasta = pysam.FastaFile(str(reference_fasta))

    matched = mismatched = skipped = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for r in sampled:
        chrom = r.get("Chromosome")
        pos = r.get("Start_Position")
        ref = r.get("Reference_Allele")
        sample = r.get("Tumor_Sample_Barcode")

        if ref is None or str(ref).strip() in ("", ".", "NA"):
            skipped += 1
            continue
        ref = str(ref).strip().upper()
        if len(ref) != 1:
            skipped += 1
            continue

        try:
            pos_i = int(str(pos).strip())  # MAF pos is 1-based
        except Exception:
            skipped += 1
            continue
        if pos_i <= 0:
            skipped += 1
            continue

        contig = _normalise_chrom(chrom, fasta)
        if contig is None:
            skipped += 1
            continue

        try:
            base = fasta.fetch(contig, pos_i - 1, pos_i).upper()
        except Exception:
            skipped += 1
            continue

        if base == ref:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < max_mismatch_examples:
                mismatch_examples.append(
                    {
                        "Tumor_Sample_Barcode": sample,
                        "Chromosome": chrom,
                        "Start_Position": pos_i,
                        "Reference_Allele_in_file": ref,
                        "Reference_Allele_in_fasta": base,
                        "Resolved_Contig": contig,
                    }
                )

    fasta.close()

    return RefCheckResult(
        reference_fasta=str(reference_fasta),
        seed=seed,
        n_tested=len(sampled),
        n_matched=matched,
        n_mismatched=mismatched,
        n_skipped=skipped,
        mismatch_examples=mismatch_examples,
    )


def check_reference_genome_uv_bed(
    mutations_path: str | Path,
    reference_fasta: str | Path,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
    max_mismatch_examples: int = 5,
) -> RefCheckResult:
    """
    UV_mutations.bed uses 0-based start coordinates (BED convention).
    We fetch base at [start, start+1).
    """
    mutations_path = Path(mutations_path)
    reference_fasta = Path(reference_fasta)

    if not mutations_path.exists():
        raise FileNotFoundError(f"Mutations file not found: {mutations_path}")
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    sampled = _reservoir_sample_rows_bed_no_header(
        path=mutations_path,
        seed=seed,
        k=n_rows_to_test,
        chunksize=chunksize,
        tumour_whitelist=tumour_whitelist,
    )

    fasta = pysam.FastaFile(str(reference_fasta))

    matched = mismatched = skipped = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for r in sampled:
        chrom = r.get("Chromosome")
        start = r.get("Start")
        ref = r.get("Ref")
        sample = r.get("Sample_ID")

        if ref is None or str(ref).strip() in ("", ".", "NA"):
            skipped += 1
            continue
        ref = str(ref).strip().upper()
        if len(ref) != 1:
            skipped += 1
            continue

        try:
            start_i = int(str(start).strip())  # BED is 0-based
        except Exception:
            skipped += 1
            continue
        if start_i < 0:
            skipped += 1
            continue

        contig = _normalise_chrom(chrom, fasta)
        if contig is None:
            skipped += 1
            continue

        try:
            base = fasta.fetch(contig, start_i, start_i + 1).upper()
        except Exception:
            skipped += 1
            continue

        if base == ref:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < max_mismatch_examples:
                mismatch_examples.append(
                    {
                        "Sample_ID": sample,
                        "Chromosome": chrom,
                        "Start_0_based": start_i,
                        "Reference_Allele_in_file": ref,
                        "Reference_Allele_in_fasta": base,
                        "Resolved_Contig": contig,
                    }
                )

    fasta.close()

    return RefCheckResult(
        reference_fasta=str(reference_fasta),
        seed=seed,
        n_tested=len(sampled),
        n_matched=matched,
        n_mismatched=mismatched,
        n_skipped=skipped,
        mismatch_examples=mismatch_examples,
    )


def check_reference_genome_uv_bed_multi(
    mutations_paths: Sequence[str | Path],
    reference_fasta: str | Path,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
    max_mismatch_examples: int = 5,
) -> RefCheckResult:
    paths = [Path(p) for p in mutations_paths]
    reference_fasta = Path(reference_fasta)

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Mutations file not found: {p}")
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    sampled = _reservoir_sample_rows_bed_no_header_multi(
        paths=paths,
        seed=seed,
        k=n_rows_to_test,
        chunksize=chunksize,
        tumour_whitelist=tumour_whitelist,
    )

    fasta = pysam.FastaFile(str(reference_fasta))

    matched = mismatched = skipped = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for r in sampled:
        chrom = r.get("Chromosome")
        start = r.get("Start")
        ref = r.get("Ref")
        sample = r.get("Sample_ID")

        if ref is None or str(ref).strip() in ("", ".", "NA"):
            skipped += 1
            continue
        ref = str(ref).strip().upper()
        if len(ref) != 1:
            skipped += 1
            continue

        try:
            start_i = int(str(start).strip())  # BED is 0-based
        except Exception:
            skipped += 1
            continue
        if start_i < 0:
            skipped += 1
            continue

        contig = _normalise_chrom(chrom, fasta)
        if contig is None:
            skipped += 1
            continue

        try:
            base = fasta.fetch(contig, start_i, start_i + 1).upper()
        except Exception:
            skipped += 1
            continue

        if base == ref:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < max_mismatch_examples:
                mismatch_examples.append(
                    {
                        "Sample_ID": sample,
                        "Chromosome": chrom,
                        "Start_0_based": start_i,
                        "Reference_Allele_in_file": ref,
                        "Reference_Allele_in_fasta": base,
                        "Resolved_Contig": contig,
                    }
                )

    fasta.close()

    return RefCheckResult(
        reference_fasta=str(reference_fasta),
        seed=seed,
        n_tested=len(sampled),
        n_matched=matched,
        n_mismatched=mismatched,
        n_skipped=skipped,
        mismatch_examples=mismatch_examples,
    )


# -----------------------------
# Stats: TCGA MAF-style TSV
# -----------------------------
def compute_stats_tcga_maf(
    mutations_path: str | Path,
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> MutationStats:
    mutations_path = Path(mutations_path)
    if not mutations_path.exists():
        raise FileNotFoundError(f"Mutations file not found: {mutations_path}")

    usecols = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Variant_Type",
    ]

    n_rows = 0
    missing_sample_barcode_rows = 0

    samples = set()
    patients = set()
    genes = set()
    variants = set()

    muts_per_sample: Dict[str, int] = {}
    muts_per_patient: Dict[str, int] = {}

    varclass_counts: Dict[str, int] = {}
    snv_count = 0
    indel_count = 0

    for chunk in pd.read_csv(
        mutations_path,
        sep="\t",
        usecols=lambda c: c in usecols,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        n_rows += len(chunk)

        sample_series = chunk.get("Tumor_Sample_Barcode")
        if sample_series is None:
            raise ValueError("Column 'Tumor_Sample_Barcode' not found in file.")

        is_missing = sample_series.isna() | (sample_series.str.strip() == "")
        missing_sample_barcode_rows += int(is_missing.sum())

        chunk = chunk.loc[~is_missing].copy()
        if chunk.empty:
            continue

        # Samples and per-sample counts
        for s in chunk["Tumor_Sample_Barcode"].astype(str):
            s = s.strip()
            if not s:
                continue
            samples.add(s)
            muts_per_sample[s] = muts_per_sample.get(s, 0) + 1

            pid = _patient_id_from_tcga_barcode(s)
            if pid:
                patients.add(pid)
                muts_per_patient[pid] = muts_per_patient.get(pid, 0) + 1

        # Genes
        if "Hugo_Symbol" in chunk.columns:
            for g in chunk["Hugo_Symbol"].dropna().astype(str):
                g = g.strip()
                if g and g != ".":
                    genes.add(g)

        # Variant classification counts
        if "Variant_Classification" in chunk.columns:
            vc = chunk["Variant_Classification"].fillna("NA").astype(str)
            for v in vc:
                varclass_counts[v] = varclass_counts.get(v, 0) + 1

        # Variant type counts
        if "Variant_Type" in chunk.columns:
            vt = chunk["Variant_Type"].fillna("NA").astype(str).str.upper()
            snv_count += int((vt == "SNP").sum()) + int((vt == "SNV").sum())
            indel_count += int(vt.isin(["INS", "DEL", "INDEL"]).sum())

        # Unique variant key per sample (sample|chrom|pos|ref|alt)
        needed = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Tumor_Sample_Barcode"]
        if all(c in chunk.columns for c in needed):
            sub = chunk[needed].fillna("").astype(str)
            for row in sub.itertuples(index=False):
                chrom, pos, ref, alt, sample = row
                variants.add(f"{sample}|{chrom}|{pos}|{ref}|{alt}")

    # Summaries
    s_counts = pd.Series(list(muts_per_sample.values()), dtype="int64")
    p_counts = pd.Series(list(muts_per_patient.values()), dtype="int64") if muts_per_patient else pd.Series([], dtype="int64")

    top_vc = "NA"
    top_vc_count = 0
    if varclass_counts:
        top_vc, top_vc_count = max(varclass_counts.items(), key=lambda kv: kv[1])

    n_unique_samples = len(samples)
    avg_per_sample = (n_rows / n_unique_samples) if n_unique_samples else 0.0

    # Optional reference check
    ref_check = None
    if reference_fasta is not None:
        ref_check = check_reference_genome_tcga_maf(
            mutations_path=mutations_path,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
        )

    return MutationStats(
        file_path=str(mutations_path),
        file_format="tcga_maf_tsv",
        n_rows=int(n_rows),
        n_unique_samples=len(samples),
        n_unique_patients=len(patients),
        n_unique_genes=len(genes),
        n_unique_variants=len(variants),
        avg_mutations_per_sample=float(avg_per_sample),
        median_mutations_per_sample=float(s_counts.median()) if len(s_counts) else 0.0,
        median_mutations_per_patient=float(p_counts.median()) if len(p_counts) else 0.0,
        top_variant_classification=top_vc,
        top_variant_classification_count=int(top_vc_count),
        snv_count=int(snv_count),
        indel_count=int(indel_count),
        missing_sample_barcode_rows=int(missing_sample_barcode_rows),
        ref_check=ref_check,
    )


def compute_stats_tcga_maf_multi(
    mutations_paths: Sequence[str | Path],
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
) -> MutationStats:
    paths = [Path(p) for p in mutations_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Mutations file not found: {p}")

    usecols = [
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Variant_Type",
    ]

    n_rows = 0
    missing_sample_barcode_rows = 0

    samples = set()
    patients = set()
    genes = set()
    variants = set()

    muts_per_sample: Dict[str, int] = {}
    muts_per_patient: Dict[str, int] = {}

    varclass_counts: Dict[str, int] = {}
    snv_count = 0
    indel_count = 0

    for mutations_path in paths:
        for chunk in pd.read_csv(
            mutations_path,
            sep="\t",
            usecols=lambda c: c in usecols,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        ):
            n_rows += len(chunk)

            sample_series = chunk.get("Tumor_Sample_Barcode")
            if sample_series is None:
                raise ValueError("Column 'Tumor_Sample_Barcode' not found in file.")

            is_missing = sample_series.isna() | (sample_series.str.strip() == "")
            missing_sample_barcode_rows += int(is_missing.sum())

            chunk = chunk.loc[~is_missing].copy()
            if chunk.empty:
                continue

            # Samples and per-sample counts
            for s in chunk["Tumor_Sample_Barcode"].astype(str):
                s = s.strip()
                if not s:
                    continue
                samples.add(s)
                muts_per_sample[s] = muts_per_sample.get(s, 0) + 1

                pid = _patient_id_from_tcga_barcode(s)
                if pid:
                    patients.add(pid)
                    muts_per_patient[pid] = muts_per_patient.get(pid, 0) + 1

            # Genes
            if "Hugo_Symbol" in chunk.columns:
                for g in chunk["Hugo_Symbol"].dropna().astype(str):
                    g = g.strip()
                    if g and g != ".":
                        genes.add(g)

            # Variant classification counts
            if "Variant_Classification" in chunk.columns:
                vc = chunk["Variant_Classification"].fillna("NA").astype(str)
                for v in vc:
                    varclass_counts[v] = varclass_counts.get(v, 0) + 1

            # Variant type counts
            if "Variant_Type" in chunk.columns:
                vt = chunk["Variant_Type"].fillna("NA").astype(str).str.upper()
                snv_count += int((vt == "SNP").sum()) + int((vt == "SNV").sum())
                indel_count += int(vt.isin(["INS", "DEL", "INDEL"]).sum())

            # Unique variant key per sample (sample|chrom|pos|ref|alt)
            needed = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Tumor_Sample_Barcode"]
            if all(c in chunk.columns for c in needed):
                sub = chunk[needed].fillna("").astype(str)
                for row in sub.itertuples(index=False):
                    chrom, pos, ref, alt, sample = row
                    variants.add(f"{sample}|{chrom}|{pos}|{ref}|{alt}")

    # Summaries
    s_counts = pd.Series(list(muts_per_sample.values()), dtype="int64")
    p_counts = pd.Series(list(muts_per_patient.values()), dtype="int64") if muts_per_patient else pd.Series([], dtype="int64")

    top_vc = "NA"
    top_vc_count = 0
    if varclass_counts:
        top_vc, top_vc_count = max(varclass_counts.items(), key=lambda kv: kv[1])

    n_unique_samples = len(samples)
    avg_per_sample = (n_rows / n_unique_samples) if n_unique_samples else 0.0

    # Optional reference check
    ref_check = None
    if reference_fasta is not None:
        ref_check = check_reference_genome_tcga_maf_multi(
            mutations_paths=paths,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
        )

    return MutationStats(
        file_path=", ".join(str(p) for p in paths),
        file_format="tcga_maf_tsv",
        n_rows=int(n_rows),
        n_unique_samples=len(samples),
        n_unique_patients=len(patients),
        n_unique_genes=len(genes),
        n_unique_variants=len(variants),
        avg_mutations_per_sample=float(avg_per_sample),
        median_mutations_per_sample=float(s_counts.median()) if len(s_counts) else 0.0,
        median_mutations_per_patient=float(p_counts.median()) if len(p_counts) else 0.0,
        top_variant_classification=top_vc,
        top_variant_classification_count=int(top_vc_count),
        snv_count=int(snv_count),
        indel_count=int(indel_count),
        missing_sample_barcode_rows=int(missing_sample_barcode_rows),
        ref_check=ref_check,
    )


# -----------------------------
# Stats: UV bed-like (no header)
# -----------------------------
def compute_stats_uv_bed(
    mutations_path: str | Path,
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> MutationStats:
    mutations_path = Path(mutations_path)
    if not mutations_path.exists():
        raise FileNotFoundError(f"Mutations file not found: {mutations_path}")

    colnames = ["Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Cancer_Type", "Trinuc_Context"]

    n_rows = 0
    missing_sample_barcode_rows = 0

    samples = set()
    patients = set()
    variants = set()

    muts_per_sample: Dict[str, int] = {}
    muts_per_patient: Dict[str, int] = {}

    snv_count = 0
    indel_count = 0

    allowed = _normalise_tumour_whitelist(tumour_whitelist)

    for chunk in pd.read_csv(
        mutations_path,
        sep="\t",
        header=None,
        names=colnames,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        if allowed:
            mask = _matches_tumour_whitelist(chunk["Cancer_Type"], allowed)
            chunk = chunk.loc[mask]
            if chunk.empty:
                continue

        n_rows += len(chunk)

        sample_series = chunk.get("Sample_ID")
        if sample_series is None:
            raise ValueError("Expected Sample_ID column could not be read (UV bed-like parsing failed).")

        is_missing = sample_series.isna() | (sample_series.astype(str).str.strip() == "")
        missing_sample_barcode_rows += int(is_missing.sum())

        chunk = chunk.loc[~is_missing].copy()
        if chunk.empty:
            continue

        # Samples and per-sample counts
        for s in chunk["Sample_ID"].astype(str):
            s = s.strip()
            if not s:
                continue
            samples.add(s)
            muts_per_sample[s] = muts_per_sample.get(s, 0) + 1

            pid = patient_id_from_mixed_sample_id(s, uv_unique_patients=True)
            if pid:
                patients.add(pid)
                muts_per_patient[pid] = muts_per_patient.get(pid, 0) + 1

        # SNV/indel inferred from allele lengths (best-effort)
        ref = chunk["Ref"].fillna("").astype(str).str.strip()
        alt = chunk["Alt"].fillna("").astype(str).str.strip()
        is_snv = (ref.str.len() == 1) & (alt.str.len() == 1)
        snv_count += int(is_snv.sum())
        indel_count += int((~is_snv & (ref != "") & (alt != "")).sum())

        # Unique variant key per sample (sample|chrom|start|ref|alt)
        needed = ["Chromosome", "Start", "Ref", "Alt", "Sample_ID"]
        sub = chunk[needed].fillna("").astype(str)
        for row in sub.itertuples(index=False):
            chrom, start, ref_a, alt_a, sample = row
            variants.add(f"{sample}|{chrom}|{start}|{ref_a}|{alt_a}")

    # Summaries
    s_counts = pd.Series(list(muts_per_sample.values()), dtype="int64")
    p_counts = pd.Series(list(muts_per_patient.values()), dtype="int64") if muts_per_patient else pd.Series([], dtype="int64")

    n_unique_samples = len(samples)
    avg_per_sample = (n_rows / n_unique_samples) if n_unique_samples else 0.0

    # Optional reference check
    ref_check = None
    if reference_fasta is not None:
        ref_check = check_reference_genome_uv_bed(
            mutations_path=mutations_path,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
            tumour_whitelist=tumour_whitelist,
        )

    # Fields not applicable to UV bed-like are set to neutral defaults
    return MutationStats(
        file_path=str(mutations_path),
        file_format="uv_bed_like",
        n_rows=int(n_rows),
        n_unique_samples=len(samples),
        n_unique_patients=len(patients),
        n_unique_genes=0,
        n_unique_variants=len(variants),
        avg_mutations_per_sample=float(avg_per_sample),
        median_mutations_per_sample=float(s_counts.median()) if len(s_counts) else 0.0,
        median_mutations_per_patient=float(p_counts.median()) if len(p_counts) else 0.0,
        top_variant_classification="NA",
        top_variant_classification_count=0,
        snv_count=int(snv_count),
        indel_count=int(indel_count),
        missing_sample_barcode_rows=int(missing_sample_barcode_rows),
        ref_check=ref_check,
    )


def compute_stats_uv_bed_multi(
    mutations_paths: Sequence[str | Path],
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> MutationStats:
    paths = [Path(p) for p in mutations_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Mutations file not found: {p}")

    colnames = ["Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Cancer_Type", "Trinuc_Context"]
    allowed = _normalise_tumour_whitelist(tumour_whitelist)

    n_rows = 0
    missing_sample_barcode_rows = 0

    samples = set()
    patients = set()
    variants = set()

    muts_per_sample: Dict[str, int] = {}
    muts_per_patient: Dict[str, int] = {}

    snv_count = 0
    indel_count = 0

    for mutations_path in paths:
        for chunk in pd.read_csv(
            mutations_path,
            sep="\t",
            header=None,
            names=colnames,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        ):
            if allowed:
                mask = _matches_tumour_whitelist(chunk["Cancer_Type"], allowed)
                chunk = chunk.loc[mask]
                if chunk.empty:
                    continue

            n_rows += len(chunk)

            sample_series = chunk.get("Sample_ID")
            if sample_series is None:
                raise ValueError("Expected Sample_ID column could not be read (UV bed-like parsing failed).")

            is_missing = sample_series.isna() | (sample_series.astype(str).str.strip() == "")
            missing_sample_barcode_rows += int(is_missing.sum())

            chunk = chunk.loc[~is_missing].copy()
            if chunk.empty:
                continue

            # Samples and per-sample counts
            for s in chunk["Sample_ID"].astype(str):
                s = s.strip()
                if not s:
                    continue
                samples.add(s)
                muts_per_sample[s] = muts_per_sample.get(s, 0) + 1

                pid = patient_id_from_mixed_sample_id(s, uv_unique_patients=True)
                if pid:
                    patients.add(pid)
                    muts_per_patient[pid] = muts_per_patient.get(pid, 0) + 1

            # SNV/indel inferred from allele lengths (best-effort)
            ref = chunk["Ref"].fillna("").astype(str).str.strip()
            alt = chunk["Alt"].fillna("").astype(str).str.strip()
            is_snv = (ref.str.len() == 1) & (alt.str.len() == 1)
            snv_count += int(is_snv.sum())
            indel_count += int((~is_snv & (ref != "") & (alt != "")).sum())

            # Unique variant key per sample (sample|chrom|start|ref|alt)
            needed = ["Chromosome", "Start", "Ref", "Alt", "Sample_ID"]
            sub = chunk[needed].fillna("").astype(str)
            for row in sub.itertuples(index=False):
                chrom, start, ref_a, alt_a, sample = row
                variants.add(f"{sample}|{chrom}|{start}|{ref_a}|{alt_a}")

    # Summaries
    s_counts = pd.Series(list(muts_per_sample.values()), dtype="int64")
    p_counts = pd.Series(list(muts_per_patient.values()), dtype="int64") if muts_per_patient else pd.Series([], dtype="int64")

    n_unique_samples = len(samples)
    avg_per_sample = (n_rows / n_unique_samples) if n_unique_samples else 0.0

    # Optional reference check
    ref_check = None
    if reference_fasta is not None:
        ref_check = check_reference_genome_uv_bed_multi(
            mutations_paths=paths,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
            tumour_whitelist=tumour_whitelist,
        )

    # Fields not applicable to UV bed-like are set to neutral defaults
    return MutationStats(
        file_path=", ".join(str(p) for p in paths),
        file_format="uv_bed_like",
        n_rows=int(n_rows),
        n_unique_samples=len(samples),
        n_unique_patients=len(patients),
        n_unique_genes=0,
        n_unique_variants=len(variants),
        avg_mutations_per_sample=float(avg_per_sample),
        median_mutations_per_sample=float(s_counts.median()) if len(s_counts) else 0.0,
        median_mutations_per_patient=float(p_counts.median()) if len(p_counts) else 0.0,
        top_variant_classification="NA",
        top_variant_classification_count=0,
        snv_count=int(snv_count),
        indel_count=int(indel_count),
        missing_sample_barcode_rows=int(missing_sample_barcode_rows),
        ref_check=ref_check,
    )


# -----------------------------
# Orchestration
# -----------------------------
def compute_mutation_stats_any(
    mutations_path: str | Path,
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> MutationStats:
    p = Path(mutations_path)
    fmt = detect_file_format(p)

    # If detection says TCGA, try it, but if key columns are missing, fall back to UV parser.
    if fmt == "tcga_maf_tsv":
        try:
            return compute_stats_tcga_maf(
                mutations_path=p,
                reference_fasta=reference_fasta,
                seed=seed,
                n_rows_to_test=n_rows_to_test,
                chunksize=chunksize,
                tumour_whitelist=tumour_whitelist,
            )
        except ValueError as e:
            msg = str(e)
            if "Tumor_Sample_Barcode" in msg or "Column" in msg:
                # Try UV bed-like as a graceful fallback
                return compute_stats_uv_bed(
                    mutations_path=p,
                    reference_fasta=reference_fasta,
                    seed=seed,
                    n_rows_to_test=n_rows_to_test,
                    chunksize=chunksize,
                    tumour_whitelist=tumour_whitelist,
                )
            raise

    if fmt == "uv_bed_like":
        return compute_stats_uv_bed(
            mutations_path=p,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
            tumour_whitelist=tumour_whitelist,
        )

    # Unknown: attempt UV bed-like if extension is .bed, otherwise attempt TCGA, otherwise minimal
    if p.suffix.lower() == ".bed":
        return compute_stats_uv_bed(
            mutations_path=p,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
            tumour_whitelist=tumour_whitelist,
        )

    try:
        return compute_stats_tcga_maf(
            mutations_path=p,
            reference_fasta=reference_fasta,
            seed=seed,
            n_rows_to_test=n_rows_to_test,
            chunksize=chunksize,
            tumour_whitelist=tumour_whitelist,
        )
    except Exception:
        # minimal fallback
        n_rows = 0
        if p.exists():
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for _ in f:
                    n_rows += 1

        return MutationStats(
            file_path=str(p),
            file_format="unknown",
            n_rows=int(n_rows),
            n_unique_samples=0,
            n_unique_patients=0,
            n_unique_genes=0,
            n_unique_variants=0,
            avg_mutations_per_sample=0.0,
            median_mutations_per_sample=0.0,
            median_mutations_per_patient=0.0,
            top_variant_classification="NA",
            top_variant_classification_count=0,
            snv_count=0,
            indel_count=0,
            missing_sample_barcode_rows=0,
            ref_check=None,
        )


def stats_as_dict(stats: MutationStats) -> Dict[str, Any]:
    return asdict(stats)


def _discover_default_files(data_dir: Path) -> List[Path]:
    """
    Default expectation: you have these two files.
    If they exist, include them. Otherwise fall back to common mutation-like files.
    """
    candidates = []
    for name in ["data_mutations.txt", "UV_mutations.bed", "uv_mutations.bed", "UV_mutations.tsv"]:
        p = data_dir / name
        if p.exists():
            candidates.append(p)

    if candidates:
        return candidates

    # fallback: pick typical extensions
    exts = (".bed", ".tsv", ".txt", ".maf")
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            candidates.append(p)
    return candidates


def _parse_files_arg(files: Optional[List[str]]) -> Optional[List[Path]]:
    if not files:
        return None
    out: List[Path] = []
    for item in files:
        # allow comma-separated in a single token
        parts = [x for x in item.split(",") if x.strip()]
        for p in parts:
            out.append(Path(p.strip()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute summary stats for TCGA MAF-style mutations and UV bed-like WGS mutations."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory to search when using --all (default: current directory).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all default mutation files in --data-dir (prefers data_mutations.txt and UV_mutations.bed if present).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Process only these files (space-separated or comma-separated). Example: --files data_mutations.txt UV_mutations.bed",
    )
    parser.add_argument(
        "--reference-fasta",
        type=str,
        default=None,
        help="Optional reference FASTA (hg19/GRCh37) for random ref allele checks (requires .fai index).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reference checking sampling.")
    parser.add_argument(
        "--n-rows-to-test",
        type=int,
        default=50,
        help="How many randomly sampled rows to check against the reference FASTA.",
    )
    parser.add_argument("--chunksize", type=int, default=250_000, help="CSV chunk size for streaming reads.")
    parser.add_argument(
        "--tumour-whitelist",
        type=str,
        default=None,
        help="Comma list of tumour codes to keep (e.g., 'SKCM,MELA'); case-insensitive.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path. If omitted, prints JSON to stdout.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    file_list: List[Path]
    explicit = _parse_files_arg(args.files)

    if args.all:
        file_list = _discover_default_files(data_dir)
    elif explicit is not None:
        file_list = explicit
    else:
        # sensible default: try the two canonical filenames in current dir
        file_list = _discover_default_files(data_dir)

    # Normalise to absolute-ish paths based on data_dir when relative
    resolved_files: List[Path] = []
    for p in file_list:
        if not p.is_absolute():
            # if user passed "UV_mutations.bed" and it's in data_dir, prefer that
            cand = (data_dir / p)
            resolved_files.append(cand if cand.exists() else p)
        else:
            resolved_files.append(p)

    ref_fa = args.reference_fasta

    tumour_whitelist = None
    if args.tumour_whitelist:
        tumour_whitelist = [t.strip() for t in args.tumour_whitelist.split(",") if t.strip()]

    results: List[Dict[str, Any]] = []
    for fp in resolved_files:
        stats = compute_mutation_stats_any(
            mutations_path=fp,
            reference_fasta=ref_fa,
            seed=args.seed,
            n_rows_to_test=args.n_rows_to_test,
            chunksize=args.chunksize,
            tumour_whitelist=tumour_whitelist,
        )
        results.append(stats_as_dict(stats))

    payload = {
        "reference_fasta": ref_fa,
        "n_files": len(results),
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


def compute_mutation_stats(
    mutations_path: str | Path | Sequence[str | Path],
    reference_fasta: str | Path | None = None,
    seed: int = 123,
    n_rows_to_test: int = 50,
    chunksize: int = 250_000,
    tumour_whitelist: Optional[Sequence[str]] = None,
) -> MutationStats:
    """
    Compute stats for either:
      - TCGA MAF-style TSV (e.g. data_mutations.txt), or
      - UV bed-like no-header file (e.g. UV_mutations.bed)

    This wrapper exists so you can do:
        from scripts.mutation_stats import compute_mutation_stats, stats_as_dict
    """
    if isinstance(mutations_path, (list, tuple)):
        paths = [Path(p) for p in mutations_path]
        if not paths:
            raise ValueError("mutations_path list is empty")
        fmts = [detect_file_format(p) for p in paths]
        if all(f == "uv_bed_like" for f in fmts):
            return compute_stats_uv_bed_multi(
                mutations_paths=paths,
                reference_fasta=reference_fasta,
                seed=seed,
                n_rows_to_test=n_rows_to_test,
                chunksize=chunksize,
                tumour_whitelist=tumour_whitelist,
            )
        if all(f == "tcga_maf_tsv" for f in fmts):
            return compute_stats_tcga_maf_multi(
                mutations_paths=paths,
                reference_fasta=reference_fasta,
                seed=seed,
                n_rows_to_test=n_rows_to_test,
                chunksize=chunksize,
            )
        raise ValueError(
            "mutations_path contains mixed or unsupported formats; "
            "pass only UV bed-like files or only TCGA MAF-style files"
        )
    return compute_mutation_stats_any(
        mutations_path=mutations_path,
        reference_fasta=reference_fasta,
        seed=seed,
        n_rows_to_test=n_rows_to_test,
        chunksize=chunksize,
        tumour_whitelist=tumour_whitelist,
    )


if __name__ == "__main__":
    main()
