"""
sample_selector.py

Purpose
-------
Select a reproducible random subset of samples from one or more BED-like mutation
files and extract all mutations for those samples, standardising to a common schema.

Supported input formats (no header, tab-separated)
--------------------------------------------------
1) UV_mutations.bed style (8 columns):
   Chrom  Start  End  Sample_ID  Ref  Alt  Tumour  Context

   Example:
   chr1    87302   87303   TCGA-DA-A1HW   G   A   SKCM   cgg

2) ICGC_WGS_Feb20_mutations.bed style (8 columns, as in your example):
   Chrom  Start  End  Donor_ID  Ref  Alt  Project  Sample_ID

   Example:
   chr1  2112412  2112413  DO1000  T  C  BRCA-UK  PD3851a

Output schema (always)
----------------------
Chromosome, Start, End, Sample_ID, Ref, Alt, Tumour, Context

Notes
-----
- Auto-detection chooses Sample_ID column and maps fields into the standard schema.
- You can override sample id column with `sample_id_col_override`.
- If you pass a list of files, selection is global across all files
  (deterministic shuffle over (file, sample) pairs).
- Larger k contains smaller k when the seed is fixed.
- Tumour labels are normalized to the base name before any "-" suffix.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from scripts.contigs import canonicalise_contig

COLNAMES = ["Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Tumour", "Context"]


def _normalise_tumour_value(value: str) -> str:
    s = str(value).strip()
    if not s:
        return ""
    return s.split("-", 1)[0].strip()


def _normalise_tumour_filter(tumour_filter: Optional[Sequence[str]]) -> Set[str]:
    if not tumour_filter:
        return set()
    return {
        _normalise_tumour_value(x).upper()
        for x in tumour_filter
        if _normalise_tumour_value(x)
    }


def _apply_tumour_filter(
        df: pd.DataFrame, tumour_filter: Optional[Sequence[str]]
) -> pd.DataFrame:
    allowed = _normalise_tumour_filter(tumour_filter)
    if not allowed or df.empty:
        return df

    t_up = df["Tumour"].fillna("").astype(str).str.strip().str.upper()
    return df.loc[t_up.isin(allowed)]


# -----------------------------
# Format detection + projection
# -----------------------------
@dataclass(frozen=True)
class MutationFormat:
    """
    Describes how to map an input file's columns onto the standard schema.
    All indices are 0-based.
    """
    n_cols: int
    chrom_col: int = 0
    start_col: int = 1
    end_col: int = 2
    sample_col: int = 3
    ref_col: int = 4
    alt_col: int = 5
    tumour_col: Optional[int] = 6
    context_col: Optional[int] = 7
    name: str = "uv_like"


def _read_head_chunk(bed_path: Path, nrows: int = 2000) -> pd.DataFrame:
    return pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        dtype=str,
        nrows=nrows,
        low_memory=False,
    )


def detect_mutation_format(
        bed_path: Union[str, Path],
        *,
        sample_id_col_override: Optional[int] = None,
) -> MutationFormat:
    """
    Detect whether the file is UV-like or ICGC-like (as per examples).
    Returns a MutationFormat that projects into the standard schema.

    If sample_id_col_override is provided, it will be used as Sample_ID column.
    """
    bed_path = Path(bed_path)
    head = _read_head_chunk(bed_path, nrows=2000)

    if head.empty:
        # Default; extraction will just produce empty output anyway.
        return MutationFormat(n_cols=8, name="empty")

    n_cols = int(head.shape[1])
    if n_cols < 6:
        raise ValueError(f"Expected at least 6 columns, got {n_cols} in {bed_path}")

    # If user overrides sample id column, respect it.
    if sample_id_col_override is not None:
        if sample_id_col_override < 0 or sample_id_col_override >= n_cols:
            raise ValueError(
                f"sample_id_col_override={sample_id_col_override} out of range for {bed_path} with {n_cols} cols"
            )
        # Best-effort: assume chrom/start/end at 0/1/2 and ref/alt at 4/5 if present.
        tumour_col = 6 if n_cols > 6 else None
        context_col = 7 if n_cols > 7 else None
        return MutationFormat(
            n_cols=n_cols,
            sample_col=sample_id_col_override,
            tumour_col=tumour_col,
            context_col=context_col,
            name=f"override_sample_col_{sample_id_col_override}",
        )

    # Heuristics for 8-column files (your two cases are both 8 columns).
    # UV-like: col3 contains TCGA-*, Zheng_*, etc. and col7 is short context (e.g. cgg/GGA)
    # ICGC-like: col3 looks like DO1234 and col7 looks like a sample id (e.g. PD3851a)
    if n_cols >= 8:
        c3 = head.iloc[:, 3].fillna("").astype(str).str.strip()
        c7 = head.iloc[:, 7].fillna("").astype(str).str.strip()

        # UV indicators
        uv_like = (
                c3.str.contains(r"^(?:TCGA-|Zheng_)", regex=True).mean() > 0.05
                or bed_path.name.lower().startswith("uv_")
                or "uv" in bed_path.name.lower()
        )

        # ICGC indicators
        donor_like = c3.str.contains(r"^DO\d+$", regex=True).mean() > 0.05
        sample_like = c7.str.contains(r"^[A-Za-z]{1,4}\d+[A-Za-z]?$", regex=True).mean() > 0.05
        icgc_like = (donor_like and sample_like) or ("icgc" in bed_path.name.lower())

        if icgc_like and not uv_like:
            # Map: tumour_col=6 (Project), context_col=None unless you want to keep something else.
            # We'll store Project into Tumour, and store "" in Context.
            return MutationFormat(
                n_cols=n_cols,
                sample_col=7,
                tumour_col=6,
                context_col=None,
                name="icgc_like",
            )

        # Default to UV-like mapping
        return MutationFormat(
            n_cols=n_cols,
            sample_col=3,
            tumour_col=6,
            context_col=7,
            name="uv_like",
        )

    # For other column counts: assume sample is col3 and best-effort tumour/context at 6/7 if present
    tumour_col = 6 if n_cols > 6 else None
    context_col = 7 if n_cols > 7 else None
    return MutationFormat(
        n_cols=n_cols,
        sample_col=3 if n_cols > 3 else 0,
        tumour_col=tumour_col,
        context_col=context_col,
        name=f"generic_{n_cols}cols",
    )


def _project_chunk_to_standard_schema(chunk: pd.DataFrame, fmt: MutationFormat) -> pd.DataFrame:
    """
    Given a raw chunk (with integer columns 0..n-1), project into COLNAMES.
    Missing tumour/context are filled with "".
    """

    def col_or_empty(idx: Optional[int]) -> pd.Series:
        if idx is None or idx >= chunk.shape[1]:
            return pd.Series([""] * len(chunk), index=chunk.index, dtype=str)
        return chunk.iloc[:, idx].fillna("").astype(str)

    out = pd.DataFrame(
        {
            "Chromosome": col_or_empty(fmt.chrom_col),
            "Start": col_or_empty(fmt.start_col),
            "End": col_or_empty(fmt.end_col),
            "Sample_ID": col_or_empty(fmt.sample_col),
            "Ref": col_or_empty(fmt.ref_col) if fmt.ref_col < chunk.shape[1] else "",
            "Alt": col_or_empty(fmt.alt_col) if fmt.alt_col < chunk.shape[1] else "",
            "Tumour": col_or_empty(fmt.tumour_col),
            "Context": col_or_empty(fmt.context_col),
        }
    )
    return out[COLNAMES]


# -----------------------------
# Selection objects
# -----------------------------
@dataclass(frozen=True)
class SampleSelection:
    seed: int
    ordered_samples: List[str]

    def first_k(self, k: int | None) -> List[str]:
        if k is None:
            return self.ordered_samples
        if k < 0:
            raise ValueError("k must be >= 0")
        return self.ordered_samples[:k]


# -----------------------------
# Single-file operations
# -----------------------------
def infer_sample_order(
        bed_path: str | Path,
        seed: int = 123,
        chunksize: int = 250_000,
        *,
        sample_id_col_override: Optional[int] = None,
) -> SampleSelection:
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"File not found: {bed_path}")

    fmt = detect_mutation_format(bed_path, sample_id_col_override=sample_id_col_override)
    samples: Set[str] = set()

    # Read only the sample column.
    usecols = [fmt.sample_col]
    for raw_chunk in pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            usecols=usecols,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
    ):
        s = raw_chunk.iloc[:, 0].fillna("").astype(str).str.strip()
        s = s[s != ""]
        samples.update(s.tolist())

    ordered = sorted(samples)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    return SampleSelection(seed=seed, ordered_samples=ordered)


def extract_mutations_for_samples(
        bed_path: str | Path,
        sample_ids: Sequence[str],
        chunksize: int = 250_000,
        *,
        sample_id_col_override: Optional[int] = None,
        tumour_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    bed_path = Path(bed_path)
    want: Set[str] = set(sample_ids)
    if not want:
        return pd.DataFrame(columns=COLNAMES)

    fmt = detect_mutation_format(bed_path, sample_id_col_override=sample_id_col_override)

    kept: List[pd.DataFrame] = []

    # We must read at least through the sample column; easiest is to read all cols then project.
    for raw_chunk in pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
    ):
        # Project to standard schema
        chunk = _project_chunk_to_standard_schema(raw_chunk, fmt)

        # ----------------------------
        # STANDARDISE EVERYTHING HERE
        # ----------------------------
        # 1) strip whitespace
        for c in COLNAMES:
            chunk[c] = chunk[c].fillna("").astype(str).str.strip()

        # 2) Chromosome: canonicalise to chr1..chr22, chrX, chrY; drop non-primary/MT
        chunk["Chromosome"] = chunk["Chromosome"].map(canonicalise_contig)
        chunk = chunk.loc[chunk["Chromosome"].notna()]

        # 3) Start/End: ensure clean ints-as-strings
        chunk["Start"] = chunk["Start"].str.replace(r"\.0$", "", regex=True)
        chunk["End"] = chunk["End"].str.replace(r"\.0$", "", regex=True)

        # 4) Sample_ID: strip (already done) but keep explicit
        chunk["Sample_ID"] = chunk["Sample_ID"].str.strip()

        # 5) Tumour: collapse to base label (before "-")
        chunk["Tumour"] = chunk["Tumour"].map(_normalise_tumour_value)

        # Apply tumour filter after standardisation
        chunk = _apply_tumour_filter(chunk, tumour_filter)

        # Filter by sample set
        mask = chunk["Sample_ID"].isin(want)
        sub = chunk.loc[mask]
        if not sub.empty:
            kept.append(sub)

    if not kept:
        return pd.DataFrame(columns=COLNAMES)

    out = pd.concat(kept, ignore_index=True)
    return out


def count_mutations_per_sample(
        bed_path: str | Path,
        chunksize: int = 250_000,
        *,
        sample_id_col_override: Optional[int] = None,
        tumour_filter: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Return per-sample mutation counts after standardisation and optional tumour filtering.
    """
    bed_path = Path(bed_path)
    fmt = detect_mutation_format(bed_path, sample_id_col_override=sample_id_col_override)
    counts: Dict[str, int] = defaultdict(int)

    for raw_chunk in pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
    ):
        chunk = _project_chunk_to_standard_schema(raw_chunk, fmt)

        for c in COLNAMES:
            chunk[c] = chunk[c].fillna("").astype(str).str.strip()

        chunk["Chromosome"] = chunk["Chromosome"].map(canonicalise_contig)
        chunk = chunk.loc[chunk["Chromosome"].notna()]

        chunk["Start"] = chunk["Start"].str.replace(r"\.0$", "", regex=True)
        chunk["End"] = chunk["End"].str.replace(r"\.0$", "", regex=True)
        chunk["Sample_ID"] = chunk["Sample_ID"].str.strip()
        chunk["Tumour"] = chunk["Tumour"].map(_normalise_tumour_value)

        chunk = _apply_tumour_filter(chunk, tumour_filter)
        chunk = chunk.loc[chunk["Sample_ID"] != ""]
        if chunk.empty:
            continue

        for sample_id, n in chunk["Sample_ID"].value_counts().items():
            counts[str(sample_id)] += int(n)

    return dict(counts)


def count_mutations_per_sample_multi(
        bed_paths: Sequence[str | Path],
        chunksize: int = 250_000,
        *,
        sample_id_col_overrides: Optional[Dict[str, int]] = None,
        tumour_filter: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Return per-(file, sample) mutation counts keyed by "<stem>::<sample_id>".
    """
    overrides = sample_id_col_overrides or {}
    counts: Dict[str, int] = defaultdict(int)

    for p in [Path(x) for x in bed_paths]:
        override = None
        if str(p) in overrides:
            override = overrides[str(p)]
        elif p.name in overrides:
            override = overrides[p.name]

        per_file = count_mutations_per_sample(
            p,
            chunksize=chunksize,
            sample_id_col_override=override,
            tumour_filter=tumour_filter,
        )
        stem = p.stem
        for sid, n in per_file.items():
            counts[f"{stem}::{sid}"] += int(n)

    return dict(counts)


# -----------------------------
# Multi-file operations
# -----------------------------
def compile_k_samples_multi(
        bed_paths: Sequence[str | Path],
        k: int | None,
        seed: int = 123,
        chunksize: int = 250_000,
        *,
        sample_id_col_overrides: Optional[Dict[str, int]] = None,
        tumour_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns (selected_sample_ids, mutations_df) for multiple mutation files.

    Selection is performed globally across all (file, sample) pairs:
      - collect unique samples per file
      - create global list of keys: "<stem>::<sample_id>"
      - shuffle deterministically using `seed`
      - take first k keys (or all if k is None)
      - extract mutations from each file for selected samples in that file
      - concatenate into one DataFrame with the standard schema

    sample_id_col_overrides: optional mapping from path (string) OR basename to sample column index.
    """
    paths = [Path(p) for p in bed_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    overrides = sample_id_col_overrides or {}

    # Collect per-file ordered samples (deterministic per file is not sufficient for global nesting),
    # so instead collect unique sample sets per file, then do ONE global shuffle.
    per_file_samples: List[Tuple[Path, List[str]]] = []
    global_keys: List[str] = []

    for p in paths:
        override = None
        if str(p) in overrides:
            override = overrides[str(p)]
        elif p.name in overrides:
            override = overrides[p.name]

        sel = infer_sample_order(p, seed=seed, chunksize=chunksize, sample_id_col_override=override)
        # sel.ordered_samples is already shuffled, but we only need the set; we will reshuffle globally below.
        uniq = list(dict.fromkeys(sel.ordered_samples))  # preserve order but unique
        per_file_samples.append((p, uniq))

        stem = p.stem
        global_keys.extend([f"{stem}::{sid}" for sid in uniq])

    # Global deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(global_keys)

    if k is not None:
        if k < 0 or k > len(global_keys):
            raise ValueError(f"k must be between 0 and {len(global_keys)} (got {k})")
        chosen_keys = global_keys[:k]
    else:
        chosen_keys = global_keys

    # Build per-file chosen sample lists
    chosen_by_file: Dict[Path, List[str]] = {p: [] for p in paths}
    for key in chosen_keys:
        stem, sid = key.split("::", 1)
        # route to the matching file(s) by stem
        matched = [p for p in paths if p.stem == stem]
        if not matched:
            continue
        # If multiple files share a stem (rare), include in all.
        for p in matched:
            chosen_by_file[p].append(sid)

    # Extract and concat
    kept_dfs: List[pd.DataFrame] = []
    for p in paths:
        override = None
        if str(p) in overrides:
            override = overrides[str(p)]
        elif p.name in overrides:
            override = overrides[p.name]

        sub = extract_mutations_for_samples(
            p,
            chosen_by_file.get(p, []),
            chunksize=chunksize,
            sample_id_col_override=override,
            tumour_filter=tumour_filter,
        )
        if not sub.empty:
            kept_dfs.append(sub)

    out = pd.concat(kept_dfs, ignore_index=True) if kept_dfs else pd.DataFrame(columns=COLNAMES)
    return chosen_keys, out


# -----------------------------
# Public API
# -----------------------------
def compile_k_samples(
        bed_path: str | Path | Sequence[str | Path],
        k: int | None,
        seed: int = 123,
        chunksize: int = 250_000,
        *,
        sample_id_col_override: Optional[int] = None,
        sample_id_col_overrides: Optional[Dict[str, int]] = None,
        tumour_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns (selected_sample_ids, mutations_df).

    - If bed_path is a single path: behaves like before, but with auto format detection.
    - If bed_path is a list/tuple of paths: selects samples globally across all files and concatenates mutations.

    Use sample_id_col_override for single-file overrides.
    Use sample_id_col_overrides for multi-file overrides:
        {"/abs/path/file.bed": 7} or {"file.bed": 7}
    """
    if isinstance(bed_path, (list, tuple)):
        return compile_k_samples_multi(
            bed_paths=bed_path,
            k=k,
            seed=seed,
            chunksize=chunksize,
            sample_id_col_overrides=sample_id_col_overrides,
            tumour_filter=tumour_filter,
        )

    # Single file path
    selection = infer_sample_order(bed_path, seed=seed, chunksize=chunksize,
                                   sample_id_col_override=sample_id_col_override)
    n = len(selection.ordered_samples)

    if k is not None and (k < 0 or k > n):
        raise ValueError(f"k must be between 0 and {n} (got {k})")

    chosen = selection.first_k(k)
    muts = extract_mutations_for_samples(
        bed_path,
        chosen,
        chunksize=chunksize,
        sample_id_col_override=sample_id_col_override,
        tumour_filter=tumour_filter,
    )
    return chosen, muts
