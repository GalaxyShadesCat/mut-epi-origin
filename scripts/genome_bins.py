"""
genome_bins.py

Utilities for reading a FASTA index (.fai) and constructing genome bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.contigs import canonical_autosomes_list, canonicalise_contig

@dataclass(frozen=True)
class ChromInfo:
    chrom: str
    length: int


def load_fai(fai_path: str | Path) -> pd.DataFrame:
    fai_path = Path(fai_path)
    if not fai_path.exists():
        raise FileNotFoundError(f"FAI not found: {fai_path}")

    fai = pd.read_csv(
        fai_path,
        sep="\t",
        header=None,
        names=["chrom", "length", "offset", "line_bases", "line_width"],
    )
    fai["length"] = fai["length"].astype(int)
    return fai


def build_bins(chrom_length: int, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (edges, centres) with edges capped at chrom_length.

    Important: last edge equals chrom_length so bigWig queries never exceed bounds.
    """
    if chrom_length <= 0:
        raise ValueError("chrom_length must be > 0")
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")

    edges = np.arange(0, chrom_length + bin_size, bin_size, dtype=int)
    if edges[-1] != chrom_length:
        edges[-1] = chrom_length

    centres = (edges[:-1] + edges[1:]) // 2
    return edges, centres


def iter_chroms(fai: pd.DataFrame, chroms: Optional[Sequence[str]] = None) -> Iterable[ChromInfo]:
    """
    Yield ChromInfo(chrom, length) for a subset of chromosomes.

    Parameters
    ----------
    fai
        DataFrame from load_fai() with at least columns: 'chrom' and 'length'.
        Typical GRCh37 FASTA indexes use contigs like: 1..22, X, Y, MT (+ extra contigs).
    chroms
        If provided, only yield these chromosomes (in this order). Chrom names may be:
          - "1".."22","X","Y","MT" (Ensembl/GRCh37 style)
          - "chr1".."chr22","chrX","chrY","chrM" (UCSC style)
        If None, default to autosomes in UCSC style:
          chr1..chr22

    Behaviour / naming
    ------------------
    - The yielded ChromInfo.chrom is always canonical ("chr1".."chr22","chrX","chrY").
    - FAI contigs are canonicalised; non-primary contigs are dropped.

    Raises
    ------
    ValueError if any requested chromosome cannot be found in the FAI (via alias mapping).
    """
    if "chrom" not in fai.columns or "length" not in fai.columns:
        raise ValueError("FAI dataframe must have columns: 'chrom' and 'length'")

    # Build a lookup from canonical contig name -> length
    length_map: dict[str, int] = {}
    for raw_contig, raw_len in zip(fai["chrom"].astype(str), fai["length"].astype(int)):
        canonical = canonicalise_contig(raw_contig)
        if canonical is None:
            continue
        if canonical in length_map and length_map[canonical] != int(raw_len):
            raise ValueError(
                f"FAI contig length mismatch for {canonical}: "
                f"{length_map[canonical]} vs {int(raw_len)}"
            )
        length_map[canonical] = int(raw_len)

    # Default: primary chromosomes only, canonical style
    if chroms is None:
        chroms = canonical_autosomes_list()

    # Canonicalise requested contigs and de-duplicate while preserving order
    requested: list[str] = []
    seen: set[str] = set()
    for req in chroms:
        canonical = canonicalise_contig(req)
        if canonical is None:
            raise ValueError(f"Chromosome '{req}' is not a primary canonical contig.")
        if canonical not in seen:
            seen.add(canonical)
            requested.append(canonical)

    # Yield in requested order
    for canonical in requested:
        if canonical not in length_map:
            raise ValueError(f"Chromosome '{canonical}' not found in FAI after canonicalisation.")
        yield ChromInfo(chrom=canonical, length=int(length_map[canonical]))
