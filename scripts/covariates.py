"""
covariates.py

Non-epigenomic covariates plus replication timing.

Non-epigenomic (from reference FASTA):
- gc_fraction per bin
- cpg_frequency per bin (count(CpG) / bin_length)
- optional trinucleotide features (simple counts/normalised frequencies)

Replication timing (from bigWig):
- mean timing signal per bin

Notes
-----
- These are used as adjustment features, not targets.
- DNase is intentionally not handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pysam
except ImportError as e:
    raise ImportError("pysam is required for FASTA-based covariates. Install with: pip install pysam") from e

from scripts.bigwig_utils import get_bigwig_chrom_lengths, mean_per_bin_bigwig
from scripts.contigs import ContigResolver, is_primary_canonical

def _fetch_seq(fasta: pysam.FastaFile, chrom: str, start: int, end: int, resolver: ContigResolver) -> str:
    # pysam fetch uses 0-based half-open coordinates
    chrom_q = resolver.resolve_for_fasta(chrom)
    s = fasta.fetch(chrom_q, start, end)
    return s.upper()


def gc_fraction_per_bin(fasta_path: str | Path, chrom: str, bin_edges: np.ndarray) -> np.ndarray:
    if not is_primary_canonical(chrom):
        raise ValueError(f"gc_fraction_per_bin requires canonical contigs (chr1..chr22, chrX, chrY). Got '{chrom}'.")
    fasta = pysam.FastaFile(str(fasta_path))
    resolver = ContigResolver(fasta_contigs=fasta.references, bigwig_contigs=None)
    out = np.zeros(len(bin_edges) - 1, dtype=float)
    for i in range(len(out)):
        s, e = int(bin_edges[i]), int(bin_edges[i + 1])
        seq = _fetch_seq(fasta, chrom, s, e, resolver)
        if not seq:
            out[i] = np.nan
            continue
        gc = seq.count("G") + seq.count("C")
        out[i] = gc / max(len(seq), 1)
    fasta.close()
    return out


def cpg_frequency_per_bin(fasta_path: str | Path, chrom: str, bin_edges: np.ndarray) -> np.ndarray:
    if not is_primary_canonical(chrom):
        raise ValueError(f"cpg_frequency_per_bin requires canonical contigs (chr1..chr22, chrX, chrY). Got '{chrom}'.")
    fasta = pysam.FastaFile(str(fasta_path))
    resolver = ContigResolver(fasta_contigs=fasta.references, bigwig_contigs=None)
    out = np.zeros(len(bin_edges) - 1, dtype=float)
    for i in range(len(out)):
        s, e = int(bin_edges[i]), int(bin_edges[i + 1])
        seq = _fetch_seq(fasta, chrom, s, e, resolver)
        if len(seq) < 2:
            out[i] = 0.0
            continue
        # Count overlapping "CG" occurrences
        cpg = sum(1 for j in range(len(seq) - 1) if seq[j : j + 2] == "CG")
        out[i] = cpg / max((e - s), 1)
    fasta.close()
    return out


def trinuc_frequency_per_bin(
    fasta_path: str | Path,
    chrom: str,
    bin_edges: np.ndarray,
    trinucs: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns dict: {trinuc: frequency_per_bin} where frequency is count / bin_length.

    Keep this small unless you really need all 64.
    Default: a small canonical set often used around CpG/UV contexts.
    """
    if trinucs is None:
        trinucs = ["TCG", "CCG", "ACG", "GCG", "TCC", "CCC", "ACC", "GCC"]

    if not is_primary_canonical(chrom):
        raise ValueError(
            f"trinuc_frequency_per_bin requires canonical contigs (chr1..chr22, chrX, chrY). Got '{chrom}'."
        )

    fasta = pysam.FastaFile(str(fasta_path))
    resolver = ContigResolver(fasta_contigs=fasta.references, bigwig_contigs=None)
    out = {t: np.zeros(len(bin_edges) - 1, dtype=float) for t in trinucs}

    for i in range(len(bin_edges) - 1):
        s, e = int(bin_edges[i]), int(bin_edges[i + 1])
        seq = _fetch_seq(fasta, chrom, s, e, resolver)
        L = max(e - s, 1)
        if len(seq) < 3:
            for t in trinucs:
                out[t][i] = 0.0
            continue

        # Sliding window trinuc count
        for t in trinucs:
            cnt = sum(1 for j in range(len(seq) - 2) if seq[j : j + 3] == t)
            out[t][i] = cnt / L

    fasta.close()
    return out



def bigwig_mean_per_bin(bigwig_path: str | Path, chrom: str, bin_edges: np.ndarray) -> np.ndarray:
    """
    Mean bigWig signal per bin, with robust contig resolution and interval clamping.

    - Resolves chrom naming differences using ContigResolver.
    - Clamps intervals to [0, chrom_length] to avoid pyBigWig "Invalid interval bounds!".
    """
    if not is_primary_canonical(chrom):
        raise ValueError(
            f"bigwig_mean_per_bin requires canonical contigs (chr1..chr22, chrX, chrY). Got '{chrom}'."
        )
    chrom_lengths = get_bigwig_chrom_lengths(bigwig_path)
    resolver = ContigResolver(fasta_contigs=None, bigwig_contigs=list(chrom_lengths.keys()))
    return mean_per_bin_bigwig(bigwig_path, chrom, bin_edges, resolver)
