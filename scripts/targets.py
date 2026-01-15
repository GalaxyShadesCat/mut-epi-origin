"""
targets.py

Target track extraction (DNase accessibility) from bigWig.

This is intentionally separate from covariates because in your experiment:
- mutations track(s) are predictors/signals
- mela DNase accessibility is the target to correlate against or predict/residualise
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from scripts.bigwig_utils import get_bigwig_chrom_lengths, mean_per_bin_bigwig
from scripts.contigs import ContigResolver, is_primary_canonical


def dnase_mean_per_bin(dnase_bigwig: str | Path, chrom: str, bin_edges: np.ndarray) -> np.ndarray:
    if not is_primary_canonical(chrom):
        raise ValueError(f"dnase_mean_per_bin requires canonical contigs (chr1..chr22, chrX, chrY). Got '{chrom}'.")

    chrom_lengths = get_bigwig_chrom_lengths(dnase_bigwig)
    resolver = ContigResolver(fasta_contigs=None, bigwig_contigs=list(chrom_lengths.keys()))
    return mean_per_bin_bigwig(dnase_bigwig, chrom, bin_edges, resolver)
