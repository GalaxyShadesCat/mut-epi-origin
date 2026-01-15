"""
bigwig_utils.py

Helpers for safe bigWig extraction with canonical contig handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig

from scripts.contigs import ContigResolver


def get_bigwig_chrom_lengths(bigwig_path: str | Path) -> dict[str, int]:
    bw = pyBigWig.open(str(bigwig_path))
    chroms = bw.chroms()
    bw.close()
    return {str(k): int(v) for k, v in chroms.items()}


def mean_per_bin_bigwig(
    bigwig_path: str | Path,
    canonical_chrom: str,
    bin_edges: np.ndarray,
    resolver: ContigResolver,
) -> np.ndarray:
    bw = pyBigWig.open(str(bigwig_path))
    chrom_sizes = bw.chroms()
    chrom_examples = sorted(chrom_sizes.keys())[:10]

    try:
        bw_chrom = resolver.resolve_for_bigwig(canonical_chrom)
    except ValueError as e:
        bw.close()
        raise ValueError(
            f"bigWig missing contig for '{canonical_chrom}'. "
            f"Example contigs: {', '.join(chrom_examples) or 'none'}."
        ) from e

    if bw_chrom not in chrom_sizes:
        bw.close()
        raise ValueError(
            f"bigWig missing contig for '{canonical_chrom}' (resolved to '{bw_chrom}'). "
            f"Example contigs: {', '.join(chrom_examples) or 'none'}."
        )

    chrom_len = int(chrom_sizes[bw_chrom])

    out = np.full(len(bin_edges) - 1, np.nan, dtype=float)
    for i in range(len(out)):
        s = int(bin_edges[i])
        e = int(bin_edges[i + 1])

        s = max(0, min(s, chrom_len))
        e = max(0, min(e, chrom_len))

        if s >= e:
            out[i] = np.nan
            continue

        v = bw.stats(bw_chrom, s, e, type="mean")[0]
        out[i] = np.nan if v is None else float(v)

    bw.close()
    return out

