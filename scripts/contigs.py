"""
contigs.py

Canonical contig handling utilities for internal pipeline use.
"""

from __future__ import annotations

from typing import Optional, Sequence


def canonical_primary_list() -> list[str]:
    return [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def canonicalise_contig(contig: str) -> Optional[str]:
    """
    Map a contig name to canonical UCSC-style primary contigs.

    Returns:
      - "chr1".."chr22","chrX","chrY" for primary contigs
      - None for mitochondrial (MT/chrM) and non-primary contigs
    """
    if contig is None:
        return None
    c = str(contig).strip()
    if not c:
        return None

    cu = c.upper()
    if cu.startswith("CHR"):
        core = cu[3:]
    else:
        core = cu

    if core in ("M", "MT"):
        return None

    if core.isdigit():
        num = int(core)
        if 1 <= num <= 22:
            return f"chr{num}"
        return None

    if core in ("X", "Y"):
        return f"chr{core}"

    return None


def is_primary_canonical(contig: str) -> bool:
    return contig in set(canonical_primary_list())


class ContigResolver:
    def __init__(
        self,
        fasta_contigs: Sequence[str] | None,
        bigwig_contigs: Sequence[str] | None,
    ) -> None:
        self._fasta_contigs = set(str(c) for c in fasta_contigs) if fasta_contigs is not None else None
        self._bigwig_contigs = set(str(c) for c in bigwig_contigs) if bigwig_contigs is not None else None
        self._fasta_map = self._build_map(self._fasta_contigs) if self._fasta_contigs is not None else {}
        self._bigwig_map = self._build_map(self._bigwig_contigs) if self._bigwig_contigs is not None else {}

    def _build_map(self, contigs: set[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for canonical in canonical_primary_list():
            preferred = [canonical, canonical[3:]]
            for cand in preferred:
                if cand in contigs:
                    mapping[canonical] = cand
                    break
        return mapping

    def _validate_canonical(self, canonical: str) -> str:
        c = canonicalise_contig(canonical)
        if c is None:
            raise ValueError(f"Contig '{canonical}' is not a primary canonical contig.")
        if c != canonical:
            raise ValueError(f"Contig '{canonical}' is not in canonical form; use '{c}'.")
        return c

    def has_canonical_in_fasta(self, canonical: str) -> bool:
        self._validate_canonical(canonical)
        return canonical in self._fasta_map

    def has_canonical_in_bigwig(self, canonical: str) -> bool:
        self._validate_canonical(canonical)
        return canonical in self._bigwig_map

    def resolve_for_fasta(self, canonical: str) -> str:
        self._validate_canonical(canonical)
        if self._fasta_contigs is None:
            raise ValueError("FASTA contigs not provided; cannot resolve contigs.")
        if canonical in self._fasta_map:
            return self._fasta_map[canonical]
        examples = ", ".join(sorted(self._fasta_contigs)[:10]) or "none"
        raise ValueError(f"FASTA missing contig for canonical '{canonical}'. Example contigs: {examples}.")

    def resolve_for_bigwig(self, canonical: str) -> str:
        self._validate_canonical(canonical)
        if self._bigwig_contigs is None:
            raise ValueError("bigWig contigs not provided; cannot resolve contigs.")
        if canonical in self._bigwig_map:
            return self._bigwig_map[canonical]
        examples = ", ".join(sorted(self._bigwig_contigs)[:10]) or "none"
        raise ValueError(f"bigWig missing contig for canonical '{canonical}'. Example contigs: {examples}.")
