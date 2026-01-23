# scripts/sanity_checks.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Sequence, Dict, Union

import numpy as np
import pandas as pd


# -------------------------
# Pretty printing helpers
# -------------------------
def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _expect(cond: bool, pass_msg: str, fail_msg: str, *, hard: bool = False) -> bool:
    if cond:
        _ok(pass_msg)
        return True
    _fail(fail_msg)
    if hard:
        raise AssertionError(fail_msg)
    return False


# -------------------------
# Canonical + resolution helpers
# -------------------------
PRIMARY_CANONICAL: List[str] = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _is_chr_style(contig: str) -> bool:
    return str(contig).startswith("chr")


def _resolve_chr_style(want_chr: str, available: set[str]) -> Optional[str]:
    w = str(want_chr)
    if w in available:
        return w
    if w.startswith("chr"):
        alt = w[3:]
        if alt in available:
            return alt
    else:
        alt = "chr" + w
        if alt in available:
            return alt
    return None


def _contig_style_summary(available: set[str]) -> str:
    has_chr1 = "chr1" in available
    has_1 = "1" in available
    if has_chr1 and not has_1:
        return "chr*"
    if has_1 and not has_chr1:
        return "numeric"
    if has_chr1 and has_1:
        return "mixed"
    chr_like = sum(1 for c in list(available)[:200] if str(c).startswith("chr"))
    return "chr*" if chr_like >= 1 else "unknown"


# -------------------------
# Config container
# -------------------------
MutPathT = Union[Path, Sequence[Path]]


@dataclass
class SanityConfig:
    project_root: Path
    mut_path: MutPathT
    fai_path: Path
    fasta_path: Path

    # check all DNase bigwigs
    dnase_bws: Dict[str, Path]  # e.g. {"mela": Path(...), "kera": Path(...), "fibr": Path(...)}

    timing_bw: Optional[Path] = None

    # how heavy to run
    k_samples: int = 1
    seed: int = 123
    chunksize: int = 250_000
    tumour_filter: Optional[Sequence[str]] = None

    # binning quick checks
    bin_size: int = 50_000
    check_chrom: str = "chr1"  # INTERNAL standard should be chr-style

    # strictness
    hard_fail: bool = False


def _as_path_list(p: MutPathT) -> List[Path]:
    if isinstance(p, (list, tuple)):
        return [Path(x) for x in p]
    return [Path(p)]


# -------------------------
# Checks
# -------------------------
def check_paths(cfg: SanityConfig) -> None:
    _expect(
        cfg.project_root.exists(),
        "PROJECT_ROOT exists",
        f"PROJECT_ROOT missing: {cfg.project_root}",
        hard=cfg.hard_fail,
    )

    # mutation paths can be list
    mut_paths = _as_path_list(cfg.mut_path)
    _expect(len(mut_paths) > 0, "mut_path provided", "mut_path is empty", hard=cfg.hard_fail)
    for mp in mut_paths:
        _expect(mp.exists(), f"mut_path exists: {mp}", f"mut_path missing: {mp}", hard=cfg.hard_fail)

    for p, name in [
        (cfg.fai_path, "fai_path"),
        (cfg.fasta_path, "fasta_path"),
    ]:
        _expect(p.exists(), f"{name} exists: {p}", f"{name} missing: {p}", hard=cfg.hard_fail)

    _expect(
        bool(cfg.dnase_bws),
        "dnase_bws provided",
        "dnase_bws is empty (provide at least one DNase bigWig)",
        hard=cfg.hard_fail,
    )
    for ct, bw in cfg.dnase_bws.items():
        _expect(Path(bw).exists(), f"DNase bigWig exists [{ct}]: {bw}", f"DNase bigWig missing [{ct}]: {bw}",
                hard=cfg.hard_fail)

    if cfg.timing_bw is not None:
        _expect(
            cfg.timing_bw.exists(),
            f"timing_bw exists: {cfg.timing_bw}",
            f"timing_bw missing: {cfg.timing_bw}",
            hard=cfg.hard_fail,
        )


def check_fasta_index(cfg: SanityConfig) -> None:
    fai = Path(str(cfg.fasta_path) + ".fai")
    _expect(
        fai.exists(),
        f"FASTA index exists: {fai}",
        f"Missing FASTA index: {fai}. Create with: samtools faidx {cfg.fasta_path}",
        hard=cfg.hard_fail,
    )


def check_imports() -> None:
    try:
        import pysam  # noqa
        _ok("pysam import OK")
    except Exception as e:
        _fail(f"pysam import FAILED: {e}. Install: pip/conda install pysam")

    try:
        import pyBigWig  # noqa
        _ok("pyBigWig import OK")
    except Exception as e:
        _fail(f"pyBigWig import FAILED: {e}. Install: pip install pyBigWig")


def check_fai_primary_contigs(cfg: SanityConfig) -> List[str]:
    from scripts.genome_bins import load_fai, iter_chroms

    fai_df = load_fai(cfg.fai_path)
    _expect(len(fai_df) > 0, "FAI loaded with rows", "FAI loaded but is empty", hard=cfg.hard_fail)

    chrom_infos = list(iter_chroms(fai_df, chroms=None))
    chroms = [c.chrom for c in chrom_infos]

    _expect(
        len(chroms) == 24,
        "iter_chroms() returned 24 primary contigs",
        f"iter_chroms() returned {len(chroms)} contigs (expected 24). Got head: {chroms[:10]}",
        hard=cfg.hard_fail,
    )

    _expect(
        chroms == PRIMARY_CANONICAL,
        "Primary contig list exactly chr1..chr22,chrX,chrY (internal standard)",
        f"Primary contig list differs. First 10: {chroms[:10]}",
        hard=cfg.hard_fail,
    )

    fai_contigs = set(fai_df["chrom"].astype(str).tolist())
    style = _contig_style_summary(fai_contigs)
    _ok(f"FAI contig naming style looks like: {style} (this can differ from internal canonical)")

    resolved = _resolve_chr_style(cfg.check_chrom, fai_contigs)
    _expect(
        resolved is not None,
        f"{cfg.check_chrom} is resolvable in FAI (maps to '{resolved}')",
        f"{cfg.check_chrom} is NOT resolvable in FAI contigs. Example contigs: {sorted(list(fai_contigs))[:10]}",
        hard=cfg.hard_fail,
    )

    return chroms


def check_bigwig_contigs_one(bw_path: Path, *, label: str, chroms: List[str], hard: bool) -> None:
    import pyBigWig

    bw = pyBigWig.open(str(bw_path))
    bw_chroms = set(map(str, bw.chroms().keys()))
    bw.close()

    style = _contig_style_summary(bw_chroms)
    _ok(f"[{label}] bigWig contig naming style looks like: {style}")

    resolved = _resolve_chr_style("chr1", bw_chroms)
    _expect(
        resolved is not None,
        f"[{label}] bigWig can resolve chr1 (maps to '{resolved}')",
        f"[{label}] bigWig cannot resolve chr1 (contig mismatch likely)",
        hard=hard,
    )

    present = [c for c in chroms if _resolve_chr_style(c, bw_chroms) is not None]
    missing = [c for c in chroms if _resolve_chr_style(c, bw_chroms) is None]
    _expect(
        len(present) >= 20,
        f"[{label}] bigWig contains/resolves most primary chroms ({len(present)}/24)",
        f"[{label}] bigWig missing many primary chroms ({len(missing)}/24 missing). Example missing: {missing[:5]}",
        hard=hard,
    )


def check_bigwig_contigs(cfg: SanityConfig, chroms: List[str]) -> None:
    for ct, bw in cfg.dnase_bws.items():
        check_bigwig_contigs_one(Path(bw), label=f"DNase:{ct}", chroms=chroms, hard=cfg.hard_fail)

    if cfg.timing_bw is not None:
        check_bigwig_contigs_one(Path(cfg.timing_bw), label="Timing", chroms=chroms, hard=False)


def check_mutation_file_shape_one(mut_path: Path, *, hard: bool) -> None:
    with mut_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))

    _expect(len(lines) > 0, f"[{mut_path.name}] mutation file has content",
            f"[{mut_path.name}] mutation file seems empty",
            hard=hard)

    bad = []
    for i, ln in enumerate(lines, start=1):
        cols = ln.split("\t")
        if len(cols) < 6:
            bad.append((i, len(cols), ln[:80]))
    _expect(
        len(bad) == 0,
        f"[{mut_path.name}] tab-delimited with >= 6 columns (OK)",
        f"[{mut_path.name}] early lines have <6 cols. Examples: {bad[:2]}",
        hard=hard,
    )


def check_mutation_file_shape(cfg: SanityConfig) -> None:
    for mp in _as_path_list(cfg.mut_path):
        check_mutation_file_shape_one(mp, hard=cfg.hard_fail)


def check_sample_selection_and_df(cfg: SanityConfig) -> pd.DataFrame:
    from scripts.sample_selector import compile_k_samples

    mut_paths = _as_path_list(cfg.mut_path)
    mut_arg: Union[Path, List[Path]] = mut_paths[0] if len(mut_paths) == 1 else mut_paths

    chosen, muts_df = compile_k_samples(
        mut_arg,
        k=cfg.k_samples,
        seed=cfg.seed,
        chunksize=cfg.chunksize,
        tumour_filter=cfg.tumour_filter,
    )

    _expect(
        len(chosen) == cfg.k_samples,
        f"Selected k={cfg.k_samples} sample keys",
        f"Expected {cfg.k_samples} samples, got {len(chosen)}",
        hard=cfg.hard_fail,
    )
    _expect(
        len(muts_df) > 0,
        "muts_df has rows for selected sample(s)",
        "muts_df is empty (selection or filtering bug)",
        hard=cfg.hard_fail,
    )

    expected_cols = {"Chromosome", "Start", "End", "Sample_ID", "Ref", "Alt", "Tumour", "Context"}
    _expect(
        expected_cols.issubset(set(muts_df.columns)),
        "muts_df has expected columns",
        f"muts_df missing columns. Has: {list(muts_df.columns)}",
        hard=cfg.hard_fail,
    )

    # INTERNAL STANDARD: chr-style
    non_chr = (~muts_df["Chromosome"].astype(str).str.startswith("chr")).sum()
    _expect(
        non_chr == 0,
        "All mutation Chromosome values are chr*-style (internal standard OK)",
        f"Found {non_chr} rows whose Chromosome is not chr*-style",
        hard=cfg.hard_fail,
    )

    starts = pd.to_numeric(muts_df["Start"], errors="coerce")
    ends = pd.to_numeric(muts_df["End"], errors="coerce")
    _expect(starts.notna().all(), "All Start values parse as integers", "Some Start values are non-numeric",
            hard=cfg.hard_fail)
    _expect(ends.notna().all(), "All End values parse as integers", "Some End values are non-numeric",
            hard=cfg.hard_fail)

    bad = (starts >= ends).sum()
    _expect(bad == 0, "All Start < End (BED-style)", f"Found {bad} rows with Start >= End",
            hard=cfg.hard_fail)

    unique_sid = muts_df["Sample_ID"].nunique(dropna=True)
    _expect(
        unique_sid == cfg.k_samples,
        f"muts_df contains exactly k unique Sample_IDs ({unique_sid})",
        f"Expected {cfg.k_samples} unique Sample_IDs, got {unique_sid}",
        hard=cfg.hard_fail,
    )

    _ok(f"muts_df rows: {len(muts_df):,}. Top chroms:\n{muts_df['Chromosome'].value_counts().head(5)}")
    return muts_df


def check_bins_and_lengths(cfg: SanityConfig) -> Tuple[np.ndarray, int, str]:
    from scripts.genome_bins import load_fai, build_bins

    fai_df = load_fai(cfg.fai_path)

    expected_cols = {"chrom", "length"}
    if not expected_cols.issubset(set(fai_df.columns)):
        _fail(f"FAI dataframe missing columns {expected_cols}. Has: {list(fai_df.columns)}")
        if cfg.hard_fail:
            raise AssertionError("FAI dataframe schema mismatch")
        return np.array([0, 1], dtype=int), 1, cfg.check_chrom

    fai_df = fai_df.copy()
    fai_df["chrom"] = fai_df["chrom"].astype(str).str.strip()
    fai_contigs = set(fai_df["chrom"].tolist())

    resolved = _resolve_chr_style(cfg.check_chrom, fai_contigs)
    _expect(
        resolved is not None,
        f"{cfg.check_chrom} resolves in FAI (maps to '{resolved}')",
        f"{cfg.check_chrom} not resolvable in FAI contigs. Example contigs: {sorted(list(fai_contigs))[:10]}",
        hard=cfg.hard_fail,
    )
    if resolved is None:
        return np.array([0, 1], dtype=int), 1, cfg.check_chrom

    sel = fai_df.loc[fai_df["chrom"] == resolved, "length"]
    if sel.empty:
        _fail(f"Resolved contig '{resolved}' not found as exact row in FAI (normalisation mismatch).")
        if cfg.hard_fail:
            raise AssertionError(f"Resolved contig '{resolved}' missing from FAI rows")
        return np.array([0, 1], dtype=int), 1, resolved

    chrom_len = int(pd.to_numeric(sel, errors="coerce").iloc[0])
    _expect(
        chrom_len > 1_000_000,
        f"{cfg.check_chrom} length looks plausible ({chrom_len:,}) [FAI contig '{resolved}']",
        f"{cfg.check_chrom} length too small ({chrom_len}) [FAI contig '{resolved}']",
        hard=cfg.hard_fail,
    )

    edges, centres = build_bins(chrom_len, cfg.bin_size)
    _expect(len(edges) == len(centres) + 1, "Bin edges and centres consistent",
            "Bin edges/centres lengths inconsistent", hard=cfg.hard_fail)
    _expect(edges[0] == 0, "Bins start at 0", f"Bins start at {edges[0]} not 0", hard=cfg.hard_fail)
    _expect(edges[-1] >= chrom_len, "Bins cover chromosome length", "Bins do not cover chromosome length",
            hard=cfg.hard_fail)

    _ok(f"Built {len(centres):,} bins for {cfg.check_chrom} at bin_size={cfg.bin_size:,}")
    return edges, chrom_len, resolved


def check_dnase_track(cfg: SanityConfig, edges: np.ndarray) -> None:
    from scripts.targets import dnase_mean_per_bin

    for ct, bw in cfg.dnase_bws.items():
        try:
            dn = dnase_mean_per_bin(bw, cfg.check_chrom, edges)
        except Exception as e:
            _fail(f"[DNase:{ct}] extraction failed for chrom={cfg.check_chrom}: {e}")
            continue

        _expect(len(dn) == len(edges) - 1, f"[DNase:{ct}] track length equals n_bins",
                f"[DNase:{ct}] track length {len(dn)} != n_bins {len(edges) - 1}", hard=cfg.hard_fail)

        nfinite = int(np.isfinite(dn).sum())
        _expect(nfinite > 0, f"[DNase:{ct}] has some finite values",
                f"[DNase:{ct}] entirely NaN (contig mismatch or bounds)", hard=cfg.hard_fail)
        _expect(nfinite > 0.8 * len(dn), f"[DNase:{ct}] mostly finite ({nfinite}/{len(dn)})",
                f"[DNase:{ct}] many NaNs ({nfinite}/{len(dn)} finite).", hard=False)

        _ok(f"[DNase:{ct}] stats: finite={nfinite}/{len(dn)}, min={np.nanmin(dn):.3g}, max={np.nanmax(dn):.3g}")


def check_timing_track(cfg: SanityConfig, edges: np.ndarray) -> None:
    if cfg.timing_bw is None:
        _warn("Timing bigWig not provided; skipping timing sanity check")
        return

    from scripts.covariates import bigwig_mean_per_bin

    try:
        t = bigwig_mean_per_bin(cfg.timing_bw, cfg.check_chrom, edges)
    except Exception as e:
        _fail(f"[Timing] extraction failed for chrom={cfg.check_chrom}: {e}")
        return

    _expect(len(t) == len(edges) - 1, "[Timing] track length equals n_bins",
            f"[Timing] track length {len(t)} != n_bins {len(edges) - 1}", hard=cfg.hard_fail)

    nfinite = int(np.isfinite(t).sum())
    _expect(nfinite > 0, "[Timing] has some finite values",
            "[Timing] entirely NaN (contig mismatch or bounds)", hard=cfg.hard_fail)
    _ok(f"[Timing] stats: finite={nfinite}/{len(t)}, min={np.nanmin(t):.3g}, max={np.nanmax(t):.3g}")


def check_fasta_fetch_covariates(cfg: SanityConfig, edges: np.ndarray) -> None:
    from scripts.covariates import gc_fraction_per_bin, cpg_frequency_per_bin

    try:
        gc = gc_fraction_per_bin(cfg.fasta_path, cfg.check_chrom, edges)
        cpg = cpg_frequency_per_bin(cfg.fasta_path, cfg.check_chrom, edges)
    except Exception as e:
        _fail(f"FASTA covariate fetch failed for chrom={cfg.check_chrom}: {e}")
        _fail("This usually means covariates.py is not resolving chr<->numeric for pysam.fetch().")
        return

    _expect(len(gc) == len(edges) - 1, "GC covariate length equals n_bins", "GC covariate length mismatch",
            hard=cfg.hard_fail)
    _expect(len(cpg) == len(edges) - 1, "CpG covariate length equals n_bins", "CpG covariate length mismatch",
            hard=cfg.hard_fail)

    ngc = int(np.isfinite(gc).sum())
    ncpg = int(np.isfinite(cpg).sum())

    _expect(ngc > 0, "GC has some finite values", "GC is entirely NaN (FASTA contig mismatch likely)",
            hard=cfg.hard_fail)
    _expect(ncpg > 0, "CpG has some finite values", "CpG is entirely NaN (FASTA contig mismatch likely)",
            hard=cfg.hard_fail)

    if np.nanmin(gc) < -1e-6 or np.nanmax(gc) > 1 + 1e-6:
        _warn(f"GC fraction outside [0,1]. min={np.nanmin(gc)}, max={np.nanmax(gc)}.")
    else:
        _ok(f"GC fraction within [0,1] (min={np.nanmin(gc):.3f}, max={np.nanmax(gc):.3f})")

    _ok(f"CpG stats: finite={ncpg}/{len(cpg)}, min={np.nanmin(cpg):.3g}, max={np.nanmax(cpg):.3g}")


def check_mutation_positions_alignment(cfg: SanityConfig, muts_df: pd.DataFrame) -> None:
    sub = muts_df.loc[muts_df["Chromosome"] == cfg.check_chrom]
    n = len(sub)
    _expect(
        n > 0,
        f"Selected sample(s) have mutations on {cfg.check_chrom}",
        f"No mutations on {cfg.check_chrom} for this selection (not an error, but reduces power)",
        hard=False,
    )

    if n > 0:
        starts = sub["Start"].astype(int).to_numpy()
        _expect(np.min(starts) >= 0, "Mutation starts are >= 0", "Found mutation starts < 0",
                hard=cfg.hard_fail)
        _ok(f"{cfg.check_chrom}: n_mutations={n:,}, start range [{starts.min():,}, {starts.max():,}]")


def run_all_sanity_checks(cfg: SanityConfig) -> None:
    mut_paths = _as_path_list(cfg.mut_path)

    print("=== SANITY CHECKS BEGIN ===")
    print(f"PROJECT_ROOT: {cfg.project_root}")
    print("mut_path(s):")
    for mp in mut_paths:
        print(f"  - {mp}")
    print(f"fai_path:     {cfg.fai_path}")
    print(f"fasta_path:   {cfg.fasta_path}")
    print("dnase_bws:")
    for ct, bw in cfg.dnase_bws.items():
        print(f"  - {ct}: {bw}")
    print(f"timing_bw:    {cfg.timing_bw}")
    print(f"k={cfg.k_samples}, seed={cfg.seed}, bin_size={cfg.bin_size}, check_chrom={cfg.check_chrom}")
    print()

    _expect(
        _is_chr_style(cfg.check_chrom),
        f"check_chrom uses internal canonical chr-style ({cfg.check_chrom})",
        f"check_chrom should be chr-style internally (got {cfg.check_chrom}). Set e.g. check_chrom='chr1'.",
        hard=cfg.hard_fail,
    )

    check_imports()
    check_paths(cfg)
    check_fasta_index(cfg)

    chroms = check_fai_primary_contigs(cfg)
    check_bigwig_contigs(cfg, chroms)

    check_mutation_file_shape(cfg)
    muts_df = check_sample_selection_and_df(cfg)

    edges, _, _ = check_bins_and_lengths(cfg)
    check_mutation_positions_alignment(cfg, muts_df)

    # bigWig target/covariates
    check_dnase_track(cfg, edges)
    check_timing_track(cfg, edges)

    # FASTA covariates
    check_fasta_fetch_covariates(cfg, edges)

    print("\n=== SANITY CHECKS END ===")
