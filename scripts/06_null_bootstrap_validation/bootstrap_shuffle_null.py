#!/usr/bin/env python3
"""Run null bootstrap checks by shuffling mutation coordinates.

This script creates null replicates using `bedtools shuffle`, reruns the
selected mutation-to-accessibility configuration, and summarises whether
downstream signals persist:

- clinical association signals (`validate_state_scores.py`)
- DESeq2 gene-level signal
- limma-voom gene-level signal
- Hallmark fgsea pathway signal

The intent is a negative-control test: under shuffled coordinates, strong
biological signals should attenuate.

Replicates that fail downstream DE because no labels survive score-gap
filtering are recorded and skipped rather than aborting the full bootstrap run.

For quick smoke checks, use `--smoke-test-fast` to disable expensive
validation modelling and reduce fgsea permutations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def empty_summary_row(rep: int, seed: int) -> Dict[str, float | str]:
    return {
        "replicate": float(rep),
        "seed": float(seed),
        "status": "failed",
        "failed_stage": "unknown",
        "failure_reason": "",
        "validation_rows": float("nan"),
        "validation_min_p": float("nan"),
        "validation_n_p_le_0.05": float("nan"),
        "validation_group_tests_rows": float("nan"),
        "validation_group_tests_min_p": float("nan"),
        "validation_group_tests_n_p_le_0.05": float("nan"),
        "validation_group_tests_mb_adjusted_rows": float("nan"),
        "validation_group_tests_mb_adjusted_min_p": float("nan"),
        "validation_group_tests_mb_adjusted_n_p_le_0.05": float("nan"),
        "deseq_significant_genes": float("nan"),
        "deseq_genes_tested": float("nan"),
        "limma_significant_genes": float("nan"),
        "limma_genes_tested": float("nan"),
        "fgsea_stat_significant_pathways": float("nan"),
        "fgsea_fc_significant_pathways": float("nan"),
    }


def failure_reason_from_error(err: BaseException) -> str:
    message = str(err)
    if "No sample rows remain after score-gap filtering." in message:
        return "no_labels_after_score_gap_filtering"
    return message.replace("\n", " ").strip()


def parse_summary_txt(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        out[key.strip()] = value.strip()
    return out


def summarise_validation(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {
            "validation_rows": 0.0,
            "validation_min_p": float("nan"),
            "validation_n_p_le_0.05": 0.0,
        }
    dt = pd.read_csv(path)
    if "p_value" not in dt.columns or dt.empty:
        return {
            "validation_rows": float(len(dt)),
            "validation_min_p": float("nan"),
            "validation_n_p_le_0.05": 0.0,
        }
    p = pd.to_numeric(dt["p_value"], errors="coerce")
    return {
        "validation_rows": float(len(dt)),
        "validation_min_p": float(p.min(skipna=True)),
        "validation_n_p_le_0.05": float((p <= 0.05).sum()),
    }


def summarise_pvalue_table(
    path: Path,
    row_key: str,
    min_p_key: str,
    n_sig_key: str,
) -> Dict[str, float]:
    if not path.exists():
        return {
            row_key: 0.0,
            min_p_key: float("nan"),
            n_sig_key: 0.0,
        }

    dt = pd.read_csv(path)
    if "p_value" not in dt.columns or dt.empty:
        return {
            row_key: float(len(dt)),
            min_p_key: float("nan"),
            n_sig_key: 0.0,
        }

    p_values = pd.to_numeric(dt["p_value"], errors="coerce")
    return {
        row_key: float(len(dt)),
        min_p_key: float(p_values.min(skipna=True)),
        n_sig_key: float((p_values <= 0.05).sum()),
    }


def maybe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def has_bed_header(path: Path) -> bool:
    """Return True when the first non-empty line looks like a header row."""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                return True
            fields = line.split("\t")
            if len(fields) < 3:
                return True
            try:
                int(fields[1])
                int(fields[2])
                return False
            except ValueError:
                return True
    raise ValueError(f"Mutation file is empty: {path}")


def infer_bed_chrom_has_chr_prefix(path: Path) -> bool:
    """Infer whether BED chromosome names use the UCSC-style `chr` prefix."""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            try:
                int(fields[1])
                int(fields[2])
            except ValueError:
                continue
            return fields[0].startswith("chr")
    raise ValueError(f"Could not infer chromosome naming from mutation file: {path}")


def format_fai_contig_for_bed(contig: str, use_chr_prefix: bool) -> str:
    """Convert FAI contig names to match BED chromosome naming style."""
    name = contig.strip()
    if use_chr_prefix:
        if name.startswith("chr"):
            return name
        if name == "MT":
            return "chrM"
        return f"chr{name}"

    if not name.startswith("chr"):
        return name
    core = name[3:]
    if core == "M":
        return "MT"
    return core


def write_shuffle_genome_file(
    fai_path: Path,
    output_path: Path,
    use_chr_prefix: bool,
) -> Dict[str, int]:
    """Write a two-column genome-size file matching BED chromosome naming."""
    length_by_chrom: Dict[str, int] = {}
    with fai_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            chrom = format_fai_contig_for_bed(fields[0], use_chr_prefix=use_chr_prefix)
            length = int(fields[1])
            if chrom in length_by_chrom and length_by_chrom[chrom] != length:
                raise ValueError(
                    f"Conflicting chromosome lengths for {chrom}: "
                    f"{length_by_chrom[chrom]} vs {length}"
                )
            length_by_chrom[chrom] = length

    if not length_by_chrom:
        raise ValueError(f"No chromosome lengths parsed from FAI: {fai_path}")

    with output_path.open("w", encoding="utf-8") as handle:
        for chrom, length in length_by_chrom.items():
            handle.write(f"{chrom}\t{length}\n")

    return length_by_chrom


def validate_shuffled_coordinates(
    shuffled_path: Path,
    length_by_chrom: Dict[str, int],
) -> None:
    """Fail fast when shuffled coordinates are invalid or out of bounds."""
    total_rows = 0
    invalid_rows = 0
    first_invalid: str | None = None

    with shuffled_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                invalid_rows += 1
                if first_invalid is None:
                    first_invalid = line
                continue

            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                invalid_rows += 1
                if first_invalid is None:
                    first_invalid = line
                continue

            total_rows += 1
            chrom_len = length_by_chrom.get(chrom)
            is_valid = (
                chrom_len is not None
                and start >= 0
                and end > start
                and end <= chrom_len
            )
            if not is_valid:
                invalid_rows += 1
                if first_invalid is None:
                    first_invalid = line

    if total_rows == 0:
        raise ValueError(f"Shuffled file has no data rows: {shuffled_path}")
    if invalid_rows > 0:
        raise ValueError(
            "Shuffled coordinates contain invalid rows "
            f"({invalid_rows}/{total_rows}). "
            f"First invalid row: {first_invalid}"
        )


def deduplicate_shuffled_rows(shuffled_path: Path) -> tuple[int, int]:
    """Deduplicate shuffled rows in-place and return (total_rows, removed_rows)."""
    seen: set[str] = set()
    kept_lines: List[str] = []
    total_rows = 0

    with shuffled_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            total_rows += 1
            if line in seen:
                continue
            seen.add(line)
            kept_lines.append(line)

    with shuffled_path.open("w", encoding="utf-8") as handle:
        if kept_lines:
            handle.write("\n".join(kept_lines) + "\n")

    removed_rows = total_rows - len(kept_lines)
    return total_rows, removed_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Null bootstrap by mutation-coordinate shuffling with rerun of "
            "grid/validation/DE/limma/fgsea."
        )
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10,
        help="Number of shuffled replicates to run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260408,
        help="Base random seed for replicate-specific shuffles.",
    )
    parser.add_argument(
        "--mut-path",
        default="data/raw/mutations/lihc_snv_mutation_table.tsv",
        help="Input mutation table (BED-like) to shuffle.",
    )
    parser.add_argument(
        "--fai-path",
        default="data/raw/reference/GRCh37.fa.fai",
        help="Reference FAI for bedtools shuffle and grid search.",
    )
    parser.add_argument(
        "--fasta-path",
        default="data/raw/reference/GRCh37.fa",
        help="Reference FASTA for grid search.",
    )
    parser.add_argument(
        "--timing-bw",
        default="data/raw/timing/repliSeq_SknshWaveSignalRep1.bigWig",
        help="Timing bigWig for grid search.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/derived/master_metadata.csv",
        help="Metadata CSV path.",
    )
    parser.add_argument(
        "--counts-path",
        default="data/raw/rna/TCGA-LIHC.star_counts.tsv",
        help="RNA count matrix path.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/null_shuffle_bootstrap_foxa2_exp_decay_500k",
        help="Output directory for all bootstrap replicates.",
    )
    parser.add_argument(
        "--dnase-map-json",
        default=json.dumps(
            {
                "foxa2_normal_pos": "data/processed/ATAC-seq/GSE281574/hg19/FOXA2_normal_pos.bigWig",
                "foxa2_abnormal_zero": "data/processed/ATAC-seq/GSE281574/hg19/FOXA2_abnormal_zero.bigWig",
            }
        ),
        help="JSON mapping of inferred state labels to track files.",
    )
    parser.add_argument(
        "--track-strategy",
        default="exp_decay",
        help="Track strategy to rerun (default: exp_decay).",
    )
    parser.add_argument(
        "--grid-per-sample-count",
        type=int,
        default=0,
        help=(
            "Number of per-sample runs for grid stage. "
            "Use 0 for all available samples."
        ),
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=500000,
        help="Bin size for rerun config.",
    )
    parser.add_argument(
        "--scoring-system",
        default="spearman_r_linear_resid",
        help="Scoring system for DE downstream script.",
    )
    parser.add_argument(
        "--state-labels",
        default="foxa2_normal_pos,foxa2_abnormal_zero",
        help="Comma-separated two-state labels.",
    )
    parser.add_argument(
        "--smoke-test-fast",
        action="store_true",
        help=(
            "Fast smoke mode: disable validation modelling/correlation and "
            "reduce fgsea permutations."
        ),
    )
    parser.add_argument(
        "--validation-group-test-vars",
        default="",
        help=(
            "Optional comma-separated override for validate_state_scores "
            "--group-test-vars."
        ),
    )
    parser.add_argument(
        "--validation-correlation-vars",
        default="",
        help=(
            "Optional comma-separated override for validate_state_scores "
            "--correlation-vars (use 'none' to disable)."
        ),
    )
    parser.add_argument(
        "--validation-modelling-targets",
        default="",
        help=(
            "Optional comma-separated override for validate_state_scores "
            "--modelling-targets (use 'none' to disable)."
        ),
    )
    parser.add_argument(
        "--validation-covariate-cols",
        default="",
        help=(
            "Optional comma-separated override for validate_state_scores "
            "--covariate-cols."
        ),
    )
    parser.add_argument(
        "--fgsea-nperm-simple",
        type=int,
        default=10000,
        help="fgsea simple permutation count per replicate.",
    )
    parser.add_argument(
        "--adjust-for-mutation-burden",
        action="store_true",
        help=(
            "Apply mutation-burden-adjusted clinical association analyses "
            "in validate_state_scores.py."
        ),
    )
    parser.add_argument(
        "--mutation-burden-col",
        default="mutations_post_downsample",
        help="Mutation burden column name in results.csv used for adjustment.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mut_path = (project_root / args.mut_path).resolve()
    fai_path = (project_root / args.fai_path).resolve()
    fasta_path = (project_root / args.fasta_path).resolve()
    timing_bw = (project_root / args.timing_bw).resolve()
    metadata_path = (project_root / args.metadata_path).resolve()
    counts_path = (project_root / args.counts_path).resolve()
    dnase_map = json.loads(args.dnase_map_json)

    required_paths = {
        "--mut-path": mut_path,
        "--fai-path": fai_path,
        "--fasta-path": fasta_path,
        "--timing-bw": timing_bw,
        "--metadata-path": metadata_path,
        "--counts-path": counts_path,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    state_labels = [x.strip() for x in args.state_labels.split(",") if x.strip()]
    if len(state_labels) != 2:
        raise ValueError("--state-labels must contain exactly two labels.")
    state_labels_csv = ",".join(state_labels)

    validation_group_test_vars = args.validation_group_test_vars.strip()
    validation_correlation_vars = args.validation_correlation_vars.strip()
    validation_modelling_targets = args.validation_modelling_targets.strip()
    validation_covariate_cols = args.validation_covariate_cols.strip()
    fgsea_nperm_simple = args.fgsea_nperm_simple
    grid_per_sample_count = args.grid_per_sample_count
    if args.smoke_test_fast:
        if not validation_correlation_vars:
            validation_correlation_vars = "none"
        if not validation_modelling_targets:
            validation_modelling_targets = "none"
        fgsea_nperm_simple = min(fgsea_nperm_simple, 500)
        if grid_per_sample_count == 0:
            grid_per_sample_count = 5

    mutation_has_chr_prefix = infer_bed_chrom_has_chr_prefix(mut_path)
    shuffle_genome_path = out_dir / "shuffle.genome"
    genome_lengths = write_shuffle_genome_file(
        fai_path=fai_path,
        output_path=shuffle_genome_path,
        use_chr_prefix=mutation_has_chr_prefix,
    )

    rows: List[Dict[str, float | str]] = []

    for rep in range(args.n_bootstrap):
        seed = args.base_seed + rep
        rep_dir = out_dir / f"replicate_{rep:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        shuffled_path = rep_dir / "mutations_shuffled.tsv"
        shuffle_input_path = mut_path
        experiment_dir = rep_dir / "experiment"
        deseq_dir = rep_dir / "de_deseq"
        limma_dir = rep_dir / "de_limma"
        fgsea_dir = rep_dir / "fgsea"
        fgsea_dir.mkdir(parents=True, exist_ok=True)

        header_line: str | None = None
        if has_bed_header(mut_path):
            with mut_path.open("r", encoding="utf-8") as handle:
                header_line = handle.readline().rstrip("\n")
                no_header_path = rep_dir / "mutations_input_no_header.tsv"
                with no_header_path.open("w", encoding="utf-8") as out_handle:
                    for line in handle:
                        out_handle.write(line)
            shuffle_input_path = no_header_path

        failed_stage = ""
        try:
            shuffle_cmd = [
                "bedtools",
                "shuffle",
                "-i",
                str(shuffle_input_path),
                "-g",
                str(shuffle_genome_path),
                "-chrom",
                "-seed",
                str(seed),
            ]
            with shuffled_path.open("w", encoding="utf-8") as handle:
                subprocess.run(shuffle_cmd, cwd=str(project_root), check=True, stdout=handle)
            validate_shuffled_coordinates(
                shuffled_path=shuffled_path,
                length_by_chrom=genome_lengths,
            )
            total_rows, removed_rows = deduplicate_shuffled_rows(shuffled_path)
            if removed_rows > 0:
                print(
                    f"Replicate {rep:03d}: removed {removed_rows} duplicate shuffled "
                    f"rows out of {total_rows}."
                )
            if header_line is not None:
                shuffled_body = shuffled_path.read_text(encoding="utf-8")
                shuffled_path.write_text(f"{header_line}\n{shuffled_body}", encoding="utf-8")

            explicit_setup = json.dumps(
                [
                    {
                        "track_strategy": args.track_strategy,
                        "bin_size": int(args.bin_size),
                        "covariates": "gc+cpg+timing",
                    }
                ],
                separators=(",", ":"),
            )

            grid_cmd = [
                "python",
                "-m",
                "scripts.grid_search.cli",
                "--mut-path",
                str(shuffled_path),
                "--fai-path",
                str(fai_path),
                "--fasta-path",
                str(fasta_path),
                "--dnase-map-json",
                json.dumps(dnase_map, separators=(",", ":")),
                "--timing-bw",
                str(timing_bw),
                "--tumour-filter",
                "LIHC",
                "--out-dir",
                str(experiment_dir),
                "--base-seed",
                str(seed),
                "--n-resamples",
                "1",
                "--k-samples",
                "1",
                "--per-sample-count",
                str(grid_per_sample_count),
                "--downsample",
                "none",
                "--track-strategies",
                args.track_strategy,
                "--covariate-sets",
                "gc+cpg+timing",
                "--explicit-setups-json",
                explicit_setup,
                "--standardise-scope",
                "per_chrom",
            ]
            if args.track_strategy == "counts_raw":
                grid_cmd.extend(["--counts-raw-bins", str(args.bin_size)])
            elif args.track_strategy == "exp_decay":
                grid_cmd.extend(["--exp-decay-bins", str(args.bin_size)])
            else:
                raise ValueError(
                    f"Unsupported --track-strategy for null bootstrap: {args.track_strategy}"
                )
            failed_stage = "grid_search"
            run_cmd(grid_cmd, cwd=project_root)

            validate_cmd = [
                "python",
                "scripts/01_pan_celltype_benchmark/validate_state_scores.py",
                "--experiment-name",
                experiment_dir.name,
                "--experiments-root",
                str(rep_dir),
                "--metadata-path",
                str(metadata_path),
                "--state-labels",
                state_labels_csv,
                "--state-suffixes",
                state_labels_csv,
                "--scoring-systems",
                args.scoring_system,
                "--allow-aggregated-results",
            ]
            if validation_group_test_vars:
                validate_cmd.extend(["--group-test-vars", validation_group_test_vars])
            if validation_correlation_vars:
                validate_cmd.extend(["--correlation-vars", validation_correlation_vars])
            if validation_modelling_targets:
                validate_cmd.extend(["--modelling-targets", validation_modelling_targets])
            if validation_covariate_cols:
                validate_cmd.extend(["--covariate-cols", validation_covariate_cols])
            if args.adjust_for_mutation_burden:
                validate_cmd.extend(
                    [
                        "--adjust-for-mutation-burden",
                        "--mutation-burden-col",
                        args.mutation_burden_col,
                    ]
                )
            failed_stage = "validation"
            run_cmd(validate_cmd, cwd=project_root)

            results_path = experiment_dir / "results.csv"
            deseq_cmd = [
                "Rscript",
                "scripts/03_differential_expression/run_differential_expression_by_inferred_labels.R",
                "--counts-path",
                str(counts_path),
                "--results-path",
                str(results_path),
                "--metadata-path",
                str(metadata_path),
                "--metadata-sample-col",
                "tumour_sample_submitter_id",
                "--covariates",
                "age_at_diagnosis,gender,ajcc_pathologic_stage",
                "--output-dir",
                str(deseq_dir),
                "--track-strategy",
                args.track_strategy,
                "--bin-size",
                str(args.bin_size),
                "--scoring-system",
                args.scoring_system,
                "--state-labels",
                state_labels_csv,
                "--contrast-case",
                state_labels[1],
                "--contrast-control",
                state_labels[0],
            ]
            failed_stage = "deseq"
            run_cmd(deseq_cmd, cwd=project_root)

            limma_cmd = [
                "Rscript",
                "scripts/03_differential_expression/run_limma_by_inferred_labels.R",
                "--counts-path",
                str(counts_path),
                "--labels-path",
                str(deseq_dir / "sample_labels_used.csv"),
                "--output-dir",
                str(limma_dir),
                "--contrast-case",
                state_labels[1],
                "--contrast-control",
                state_labels[0],
            ]
            failed_stage = "limma"
            run_cmd(limma_cmd, cwd=project_root)

            fgsea_stat_prefix = fgsea_dir / "fgsea_stat"
            fgsea_stat_cmd = [
                "Rscript",
                "scripts/05_pathway_enrichment/run_fgsea_from_de.R",
                "--de-results",
                str(deseq_dir / "differential_expression_results_all.csv"),
                "--rank-metric",
                "stat",
                "--gene-sets-source",
                "msigdb",
                "--msigdb-collection",
                "H",
                "--out-prefix",
                str(fgsea_stat_prefix),
                "--nperm-simple",
                str(fgsea_nperm_simple),
            ]
            failed_stage = "fgsea_stat"
            run_cmd(fgsea_stat_cmd, cwd=project_root)

            fgsea_fc_prefix = fgsea_dir / "fgsea_logfc_times_neglog10p"
            fgsea_fc_cmd = [
                "Rscript",
                "scripts/05_pathway_enrichment/run_fgsea_from_de.R",
                "--de-results",
                str(deseq_dir / "differential_expression_results_all.csv"),
                "--rank-metric",
                "logfc_times_neglog10p",
                "--gene-sets-source",
                "msigdb",
                "--msigdb-collection",
                "H",
                "--out-prefix",
                str(fgsea_fc_prefix),
                "--nperm-simple",
                str(fgsea_nperm_simple),
            ]
            failed_stage = "fgsea_logfc"
            run_cmd(fgsea_fc_cmd, cwd=project_root)

            row = empty_summary_row(rep=rep, seed=seed)
            row["status"] = "ok"
            row["failed_stage"] = ""
            row["failure_reason"] = ""

            validation = summarise_validation(experiment_dir / "validation_label_associations.csv")
            validation_group_tests = summarise_pvalue_table(
                path=experiment_dir / "validation_group_tests.csv",
                row_key="validation_group_tests_rows",
                min_p_key="validation_group_tests_min_p",
                n_sig_key="validation_group_tests_n_p_le_0.05",
            )
            validation_group_tests_adjusted = summarise_pvalue_table(
                path=(
                    experiment_dir
                    / "validation_group_tests_mutation_burden_adjusted.csv"
                ),
                row_key="validation_group_tests_mb_adjusted_rows",
                min_p_key="validation_group_tests_mb_adjusted_min_p",
                n_sig_key="validation_group_tests_mb_adjusted_n_p_le_0.05",
            )
            deseq_summary = parse_summary_txt(deseq_dir / "run_summary.txt")
            limma_summary = parse_summary_txt(limma_dir / "limma_run_summary.txt")
            fgsea_stat_summary = parse_summary_txt(
                Path(str(fgsea_stat_prefix) + "_fgsea_summary.txt")
            )
            fgsea_fc_summary = parse_summary_txt(
                Path(str(fgsea_fc_prefix) + "_fgsea_summary.txt")
            )
            row.update(
                {
                    "validation_rows": validation["validation_rows"],
                    "validation_min_p": validation["validation_min_p"],
                    "validation_n_p_le_0.05": validation["validation_n_p_le_0.05"],
                    "validation_group_tests_rows": validation_group_tests[
                        "validation_group_tests_rows"
                    ],
                    "validation_group_tests_min_p": validation_group_tests[
                        "validation_group_tests_min_p"
                    ],
                    "validation_group_tests_n_p_le_0.05": validation_group_tests[
                        "validation_group_tests_n_p_le_0.05"
                    ],
                    "validation_group_tests_mb_adjusted_rows": (
                        validation_group_tests_adjusted[
                            "validation_group_tests_mb_adjusted_rows"
                        ]
                    ),
                    "validation_group_tests_mb_adjusted_min_p": (
                        validation_group_tests_adjusted[
                            "validation_group_tests_mb_adjusted_min_p"
                        ]
                    ),
                    "validation_group_tests_mb_adjusted_n_p_le_0.05": (
                        validation_group_tests_adjusted[
                            "validation_group_tests_mb_adjusted_n_p_le_0.05"
                        ]
                    ),
                    "deseq_significant_genes": maybe_float(deseq_summary.get("significant_genes")),
                    "deseq_genes_tested": maybe_float(deseq_summary.get("genes_tested")),
                    "limma_significant_genes": maybe_float(
                        limma_summary.get("significant_genes_fdr_0.05")
                    ),
                    "limma_genes_tested": maybe_float(limma_summary.get("genes_tested")),
                    "fgsea_stat_significant_pathways": maybe_float(
                        fgsea_stat_summary.get("significant_pathways_fdr_0.05")
                    ),
                    "fgsea_fc_significant_pathways": maybe_float(
                        fgsea_fc_summary.get("significant_pathways_fdr_0.05")
                    ),
                }
            )
        except Exception as err:  # pylint: disable=broad-except
            row = empty_summary_row(rep=rep, seed=seed)
            row["failed_stage"] = failed_stage or "unknown"
            row["failure_reason"] = failure_reason_from_error(err)
            print(
                "Replicate "
                f"{rep:03d} failed at stage '{row['failed_stage']}' "
                f"with reason '{row['failure_reason']}'. Continuing."
            )
            tb_path = rep_dir / "failure_traceback.txt"
            tb_path.write_text(traceback.format_exc(), encoding="utf-8")

        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "bootstrap_summary.csv", index=False)

    summary_path = out_dir / "bootstrap_summary.csv"
    summary = pd.read_csv(summary_path)
    agg = {
        "n_replicates": len(summary),
        "mean_validation_n_p_le_0.05": summary["validation_n_p_le_0.05"].mean(),
        "mean_validation_group_tests_n_p_le_0.05": summary[
            "validation_group_tests_n_p_le_0.05"
        ].mean(),
        "mean_validation_group_tests_mb_adjusted_n_p_le_0.05": summary[
            "validation_group_tests_mb_adjusted_n_p_le_0.05"
        ].mean(),
        "mean_deseq_significant_genes": summary["deseq_significant_genes"].mean(),
        "mean_limma_significant_genes": summary["limma_significant_genes"].mean(),
        "mean_fgsea_stat_significant_pathways": summary[
            "fgsea_stat_significant_pathways"
        ].mean(),
        "mean_fgsea_fc_significant_pathways": summary[
            "fgsea_fc_significant_pathways"
        ].mean(),
    }
    with (out_dir / "bootstrap_aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(agg, handle, indent=2)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {out_dir / 'bootstrap_aggregate_summary.json'}")


if __name__ == "__main__":
    main()
