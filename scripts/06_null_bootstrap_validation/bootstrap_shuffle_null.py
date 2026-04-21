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
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


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


def maybe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


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

    rows: List[Dict[str, float | str]] = []

    for rep in range(args.n_bootstrap):
        seed = args.base_seed + rep
        rep_dir = out_dir / f"replicate_{rep:03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        shuffled_path = rep_dir / "mutations_shuffled.tsv"
        experiment_dir = rep_dir / "experiment"
        deseq_dir = rep_dir / "de_deseq"
        limma_dir = rep_dir / "de_limma"
        fgsea_dir = rep_dir / "fgsea"
        fgsea_dir.mkdir(parents=True, exist_ok=True)

        shuffle_cmd = [
            "bedtools",
            "shuffle",
            "-i",
            str(mut_path),
            "-g",
            str(fai_path),
            "-chrom",
            "-seed",
            str(seed),
            "-header",
        ]
        with shuffled_path.open("w", encoding="utf-8") as handle:
            subprocess.run(shuffle_cmd, cwd=str(project_root), check=True, stdout=handle)

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
            "0",
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
        run_cmd(validate_cmd, cwd=project_root)

        results_path = experiment_dir / "results.csv"
        deseq_cmd = [
            "Rscript",
            "scripts/04_differential_expression/run_differential_expression_by_inferred_labels.R",
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
        run_cmd(deseq_cmd, cwd=project_root)

        limma_cmd = [
            "Rscript",
            "scripts/04_differential_expression/run_limma_by_inferred_labels.R",
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
            "10000",
        ]
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
            "10000",
        ]
        run_cmd(fgsea_fc_cmd, cwd=project_root)

        validation = summarise_validation(experiment_dir / "validation_label_associations.csv")
        deseq_summary = parse_summary_txt(deseq_dir / "run_summary.txt")
        limma_summary = parse_summary_txt(limma_dir / "limma_run_summary.txt")
        fgsea_stat_summary = parse_summary_txt(Path(str(fgsea_stat_prefix) + "_fgsea_summary.txt"))
        fgsea_fc_summary = parse_summary_txt(Path(str(fgsea_fc_prefix) + "_fgsea_summary.txt"))

        row: Dict[str, float | str] = {
            "replicate": float(rep),
            "seed": float(seed),
            "validation_rows": validation["validation_rows"],
            "validation_min_p": validation["validation_min_p"],
            "validation_n_p_le_0.05": validation["validation_n_p_le_0.05"],
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
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "bootstrap_summary.csv", index=False)

    summary_path = out_dir / "bootstrap_summary.csv"
    summary = pd.read_csv(summary_path)
    agg = {
        "n_replicates": len(summary),
        "mean_validation_n_p_le_0.05": summary["validation_n_p_le_0.05"].mean(),
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
