#!/usr/bin/env python3
"""Run gene-level differential mutation analysis using inferred label groups.

This script compares gene-level mutation frequencies between two inferred
sample groups from `validation_score_rankings.csv`.

Workflow:
1. Read inferred labels and filter to one config and scoring system.
2. Keep samples passing a score-gap threshold.
3. Read a gene-annotated mutation table.
4. Build a per-gene mutated or not-mutated table by sample.
5. Run Fisher's exact test per gene between the two labels.
6. Apply Benjamini-Hochberg FDR correction.
7. Write full and significant results tables.

Notes:
- TCGA barcodes are harmonised to the first 15 characters for joining.
- This is differential mutation analysis, not RNA expression analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mutation-path",
        type=Path,
        required=True,
        help="Path to a gene-annotated mutation table.",
    )
    parser.add_argument(
        "--rankings-path",
        type=Path,
        default=Path(
            "outputs/experiments/"
            "lihc_foxa2_clinical_complete/"
            "validation_score_rankings.csv"
        ),
        help="Path to validation_score_rankings.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/"
            "lihc_foxa2_clinical_complete/"
            "dm_counts_raw_500k_spearman_r_linear_resid"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--config-id",
        default="track_strategy=counts_raw|bin_size=500000.0",
        help="Configuration ID filter.",
    )
    parser.add_argument(
        "--scoring-system",
        default="spearman_r_linear_resid",
        help="Scoring-system filter.",
    )
    parser.add_argument(
        "--score-gap-threshold",
        type=float,
        default=0.0,
        help="Minimum score-gap threshold.",
    )
    parser.add_argument(
        "--rankings-sample-col",
        default="sample",
        help="Sample column name in rankings file.",
    )
    parser.add_argument(
        "--rankings-label-col",
        default="best_cell_state",
        help="Label column name in rankings file.",
    )
    parser.add_argument(
        "--mutation-sample-col",
        default="",
        help="Sample column in mutation table (auto-detected if blank).",
    )
    parser.add_argument(
        "--mutation-gene-col",
        default="",
        help="Gene column in mutation table (auto-detected if blank).",
    )
    parser.add_argument(
        "--variant-class-col",
        default="Variant_Classification",
        help="Variant class column for optional silent filtering.",
    )
    parser.add_argument(
        "--exclude-silent",
        choices=["true", "false"],
        default="true",
        help="Exclude silent or synonymous variants when possible.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=3,
        help="Minimum samples required per label.",
    )
    parser.add_argument(
        "--min-mutated-samples",
        type=int,
        default=3,
        help="Minimum mutated samples required per gene.",
    )
    parser.add_argument(
        "--id-prefix-length",
        type=int,
        default=15,
        help="TCGA ID prefix length used for joining.",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR significance threshold.",
    )
    parser.add_argument(
        "--delim",
        default="tab",
        help="Mutation table delimiter: tab, comma, or a literal separator.",
    )
    return parser.parse_args()


def normalise_tcga_id(values: pd.Series, prefix_len: int) -> pd.Series:
    """Normalise sample identifiers to a harmonised TCGA-style prefix."""
    clean = (
        values.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .str.slice(0, prefix_len)
    )
    return clean


def infer_delimiter(delim_arg: str) -> str:
    """Return a concrete separator from a user delimiter argument."""
    lowered = delim_arg.lower()
    if lowered == "tab":
        return "\t"
    if lowered == "comma":
        return ","
    return delim_arg


def pick_column(
    columns: pd.Index,
    user_value: str,
    candidates: list[str],
    label: str,
) -> str:
    """Pick a user-specified or auto-detected column name."""
    if user_value:
        if user_value not in columns:
            raise ValueError(
                f"Requested {label} not found in mutation table: {user_value}"
            )
        return user_value
    for candidate in candidates:
        if candidate in columns:
            return candidate
    joined = ", ".join(candidates)
    raise ValueError(f"Could not auto-detect {label}. Candidates checked: {joined}")


def trim_to_na(series: pd.Series) -> pd.Series:
    """Trim whitespace and convert empty strings to missing."""
    clean = series.astype(str).str.strip()
    clean = clean.replace("", pd.NA)
    clean = clean.where(~clean.str.lower().eq("nan"), pd.NA)
    return clean


def main() -> None:
    """Run differential mutation analysis and write outputs."""
    args = parse_args()

    if not args.mutation_path.exists():
        raise FileNotFoundError(f"Mutation table not found: {args.mutation_path}")
    if not args.rankings_path.exists():
        raise FileNotFoundError(f"Rankings file not found: {args.rankings_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclude_silent = args.exclude_silent == "true"
    delim = infer_delimiter(args.delim)

    rankings = pd.read_csv(args.rankings_path)
    required_rank_cols = [
        args.rankings_sample_col,
        args.rankings_label_col,
        "config_id",
        "scoring_system",
        "score_gap",
    ]
    missing_rank_cols = [c for c in required_rank_cols if c not in rankings.columns]
    if missing_rank_cols:
        missing = ", ".join(missing_rank_cols)
        raise ValueError(f"Missing columns in rankings file: {missing}")

    keep_rank = (
        (rankings["config_id"] == args.config_id)
        & (rankings["scoring_system"] == args.scoring_system)
        & (rankings["score_gap"] >= args.score_gap_threshold)
    )
    label_df = rankings.loc[
        keep_rank,
        [args.rankings_sample_col, args.rankings_label_col, "score_gap"],
    ].copy()
    label_df.columns = ["sample_raw", "label", "score_gap"]
    if label_df.empty:
        raise ValueError("No ranking rows left after config, scoring, and threshold filters.")

    label_df["sample_id"] = normalise_tcga_id(
        label_df["sample_raw"], prefix_len=args.id_prefix_length
    )
    label_df["label"] = trim_to_na(label_df["label"])
    label_df = label_df.dropna(subset=["label"])
    label_df = label_df.drop_duplicates(subset=["sample_id"], keep="first")
    if label_df.empty:
        raise ValueError("No valid label assignments after cleaning.")

    mutation_df = pd.read_csv(args.mutation_path, sep=delim, low_memory=False)
    sample_col = pick_column(
        mutation_df.columns,
        args.mutation_sample_col,
        [
            "Tumor_Sample_Barcode",
            "tumour_sample_submitter_id",
            "sample",
            "Sample_ID",
            "sample_id",
        ],
        "sample column",
    )
    gene_col = pick_column(
        mutation_df.columns,
        args.mutation_gene_col,
        ["Hugo_Symbol", "gene_symbol", "Gene", "gene", "gene_name"],
        "gene column",
    )

    mutation_df["sample_id"] = normalise_tcga_id(
        mutation_df[sample_col], prefix_len=args.id_prefix_length
    )
    mutation_df["gene_name"] = trim_to_na(mutation_df[gene_col])
    mutation_df = mutation_df.dropna(subset=["sample_id", "gene_name"])

    if exclude_silent and args.variant_class_col in mutation_df.columns:
        variant_class = mutation_df[args.variant_class_col].astype(str).str.strip().str.lower()
        keep_variant = ~variant_class.isin({"silent", "synonymous", "synonymous_snv"})
        mutation_df = mutation_df.loc[keep_variant].copy()

    merged_samples = (
        pd.DataFrame({"sample_id": mutation_df["sample_id"].drop_duplicates()})
        .merge(label_df, on="sample_id", how="inner")
        .copy()
    )
    if merged_samples.empty:
        rank_examples = ", ".join(sorted(label_df["sample_id"].unique())[:5])
        mut_examples = ", ".join(sorted(mutation_df["sample_id"].unique())[:5])
        raise ValueError(
            "No overlapping samples between mutation table and label assignments.\n"
            f"Ranking sample examples: {rank_examples}\n"
            f"Mutation sample examples: {mut_examples}\n"
            "Please use a mutation table from the same cohort (TCGA-LIHC)."
        )

    group_sizes = merged_samples["label"].value_counts()
    keep_labels = group_sizes[group_sizes >= args.min_group_size].index
    merged_samples = merged_samples[merged_samples["label"].isin(keep_labels)].copy()
    group_sizes = merged_samples["label"].value_counts()
    if len(group_sizes) != 2:
        raise ValueError(
            "This script currently requires exactly two labels after filtering. "
            f"Found {len(group_sizes)} labels."
        )

    label_a, label_b = sorted(group_sizes.index.tolist())
    n_a = int((merged_samples["label"] == label_a).sum())
    n_b = int((merged_samples["label"] == label_b).sum())

    samples_used = set(merged_samples["sample_id"].tolist())
    sample_to_label = dict(
        zip(merged_samples["sample_id"], merged_samples["label"], strict=False)
    )
    mutation_df = mutation_df[mutation_df["sample_id"].isin(samples_used)].copy()
    if mutation_df.empty:
        raise ValueError("No mutation rows left after sample matching.")

    gene_sample = mutation_df[["gene_name", "sample_id"]].drop_duplicates().copy()
    gene_sample["label"] = gene_sample["sample_id"].map(sample_to_label)

    mutated_total = gene_sample["gene_name"].value_counts()
    genes_keep = mutated_total[mutated_total >= args.min_mutated_samples].index
    gene_sample = gene_sample[gene_sample["gene_name"].isin(genes_keep)].copy()
    if gene_sample.empty:
        raise ValueError("No genes left after min-mutated-samples filtering.")

    counts = (
        gene_sample.groupby(["gene_name", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=[label_a, label_b], fill_value=0)

    records: list[dict[str, object]] = []
    for gene_name, row in counts.iterrows():
        m_a = int(row[label_a])
        m_b = int(row[label_b])
        u_a = n_a - m_a
        u_b = n_b - m_b
        odds_ratio, p_value = fisher_exact([[m_a, u_a], [m_b, u_b]])
        frac_a = m_a / n_a
        frac_b = m_b / n_b
        records.append(
            {
                "gene": gene_name,
                "label_a": label_a,
                "label_b": label_b,
                "n_label_a": n_a,
                "n_label_b": n_b,
                "mutated_label_a": m_a,
                "mutated_label_b": m_b,
                "mutated_frac_label_a": frac_a,
                "mutated_frac_label_b": frac_b,
                "delta_mutated_frac_a_minus_b": frac_a - frac_b,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
            }
        )

    result_df = pd.DataFrame.from_records(records)
    reject, fdr_bh, _, _ = multipletests(result_df["p_value"], method="fdr_bh")
    result_df["fdr_bh"] = fdr_bh
    result_df["is_significant_fdr"] = reject
    result_df = result_df.sort_values(["fdr_bh", "p_value"], kind="mergesort")
    sig_df = result_df[result_df["fdr_bh"] <= args.fdr_alpha].copy()

    sample_out_path = args.output_dir / "sample_labels_used.csv"
    all_out_path = args.output_dir / "differential_mutation_results_all.csv"
    sig_out_path = args.output_dir / "differential_mutation_results_significant.csv"
    summary_out_path = args.output_dir / "run_summary.txt"

    merged_samples.to_csv(sample_out_path, index=False)
    result_df.to_csv(all_out_path, index=False)
    sig_df.to_csv(sig_out_path, index=False)

    summary_lines = [
        f"mutation_path: {args.mutation_path}",
        f"rankings_path: {args.rankings_path}",
        f"config_id: {args.config_id}",
        f"scoring_system: {args.scoring_system}",
        f"score_gap_threshold: {args.score_gap_threshold}",
        f"mutation_sample_col: {sample_col}",
        f"mutation_gene_col: {gene_col}",
        f"exclude_silent: {exclude_silent}",
        f"samples_used: {len(merged_samples)}",
        f"n_label_a: {n_a}",
        f"n_label_b: {n_b}",
        f"genes_tested: {len(result_df)}",
        f"fdr_alpha: {args.fdr_alpha}",
        f"significant_genes: {len(sig_df)}",
    ]
    summary_out_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Done.")
    print("Outputs:")
    print(f"  {sample_out_path}")
    print(f"  {all_out_path}")
    print(f"  {sig_out_path}")
    print(f"  {summary_out_path}")


if __name__ == "__main__":
    main()
