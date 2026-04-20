#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin

RESULT_NAME="04_differential_expression"

# Reproduce primary thesis DE output:
# outputs/thesis/04_differential_expression/04_differential_expression_deseq2_binary_all_genes.csv
# Binary DESeq2 run (main thesis DE table).
Rscript scripts/04_differential_expression/run_differential_expression_by_inferred_labels.R \
  --counts-path data/raw/rna/TCGA-LIHC.star_counts.tsv \
  --results-path outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/results.csv \
  --metadata-path data/derived/master_metadata.csv \
  --metadata-sample-col tumour_sample_submitter_id \
  --covariates age_at_diagnosis,gender,ajcc_pathologic_stage \
  --track-strategy counts_raw \
  --bin-size 500000 \
  --scoring-system spearman_r_linear_resid \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --contrast-case foxa2_abnormal_zero \
  --contrast-control foxa2_normal_pos

# Stepwise follow-up run producing thesis-linked outputs under:
# outputs/thesis/04_differential_expression/
Rscript scripts/04_differential_expression/run_de_followups_stepwise.R

echo "Done: ${RESULT_NAME} regenerated."
