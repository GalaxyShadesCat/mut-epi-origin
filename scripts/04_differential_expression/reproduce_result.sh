#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
else
  CONDA_BASE="$(conda info --base)"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Conda deactivate/activate hooks may read unset backup vars under `set -u`.
set +u
if [[ "${CONDA_DEFAULT_ENV:-}" != "mut-epi-origin" ]]; then
  conda activate mut-epi-origin
fi
set -u

RESULT_NAME="04_differential_expression"
DE_OUT_DIR="outputs/experiments/lihc_foxa2_all_samples/de_exp_decay_500k_spearman_r_linear_resid"
LIMMA_OUT_DIR="outputs/experiments/lihc_foxa2_all_samples/de_limma_exp_decay_500k_spearman_r_linear_resid"

# Reproduce primary thesis DE output:
# outputs/thesis/04_differential_expression/04_differential_expression_deseq2_binary_all_genes.csv
# Binary DESeq2 run (main thesis DE table).
Rscript scripts/04_differential_expression/run_differential_expression_by_inferred_labels.R \
  --counts-path data/raw/rna/TCGA-LIHC.star_counts.tsv \
  --results-path outputs/experiments/lihc_foxa2_all_samples/results.csv \
  --metadata-path data/derived/master_metadata.csv \
  --metadata-sample-col tumour_sample_submitter_id \
  --covariates age_at_diagnosis,gender,ajcc_pathologic_stage \
  --output-dir "${DE_OUT_DIR}" \
  --track-strategy exp_decay \
  --bin-size 500000 \
  --scoring-system spearman_r_linear_resid \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --contrast-case foxa2_abnormal_zero \
  --contrast-control foxa2_normal_pos

# Binary limma-voom run (paired with DESeq2 labels from this run).
Rscript scripts/04_differential_expression/run_limma_by_inferred_labels.R \
  --counts-path data/raw/rna/TCGA-LIHC.star_counts.tsv \
  --labels-path "${DE_OUT_DIR}/sample_labels_used.csv" \
  --output-dir "${LIMMA_OUT_DIR}" \
  --contrast-case foxa2_abnormal_zero \
  --contrast-control foxa2_normal_pos

# Stepwise follow-up run producing thesis-linked outputs under:
# outputs/thesis/04_differential_expression/
Rscript scripts/04_differential_expression/run_de_followups_stepwise.R

echo "Done: ${RESULT_NAME} regenerated."
