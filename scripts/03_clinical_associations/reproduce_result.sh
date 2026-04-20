#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin

RESULT_NAME="03_clinical_associations"

# Reproduce:
# outputs/thesis/03_clinical_associations/03_clinical_associations_validation_summary.csv
# outputs/thesis/03_clinical_associations/03_clinical_associations_group_association_tests.csv
# outputs/thesis/03_clinical_associations/03_clinical_associations_label_association_tests.csv
python scripts/01_pan_celltype_benchmark/validate_state_scores.py \
  --experiment-name lihc_foxa2_top4_all_samples_per_sample_merged \
  --metadata-path data/derived/master_metadata.csv \
  --metadata-sample-col tumour_sample_submitter_id \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --allow-aggregated-results

echo "Done: ${RESULT_NAME} regenerated."
