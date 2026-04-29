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

RESULT_NAME="03_hepatocyte_clinical_associations"

# Reproduce:
# outputs/thesis/03_hepatocyte_clinical_associations/03_hepatocyte_state_benchmark_results.csv
# outputs/experiments/lihc_foxa2_clinical_complete/validation_summary.csv
# outputs/experiments/lihc_foxa2_clinical_complete/validation_group_tests.csv
# outputs/experiments/lihc_foxa2_clinical_complete/validation_group_tests_mutation_burden_adjusted.csv
# outputs/experiments/lihc_foxa2_clinical_complete/validation_label_associations.csv
CMD_FILE="outputs/thesis/03_hepatocyte_clinical_associations/03_hepatocyte_state_benchmark_run_command.txt"
if [[ ! -f "${CMD_FILE}" ]]; then
  echo "Missing command file: ${CMD_FILE}" >&2
  exit 1
fi
eval "$(cat "${CMD_FILE}")"

python scripts/01_pan_celltype_benchmark/validate_state_scores.py \
  --experiment-name lihc_foxa2_clinical_complete \
  --metadata-path data/derived/master_metadata.csv \
  --metadata-sample-col tumour_sample_submitter_id \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --adjust-for-mutation-burden \
  --mutation-burden-col mutations_post_downsample \
  --allow-aggregated-results

echo "Done: ${RESULT_NAME} regenerated."
