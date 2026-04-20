#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin

RESULT_NAME="02_hepatocyte_state_identification"

# Reproduce:
# outputs/thesis/02_hepatocyte_state_identification/02_hepatocyte_state_benchmark_results.csv
CMD_FILE="outputs/thesis/02_hepatocyte_state_identification/02_hepatocyte_state_benchmark_run_command.txt"
if [[ ! -f "${CMD_FILE}" ]]; then
  echo "Missing command file: ${CMD_FILE}" >&2
  exit 1
fi
eval "$(cat "${CMD_FILE}")"

echo "Done: ${RESULT_NAME} regenerated."
