#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin

RESULT_NAME="01_pan_celltype_benchmark"

# Reproduce:
# outputs/thesis/01_pan_celltype_benchmark/01_pan_celltype_benchmark_results.csv
CMD_FILE="outputs/thesis/01_pan_celltype_benchmark/01_pan_celltype_benchmark_run_command.txt"
if [[ ! -f "${CMD_FILE}" ]]; then
  echo "Missing command file: ${CMD_FILE}" >&2
  exit 1
fi
eval "$(cat "${CMD_FILE}")"

echo "Done: ${RESULT_NAME} regenerated."
