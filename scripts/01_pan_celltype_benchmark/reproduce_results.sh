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

RESULT_NAME="01_pan_celltype_benchmark"

# Reproduce:
# outputs/thesis/01_pan_celltype_benchmark/pan_celltype_benchmark_results.csv
CMD_FILE="scripts/01_pan_celltype_benchmark/pan_celltype_benchmark_run_command.txt"
if [[ ! -f "${CMD_FILE}" ]]; then
  echo "Missing command file: ${CMD_FILE}" >&2
  exit 1
fi
eval "$(cat "${CMD_FILE}")"

echo "Done: ${RESULT_NAME} regenerated."
