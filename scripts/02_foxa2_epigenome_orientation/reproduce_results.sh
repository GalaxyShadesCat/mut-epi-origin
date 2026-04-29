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
set +u
if [[ "${CONDA_DEFAULT_ENV:-}" != "mut-epi-origin" ]]; then
  conda activate mut-epi-origin
fi
set -u

jupyter nbconvert \
  --to notebook \
  --execute outputs/thesis/02_foxa2_epigenome_orientation/foxa2_epigenome_orientation.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=1200

echo "Done: 02_foxa2_epigenome_orientation regenerated."
