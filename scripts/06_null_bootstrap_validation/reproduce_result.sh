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

RESULT_NAME="06_null_bootstrap_validation"

# Reproduce null-bootstrap outputs in:
# outputs/experiments/null_shuffle_bootstrap_foxa2_exp_decay_500k/
# Null bootstrap reproducer.
# This runs the full negative-control pipeline per replicate:
# shuffle -> benchmark rerun -> validation -> DESeq2 -> limma -> fgsea.
#
# Default below is 10 permutations; pass another count
# as the first positional argument if needed.
# Pass a second positional argument to change maximum parallel jobs.
# Pass a third positional argument to control resume mode:
#   resume (default)   -> skip completed replicate directories
#   no-resume          -> rerun all requested replicate indices
N_BOOTSTRAP="${1:-10}"
MAX_PARALLEL="${2:-10}"
RESUME_MODE="${3:-resume}"
BASE_SEED=20260408
OUT_DIR="outputs/experiments/null_shuffle_bootstrap_foxa2_exp_decay_500k"
JOB_ROOT="${OUT_DIR}/_parallel_jobs"

mkdir -p "${OUT_DIR}" "${JOB_ROOT}"

if [[ "${RESUME_MODE}" != "resume" && "${RESUME_MODE}" != "no-resume" ]]; then
  echo "Invalid RESUME_MODE: ${RESUME_MODE}. Use 'resume' or 'no-resume'." >&2
  exit 1
fi

run_one_permutation() {
  local rep="$1"
  local rep_label seed job_dir rep_dir

  rep_label="$(printf "%03d" "${rep}")"
  seed="$((BASE_SEED + rep))"
  job_dir="${JOB_ROOT}/replicate_${rep_label}"
  rep_dir="${OUT_DIR}/replicate_${rep_label}"

  rm -rf "${job_dir}"
  mkdir -p "${job_dir}"

  echo "[replicate_${rep_label}] start (seed=${seed})"

  # Do not abort the whole launcher if this replicate fails internally
  # (for example FGSEA package availability issues on HPC nodes).
  if ! python scripts/06_null_bootstrap_validation/bootstrap_shuffle_null.py \
    --n-bootstrap 1 \
    --base-seed "${seed}" \
    --mut-path data/raw/mutations/lihc_snv_mutation_table.tsv \
    --fai-path data/raw/reference/GRCh37.fa.fai \
    --fasta-path data/raw/reference/GRCh37.fa \
    --timing-bw data/raw/timing/repliSeq_SknshWaveSignalRep1.bigWig \
    --metadata-path data/derived/master_metadata.csv \
    --counts-path data/raw/rna/TCGA-LIHC.star_counts.tsv \
    --out-dir "${job_dir}" \
    --track-strategy exp_decay \
    --bin-size 500000 \
    --scoring-system spearman_r_linear_resid \
    --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
    --adjust-for-mutation-burden \
    --mutation-burden-col mutations_post_downsample \
    > "${job_dir}/run.log" 2>&1; then
    echo "[replicate_${rep_label}] launcher command failed; continuing."
  fi

  if [[ -d "${job_dir}/replicate_000" ]]; then
    rm -rf "${rep_dir}"
    mv "${job_dir}/replicate_000" "${rep_dir}"
  else
    mkdir -p "${rep_dir}"
    echo "[replicate_${rep_label}] missing replicate output directory."
  fi

  if [[ -f "${job_dir}/bootstrap_summary.csv" ]]; then
    cp "${job_dir}/bootstrap_summary.csv" "${rep_dir}/bootstrap_summary_single.csv"
  fi

  cp "${job_dir}/run.log" "${rep_dir}/run.log" 2>/dev/null || true
  echo "[replicate_${rep_label}] done"
}

running_jobs=0
queued_jobs=0
for rep in $(seq 0 $((N_BOOTSTRAP - 1))); do
  rep_label="$(printf "%03d" "${rep}")"
  rep_dir="${OUT_DIR}/replicate_${rep_label}"
  single_summary="${rep_dir}/bootstrap_summary_single.csv"

  if [[ "${RESUME_MODE}" == "resume" && -f "${single_summary}" ]]; then
    echo "[replicate_${rep_label}] already completed; skipping."
    continue
  fi

  run_one_permutation "${rep}" &
  running_jobs=$((running_jobs + 1))
  queued_jobs=$((queued_jobs + 1))

  if (( running_jobs >= MAX_PARALLEL )); then
    wait -n || true
    running_jobs=$((running_jobs - 1))
  fi
done

while (( running_jobs > 0 )); do
  wait -n || true
  running_jobs=$((running_jobs - 1))
done

if (( queued_jobs == 0 )); then
  echo "No new permutations were launched."
fi

python - <<PY
import json
from pathlib import Path

import pandas as pd

n_bootstrap = int("${N_BOOTSTRAP}")
base_seed = int("${BASE_SEED}")
out_dir = Path("${OUT_DIR}")

rows = []
default_cols = [
    "replicate",
    "seed",
    "status",
    "failed_stage",
    "failure_reason",
    "validation_rows",
    "validation_min_p",
    "validation_n_p_le_0.05",
    "validation_group_tests_rows",
    "validation_group_tests_min_p",
    "validation_group_tests_n_p_le_0.05",
    "validation_group_tests_mb_adjusted_rows",
    "validation_group_tests_mb_adjusted_min_p",
    "validation_group_tests_mb_adjusted_n_p_le_0.05",
    "deseq_significant_genes",
    "deseq_genes_tested",
    "limma_significant_genes",
    "limma_genes_tested",
    "fgsea_stat_significant_pathways",
    "fgsea_fc_significant_pathways",
]

for rep in range(n_bootstrap):
    rep_dir = out_dir / f"replicate_{rep:03d}"
    single_summary = rep_dir / "bootstrap_summary_single.csv"
    if single_summary.exists():
        dt = pd.read_csv(single_summary)
        if not dt.empty:
            row = dt.iloc[0].to_dict()
        else:
            row = {}
    else:
        row = {}

    row["replicate"] = float(rep)
    row["seed"] = float(base_seed + rep)
    row.setdefault("status", "failed")
    row.setdefault("failed_stage", "unknown")
    row.setdefault("failure_reason", "missing_single_summary")
    rows.append(row)

summary = pd.DataFrame(rows)
for col in default_cols:
    if col not in summary.columns:
        summary[col] = pd.NA
summary = summary[default_cols]

summary_path = out_dir / "bootstrap_summary.csv"
summary.to_csv(summary_path, index=False)

agg = {
    "n_replicates": len(summary),
    "mean_validation_n_p_le_0.05": float(summary["validation_n_p_le_0.05"].mean()),
    "mean_validation_group_tests_n_p_le_0.05": float(
        summary["validation_group_tests_n_p_le_0.05"].mean()
    ),
    "mean_validation_group_tests_mb_adjusted_n_p_le_0.05": float(
        summary["validation_group_tests_mb_adjusted_n_p_le_0.05"].mean()
    ),
    "mean_deseq_significant_genes": float(summary["deseq_significant_genes"].mean()),
    "mean_limma_significant_genes": float(summary["limma_significant_genes"].mean()),
    "mean_fgsea_stat_significant_pathways": float(
        summary["fgsea_stat_significant_pathways"].mean()
    ),
    "mean_fgsea_fc_significant_pathways": float(
        summary["fgsea_fc_significant_pathways"].mean()
    ),
}
with (out_dir / "bootstrap_aggregate_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(agg, handle, indent=2)

print(f"Wrote: {summary_path}")
print(f"Wrote: {out_dir / 'bootstrap_aggregate_summary.json'}")
PY

echo "Done: ${RESULT_NAME} regenerated."
