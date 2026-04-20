#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source /home/lem/miniconda3/etc/profile.d/conda.sh
conda activate mut-epi-origin

RESULT_NAME="06_null_bootstrap_validation"

# Reproduce null-bootstrap outputs in:
# outputs/experiments/null_shuffle_bootstrap_foxa2_counts_raw_500k/
# Null bootstrap reproducer.
# This runs the full negative-control pipeline per replicate:
# shuffle -> benchmark rerun -> validation -> DESeq2 -> limma -> fgsea.
#
# Default below is 10 replicates (same as script default); pass another count
# as first positional argument if needed.
N_BOOTSTRAP="${1:-10}"

python scripts/06_null_bootstrap_validation/bootstrap_shuffle_null.py \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --base-seed 20260408 \
  --mut-path data/raw/mutations/lihc_snv_mutation_table.tsv \
  --fai-path data/raw/reference/GRCh37.fa.fai \
  --fasta-path data/raw/reference/GRCh37.fa \
  --timing-bw data/raw/timing/repliSeq_SknshWaveSignalRep1.bigWig \
  --metadata-path data/derived/master_metadata.csv \
  --counts-path data/raw/rna/TCGA-LIHC.star_counts.tsv \
  --out-dir outputs/experiments/null_shuffle_bootstrap_foxa2_counts_raw_500k \
  --track-strategy counts_raw \
  --bin-size 500000 \
  --scoring-system spearman_r_linear_resid \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero

echo "Done: ${RESULT_NAME} regenerated."
