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
# outputs/thesis/04_differential_expression/data/source_inputs/limma_voom_binary_results.csv
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

# Build the pathway-enrichment grid across model variants and rank metrics.
FGSEA_OUT_DIR="outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step7_fgsea_deseq_grid"
FGSEA_FIG_DIR="outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/figures_fgsea"
mkdir -p "${FGSEA_OUT_DIR}" "${FGSEA_FIG_DIR}"

for spec in \
  "step1_continuous|outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step1_continuous_deseq_results.csv" \
  "step3_continuous_sva|outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step3_continuous_deseq_with_sva_results.csv" \
  "step4_hallmark_universe|outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step4_continuous_deseq_hallmark_universe_results.csv" \
  "step6_limma_gene_pool|outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step6_deseq2_limma_gene_pool_results.csv" \
  "deseq2_binary_main|outputs/experiments/lihc_foxa2_all_samples/de_exp_decay_500k_spearman_r_linear_resid/differential_expression_results_all.csv"
do
  name="${spec%%|*}"
  de_path="${spec##*|}"
  for metric in stat sign_logfc_neglog10p logfc_times_neglog10p
  do
    Rscript scripts/04_differential_expression/run_fgsea_from_de.R \
      --de-results "${de_path}" \
      --rank-metric "${metric}" \
      --gene-sets-source msigdb \
      --msigdb-collection H \
      --out-prefix "${FGSEA_OUT_DIR}/${name}__${metric}"
  done
done

python - <<'PY'
from pathlib import Path

import pandas as pd

out_dir = Path("outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step7_fgsea_deseq_grid")
rows = []
for summary in sorted(out_dir.glob("*_fgsea_summary.txt")):
    stem = summary.name.replace("_fgsea_summary.txt", "")
    run_name, metric = stem.split("__", 1)
    vals = {}
    for line in summary.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            vals[k.strip()] = v.strip()
    rows.append(
        {
            "input": run_name,
            "metric": metric,
            "ok": True,
            "genes_ranked": int(float(vals.get("genes_ranked", "nan"))),
            "pathways_tested": int(float(vals.get("pathways_tested", "nan"))),
            "significant_pathways_fdr_0.05": int(float(vals.get("significant_pathways_fdr_0.05", "0"))),
            "best_pathway": vals.get("best_pathway", ""),
            "best_pathway_padj": vals.get("best_pathway_padj", ""),
            "stderr_head": "[]",
        }
    )
pd.DataFrame(rows).sort_values(["input", "metric"]).to_csv(out_dir / "grid_summary.csv", index=False)
PY

Rscript - <<'RS'
suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
  library(ggplot2)
})
rank_path <- "outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step7_fgsea_deseq_grid/step1_continuous__stat_ranked_genes.csv"
fig_dir <- "outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/figures_fgsea"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

rk <- fread(rank_path)
stats <- rk$score
names(stats) <- rk$gene
stats <- sort(stats, decreasing = TRUE)

hallmark <- as.data.table(msigdbr(species = "Homo sapiens", collection = "H"))
pathways <- split(hallmark$gene_symbol, hallmark$gs_name)

for (pw in c("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_TNFA_SIGNALING_VIA_NFKB")) {
  p <- plotEnrichment(pathways[[pw]], stats) + ggtitle(pw)
  slug <- gsub("[^A-Za-z0-9_]+", "_", pw)
  out <- file.path(fig_dir, paste0("step1_stat_", slug, "_enrichment.png"))
  ggsave(out, p, width = 8, height = 6, dpi = 150)
}
RS

echo "Done: ${RESULT_NAME} regenerated."
