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

RESULT_NAME="05_pathway_enrichment"

# Reproduce thesis-linked pathway outputs under:
# outputs/thesis/05_pathway_enrichment/
OUT_DIR="outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step7_fgsea_deseq_grid"
FIG_DIR="outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/figures_fgsea"
mkdir -p "${OUT_DIR}" "${FIG_DIR}"

# Ensure DE inputs exist.
bash scripts/03_differential_expression/reproduce_result.sh

# Build fgsea grid across model variants and rank metrics.
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
    Rscript scripts/05_pathway_enrichment/run_fgsea_from_de.R \
      --de-results "${de_path}" \
      --rank-metric "${metric}" \
      --gene-sets-source msigdb \
      --msigdb-collection H \
      --out-prefix "${OUT_DIR}/${name}__${metric}"
  done
done

# Summarise fgsea grid results.
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

# Rebuild two thesis figures from the step1/stat ranking.
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

hallmark <- as.data.table(msigdbr(species = "Homo sapiens", category = "H"))
pathways <- split(hallmark$gene_symbol, hallmark$gs_name)

for (pw in c("HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_TNFA_SIGNALING_VIA_NFKB")) {
  p <- plotEnrichment(pathways[[pw]], stats) + ggtitle(pw)
  slug <- gsub("[^A-Za-z0-9_]+", "_", pw)
  out <- file.path(fig_dir, paste0("step1_stat_", slug, "_enrichment.png"))
  ggsave(out, p, width = 8, height = 6, dpi = 150)
}
RS

echo "Done: ${RESULT_NAME} regenerated."
