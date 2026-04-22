# Thesis Output Index

This directory is a curated, thesis-facing view of key outputs.
Pipeline outputs remain in `outputs/experiments/`.

## Sections
- `01_pan_celltype_benchmark`
  - `01_pan_celltype_benchmark_results.csv`
  - `01_pan_celltype_benchmark_run_command.txt`
- `02_hepatocyte_clinical_associations`
  - `02_hepatocyte_state_benchmark_results.csv`
  - `02_hepatocyte_state_benchmark_run_command.txt`
- `03_differential_expression`
  - `03_differential_expression_deseq2_binary_all_genes.csv`
  - `03_differential_expression_deseq2_binary_limma_gene_pool_all_genes.csv`
  - `03_differential_expression_deseq2_continuous_all_genes.csv`
  - `03_differential_expression_limma_voom_binary_all_genes.csv`
  - `03_differential_expression_deseq2_continuous_pca.png`
  - `03_differential_expression_deseq2_continuous_volcano_pvalue.png`
- `05_pathway_enrichment`
  - `05_pathway_enrichment_deseq_model_comparison_summary.csv`
  - `05_pathway_enrichment_deseq_continuous_stat_results.csv`
  - `05_pathway_enrichment_oxphos_deseq_continuous_stat.png`
  - `05_pathway_enrichment_tnfa_nfkb_deseq_continuous_stat.png`
- `06_null_bootstrap_validation`
  - `06_null_bootstrap_validation_pipeline.py`
  - `06_null_bootstrap_validation_run_limma_by_inferred_labels.R`
- Root report
  - `foxa2_followup_analysis_report.md`

## Notes
- Files here are symlinks to source outputs/scripts unless otherwise stated.
- Add final thesis figures/tables under `outputs/thesis/figures/` if needed.
