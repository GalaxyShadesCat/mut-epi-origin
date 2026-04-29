# Thesis Output Index

This directory is a curated, thesis-facing view of key outputs.
Pipeline outputs remain outside this curated thesis view.

## Sections
- `01_pan_celltype_benchmark`
  - `pan_celltype_benchmark_results.csv`
  - `figures/01_overall_benchmark_performance.png`
  - `figures/02_performance_by_cancer_type_best_setup.png`
  - `figures/03_score_gap_vs_mutation_burden_best_setup.png`
- `02_foxa2_epigenome_orientation`
  - `figures/01_lihc_mutation_burden.png`
  - `figures/02_lihc_clinical_feature_summary.png`
  - `figures/03_foxa2_umap_reference_cells.png`
  - `figures/04_selected_locus_foxa2_atac_with_sample_exp_decay_mutations.png`
- `03_hepatocyte_clinical_associations`
  - `figures/01_state_distribution_selected_setup.png`
  - `figures/02_label_sensitivity_state_composition.png`
  - `figures/03_clinical_boxplots_adjusted_scores_by_viral_status.png`
  - `figures/04_overall_survival_by_foxa2_state.png`
- `04_differential_expression`
  - `figures/01_pathway_nes_plot.png`
  - `figures/02_volcano_limma_binary.png`
  - `figures/03_representative_gene_expression_panel.png`
- `05_null_bootstrap_validation`
  - `reproduce_results.sh`
  - `figures/01_null_benchmark_clinical_tests.png`
  - `figures/02_null_benchmark_genes.png`
  - `figures/03_null_benchmark_pathways.png`

## Notes
- Executable scripts live under the repository-level `scripts/` directory; thesis folders keep only lightweight reproduction wrappers.
