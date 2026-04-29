# FOXA2-Inferred State Expression Follow-up

## 1) What was analysed

### Input files
- Counts: `data/raw/rna/TCGA-LIHC.star_counts.tsv`
- Inferred labels and scores: `outputs/thesis/04_differential_expression/data/source_inputs/sample_labels_used.csv`
- Metadata: `data/derived/master_metadata.csv`

### Samples used
- Complete-case samples: `n = 271`
- Group sizes:
  - `foxa2_normal_pos = 220`
  - `foxa2_abnormal_zero = 51`

### Main covariates in adjusted models
- `age_at_diagnosis` (scaled)
- `gender`
- `ajcc_pathologic_stage`

### Modelling approach
- **limma-voom sensitivity analysis**
  - `edgeR::calcNormFactors()` for TMM normalisation
  - `limma::voom()` for logCPM precision weights
- **DESeq2 follow-up**
  - standard DESeq2 normalisation and dispersion framework

### Scripts used
- Stepwise follow-up: `scripts/04_differential_expression/run_de_followups_stepwise.R`
- limma immune-partition script: `scripts/04_differential_expression/partition_limma_hits_by_immune_signal.R`

---

## 2) Main results

### Summary
From `stepwise_report.txt`:

- **Continuous DESeq2 analysis (score-based):**
  - no FDR-significant genes

- **Hallmark fgsea on ranked continuous statistics:**
  - strong pathway-level signal
  - `29/50` Hallmark pathways at `FDR < 0.05`

- **limma-voom binary sensitivity analysis (`group` term):**
  - `17` significant genes at `FDR < 0.05`

### Key output files
- `stepwise_report.txt`
- `step5_limma_voom_binary_results.csv`
- `step2_fgsea_hallmark.csv`

---

## 3) What the results mean

The main question was whether the inferred FOXA2-based groups capture a biologically meaningful difference in TCGA-LIHC tumours.

At a broad level, the answer is **yes, partly**.

- The **pathway-level signal is strong**
- The **single-gene signal is weaker and depends on the method**
- A large part of the single-gene limma signal appears to be related to **immune or microenvironment differences**
- One gene, **FZD10**, remains the strongest candidate for a more **tumour-intrinsic** FOXA2-related signal

Importantly, these are **all tumour samples**. This is **not** a tumour-versus-normal analysis.

Instead, the analysis asks:

**Which tumour samples look more similar to the FOXA2+ reference track, and which look more similar to the FOXA2- reference track?**

So the result is best interpreted as a comparison of **two tumour subgroups**, not tumour versus normal liver.

---

## 4) Immune-related signal and why it still matters

Although all samples are tumours, TCGA RNA-seq is **bulk tumour RNA-seq**. That means each sample can still contain a mixture of:

- tumour cells
- immune cells
- stromal cells
- other cells in the tumour microenvironment

Because of this, gene-expression differences between tumour groups can reflect both:

1. **tumour-cell state**
2. **differences in immune or stromal composition**

So the immune-related genes are still important. They suggest that the two inferred tumour groups may differ not only in tumour biology, but also in their surrounding microenvironment.

However, for validating the FOXA2-based grouping itself, the more important evidence is the part that remains after accounting for immune-related signal.

---

## 5) Immune-associated filtering and its impact

Baseline significant limma genes were split into:

- `likely_immune_associated`
- `candidate_tumour_intrinsic`

### Rules used
A gene was treated as **likely immune-associated** if it showed:
- immunoglobulin, TCR, HLA, or CD-marker patterns
- and/or membership in immune-related Hallmark gene sets

### Result of this split
- Total baseline significant genes: `17`
- Likely immune-associated: `12`
- Candidate tumour-intrinsic: `5`

### Immune-adjusted sensitivity rerun
An additional immune-score covariate was added.

After this adjustment:
- Significant genes remaining: `2`
- Baseline retained: `2`
- Baseline lost: `15`

### Genes retained after immune adjustment
- `IGHV1-3`  
- `FZD10`

### Interpretation
This suggests that **most of the limma gene-level signal is probably driven by immune composition**.

After adjusting for this, **FZD10** remains the strongest non-immune candidate, making it the best current tumour-intrinsic gene signal in this FOXA2-state framework.

### Relevant files
- `step5_limma_hit_partition_summary.txt`
- `step5_limma_significant_genes_annotated.csv`
- `step5_limma_immune_adjusted_results.csv`
- `step5_limma_vs_immune_adjusted_comparison.csv`

---

## 6) FZD10 finding in simple terms

In this cohort:

- `FZD10` is higher in `foxa2_abnormal_zero` than in `foxa2_normal_pos`
- `FZD10` remains significant after immune-score adjustment
- Across all LIHC samples, FOXA2 and FZD10 show a weak inverse relationship:
  - Spearman = `-0.139`
  - Pearson = `-0.174`

### Simple interpretation
The **FOXA2- aligned state** is associated with **higher FZD10**.

This is important because FOXA2 is generally linked to a more normal, differentiated liver-cell state, while higher FZD10 is more consistent with a more aggressive or stem-like tumour state.

So the direction of the result makes biological sense:

- **more FOXA2-** -> **higher FZD10**
- **more FOXA2+** -> **lower FZD10**

This does **not** prove that FOXA2 directly regulates FZD10, but it does support a biologically coherent inverse relationship.

---

## 7) Literature support for FOXA2 and FZD10

### FZD10 in HCC
FZD10 has been linked to:
- liver cancer stem-cell features
- lenvatinib resistance
- WNT/beta-catenin and Hippo signalling

References:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/36764493/
- DOI: https://doi.org/10.1053/j.gastro.2023.01.041

### FOXA2 in HCC
FOXA2 is generally reported as:
- anti-metastatic
- tumour-suppressive
- associated with a more differentiated liver-cell programme

References:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/25142974/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5836661/

### Overall consistency
Our finding of **higher FZD10 in the FOXA2-FOXA2- group** is directionally consistent with a **more aggressive, less differentiated tumour state**.

---

## 8) Other genes worth follow-up

Baseline candidate tumour-intrinsic genes:
- `FZD10`
- `STMN2`
- `TWIST2`
- `NCEH1`
- `SLC28A3`

### 1. `FZD10`
Strongest current candidate.
- Survives immune adjustment
- Fits the FOXA2-abnormal/aggressive-state interpretation
- Best gene to prioritise

### 2. `STMN2`
Supported in HCC.
- Reported to promote EMT through TGF-beta/Smad2/3 signalling

Reference:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/33705863/

### 3. `TWIST2`
Supportive but less direct.
- TWIST-family biology is linked to EMT, migration, and invasion in HCC
- Evidence is often stronger for the TWIST family overall than for TWIST2 alone

References:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/19615090/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/20219012/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3922390/

### 4. `NCEH1`
Currently weaker HCC-specific support.
- Broader cancer literature suggests roles in lipid metabolism and oncogenic processes
- Direct FOXA2/HCC relevance is still limited here

References:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9229287/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7748380/

### 5. `SLC28A3`
Currently weaker HCC-specific support.
- Better known for nucleoside transport and chemotherapy-response contexts in other cancers
- Direct FOXA2/HCC evidence is limited in this review

References:
- https://pubmed.ncbi.nlm.nih.gov/30778771/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11322974/

---

## 9) Immune-context genes

Some significant genes, such as:
- `TNFSF15`
- `ICOS`
- immunoglobulin genes

appear to be mainly related to immune or microenvironment differences.

Most of these were lost after immune-score adjustment, which supports the idea that they are **real but mostly microenvironment-driven**, rather than strong tumour-intrinsic markers of the FOXA2 state.

This means the immune-related genes are useful for understanding the tumour groups more broadly, but they should not be the main headline if the goal is to validate a FOXA2-related tumour-cell programme.

Reference:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/40762626/

---

## 10) Practical conclusion

### Main conclusion
The FOXA2-based inferred grouping appears to separate **TCGA-LIHC tumours into biologically distinct subgroups**.

### What is strongest
- Strong **pathway-level** signal
- A biologically coherent tumour-intrinsic candidate: **FZD10**

### What is weaker
- Single-gene results are **method-sensitive**
- Much of the baseline limma signal appears to be driven by **immune or microenvironment differences**

### Best interpretation
This grouping is likely capturing a mixture of:
- **tumour-cell state**
- **tumour microenvironment differences**

### Best current gene-level take-home message
**FZD10 is the strongest retained tumour-intrinsic candidate in this FOXA2-state framework.**

### Suggested next priority
Prioritise:
1. **FZD10**
2. pathway-level interpretation
3. immune-context findings as supporting context, not the main claim

### One-sentence summary
The inferred FOXA2 tracks seem to identify meaningful tumour subgroups in TCGA-LIHC, and while much of the gene-level signal is influenced by immune composition, **FZD10** remains the clearest tumour-intrinsic candidate linked to the FOXA2-abnormal state.


## 8) Direct config-alignment check (score_delta)

To directly test whether expression aligns with the selected config (not just group labels), we correlated per-sample expression with:
- `score_delta = abnormal_score - normal_score` from the selected config (`counts_raw|500000` + `spearman_r_linear_resid`).

Output files:
- `config_alignment_score_delta_vs_FOXA2_FZD10_summary.txt`
- `config_alignment_score_delta_vs_FOXA2_FZD10_samples.csv`

Results (`n = 283` matched samples):

1. `score_delta` vs `FOXA2` expression
- Spearman `r = 0.027`, `p = 0.651`
- Pearson `r = 0.033`, `p = 0.575`
- Interpretation: no meaningful direct association between score_delta and FOXA2 mRNA abundance.

2. `score_delta` vs `FZD10` expression
- Spearman `r = 0.123`, `p = 0.039`
- Pearson `r = 0.124`, `p = 0.037`
- Interpretation: weak but statistically significant positive association (higher FOXA2- score with higher FZD10).

3. Groupwise sanity check (wilcoxon)
- FOXA2 expression (`foxa2_abnormal_zero` vs `foxa2_normal_pos`): `p = 0.878` (not different)
- FZD10 expression (`foxa2_abnormal_zero` vs `foxa2_normal_pos`): `p = 0.00615` (higher in FOXA2- group)

Implication for config validation:
- In this dataset, the config captures a programme-level state that aligns with FZD10 and pathway shifts, but not with a strong FOXA2 mRNA shift itself.
- This does not invalidate the config; it indicates the selected state behaves more like a transcriptional programme than a single-gene (FOXA2 mRNA) threshold.

## 11) Consolidated gene results table

This table summarises the top limma hits with immune-adjustment status and DESeq2 FDR from a rerun restricted to the same limma gene universe (`20,279` genes).

| Gene (Ensembl) | Symbol | Direction | limma logFC | limma p | limma FDR | Sig after immune adj | Immune-adj FDR | Partition | DESeq2 FDR (limma pool) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ENSG00000211935.3 | IGHV1-3 | higher in foxa2_abnormal_zero | 0.972 | 3.16e-08 | 0.000641 | True | 0.000945 | likely_immune_associated | 1 |
| ENSG00000181634.8 | TNFSF15 | higher in foxa2_abnormal_zero | 0.272 | 2.46e-06 | 0.0168 | False | 0.18 | likely_immune_associated | 1 |
| ENSG00000111432.4 | FZD10 | higher in foxa2_abnormal_zero | 0.644 | 3.40e-06 | 0.0168 | True | 0.0459 | candidate_tumour_intrinsic | 1 |
| ENSG00000104435.14 | STMN2 | higher in foxa2_abnormal_zero | 0.873 | 3.60e-06 | 0.0168 | False | 0.229 | candidate_tumour_intrinsic | 1 |
| ENSG00000224041.3 | IGKV3D-15 | higher in foxa2_abnormal_zero | 0.858 | 4.13e-06 | 0.0168 | False | 0.126 | likely_immune_associated | 1 |
| ENSG00000239951.1 | IGKV3-20 | higher in foxa2_abnormal_zero | 0.535 | 7.74e-06 | 0.0262 | False | 0.479 | likely_immune_associated | 1 |
| ENSG00000241351.3 | IGKV3-11 | higher in foxa2_abnormal_zero | 0.626 | 1.32e-05 | 0.0363 | False | 0.432 | likely_immune_associated | 1 |
| ENSG00000233608.4 | TWIST2 | higher in foxa2_abnormal_zero | 0.639 | 1.43e-05 | 0.0363 | False | 0.332 | candidate_tumour_intrinsic | 1 |
| ENSG00000276775.1 | IGHV4-4 | higher in foxa2_abnormal_zero | 0.793 | 2.06e-05 | 0.038 | False | 0.437 | likely_immune_associated | 1 |
| ENSG00000163600.13 | ICOS | higher in foxa2_abnormal_zero | 0.561 | 1.93e-05 | 0.038 | False | 0.327 | likely_immune_associated | 1 |
| ENSG00000244575.3 | IGKV1-27 | higher in foxa2_abnormal_zero | 0.772 | 1.79e-05 | 0.038 | False | 0.491 | likely_immune_associated | 1 |
| ENSG00000144959.11 | NCEH1 | higher in foxa2_abnormal_zero | 0.122 | 2.57e-05 | 0.0435 | False | 0.19 | candidate_tumour_intrinsic | 1 |

Notes:
- `Sig after immune adj` comes from `step5_limma_vs_immune_adjusted_comparison.csv`.
- `DESeq2 FDR (limma pool)` comes from `step6_deseq2_limma_gene_pool_results.csv` (DESeq2 rerun on the same limma gene pool).
- In that rerun, `samples_used = 271`, `n_fdr_le_0.05 = 0`, and `min_padj = 0.999988`.

## 12) DESeq-only GSEA: standard-practice choice and results

### What was run
Preranked fgsea was run from DESeq2 result tables using:
- Script: `scripts/run_fgsea_from_de.R`
- Gene sets: MSigDB Hallmark (`H`)
- Multiple ranking metrics:
  - `stat` (DESeq2 Wald statistic; standard primary choice)
  - `sign(log2FC) * -log10(p)` (common published alternative)
  - `log2FC * -log10(p)` (the "FC × p-value signal" idea)

Grid output directory:
- `outputs/thesis/04_differential_expression/data/source_inputs/`
- Summary table: `.../step7_fgsea_deseq_grid/grid_summary.csv`

### Which approach is standard practice here
Primary analysis:
- `step1_continuous + stat`

Sensitivity checks:
- `step3_continuous_sva + stat`
- `deseq2_binary_main + stat`

Method-comparison/supplementary:
- `step6_limma_gene_pool + stat` (DESeq2 rerun restricted to limma gene universe)

### Results (FDR <= 0.05 Hallmark pathways)
- `step1_continuous + stat`: `29` significant pathways
- `deseq2_binary_main + stat`: `27` significant pathways
- `step6_limma_gene_pool + stat`: `25` significant pathways
- `step3_continuous_sva + stat`: `2` significant pathways

### About the FC × p-value ranking metric
Using `log2FC * -log10(p)` did work but was much weaker in this dataset:
- `step1_continuous`: `2` significant pathways
- `deseq2_binary_main`: `1` significant pathway
- `step6_limma_gene_pool`: `1` significant pathway
- `step3_continuous_sva`: `0` significant pathways

Interpretation:
- The professor-suggested approach is valid and seen in literature, but in this dataset the DESeq2 `stat` ranking is clearly more sensitive and is the better primary choice.

### Top recurrent pathways in DESeq-only `stat` runs
Consistently enriched pathways included:
- `HALLMARK_OXIDATIVE_PHOSPHORYLATION`
- `HALLMARK_MYC_TARGETS_V1`
- `HALLMARK_MYC_TARGETS_V2`
- `HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION`
- `HALLMARK_INFLAMMATORY_RESPONSE`
- `HALLMARK_TNFA_SIGNALING_VIA_NFKB`
