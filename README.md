## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Shared machine-specific data paths

Set machine-specific roots in:

- `config/data_paths.json`

Required key:

- `wgs_tcga25_root`

Example:

```json
{
  "wgs_tcga25_root": "data/raw/WGS_TCGA25"
}
```

Use an absolute path if your local `WGS_TCGA25` lives outside this repository.

## Active mutation inputs

The current pipeline keeps and uses only:

- `data/raw/mutations/filtered_mutations.bed`
- `data/raw/mutations/ICGC_WGS_Feb20_mutations.LIHC_LIRI.bed`
- `data/raw/mutations/lihc_snv_mutation_table.tsv`

## LIHC data pipeline (ATAC pseudobulk -> metadata -> transfer -> SNV table -> grid search -> state-score validation)

Run the steps in this order to build the LIHC SNV mutation table, run mutation-vs-accessibility grid search, and validate inferred state scores.

### Step 1: Build ATAC pseudobulk bigWig tracks

Script: `scripts/99_data_build/make_atac_pseudobulk.R`

```bash
Rscript scripts/99_data_build/make_atac_pseudobulk.R
```

Important:
- Requires external command-line tools in `PATH`: `bedGraphToBigWig`, `liftOver`, `bedtools`, `sort`, `gzip`, `gunzip`.
- Uses the multiome Seurat object at `data/raw/multiome/GSE281574_Liver_Multiome_Seurat_GEO.rds`.
- Produces hg38 and hg19 bigWig tracks plus QC summaries under `data/processed/ATAC-seq/GSE281574/`.

### Step 2: Build master metadata

Script: `scripts/99_data_build/build_master_metadata.py`

```bash
python scripts/99_data_build/build_master_metadata.py
```

Output:
- `data/derived/master_metadata.csv`

### Step 3: Transfer VCFs and build VCF candidate manifest

Script: `scripts/99_data_build/transfer_lihc_vcfs.sh`

Dry run first:

```bash
bash scripts/99_data_build/transfer_lihc_vcfs.sh --test
```

Then run the actual transfer:

```bash
bash scripts/99_data_build/transfer_lihc_vcfs.sh
```

By default, this step does not enforce additional complete-case filtering beyond
the metadata file you provide.

NAFLD-oriented transfer (keeps fibrosis optional):

```bash
bash scripts/99_data_build/transfer_lihc_vcfs.sh \
  --metadata-csv data/derived/master_metadata.csv \
  --cohort-label nafld
```

If you want strict complete-case filtering, set it explicitly:

```bash
bash scripts/99_data_build/transfer_lihc_vcfs.sh \
  --metadata-csv data/derived/master_metadata.csv \
  --cohort-label nafld \
  --required-complete-fields alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class
```

Key outputs:
- `data/derived/manifests/lihc_tumour_vcf_candidates.tsv` (default fibrosis cohort)
- `data/derived/manifests/lihc_tumour_vcf_candidates_<cohort_label>.tsv` (custom cohort label, e.g. `nafld`)
- `data/raw/WGS_TCGA25/AtoL/VCF/` (mirrored VCF files and index sidecars, including `.vcf.gz.tbi`)

### Step 4: Build the SNV mutation table

Script: `scripts/99_data_build/build_snv_mutation_table.py`

```bash
python scripts/99_data_build/build_snv_mutation_table.py
```

Output:
- `data/raw/mutations/lihc_snv_mutation_table.tsv`

### Step 5: Run mutation-vs-accessibility grid search

Entry point:
- `python -m scripts.grid_search.cli ...`

Minimum inputs:
- `--mut-path data/raw/mutations/lihc_snv_mutation_table.tsv`
- ATAC map via `--atac-map-path` or inline mapping via `--dnase-map-json`
- `--fai-path data/raw/reference/GRCh37.fa.fai`
- `--fasta-path data/raw/reference/GRCh37.fa`

Output directory:
- `outputs/experiments/<run_name>/`
- Includes `results.csv`, run configs, and resume commands.

### Step 6: Validate inferred state scores against metadata

Script: `scripts/01_pan_celltype_benchmark/validate_state_scores.py`

For the 3-state setup:

```bash
python scripts/01_pan_celltype_benchmark/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --allow-aggregated-results
```

For a NAFLD-focused validation run (fibrosis columns optional):

```bash
python scripts/01_pan_celltype_benchmark/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --metadata-path data/derived/master_metadata.csv \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --modelling-targets nafld_status \
  --correlation-vars nafld_status \
  --group-test-vars nafld_status,obesity_class \
  --covariate-cols alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class \
  --allow-aggregated-results
```

For the 2-state FOXA2 setup:

```bash
python scripts/01_pan_celltype_benchmark/validate_state_scores.py \
  --experiment-name lihc_foxa2_top4 \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --allow-aggregated-results
```

### Step 7: Run gene-level differential mutation analysis from inferred labels

Script: `scripts/03_clinical_associations/run_differential_mutation_by_inferred_labels.py`

This step tests gene-level mutation frequency differences between two inferred
label groups using Fisher's exact test. It is differential mutation analysis,
not RNA-seq differential expression.

Example command:

```bash
python scripts/03_clinical_associations/run_differential_mutation_by_inferred_labels.py \
  --mutation-path data/raw/mutations/lihc_snv_mutation_table.tsv \
  --delim tab \
  --output-dir outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/dm_counts_raw_500k_spearman_r_linear_resid_from_lihc_snv_table
```

Minimum input requirements:
- A mutation table with sample and gene columns (for example `Tumor_Sample_Barcode` and `Hugo_Symbol`).
- Inferred labels file at `outputs/experiments/<run_name>/validation_score_rankings.csv` (default path is preconfigured in the script).
- Exactly two label groups remaining after script filters.

Outputs:
- `sample_labels_used.csv`
- `differential_mutation_results_all.csv`
- `differential_mutation_results_significant.csv`
- `run_summary.txt`

Significance reporting:
- `differential_mutation_results_all.csv` includes `fdr_bh` and `is_significant_fdr`.
- `differential_mutation_results_significant.csv` contains rows with `fdr_bh <= 0.05` by default.

### Dependency flow (quick view)

Grid search needs two inputs:
- Accessibility tracks (ATAC, hg19)
- Mutation table (LIHC SNVs)

Flow:
1. `scripts/99_data_build/make_atac_pseudobulk.R`
: builds `data/processed/ATAC-seq/GSE281574/hg19/*.bigWig`
2. `scripts/99_data_build/build_master_metadata.py` -> `scripts/99_data_build/transfer_lihc_vcfs.sh` -> `scripts/99_data_build/build_snv_mutation_table.py`
: builds `data/raw/mutations/lihc_snv_mutation_table.tsv`
3. `python -m scripts.grid_search.cli`
: consumes both inputs above and writes `outputs/experiments/<run_name>/results.csv` plus `outputs/experiments/<run_name>/runs/`
4. `python scripts/01_pan_celltype_benchmark/validate_state_scores.py ...`
: consumes `outputs/experiments/<run_name>/results.csv` and metadata file `data/derived/master_metadata.csv`

### Notes on cohort logic

- Project focus is `TCGA-LIHC`.
- Metadata source of truth is `data/derived/master_metadata.csv`.
- Fibrosis source of truth is the clinical Ishak field from `clinical.tsv` (case-level aggregated).
- HBV/HCV harmonisation uses `data/raw/annotations/mmc1.xlsx` consensus calls first, then fallback fields.
- Obesity class is derived from BMI using WHO categories.

## Grid search

For a full, practical guide to the mutation-vs-accessibility grid search runner (inputs, outputs, configuration modes, explicit setups, and resume workflow), see:

- [`scripts/grid_search/README.md`](scripts/grid_search/README.md)

## Streamlit apps

Track visualisation:

```bash
streamlit run tools/track_visualisation_dashboard.py
```

Results dashboard:

```bash
streamlit run tools/results_dashboard/run.py
```

State validation dashboard:

```bash
streamlit run tools/state_validation_dashboard.py
```
