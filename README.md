## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## LIHC data pipeline (ATAC pseudobulk -> metadata -> QC -> transfer -> SNV table -> grid search -> state-score validation)

Run the steps in this order to build the LIHC SNV mutation table, run mutation-vs-accessibility grid search, and validate inferred state scores.

### Step 1: Build ATAC pseudobulk bigWig tracks

Script: `scripts/make_atac_pseudobulk.R`

```bash
Rscript scripts/make_atac_pseudobulk.R
```

Important:
- Requires external command-line tools in `PATH`: `bedGraphToBigWig`, `liftOver`, `bedtools`, `sort`, `gzip`, `gunzip`.
- Uses the multiome Seurat object at `data/raw/multiome/GSE281574_Liver_Multiome_Seurat_GEO.rds`.
- Produces hg38 and hg19 bigWig tracks plus QC summaries under `data/processed/ATAC-seq/GSE281574/`.

### Step 2: Build master metadata

Script: `scripts/build_master_metadata.py`

```bash
python scripts/build_master_metadata.py
```

Output:
- `data/derived/master_sample_metadata.csv`

### Step 3: QC and cohort filtering

Script: `scripts/qc_lihc_cohort.py`

```bash
python scripts/qc_lihc_cohort.py
```

Outputs:
- `data/derived/master_sample_metadata_cleaned.csv`
- `data/derived/master_sample_metadata_lihc_all.csv`
- `data/derived/master_sample_metadata_lihc_fibrosis.csv`
- `data/derived/master_sample_metadata_lihc_nafld.csv`
- `data/derived/master_sample_metadata_lihc_hcv.csv`
- `data/derived/master_sample_metadata_rows_with_issues.csv`
- `data/derived/master_sample_metadata_qc_report.txt`

### Step 4: Transfer VCFs and build VCF candidate manifest

Script: `scripts/transfer_lihc_vcfs.sh`

Dry run first:

```bash
bash scripts/transfer_lihc_vcfs.sh --test
```

Then run the actual transfer:

```bash
bash scripts/transfer_lihc_vcfs.sh
```

By default, this step does not enforce additional complete-case filtering beyond
the metadata file you provide.

NAFLD-oriented transfer (keeps fibrosis optional):

```bash
bash scripts/transfer_lihc_vcfs.sh \
  --metadata-csv data/derived/master_sample_metadata_lihc_nafld.csv \
  --cohort-label nafld
```

If you want strict complete-case filtering, set it explicitly:

```bash
bash scripts/transfer_lihc_vcfs.sh \
  --metadata-csv data/derived/master_sample_metadata_lihc_nafld.csv \
  --cohort-label nafld \
  --required-complete-fields alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class
```

Key outputs:
- `data/derived/manifests/lihc_tumour_vcf_candidates.tsv` (default fibrosis cohort)
- `data/derived/manifests/lihc_tumour_vcf_candidates_<cohort_label>.tsv` (custom cohort label, e.g. `nafld`)
- `data/raw/WGS_TCGA25/AtoL/VCF/` (mirrored VCF files and index sidecars, including `.vcf.gz.tbi`)

### Step 5: Build the SNV mutation table

Script: `scripts/build_snv_mutation_table.py`

```bash
python scripts/build_snv_mutation_table.py
```

Output:
- `data/raw/mutations/lihc_snv_mutation_table.tsv`

### Step 6: Run mutation-vs-accessibility grid search

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

### Step 7: Validate inferred state scores against metadata

Script: `scripts/validate_state_scores.py`

For the 3-state setup:

```bash
python scripts/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --allow-aggregated-results
```

For a NAFLD-focused validation run (fibrosis columns optional):

```bash
python scripts/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --metadata-path data/derived/master_sample_metadata_lihc_nafld.csv \
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
python scripts/validate_state_scores.py \
  --experiment-name lihc_foxa2_top4 \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --allow-aggregated-results
```

### Dependency flow (quick view)

Grid search needs two inputs:
- Accessibility tracks (ATAC, hg19)
- Mutation table (LIHC SNVs)

Flow:
1. `scripts/make_atac_pseudobulk.R`
: builds `data/processed/ATAC-seq/GSE281574/hg19/*.bigWig`
2. `scripts/build_master_metadata.py` -> `scripts/qc_lihc_cohort.py` -> `scripts/transfer_lihc_vcfs.sh` -> `scripts/build_snv_mutation_table.py`
: builds `data/raw/mutations/lihc_snv_mutation_table.tsv`
3. `python -m scripts.grid_search.cli`
: consumes both inputs above and writes `outputs/experiments/<run_name>/results.csv` plus `outputs/experiments/<run_name>/runs/`
4. `python scripts/validate_state_scores.py ...`
: consumes `outputs/experiments/<run_name>/results.csv` and a chosen metadata cohort file (default `data/derived/master_sample_metadata_lihc_fibrosis.csv`)

### Notes on cohort logic

- Project focus is `TCGA-LIHC`.
- Fibrosis source of truth is the clinical Ishak field from `clinical.tsv` (case-level aggregated).
- HBV/HCV harmonisation uses `data/raw/annotations/mmc1.xlsx` consensus calls first, then fallback fields.
- Obesity class is derived from BMI using WHO categories.
- Default fibrosis complete-case rows require non-missing:
  - `alcohol_status`
  - `hbv_status`
  - `hcv_status`
  - `nafld_status`
  - `obesity_class`
  - `fibrosis_ishak_score`
- All-phenotype rows (`master_sample_metadata_lihc_all.csv`) require non-missing:
  - `alcohol_status`
  - `hbv_status`
  - `hcv_status`
  - `nafld_status`
  - `obesity_class`
  - `fibrosis_ishak_score`
- NAFLD-focused rows require non-missing:
  - `alcohol_status`
  - `hbv_status`
  - `hcv_status`
  - `nafld_status`
  - `obesity_class`
- HCV-focused rows require non-missing:
  - `alcohol_status`
  - `hbv_status`
  - `hcv_status`
  - `nafld_status`
  - `obesity_class`

## Grid search

For a full, practical guide to the mutation-vs-accessibility grid search runner (inputs, outputs, configuration modes, explicit setups, and resume workflow), see:

- [`scripts/grid_search/README.md`](scripts/grid_search/README.md)

## Streamlit apps

Track visualisation:

```bash
streamlit run tools/track_visualisation.py
```

Results dashboard:

```bash
streamlit run tools/results_dashboard/run.py
```

State validation dashboard:

```bash
streamlit run tools/state_validation_dashboard.py
```
