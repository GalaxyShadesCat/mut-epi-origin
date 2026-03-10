## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## LIHC data pipeline (metadata -> QC -> transfer -> SNV table)

Run the steps in this order to build the LIHC SNV mutation table.

### Step 1: Build master metadata

Script: `scripts/build_master_metadata.py`

```bash
python scripts/build_master_metadata.py
```

Output:
- `data/derived/master_sample_metadata.csv`

### Step 2: QC and cohort filtering

Script: `scripts/qc_lihc_cohort.py`

```bash
python scripts/qc_lihc_cohort.py
```

Outputs:
- `data/derived/master_sample_metadata_cleaned.csv`
- `data/derived/master_sample_metadata_lihc_fibrosis.csv`
- `data/derived/master_sample_metadata_rows_with_issues.csv`
- `data/derived/master_sample_metadata_qc_report.txt`

### Step 3: Transfer complete-case VCFs and build VCF candidate manifest

Script: `scripts/transfer_lihc_vcfs.sh`

Dry run first:

```bash
bash scripts/transfer_lihc_vcfs.sh --test
```

Then run the actual transfer:

```bash
bash scripts/transfer_lihc_vcfs.sh
```

Key outputs:
- `data/derived/manifests/lihc_tumour_vcf_candidates.tsv`
- `data/raw/WGS_TCGA25/AtoL/VCF/` (mirrored VCF files and index sidecars, including `.vcf.gz.tbi`)

### Step 4: Build the SNV mutation table

Script: `scripts/build_snv_mutation_table.py`

```bash
python scripts/build_snv_mutation_table.py
```

Output:
- `data/raw/mutations/lihc_snv_mutation_table.tsv`

### Dependency chain

`scripts/build_master_metadata.py` -> `data/derived/master_sample_metadata.csv`  
`scripts/qc_lihc_cohort.py` -> `data/derived/master_sample_metadata_lihc_fibrosis.csv`  
`scripts/transfer_lihc_vcfs.sh` -> `data/derived/manifests/lihc_tumour_vcf_candidates.tsv` + `data/raw/WGS_TCGA25/AtoL/VCF/`  
`scripts/build_snv_mutation_table.py` -> `data/raw/mutations/lihc_snv_mutation_table.tsv`

### Notes on cohort logic

- Project focus is `TCGA-LIHC`.
- Fibrosis source of truth is the clinical Ishak field from `clinical.tsv` (case-level aggregated).
- HBV/HCV harmonisation uses `data/raw/annotations/mmc1.xlsx` consensus calls first, then fallback fields.
- Obesity class is derived from BMI using WHO categories.
- Complete-case rows require non-missing:
  - `alcohol_status`
  - `hbv_status`
  - `hcv_status`
  - `nafld_status`
  - `obesity_class`
  - `fibrosis_ishak_score`

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
