## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Streamlit apps

Track Visualisation:

```bash
streamlit run tools/track_visualisation.py
```

Results Dashboard:

```bash
streamlit run tools/results_dashboard/run.py
```

## Build master metadata

Build the LIHC-focused master metadata table with harmonised alcohol, virus, obesity, and fibrosis fields:

```bash
python scripts/build_master_metadata.py
```

Output:
- `data/derived/master_sample_metadata.csv`

## QC LIHC cohort

Run QC/cleaning on `master_sample_metadata.csv`, including fibrosis filtering and missingness logs for alcohol/virus/obesity:

```bash
python scripts/qc_lihc_cohort.py
```

Outputs:
- `data/derived/master_sample_metadata_cleaned.csv`
- `data/derived/master_sample_metadata_lihc_fibrosis.csv`
- `data/derived/master_sample_metadata_rows_with_issues.csv`
- `data/derived/master_sample_metadata_qc_report.txt`

## Important considerations

- **Project focus**: downstream cohort outputs are restricted to `TCGA-LIHC`.
- **Fibrosis source of truth**: fibrosis uses the clinical Ishak field from `clinical.tsv` (case-level aggregated), then standardised to:
  - `0 - No Fibrosis`
  - `1,2 - Portal Fibrosis`
  - `3,4 - Fibrous Septa`
  - `5 - Nodular Formation and Incomplete Cirrhosis`
  - `6 - Established Cirrhosis`
- **Virus harmonisation**:
  - HBV/HCV use `data/raw/annotations/mmc1.xlsx` consensus calls first (`HBV_consensus`, `HCV_consensus`) when available.
  - If consensus is missing, fallback uses `Hepatitis B/C`, then annotation evidence from clinical matrix fields.
- **Obesity standardisation**:
  - `bmi_final` prioritises BMI calculated from height/weight.
  - If unavailable, `data/raw/annotations/mmc1.xlsx` BMI is used as fallback.
  - Final obesity class is derived from BMI using WHO classes:
    - `Underweight`, `Normal`, `Overweight`, `Obesity Class I`, `Obesity Class II`, `Obesity Class III`.
- **QC fibrosis cohort filter** (`master_sample_metadata_lihc_fibrosis.csv`):
  - LIHC only
  - non-missing fibrosis
  - `primary_diagnosis` in:
    - `Hepatocellular carcinoma, NOS`
    - `Hepatocellular carcinoma, clear cell type`
  - `tumour_sample_type == Primary Tumor`
- **Complete-case definition**: a row is treated as complete when all of these are non-missing:
  - `alcohol_status`, `hbv_status`, `hcv_status`, `nafld_status`, `obesity_class`, `fibrosis_ishak_score`.
