# Annotations Provenance (LIHC)

This directory contains external annotation resources used to enrich LIHC metadata in `scripts/99_data_build/build_master_metadata.py`.

## Source publications and portals

- Cell paper (LIHC cohort context and supplementary resources):  
  https://www.cell.com/cms/10.1016/j.cell.2017.05.046/
- UCSC Xena clinical matrix dataset page:  
  https://xenabrowser.net/datapages/?dataset=TCGA.LIHC.sampleMap%2FLIHC_clinicalMatrix&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443

## Files used

- `mmc1.xlsx`  
  Supplementary annotation table used for curated risk and clinical fields, including:
  - `HBV_consensus`, `HCV_consensus`
  - `Hepatitis B`, `Hepatitis C`
  - `Alcoholic liver disease`
  - `NAFLD`
  - `BMI`, `ObesityClass1`, `ObesityClass2`
  - `liver_fibrosis_ishak_score_category`

- `TCGA.LIHC.sampleMap_LIHC_clinicalMatrix.tsv`  
  UCSC Xena LIHC clinical matrix used for complementary annotation fields, including:
  - `hist_hepato_carc_fact`
  - `viral_hepatitis_serology`
  - `height`, `weight`
  - `fibrosis_ishak_score`

## Integration notes

- In the metadata build workflow, these files are joined to LIHC tumour samples by UUID/barcode mappings.
- Virus status prioritises consensus calls from `mmc1.xlsx` where available.
- Obesity is standardised from BMI using WHO classes; BMI is calculated from height/weight first, then falls back to curated BMI.
- Fibrosis source of truth is the clinical Ishak field from the TCGA clinical join path; annotation fibrosis fields are retained for consistency checks.
