## Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate mut-epi-origin
pip install -r requirements.txt
```

If `conda activate` fails, run `conda init` once and restart your shell.

## Reproduce thesis results

The thesis-facing results are organised under `outputs/thesis/` and mirror the
Results section figure order. To regenerate the active thesis outputs, run the
section scripts from the repository root after activating the environment:

```bash
conda activate mut-epi-origin

bash scripts/01_pan_celltype_benchmark/reproduce_results.sh
bash scripts/02_foxa2_epigenome_orientation/reproduce_results.sh
bash scripts/03_hepatocyte_clinical_associations/reproduce_results.sh
bash scripts/04_differential_expression/reproduce_results.sh
bash scripts/05_null_bootstrap_validation/reproduce_results.sh
```

The null bootstrap script accepts optional arguments:

```bash
bash scripts/05_null_bootstrap_validation/reproduce_results.sh 10
```

where the argument is the number of bootstrap replicates.

`outputs/thesis/05_null_bootstrap_validation/reproduce_results.sh` is a
lightweight wrapper around the root-level null-bootstrap reproducer. Executable
analysis code lives under `scripts/`; thesis folders contain notebooks, curated
data, figures, and wrappers only.

For a compact list of script entrypoints, see
[`scripts/README.md`](scripts/README.md). For the curated thesis output index,
see [`outputs/thesis/index.md`](outputs/thesis/index.md).

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

## Thesis workflow details

The five thesis sections are reproduced by the `scripts/<section>/reproduce_results.sh` entrypoints listed above. The scripts assume that the shared raw and processed inputs are already present. If you need to rebuild those inputs, use the data-build steps below first.

### Build shared inputs

1. Build FOXA2 hepatocyte ATAC pseudo-bulk tracks:

```bash
Rscript scripts/99_data_build/make_atac_pseudobulk.R
```

This requires `bedGraphToBigWig`, `liftOver`, `bedtools`, `sort`, `gzip`, and `gunzip` in `PATH`, and the multiome object at `data/raw/multiome/GSE281574_Liver_Multiome_Seurat_GEO.rds`.

2. Build the LIHC metadata table:

```bash
python scripts/99_data_build/build_master_metadata.py
```

This writes `data/derived/master_metadata.csv`.

3. Transfer LIHC VCFs:

```bash
bash scripts/99_data_build/transfer_lihc_vcfs.sh --test
bash scripts/99_data_build/transfer_lihc_vcfs.sh
```

The transfer step uses `config/data_paths.json` for the local WGS root and writes the LIHC VCF manifest under `data/derived/manifests/`.

4. Build the LIHC SNV table:

```bash
python scripts/99_data_build/build_snv_mutation_table.py
```

This writes `data/raw/mutations/lihc_snv_mutation_table.tsv`.

### Thesis section coverage

- Figure 1: pan-cell-type benchmark.
- Figure 2: TCGA-LIHC cohort orientation and FOXA2 hepatocyte references.
- Figure 3: FOXA2 clinical association analyses.
- Figure 4: differential expression and pathway analysis.
- Figure 5: mutation-randomised null benchmark.

### Core analysis components

The thesis scripts call these core tools:

- `python -m scripts.grid_search.cli` for mutation-to-chromatin scoring.
- `scripts/01_pan_celltype_benchmark/validate_state_scores.py` for label and score validation.
- `scripts/04_differential_expression/run_differential_expression_by_inferred_labels.R` for DESeq2 RNA-seq differential expression.
- `scripts/04_differential_expression/run_limma_by_inferred_labels.R` for limma-voom RNA-seq differential expression.
- `scripts/04_differential_expression/run_fgsea_from_de.R` for pathway enrichment from ranked DE results.
- `scripts/05_null_bootstrap_validation/bootstrap_shuffle_null.py` for the mutation-randomised null benchmark.

For the grid-search runner details, see [`scripts/grid_search/README.md`](scripts/grid_search/README.md). For the full script layout, see [`scripts/README.md`](scripts/README.md).

### Notes on cohort logic

- Project focus is `TCGA-LIHC` for the FOXA2 downstream analyses.
- Metadata source of truth is `data/derived/master_metadata.csv`.
- Fibrosis source of truth is the clinical Ishak field from `clinical.tsv` after case-level aggregation.
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
