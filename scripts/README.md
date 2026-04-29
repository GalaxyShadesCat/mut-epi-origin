# Scripts Layout

Scripts are organised by thesis result themes.

- `01_pan_celltype_benchmark/`: benchmark and scoring pipeline entrypoints.
- `02_foxa2_epigenome_orientation/`: FOXA2 epigenome orientation figure regeneration.
- `03_hepatocyte_clinical_associations/`: hepatocyte state and clinical association analyses.
- `04_differential_expression/`: DE, limma-voom, and pathway enrichment analyses.
- `05_null_bootstrap_validation/`: null/bootstrap validation workflows.
- `common/`: shared Python modules used across workflows.
- `99_data_build/`: metadata and mutation table build scripts.

The `grid_search/` directory remains a package for the core benchmark engine.

## Reproduce thesis results

Each result section provides one executable bash script named
`reproduce_results.sh`:

- `scripts/01_pan_celltype_benchmark/reproduce_results.sh`
- `scripts/02_foxa2_epigenome_orientation/reproduce_results.sh`
- `scripts/03_hepatocyte_clinical_associations/reproduce_results.sh`
- `scripts/04_differential_expression/reproduce_results.sh`
- `scripts/05_null_bootstrap_validation/reproduce_results.sh`

Pathway enrichment is reproduced as part of
`scripts/04_differential_expression/reproduce_results.sh`.

Run from repo root, for example:

```bash
bash scripts/04_differential_expression/reproduce_results.sh
```

For null bootstrap, you may pass the replicate count:

```bash
bash scripts/05_null_bootstrap_validation/reproduce_results.sh 10
```
