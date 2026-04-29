# Scripts Layout

Scripts are organised by thesis result themes.

- `01_pan_celltype_benchmark/`: benchmark and scoring pipeline entrypoints.
- `03_hepatocyte_clinical_associations/`: hepatocyte state and clinical association analyses.
- `04_differential_expression/`: DE and follow-up expression analyses.
- `05_null_bootstrap_validation/`: null/bootstrap validation workflows.
- `pathway_enrichment/`: unnumbered helper workflow used by expression and null analyses.
- `common/`: shared Python modules used across workflows.
- `99_data_build/`: metadata and mutation table build scripts.

The `grid_search/` directory remains a package for the core benchmark engine.

## Reproduce thesis results

Each result section provides one executable bash script named
`reproduce_result.sh`:

- `scripts/01_pan_celltype_benchmark/reproduce_result.sh`
- `scripts/03_hepatocyte_clinical_associations/reproduce_result.sh`
- `scripts/04_differential_expression/reproduce_result.sh`
- `scripts/05_null_bootstrap_validation/reproduce_result.sh`

The pathway enrichment helper is available separately at
`scripts/pathway_enrichment/reproduce_result.sh`.

Run from repo root, for example:

```bash
bash scripts/04_differential_expression/reproduce_result.sh
```

For null bootstrap, you may pass the replicate count:

```bash
bash scripts/05_null_bootstrap_validation/reproduce_result.sh 10
```
