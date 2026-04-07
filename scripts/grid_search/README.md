# Grid Search: Mutation Tracks vs Accessibility Tracks

This module runs repeatable experiments that compare mutation-derived genomic tracks with accessibility tracks (DNase-seq or ATAC-seq) across cell types.

The entry point is:

```bash
python -m scripts.grid_search.cli ...
```

## What it does

For each run configuration, the pipeline:

1. Selects mutation rows (from one or more mutation tables).
2. Builds a mutation track per chromosome (for example `counts_raw` or `exp_decay`).
3. Builds optional covariates (`gc`, `cpg`, `timing`, and optional trinucleotide features).
4. Compares mutation tracks against each cell-type accessibility track.
5. Computes multiple metrics (Pearson, Spearman, local score variants, RF-residualised Pearson).
6. Predicts the best-matching cell type and writes summary outputs.

The same run emits several metrics, so one configuration can appear in multiple ranking tables.

## Main input files

- Mutation table: tab-delimited table in the standard internal format (`chrom`, `start`, `end`, `sample`, `ref`, `alt`, `cohort`), or other supported BED-like variants handled by sample selection utilities.
- Genome index: `--fai-path` (for example `data/raw/reference/GRCh37.fa.fai`).
- Genome FASTA: `--fasta-path` (for example `data/raw/reference/GRCh37.fa`).
- Accessibility tracks:
  - `--dnase-map-path` (default map file), or
  - `--atac-map-path`, or
  - `--dnase-map-json` inline mapping.
- Optional timing bigWig: `--timing-bw`.

## Outputs

By default the run directory is under `outputs/experiments/...`.

Key files:

- `results.csv`: one row per completed run configuration.
- `grid_search_params.json`: saved parameters for exact reproducibility.
- `command_init.txt`: reconstructed command used to start the run.
- `command_resume.txt`: resume command.
- `runs/<run_id>/config.json`: per-run resolved configuration.
- `runs/<run_id>/chrom_summary.csv`: per-chromosome summary.
- `runs/<run_id>/per_bin.csv`: optional per-bin table (if `--save-per-bin`).

## Configuration styles

There are two ways to define what gets run.

### 1) Cartesian grid mode (legacy/default behaviour)

You provide strategy lists and parameter grids, and the runner executes all combinations.

Examples of grid controls:

- `--track-strategies`
- `--counts-raw-bins`, `--exp-decay-bins`, etc.
- `--covariate-sets`

### 2) Explicit setup mode (recommended for targeted runs)

Use `--explicit-setups-json` to run only named setups, with no Cartesian expansion.

Each setup object supports:

- `track_strategy` (required)
- `bin_size` (required)
- `covariates` (optional; defaults to `--covariate-sets`)
- Track-specific parameters when needed (for example `exp_decay_bp`, `exp_max_distance_bp`)

## Defaults that matter

Current CLI defaults favour full per-sample coverage:

- `--k-samples 1`
- `--n-resamples 1`
- `--downsample none`
- `--per-sample-count 0` (run all samples one-at-a-time)

So, without overriding these, run count is usually:

`number_of_samples × number_of_setups × number_of_downsample_targets`

In grid mode, run count becomes:

`len(k_samples) × n_resamples × number_of_grid_combinations × number_of_downsample_targets`

## `k_samples`, `n_resamples`, and per-sample mode

- `--k-samples`: cohort size per run (`1`, `5`, `10`, `20`, `all`, etc.).
- `--n-resamples`: number of repeated sample draws per `k`.
- `--per-sample-count N`: runs the first `N` individual samples (`k=1` per run).
- `--per-sample-count 0`: runs all samples individually (default).
- `--per-sample-count none` (or `off`): disables per-sample mode and uses the `k_samples`/`n_resamples` grid.

If you are testing configuration impact only, keep `k=1` and `n_resamples=1`.

## Reproducibility and randomness

Runs are deterministic for a fixed command and seed values:

- sample planning uses `--base-seed`,
- downsampling uses deterministic seeds derived from base seed and target,
- RF components use deterministic `random_state` derived from base seed and repeat index.

Changing `base_seed`, `k_samples`, `n_resamples`, or downsampling settings will change sampled data and results.

## Minimal examples

### A) Targeted explicit setups

```bash
python -m scripts.grid_search.cli \
  --mut-path data/raw/mutations/lihc_snv_mutation_table.tsv \
  --fai-path data/raw/reference/GRCh37.fa.fai \
  --fasta-path data/raw/reference/GRCh37.fa \
  --dnase-map-json '{"hepatocyte_ac":"data/processed/ATAC-seq/GSE281574/hg19/Hepatocyte__AC.bigWig","hepatocyte_ah":"data/processed/ATAC-seq/GSE281574/hg19/Hepatocyte__AH.bigWig","hepatocyte_normal":"data/processed/ATAC-seq/GSE281574/hg19/Hepatocyte__Normal.bigWig"}' \
  --timing-bw data/raw/timing/repliSeq_SknshWaveSignalRep1.bigWig \
  --covariate-sets "gc+cpg+timing" \
  --out-dir outputs/experiments/lihc_top4 \
  --explicit-setups-json '[{"track_strategy":"counts_raw","bin_size":500000,"covariates":["gc","cpg","timing"]},{"track_strategy":"counts_raw","bin_size":1000000,"covariates":["gc","cpg","timing"]},{"track_strategy":"exp_decay","bin_size":500000,"exp_decay_bp":200000,"exp_max_distance_bp":1000000,"covariates":["gc","cpg","timing"]},{"track_strategy":"exp_decay","bin_size":1000000,"exp_decay_bp":200000,"exp_max_distance_bp":1000000,"covariates":["gc","cpg","timing"]}]'
```

### B) Quick per-sample sanity run (2 samples)

Add:

```bash
--per-sample-count 2
```

to the same command.

## Resume an interrupted run

```bash
python -m scripts.grid_search.cli resume-experiment outputs/experiments/<run_dir>
```

## Post-run sample-level modelling

After a grid-search run, you can build a modelling matrix and run model selection
directly from an experiment directory:

```bash
python scripts/score_clinvar_grid.py \
  --experiment-name <run_dir_name>
```

Important defaults in `score_clinvar_grid.py`:

- target: `fibrosis_ishak_score` (regression)
- scoring systems: `rf_resid,pearson_local_score,spearman_local_score,pearson_r_linear_resid,spearman_r_linear_resid`
- maximum CV folds: `5` (uses `min(cv_folds, n_samples)` per group)
- score inversion: enabled by default (because mutation-accessibility correlation is expected to be negative)
- metadata file: `data/derived/master_metadata.csv`
- metadata sample column: `tumour_sample_submitter_id`

Useful optional flags:

- `--target fibrosis_present` for binary classification
- `--no-invert-scores` to disable score inversion
- `--scoring-systems <comma-list>` to restrict modelled score systems
- `--cv-folds <N>` to reduce CV cost
- `--allow-aggregated-results` only if your `results.csv` rows represent multiple samples

For a NAFLD-oriented run (without requiring fibrosis columns in metadata), pass:

- `--metadata-path data/derived/master_metadata.csv`
- `--modelling-targets nafld_status`
- `--group-test-vars nafld_status,obesity_class`
- `--correlation-vars nafld_status`
- `--covariate-cols alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class`

This writes the following files into `outputs/experiments/<run_dir_name>/`:

- `model_matrix.csv`
- `model_scores.csv`
- `grid_search_summary.csv`
- `feature_importance.csv`

`grid_search_summary.csv` includes:

- `best_score`: best cross-validation score for the model/grid.
- `score_gap`: `best_score - second_best_score` (larger means clearer separation).
- `is_best_model_for_group`: selected model per `(config_id, scoring_system, feature_set, target)`
  using best score first, then score gap as tie-break.

The modelling outputs now include `feature_set`:

- `full`: chromatin scores + covariates
- `covariates_only`: baseline comparator without chromatin scores

This enables direct assessment of whether chromatin adds predictive value over
clinical covariates alone.

## Post-run state-score validation

Use `validate_state_scores.py` to validate inferred state scores against clinical metadata.

3-state setup:

```bash
python scripts/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --allow-aggregated-results
```

2-state FOXA2 setup:

```bash
python scripts/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --state-labels normal_FOXA2_pos,abnormal_FOXA2_zero \
  --state-suffixes normal_FOXA2_pos,abnormal_FOXA2_zero \
  --allow-aggregated-results
```

NAFLD-focused setup (fibrosis optional):

```bash
python scripts/validate_state_scores.py \
  --experiment-name YOUR_EXPERIMENT \
  --metadata-path data/derived/master_metadata.csv \
  --state-labels hepatocyte_normal,hepatocyte_ac,hepatocyte_ah \
  --state-suffixes normal,ac,ah \
  --modelling-targets nafld_status \
  --group-test-vars nafld_status,obesity_class \
  --correlation-vars nafld_status \
  --covariate-cols alcohol_status,hbv_status,hcv_status,nafld_status,obesity_class \
  --allow-aggregated-results
```

Model hyperparameter grids used by `score_clinvar_grid.py`:

- Random Forest:
  - `n_estimators`: `[300, 800]`
  - `max_depth`: `[3, 5, None]`
  - `min_samples_split`: `[5, 10]`
  - `min_samples_leaf`: `[2, 4]`
  - `max_features`: `["sqrt"]`
  - `bootstrap`: `[True]`
- Ridge / RidgeClassifier:
  - `alpha`: `[0.1, 1.0, 10.0, 100.0]`

## Common gotchas

- Keep all CLI flags in the same shell command; typing a flag after the command has started does not apply it.
- `--dnase-map-json`, `--dnase-map-path`, and `--atac-map-path` are mutually constrained; use one route per run.
- In ranking outputs, multiple metric leaders may come from the same underlying run configuration.
- `score_clinvar_grid.py` expects sample-level runs by default (`n_selected_samples == 1`).
  Aggregated runs are blocked unless `--allow-aggregated-results` is explicitly set.
- Very small sample groups can yield unstable or undefined regression CV scores (for example `R²` with tiny folds).
