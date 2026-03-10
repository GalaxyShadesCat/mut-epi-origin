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

Current CLI defaults are tuned for a small run:

- `--k-samples 1`
- `--n-resamples 1`
- `--downsample none`

So, without overriding these, run count is usually:

`number_of_setups × number_of_downsample_targets`

In grid mode, run count becomes:

`len(k_samples) × n_resamples × number_of_grid_combinations × number_of_downsample_targets`

## `k_samples`, `n_resamples`, and per-sample mode

- `--k-samples`: cohort size per run (`1`, `5`, `10`, `20`, `all`, etc.).
- `--n-resamples`: number of repeated sample draws per `k`.
- `--per-sample-count N`: switches to per-sample mode and runs the first `N` individual samples (`k=1` per run), instead of iterating a `k_samples` grid.

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
  --out-dir outputs/experiments/lihc_top5 \
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

## Common gotchas

- Keep all CLI flags in the same shell command; typing a flag after the command has started does not apply it.
- `--dnase-map-json`, `--dnase-map-path`, and `--atac-map-path` are mutually constrained; use one route per run.
- In ranking outputs, multiple metric leaders may come from the same underlying run configuration.
