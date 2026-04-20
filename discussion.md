# LIHC FOXA2 Validation Discussion

## Scope

This document summarises the full discussion and analysis around adjusted state-score validation for:

- `lihc_foxa2_top4`
- `lihc_foxa2_top4_nafld`

The main goals were:

1. Adjust associations for mutation burden (and clinical confounders).
2. Decide whether confidence filtering by score gap is useful and defensible.
3. Identify the strongest track strategy / bin size / metric setting.
4. Check whether findings are statistically significant and directionally interpretable.
5. Understand why some significant rows have small strata.

---

## What Was Implemented

A separate script was used so the original validation script stayed unchanged:

- `scripts/03_clinical_associations/validate_state_scores_adjusted.py`

It produces adjusted outputs (new files) without replacing the existing unadjusted pipeline.

### Statistical methods used

For score-level outcomes:

- Multiple linear regression with a likelihood-ratio test (full model vs reduced model without the exposure term).

For label-level outcomes:

- Logistic regression (one-vs-rest) with a likelihood-ratio test (full model vs reduced model without the exposure term).

Multiple testing correction:

- Benjamini-Hochberg FDR.

---

## What "adjusted" means here

Each association test includes:

- Mutation burden: `log1p(n_mutations_total)`
- Clinical covariates (primary model): `alcohol_status`, `hbv_status`, `hcv_status`, `nafld_status`, `fibrosis_ishak_score`

Important detail:

- When testing one exposure (for example `hbv_status`), that same exposure is removed from the adjustment set for that specific model.

---

## Missing-data handling

The adjusted script uses complete-case handling per test:

- It builds the required columns for that exact model.
- It drops rows with any missing values in those required columns.

So sample size (`n_total`) can change:

- across exposures,
- across covariate sets,
- across confidence filters.

No imputation is performed in the current adjusted script.

---

## Confidence threshold meaning

`score_gap` is:

- best state score minus second-best state score.

The confidence filter `score_gap_ge_X` keeps only rows with clearer separation between top two labels.

In the adjusted script:

- confidence threshold is used for label-level association rows (`confidence_filter` column),
- score-level adjusted tests are not reduced by this threshold in the same way.

---

## Why strata can become small

Small strata mainly came from:

1. Intersection of experiment samples with metadata availability.
2. Complete-case filtering for exposure plus covariates.
3. Confidence filtering (`score_gap` cutoff), which reduces rows further.
4. Class imbalance in the target label.

So a significant row like `5 vs 27` is possible even with a larger raw cohort.

---

## Threshold sweep performed

A threshold range was tested with the rule:

- robust row = `FDR < 0.05` and minority class size `>= 5`.

Thresholds tested:

- `0`
- `0.0025`
- `0.005`
- `0.0075`
- `0.01`
- `0.0125`
- `0.015`
- `0.02`

This was run for `lihc_foxa2_top4_nafld`, and compared with the prior sweep set.

### Robust counts by threshold

- `0`: 0 robust rows
- `0.0025`: 2 robust rows
- `0.005`: 4 robust rows
- `0.0075`: 6 robust rows (best)
- `0.01`: 2 robust rows
- `0.0125`: 0 robust rows
- `0.015`: 2 robust rows
- `0.02`: 0 robust rows

Conclusion:

- `0.0075` was the best threshold under this robustness criterion.

---

## Best-performing model setting

Across the robust threshold analysis, the strongest recurring setting was:

- `track_strategy=exp_decay`
- `bin_size=500000`
- `scoring_system=spearman_local_score`

Secondary robust signal appeared for:

- `exp_decay + 1M + pearson_local_score`

---

## Significant rows at best threshold (`0.0075`)

Primary robust rows (best setting, `exp_decay + 500k + spearman_local_score`):

1. `hbv_status` vs `foxa2_abnormal_zero` (`n=32`, `5 vs 27`, `FDR=0.014353`)
2. `hbv_status` vs `foxa2_normal_pos` (`n=32`, `27 vs 5`, `FDR=0.014353`)
3. `nafld_status` vs `foxa2_abnormal_zero` (`n=32`, `5 vs 27`, `FDR=0.014353`)
4. `nafld_status` vs `foxa2_normal_pos` (`n=32`, `27 vs 5`, `FDR=0.014353`)

Secondary rows at same threshold:

5. `alcohol_status` vs `foxa2_abnormal_zero` under `exp_decay + 1M + pearson_local_score` (`n=46`, `5 vs 41`, `FDR=0.049605`)
6. `alcohol_status` vs `foxa2_normal_pos` under same setting (`n=46`, `41 vs 5`, `FDR=0.049605`)

---

## Direction of association for the 4 main rows

For `exp_decay + 500k + spearman_local_score` at `0.0075`:

- `hbv_status__yes`:
  - lower odds of `foxa2_abnormal_zero`,
  - higher odds of `foxa2_normal_pos`,
  - both significant (`FDR=0.014353`).

- `nafld_status__yes`:
  - higher odds of `foxa2_abnormal_zero`,
  - lower odds of `foxa2_normal_pos`,
  - both significant (`FDR=0.014353`).

Interpretation note:

- The two label rows are complementary for a two-label problem.
- Odds-ratio magnitudes were extreme because strata were small, so direction and significance are more stable than effect-size magnitude.

---

## Clarification about significance vs label imbalance

Significance here does not simply mean one label had more samples.

It means:

- after adjustment for confounders,
- including exposure improves model fit relative to excluding exposure,
- and that improvement remains significant after FDR correction.

Label imbalance alone does not guarantee significance.

---

## Mutation-only adjustment check

A mutation-only adjustment run (without clinical covariates) was checked for `lihc_foxa2_top4_nafld` at `0.005` and `0.0075` for:

- `exp_decay + 500k + spearman_local_score`
- confident subset (`score_gap_ge_0.01` rows in output)

Result:

- 0 FDR-significant rows in this specific check.

This supports that clinical covariate structure changes the fitted association results materially.

---

## Score-level association check (`lihc_foxa2_top4`)

To complement label-level results, score-level associations were also tested for `lihc_foxa2_top4` using adjusted linear-model likelihood-ratio tests (FDR corrected).

### A) Mutation burden only

- Adjustment: `log1p_mutation_burden`
- FDR-significant rows: 8
- Pattern: all 8 significant rows were `hcv_status` associated with `score_foxa2_abnormal_zero` across multiple config/scoring combinations.

### B) Full adjustment (mutation burden + clinical confounders)

- Adjustment: `log1p_mutation_burden`, `alcohol_status`, `hbv_status`, `hcv_status`, `nafld_status`, `fibrosis_ishak_score` (excluding the exposure being tested)
- FDR-significant rows: 3

Significant rows:

1. `exp_decay` 1M + `pearson_local_score`, `score_foxa2_normal_pos ~ alcohol_status` (FDR `0.0451`)
2. `exp_decay` 500k + `spearman_r_linear_resid`, `score_foxa2_abnormal_zero ~ hcv_status` (FDR `0.0492`)
3. `counts_raw` 500k + `spearman_r_linear_resid`, `score_foxa2_abnormal_zero ~ hcv_status` (FDR `0.0492`)

Majority vs minority exposure-group breakdown for these 3 rows:

- Alcohol row (`n=68`): alcohol yes = 27, alcohol no = 41 (majority/minority = 41 vs 27)
- HCV rows (`n=68` each): hcv yes = 19, hcv no = 49 (majority/minority = 49 vs 19)

Interpretation:

- Adding clinical confounders reduced score-level significant hits from 8 to 3, suggesting part of the mutation-only signal was explained by correlated clinical structure.

---

## Cirrhosis sensitivity check (using `fibrosis_present` proxy)

There is no explicit `cirrhosis` column in the cleaned metadata files used by the adjusted script.  
The available cirrhosis-like proxy is:

- `fibrosis_present` (binary yes/no)

So a sensitivity run replaced `fibrosis_ishak_score` with `fibrosis_present`.

### Label-level (`lihc_foxa2_top4_nafld`, threshold `0.0075`)

Baseline model (`fibrosis_ishak_score`):

- robust rows (FDR < 0.05 and minority >= 5): 6
- robust unique associations: 6

Cirrhosis-proxy model (`fibrosis_present`):

- robust rows: 2
- robust unique associations: 2
- robust rows retained:
  - `nafld_status -> foxa2_abnormal_zero` (`n=32`, `5 vs 27`, FDR `0.0325`)
  - `nafld_status -> foxa2_normal_pos` (`n=32`, `27 vs 5`, FDR `0.0325`)

In this label-level setting, replacing Ishak stage with the coarse binary fibrosis proxy weakened the overall robust signal.

### Score-level (`lihc_foxa2_top4`, full adjustment)

Baseline model (`fibrosis_ishak_score`):

- FDR-significant rows: 3

Cirrhosis-proxy model (`fibrosis_present`):

- FDR-significant rows: 17

This increase suggests that the binary fibrosis proxy controls confounding less specifically than ordinal Ishak stage and can leave more residual signal.

### Recommended default

Use:

- `fibrosis_ishak_score` as the primary fibrosis/cirrhosis adjustment variable
- `fibrosis_present` as a sensitivity-only alternative

Rationale: Ishak stage is more granular and yielded more stable/credible adjusted behaviour in these runs.

---

## Raw metadata prevalence (no thresholding)

For `master_sample_metadata_lihc_nafld.csv`, in the unique-sample overlap with the experiment (`n=126`):

- `hcv_status`: 30 positive, 90 negative, 6 missing
- `hbv_status`: 74 positive, 47 negative, 5 missing
- `alcohol_status`: 55 positive, 69 negative, 2 missing
- `nafld_status`: 13 positive, 113 negative, 0 missing

These are raw attribute counts, not model-specific complete-case counts.

---

## Is it defensible to carry this into DE and Kaplan-Meier?

Yes, defensible as a discovery-led choice, with caveats:

1. Use the chosen setting as pre-specified for downstream analyses.
2. Treat DE and survival as confirmation only if done in independent or held-out data.
3. Report small strata explicitly.
4. Include sensitivity checks at nearby thresholds (`0.005`, `0.01`).

Using the rest of TCGA-LIHC as validation is reasonable if:

- it was not used for model/threshold selection,
- rules are pre-specified,
- missingness handling is pre-defined (not ad hoc).

---

## Recommended practical summary sentence

"After adjustment for mutation burden and clinical covariates using likelihood-ratio testing in logistic one-vs-rest models with FDR correction, the most robust label-level signal was observed with exp_decay (500 kb bins) and spearman_local_score, with best performance at a score-gap confidence threshold of 0.0075; significant associations were observed for HBV and NAFLD status, with minority class size at least 5."

---

## Update: Best 3 label configs from clean standalone runs

This update is based on clean reruns for `lihc_foxa2_top4` using:

- output file: `outputs/experiments/lihc_foxa2_top4/standalone_rawp_associations_all_outcomes.csv`
- selection: label outcomes only (`label_foxa2_abnormal_zero`, `label_foxa2_normal_pos`)
- significance rule in this section: `p < 0.05` per standalone config (no pooled FDR across all configs/metrics)

Top 3 config + metric pairs by number of significant label associations:

1. `track_strategy=counts_raw`, `bin_size=1000000`, `scoring_system=rf_resid` (`n_assoc=10`)
2. `track_strategy=exp_decay`, `bin_size=1000000`, `scoring_system=pearson_r_linear_resid` (`n_assoc=10`)
3. `track_strategy=counts_raw`, `bin_size=1000000`, `scoring_system=pearson_local_score` (`n_assoc=8`)

### Significant associations in each top config (with `race` and `gender` removed)

1. `track_strategy=counts_raw`, `bin_size=1000000`, `scoring_system=rf_resid`
- `age_at_initial_pathologic_diagnosis -> foxa2_normal_pos` (`p=0.001941`, coef on abnormal-label model `-0.229358`, `n=60`)
- `alcohol_status__yes -> foxa2_abnormal_zero` (`p=0.008886`, coef `+3.888809`, `n=60`)
- `cirrhosis__yes -> foxa2_abnormal_zero` (`p=0.002724`, coef `+5.260625`, `n=60`)

2. `track_strategy=exp_decay`, `bin_size=1000000`, `scoring_system=pearson_r_linear_resid`
- `age_at_initial_pathologic_diagnosis -> foxa2_normal_pos` (`p=0.002756`, coef on abnormal-label model `-0.299394`, `n=60`)
- `alcohol_status__yes -> foxa2_abnormal_zero` (`p=0.019783`, coef `+6.227161`, `n=60`)
- `cirrhosis__yes -> foxa2_abnormal_zero` (`p=0.004583`, coef `+8.675911`, `n=60`)

3. `track_strategy=counts_raw`, `bin_size=1000000`, `scoring_system=pearson_local_score`
- `obesity_class__overweight -> foxa2_abnormal_zero` (`p=0.016983`, coef `+23.589226`, `n=60`)

---

## Reproducibility notes

Environment:

- Conda environment used: `mut-epi-origin`

Script:

- `scripts/03_clinical_associations/validate_state_scores_adjusted.py`

Example run:

```bash
python scripts/03_clinical_associations/validate_state_scores_adjusted.py \
  --experiment-name lihc_foxa2_top4_nafld \
  --metadata-path data/derived/master_sample_metadata_lihc_nafld.csv \
  --state-labels foxa2_normal_pos,foxa2_abnormal_zero \
  --state-suffixes foxa2_normal_pos,foxa2_abnormal_zero \
  --score-gap-threshold 0.0075 \
  --skip-sensitivity-model \
  --allow-aggregated-results
```

---

## Update: Deduplicated + Mutation-Burden-Adjusted Results (Recomputed From Scratch)

Conditions used for this recomputation:

- Metadata deduplicated by `tumour_sample_submitter_id` (first row retained per ID)
- Mutation burden covariate derived from `results.csv` as `log1p(n_mutations_total)`
- Validation source: `validation_score_rankings.csv`
- Threshold grid: `score_gap >= {0.0, 0.005, 0.01, 0.02, 0.03}`
- Inclusion filter for this table: `n_total >= 50`
- Label-level test: logistic likelihood-ratio test (LRT) against burden-only model
- Score-level test: OLS nested-model F-test against burden-only model
- Significance rule for this table: raw `p < 0.05`

### Verified significant rows (`p < 0.05`, `n_total >= 50`)

| level | threshold | config | scoring | attribute | n | test | p-value |
|---|---:|---|---|---|---:|---|---:|
| label | 0.000 | `counts_raw|500000` | `spearman_local_score` | `nafld_status` | 72 | logistic LRT (adj.) | 0.006577 |
| label | 0.000 | `exp_decay|1000000` | `pearson_r_linear_resid` | `obesity_class` | 152 | logistic LRT (adj.) | 0.006921 |
| label | 0.010 | `exp_decay|1000000` | `spearman_r_linear_resid` | `obesity_class` | 61 | logistic LRT (adj.) | 0.008254 |
| label | 0.010 | `counts_raw|1000000` | `spearman_r_linear_resid` | `obesity_class` | 64 | logistic LRT (adj.) | 0.009103 |
| label | 0.000 | `counts_raw|500000` | `pearson_r_linear_resid` | `nafld_status` | 72 | logistic LRT (adj.) | 0.010890 |
| label | 0.010 | `counts_raw|1000000` | `rf_resid` | `obesity_class` | 52 | logistic LRT (adj.) | 0.011281 |
| label | 0.000 | `counts_raw|1000000` | `pearson_r_linear_resid` | `obesity_class` | 152 | logistic LRT (adj.) | 0.020304 |
| score | 0.000 | `counts_raw|1000000` | `rf_resid` | `obesity_class` | 152 | OLS F-test (adj.) | 0.021136 |
| score | 0.010 | `counts_raw|1000000` | `rf_resid` | `obesity_class` | 52 | OLS F-test (adj.) | 0.023488 |
| label | 0.005 | `exp_decay|1000000` | `rf_resid` | `obesity_class` | 103 | logistic LRT (adj.) | 0.023786 |
| score | 0.005 | `counts_raw|1000000` | `rf_resid` | `obesity_class` | 105 | OLS F-test (adj.) | 0.025825 |
| label | 0.000 | `counts_raw|500000` | `spearman_r_linear_resid` | `nafld_status` | 72 | logistic LRT (adj.) | 0.026156 |
| score | 0.005 | `exp_decay|500000` | `spearman_local_score` | `hbv_status` | 69 | OLS F-test (adj.) | 0.030074 |
| label | 0.000 | `counts_raw|500000` | `rf_resid` | `nafld_status` | 72 | logistic LRT (adj.) | 0.035649 |
| label | 0.000 | `counts_raw|500000` | `spearman_r_linear_resid` | `obesity_class` | 152 | logistic LRT (adj.) | 0.040241 |
| score | 0.010 | `exp_decay|500000` | `pearson_r_linear_resid` | `fibrosis_present` | 55 | OLS F-test (adj.) | 0.041929 |
| label | 0.000 | `counts_raw|500000` | `spearman_r_linear_resid` | `fibrosis_present` | 152 | logistic LRT (adj.) | 0.047480 |
| label | 0.005 | `counts_raw|1000000` | `pearson_r_linear_resid` | `obesity_class` | 112 | logistic LRT (adj.) | 0.049276 |

---

## Highlighted config: `counts_raw|500000` + `spearman_r_linear_resid`

At `score_gap >= 0.000`, this highlighted configuration has three adjusted label-level signals:

- `nafld_status` (`n=72`, logistic LRT adjusted `p=0.026156`)
- `fibrosis_present` (`n=152`, logistic LRT adjusted `p=0.047480`)
- `obesity_class` (`n=152`, logistic LRT adjusted `p=0.040241`)

Direction (from observed label proportions):

- `nafld_status_yes -> abnormal`
- `fibrosis_present_yes -> abnormal`
- `obesity_class_underweight -> abnormal` (highest abnormal proportion among obesity categories)

### Contingency tables for highlighted config (`score_gap >= 0.000`)

`nafld_status` (`n_total=72`)

| nafld_status | foxa2_abnormal_zero | foxa2_normal_pos | total | abnormal % |
|---|---:|---:|---:|---:|
| no  | 8 | 55 | 63 | 12.7 |
| yes | 4 | 5  | 9  | 44.4 |

`fibrosis_present` (`n_total=152`)

| fibrosis_present | foxa2_abnormal_zero | foxa2_normal_pos | total | abnormal % |
|---|---:|---:|---:|---:|
| no  | 4  | 38 | 42  | 9.5 |
| yes | 25 | 85 | 110 | 22.7 |

`obesity_class` (`n_total=152`)

| obesity_class | foxa2_abnormal_zero | foxa2_normal_pos | total | abnormal % |
|---|---:|---:|---:|---:|
| normal | 8 | 43 | 51 | 15.7 |
| obesity class i | 3 | 13 | 16 | 18.8 |
| obesity class ii | 0 | 4 | 4 | 0.0 |
| obesity class iii | 5 | 40 | 45 | 11.1 |
| overweight | 9 | 20 | 29 | 31.0 |
| underweight | 4 | 3 | 7 | 57.1 |
