#!/usr/bin/env Rscript

#' Run gene-level differential expression using inferred FOXA2 label groups.
#'
#' This script derives per-sample inferred labels from one experiment results CSV,
#' filters to one configuration and scoring system, then runs DESeq2 between the
#' two inferred label groups on a STAR counts matrix.
#'
#' Workflow:
#' 1. Read `results.csv` from one experiment run.
#' 2. Filter rows to `track_strategy` and `bin_size`.
#' 3. Build per-sample scores for two states from one scoring system.
#' 4. Derive `best_cell_state` and `score_gap` (best minus second best).
#' 5. Apply a score-gap threshold and minimum group size.
#' 6. Match inferred labels to count-matrix columns by harmonised TCGA prefix.
#' 7. Optionally apply a group-aware prefilter using the smaller label group.
#' 8. Run DESeq2 and write full plus significant DE results.

suppressPackageStartupMessages({
  library(data.table)
  library(DESeq2)
})

`%||%` <- function(x, y) {
  if (is.null(x)) {
    return(y)
  }
  x
}

usage <- function() {
  cat(
    paste(
      "Usage:",
      "Rscript scripts/04_differential_expression/run_differential_expression_by_inferred_labels.R \\",
      "  --counts-path <path> \\",
      "  [--results-path <path>] \\",
      "  [--metadata-path <path>] \\",
      "  [--metadata-sample-col tumour_sample_submitter_id] \\",
      "  [--covariates age_at_diagnosis,gender,ajcc_pathologic_stage] \\",
      "  [--output-dir <path>] \\",
      "  [--track-strategy counts_raw] \\",
      "  [--bin-size 500000] \\",
      "  [--scoring-system spearman_r_linear_resid] \\",
      "  [--state-labels foxa2_normal_pos,foxa2_abnormal_zero] \\",
      "  [--score-gap-threshold 0.0] \\",
      "  [--min-group-size 3] \\",
      "  [--id-prefix-length 15] \\",
      "  [--invert-scores true] \\",
      "  [--contrast-case foxa2_abnormal_zero] \\",
      "  [--contrast-control foxa2_normal_pos] \\",
      "  [--gene-col auto] \\",
      "  [--fdr-alpha 0.05] \\",
      "  [--min-abs-log2fc 0.0] \\",
      "  [--deseq-fit-type local] \\",
      "  [--min-count 10] \\",
      "  [--min-sample-frac 0.1] \\",
      "  [--group-aware-filter false] \\",
      "  [--group-filter-min-count 10] \\",
      "  [--group-filter-min-frac 0.1] \\",
      "  [--lfc-threshold 0.0] \\",
      "  [--lfc-alt-hypothesis greaterAbs]",
      sep = "\n"
    ),
    "\n"
  )
}

parse_args <- function(argv) {
  defaults <- list(
    "results-path" = "outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/results.csv",
    "metadata-path" = "data/derived/master_metadata.csv",
    "metadata-sample-col" = "tumour_sample_submitter_id",
    "covariates" = "age_at_diagnosis,gender,ajcc_pathologic_stage",
    "output-dir" = "outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/de_counts_raw_500k_spearman_r_linear_resid",
    "track-strategy" = "counts_raw",
    "bin-size" = "500000",
    "scoring-system" = "spearman_r_linear_resid",
    "state-labels" = "foxa2_normal_pos,foxa2_abnormal_zero",
    "score-gap-threshold" = "0.0",
    "min-group-size" = "3",
    "id-prefix-length" = "15",
    "invert-scores" = "true",
    "contrast-case" = "foxa2_abnormal_zero",
    "contrast-control" = "foxa2_normal_pos",
    "gene-col" = "auto",
    "fdr-alpha" = "0.05",
    "min-abs-log2fc" = "0.0",
    "deseq-fit-type" = "local",
    "min-count" = "10",
    "min-sample-frac" = "0.1",
    "group-aware-filter" = "false",
    "group-filter-min-count" = "10",
    "group-filter-min-frac" = "0.1",
    "lfc-threshold" = "0.0",
    "lfc-alt-hypothesis" = "greaterAbs"
  )

  parsed <- list()
  i <- 1
  while (i <= length(argv)) {
    token <- argv[[i]]
    if (token %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0)
    }
    if (!startsWith(token, "--")) {
      stop("Unexpected argument: ", token)
    }
    key <- substring(token, 3)
    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) {
      stop("Missing value for --", key)
    }
    parsed[[key]] <- argv[[i + 1]]
    i <- i + 2
  }

  if (is.null(parsed[["counts-path"]])) {
    usage()
    stop("--counts-path is required.")
  }

  for (key in names(defaults)) {
    parsed[[key]] <- parsed[[key]] %||% defaults[[key]]
  }
  parsed
}

normalise_tcga_id <- function(x, prefix_len) {
  y <- toupper(trimws(as.character(x)))
  y <- gsub("\\.", "-", y)
  substr(y, 1, prefix_len)
}

parse_state_labels <- function(x) {
  labels <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  labels <- labels[nzchar(labels)]
  if (length(labels) != 2) {
    stop("Exactly two state labels are required in --state-labels.")
  }
  labels
}

to_bool <- function(x) {
  value <- tolower(trimws(as.character(x)))
  if (value %in% c("true", "1", "yes", "y")) {
    return(TRUE)
  }
  if (value %in% c("false", "0", "no", "n")) {
    return(FALSE)
  }
  stop("Boolean value expected, got: ", x)
}

parse_csv_list <- function(x) {
  values <- trimws(strsplit(as.character(x), ",", fixed = TRUE)[[1]])
  values <- values[nzchar(values)]
  if (length(values) == 1 && tolower(values[[1]]) == "none") {
    return(character(0))
  }
  values
}

normalise_missing_token <- function(x) {
  value <- trimws(as.character(x))
  if (!nzchar(value) || is.na(value)) {
    return(NA_character_)
  }
  lower <- tolower(value)
  if (lower %in% c("na", "nan", "none", "null", "n/a", "unknown", "not reported", "not applicable")) {
    return(NA_character_)
  }
  if (value %in% c("[Not Available]", "[Unknown]", "---", "'--", "--")) {
    return(NA_character_)
  }
  value
}

first_sample_from_selected <- function(x) {
  text <- trimws(as.character(x))
  if (!nzchar(text) || is.na(text)) {
    return(NA_character_)
  }
  parts <- trimws(unlist(strsplit(text, "[,;]")))
  parts <- parts[nzchar(parts)]
  if (length(parts) != 1) {
    return(NA_character_)
  }
  parts[[1]]
}

derive_labels_from_results <- function(results_dt, track_strategy_value, bin_size, scoring_system,
                                       state_labels, score_gap_threshold, id_prefix_length,
                                       invert_scores) {
  required_cols <- c("selected_sample_ids", "track_strategy")
  missing_required <- setdiff(required_cols, names(results_dt))
  if (length(missing_required) > 0) {
    stop("Missing required columns in results.csv: ", paste(missing_required, collapse = ", "))
  }

  score_cols <- paste0(scoring_system, "_", state_labels, "_mean_weighted")
  missing_scores <- setdiff(score_cols, names(results_dt))
  if (length(missing_scores) > 0) {
    stop("Missing score columns in results.csv: ", paste(missing_scores, collapse = ", "))
  }

  dt <- copy(results_dt)
  dt[, sample_raw := vapply(selected_sample_ids, first_sample_from_selected, character(1))]
  dt <- dt[!is.na(sample_raw) & sample_raw != ""]

  strategy_bin_col <- paste0(track_strategy_value, "_bin")
  has_strategy_bin_col <- strategy_bin_col %in% names(dt)
  has_generic_bin_col <- "bin_size" %in% names(dt)

  if (has_strategy_bin_col) {
    dt <- dt[
      track_strategy == track_strategy_value &
        as.numeric(get(strategy_bin_col)) == as.numeric(bin_size)
    ]
  } else if (has_generic_bin_col) {
    dt <- dt[
      track_strategy == track_strategy_value &
        as.numeric(get("bin_size")) == as.numeric(bin_size)
    ]
  } else {
    dt <- dt[track_strategy == track_strategy_value]
    warning(
      "No strategy-specific bin column found (expected '",
      strategy_bin_col,
      "'). Falling back to track_strategy-only filtering."
    )
  }

  if (nrow(dt) == 0) {
    stop("No rows remain after filtering by track strategy and bin size.")
  }

  for (sc in score_cols) {
    dt[[sc]] <- as.numeric(dt[[sc]])
    if (invert_scores) {
      dt[[sc]] <- -1.0 * dt[[sc]]
    }
  }

  agg <- dt[
    ,
    c(
      lapply(.SD, mean, na.rm = TRUE),
      list(sample_raw = sample_raw[1])
    ),
    by = .(sample = normalise_tcga_id(sample_raw, id_prefix_length)),
    .SDcols = score_cols
  ]

  agg <- agg[!is.na(sample) & sample != ""]
  if (nrow(agg) == 0) {
    stop("No per-sample rows remain after aggregation.")
  }

  s1 <- score_cols[[1]]
  s2 <- score_cols[[2]]

  agg[, best_cell_state := ifelse(get(s1) >= get(s2), state_labels[[1]], state_labels[[2]])]
  agg[, best_score := pmax(get(s1), get(s2))]
  agg[, second_best_score := pmin(get(s1), get(s2))]
  agg[, score_gap := best_score - second_best_score]

  agg <- agg[score_gap >= score_gap_threshold]
  if (nrow(agg) == 0) {
    stop("No sample rows remain after score-gap filtering.")
  }

  agg
}

collapse_counts_by_sample <- function(count_matrix, norm_ids) {
  groups <- split(seq_along(norm_ids), norm_ids)
  collapsed <- lapply(groups, function(idx) {
    if (length(idx) == 1) {
      count_matrix[, idx]
    } else {
      rowSums(count_matrix[, idx, drop = FALSE])
    }
  })
  collapsed_matrix <- do.call(cbind, collapsed)
  colnames(collapsed_matrix) <- names(groups)
  collapsed_matrix
}

run_deseq <- function(counts_df, gene_col, labels_dt, id_prefix_length,
                      contrast_case, contrast_control, metadata_dt, covariate_cols,
                      deseq_fit_type, min_count, min_sample_frac, min_group_size,
                      group_aware_filter, group_filter_min_count, group_filter_min_frac,
                      lfc_threshold, lfc_alt_hypothesis) {
  gene_ids <- as.character(counts_df[[gene_col]])
  counts_only <- counts_df[, setdiff(names(counts_df), gene_col), with = FALSE]
  if (ncol(counts_only) == 0) {
    stop("No sample columns found in count matrix.")
  }

  sample_cols <- names(counts_only)
  sample_norm <- normalise_tcga_id(sample_cols, id_prefix_length)
  keep <- sample_norm %in% labels_dt$sample
  counts_only <- counts_only[, keep, with = FALSE]
  sample_cols <- sample_cols[keep]
  sample_norm <- sample_norm[keep]

  if (length(sample_cols) == 0) {
    stop("No overlapping samples between count matrix and inferred labels.")
  }

  count_matrix <- as.matrix(counts_only)
  storage.mode(count_matrix) <- "numeric"
  rownames(count_matrix) <- gene_ids

  collapsed_matrix <- collapse_counts_by_sample(count_matrix, sample_norm)

  labels_unique <- unique(labels_dt[, .(sample, best_cell_state, score_gap, best_score, second_best_score)])
  labels_unique <- labels_unique[match(colnames(collapsed_matrix), sample)]

  col_data <- data.table(sample = colnames(collapsed_matrix))
  col_data <- merge(col_data, labels_unique, by = "sample", all.x = TRUE)
  col_data <- merge(col_data, metadata_dt, by = "sample", all.x = TRUE)

  required_design_cols <- c("best_cell_state", covariate_cols)
  for (col_name in required_design_cols) {
    col_data[[col_name]] <- vapply(col_data[[col_name]], normalise_missing_token, character(1))
  }
  keep_complete <- complete.cases(col_data[, ..required_design_cols])
  col_data <- col_data[keep_complete]
  if (nrow(col_data) == 0) {
    stop("No samples remain after complete-case filtering for design covariates.")
  }

  collapsed_matrix <- collapsed_matrix[, col_data$sample, drop = FALSE]
  genes_before_prefilter <- nrow(collapsed_matrix)

  min_samples_required <- max(1L, ceiling(min_sample_frac * ncol(collapsed_matrix)))
  keep_genes <- rowSums(collapsed_matrix >= min_count) >= min_samples_required
  collapsed_matrix <- collapsed_matrix[keep_genes, , drop = FALSE]
  genes_after_global_prefilter <- nrow(collapsed_matrix)

  group <- factor(col_data$best_cell_state)
  group_counts <- table(group)
  if (any(group_counts < min_group_size)) {
    stop(
      "After covariate complete-case filtering, group sizes are below min-group-size: ",
      paste(names(group_counts), as.integer(group_counts), sep = "=", collapse = ", ")
    )
  }
  if (!(contrast_control %in% levels(group) && contrast_case %in% levels(group))) {
    stop("contrast labels are not both present after sample matching.")
  }
  group <- stats::relevel(group, ref = contrast_control)
  col_data$group <- group

  if (group_aware_filter) {
    min_group_n <- min(as.integer(group_counts))
    min_group_samples_required <- max(1L, ceiling(group_filter_min_frac * min_group_n))
    keep_by_group <- rep(FALSE, nrow(collapsed_matrix))
    for (group_level in levels(group)) {
      in_group <- col_data$sample[group == group_level]
      pass_group <- rowSums(
        collapsed_matrix[, in_group, drop = FALSE] >= group_filter_min_count
      ) >= min_group_samples_required
      keep_by_group <- keep_by_group | pass_group
    }
    collapsed_matrix <- collapsed_matrix[keep_by_group, , drop = FALSE]
  }

  if (nrow(collapsed_matrix) == 0) {
    stop("No genes remain after count prefiltering.")
  }

  design_data <- data.frame(
    row.names = col_data$sample,
    group = col_data$group,
    score_gap = col_data$score_gap,
    best_score = col_data$best_score,
    second_best_score = col_data$second_best_score,
    stringsAsFactors = FALSE
  )

  for (covariate in covariate_cols) {
    raw_values <- as.character(col_data[[covariate]])
    numeric_values <- suppressWarnings(as.numeric(raw_values))
    if (all(!is.na(numeric_values))) {
      design_data[[covariate]] <- numeric_values
    } else {
      design_data[[covariate]] <- factor(raw_values)
      levels(design_data[[covariate]]) <- make.names(levels(design_data[[covariate]]), unique = TRUE)
    }
  }

  levels(design_data$group) <- make.names(levels(design_data$group), unique = TRUE)

  for (covariate in covariate_cols) {
    if (is.numeric(design_data[[covariate]])) {
      sd_value <- stats::sd(design_data[[covariate]], na.rm = TRUE)
      if (!is.na(sd_value) && sd_value > 0) {
        design_data[[covariate]] <- as.numeric(scale(design_data[[covariate]]))
      }
    }
  }

  design_formula <- as.formula(paste("~", paste(c(covariate_cols, "group"), collapse = " + ")))

  dds <- DESeqDataSetFromMatrix(
    countData = round(collapsed_matrix),
    colData = design_data,
    design = design_formula
  )

  fit_method <- paste0("DESeq_fitType_", deseq_fit_type)
  dds <- tryCatch(
    DESeq(dds, fitType = deseq_fit_type),
    error = function(e) {
      if (grepl("all gene-wise dispersion estimates are within 2 orders of magnitude", conditionMessage(e), fixed = TRUE)) {
        fit_method <<- "gene_wise_dispersion_fallback"
        dds_fallback <- estimateSizeFactors(dds)
        dds_fallback <- estimateDispersionsGeneEst(dds_fallback)
        dispersions(dds_fallback) <- mcols(dds_fallback)$dispGeneEst
        return(nbinomWaldTest(dds_fallback))
      }
      stop(e)
    }
  )
  if (lfc_threshold > 0) {
    res <- results(
      dds,
      contrast = c("group", contrast_case, contrast_control),
      lfcThreshold = lfc_threshold,
      altHypothesis = lfc_alt_hypothesis
    )
  } else {
    res <- results(dds, contrast = c("group", contrast_case, contrast_control))
  }
  list(
    dds = dds,
    res = res,
    col_data = design_data,
    design_formula = design_formula,
    fit_method = fit_method,
    genes_before_prefilter = genes_before_prefilter,
    genes_after_global_prefilter = genes_after_global_prefilter,
    genes_after_prefilter = nrow(collapsed_matrix),
    prefilter_min_samples_required = min_samples_required
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  counts_path <- args[["counts-path"]]
  results_path <- args[["results-path"]]
  metadata_path <- args[["metadata-path"]]
  metadata_sample_col <- args[["metadata-sample-col"]]
  covariate_cols <- parse_csv_list(args[["covariates"]])
  output_dir <- args[["output-dir"]]

  track_strategy <- args[["track-strategy"]]
  bin_size <- as.numeric(args[["bin-size"]])
  scoring_system <- args[["scoring-system"]]
  state_labels <- parse_state_labels(args[["state-labels"]])
  score_gap_threshold <- as.numeric(args[["score-gap-threshold"]])
  min_group_size <- as.integer(args[["min-group-size"]])
  id_prefix_length <- as.integer(args[["id-prefix-length"]])
  invert_scores <- to_bool(args[["invert-scores"]])
  contrast_case <- args[["contrast-case"]]
  contrast_control <- args[["contrast-control"]]
  gene_col_arg <- args[["gene-col"]]
  fdr_alpha <- as.numeric(args[["fdr-alpha"]])
  min_abs_log2fc <- as.numeric(args[["min-abs-log2fc"]])
  deseq_fit_type <- args[["deseq-fit-type"]]
  min_count <- as.integer(args[["min-count"]])
  min_sample_frac <- as.numeric(args[["min-sample-frac"]])
  group_aware_filter <- to_bool(args[["group-aware-filter"]])
  group_filter_min_count <- as.integer(args[["group-filter-min-count"]])
  group_filter_min_frac <- as.numeric(args[["group-filter-min-frac"]])
  lfc_threshold <- as.numeric(args[["lfc-threshold"]])
  lfc_alt_hypothesis <- args[["lfc-alt-hypothesis"]]
  if (is.na(min_count) || min_count < 0) {
    stop("--min-count must be a non-negative integer.")
  }
  if (is.na(min_sample_frac) || min_sample_frac <= 0 || min_sample_frac > 1) {
    stop("--min-sample-frac must be in (0, 1].")
  }
  if (is.na(group_filter_min_count) || group_filter_min_count < 0) {
    stop("--group-filter-min-count must be a non-negative integer.")
  }
  if (is.na(group_filter_min_frac) || group_filter_min_frac <= 0 || group_filter_min_frac > 1) {
    stop("--group-filter-min-frac must be in (0, 1].")
  }
  if (is.na(lfc_threshold) || lfc_threshold < 0) {
    stop("--lfc-threshold must be non-negative.")
  }
  if (!(lfc_alt_hypothesis %in% c("greaterAbs", "lessAbs", "greater", "less"))) {
    stop("--lfc-alt-hypothesis must be one of: greaterAbs, lessAbs, greater, less.")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  results_dt <- fread(results_path)
  labels_dt <- derive_labels_from_results(
    results_dt = results_dt,
    track_strategy_value = track_strategy,
    bin_size = bin_size,
    scoring_system = scoring_system,
    state_labels = state_labels,
    score_gap_threshold = score_gap_threshold,
    id_prefix_length = id_prefix_length,
    invert_scores = invert_scores
  )

  label_counts <- table(labels_dt$best_cell_state)
  keep_labels <- names(label_counts[label_counts >= min_group_size])
  labels_dt <- labels_dt[best_cell_state %in% keep_labels]

  if (length(unique(labels_dt$best_cell_state)) != 2) {
    stop("Exactly two label groups are required after min-group-size filtering.")
  }

  counts_dt <- fread(counts_path)
  metadata_dt <- fread(metadata_path)
  if (!(metadata_sample_col %in% names(metadata_dt))) {
    stop("metadata sample column not found: ", metadata_sample_col)
  }
  missing_covariates <- setdiff(covariate_cols, names(metadata_dt))
  if (length(missing_covariates) > 0) {
    stop("Missing covariate columns in metadata: ", paste(missing_covariates, collapse = ", "))
  }
  metadata_dt <- metadata_dt[, c(metadata_sample_col, covariate_cols), with = FALSE]
  setnames(metadata_dt, old = metadata_sample_col, new = "sample_raw")
  metadata_dt[, sample := normalise_tcga_id(sample_raw, id_prefix_length)]
  metadata_dt <- metadata_dt[!is.na(sample) & sample != ""]
  metadata_dt <- metadata_dt[!duplicated(sample)]

  gene_col <- if (tolower(gene_col_arg) == "auto") names(counts_dt)[[1]] else gene_col_arg
  if (!(gene_col %in% names(counts_dt))) {
    stop("gene column not found in count matrix: ", gene_col)
  }

  fit <- run_deseq(
    counts_df = counts_dt,
    gene_col = gene_col,
    labels_dt = labels_dt,
    id_prefix_length = id_prefix_length,
    contrast_case = contrast_case,
    contrast_control = contrast_control,
    metadata_dt = metadata_dt,
    covariate_cols = covariate_cols,
    deseq_fit_type = deseq_fit_type,
    min_count = min_count,
    min_sample_frac = min_sample_frac,
    min_group_size = min_group_size,
    group_aware_filter = group_aware_filter,
    group_filter_min_count = group_filter_min_count,
    group_filter_min_frac = group_filter_min_frac,
    lfc_threshold = lfc_threshold,
    lfc_alt_hypothesis = lfc_alt_hypothesis
  )

  res_df <- as.data.frame(fit$res)
  res_df$gene <- rownames(res_df)
  res_df <- res_df[, c("gene", setdiff(names(res_df), "gene"))]
  res_df <- res_df[order(res_df$pvalue, na.last = TRUE), ]

  sig_df <- res_df[
    !is.na(res_df$padj) &
      res_df$padj <= fdr_alpha &
      abs(res_df$log2FoldChange) >= min_abs_log2fc,
  ]

  sample_labels_used <- as.data.table(fit$col_data, keep.rownames = "sample")
  fwrite(sample_labels_used, file.path(output_dir, "sample_labels_used.csv"))
  fwrite(res_df, file.path(output_dir, "differential_expression_results_all.csv"))
  fwrite(sig_df, file.path(output_dir, "differential_expression_results_significant.csv"))

  summary_lines <- c(
    paste0("counts_path: ", counts_path),
    paste0("results_path: ", results_path),
    paste0("metadata_path: ", metadata_path),
    paste0("metadata_sample_col: ", metadata_sample_col),
    paste0("track_strategy: ", track_strategy),
    paste0("bin_size: ", bin_size),
    paste0("scoring_system: ", scoring_system),
    paste0("state_labels: ", paste(state_labels, collapse = ",")),
    paste0("score_gap_threshold: ", score_gap_threshold),
    paste0("invert_scores: ", invert_scores),
    paste0("id_prefix_length: ", id_prefix_length),
    paste0("min_group_size: ", min_group_size),
    paste0("contrast_case: ", contrast_case),
    paste0("contrast_control: ", contrast_control),
    paste0("covariates: ", paste(covariate_cols, collapse = ",")),
    paste0("design_formula: ", deparse(fit$design_formula)),
    paste0("deseq_fit_type: ", deseq_fit_type),
    paste0("lfc_threshold: ", lfc_threshold),
    paste0("lfc_alt_hypothesis: ", lfc_alt_hypothesis),
    paste0("fit_method: ", fit$fit_method),
    paste0("min_count: ", min_count),
    paste0("min_sample_frac: ", min_sample_frac),
    paste0("group_aware_filter: ", group_aware_filter),
    paste0("group_filter_min_count: ", group_filter_min_count),
    paste0("group_filter_min_frac: ", group_filter_min_frac),
    paste0("prefilter_min_samples_required: ", fit$prefilter_min_samples_required),
    paste0("gene_col: ", gene_col),
    paste0("samples_used: ", nrow(fit$col_data)),
    paste0("genes_before_prefilter: ", fit$genes_before_prefilter),
    paste0("genes_after_global_prefilter: ", fit$genes_after_global_prefilter),
    paste0("genes_after_prefilter: ", fit$genes_after_prefilter),
    paste0("label_counts: ", paste(names(table(fit$col_data$group)), as.integer(table(fit$col_data$group)), sep = "=", collapse = ", ")),
    paste0("genes_tested: ", nrow(res_df)),
    paste0("fdr_alpha: ", fdr_alpha),
    paste0("min_abs_log2fc: ", min_abs_log2fc),
    paste0("significant_genes: ", nrow(sig_df))
  )
  writeLines(summary_lines, file.path(output_dir, "run_summary.txt"))

  message("Done.")
  message("Outputs:")
  message("  ", file.path(output_dir, "sample_labels_used.csv"))
  message("  ", file.path(output_dir, "differential_expression_results_all.csv"))
  message("  ", file.path(output_dir, "differential_expression_results_significant.csv"))
  message("  ", file.path(output_dir, "run_summary.txt"))
}

main()
