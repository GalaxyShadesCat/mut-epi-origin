#!/usr/bin/env Rscript

#' Run limma-voom differential expression from inferred label groups.
#'
#' This script consumes a sample-label table (for example from
#' `run_differential_expression_by_inferred_labels.R`) and runs limma-voom with
#' optional metadata covariates present in that table.

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(limma)
})

usage <- function() {
  cat(
    paste(
      "Usage:",
      "Rscript scripts/03_differential_expression/run_limma_by_inferred_labels.R \\",
      "  --counts-path <path> \\",
      "  --labels-path <path> \\",
      "  [--output-dir <path>] \\",
      "  [--id-prefix-length 15] \\",
      "  [--contrast-case foxa2_abnormal_zero] \\",
      "  [--contrast-control foxa2_normal_pos] \\",
      "  [--min-count 5] \\",
      "  [--min-sample-frac 0.05] \\",
      "  [--fdr-alpha 0.05]",
      sep = "\n"
    ),
    "\n"
  )
}

parse_args <- function(argv) {
  defaults <- list(
    "output-dir" = "outputs/experiments/lihc_foxa2_all_samples/de_limma_exp_decay_500k_spearman_r_linear_resid",
    "id-prefix-length" = "15",
    "contrast-case" = "foxa2_abnormal_zero",
    "contrast-control" = "foxa2_normal_pos",
    "min-count" = "5",
    "min-sample-frac" = "0.05",
    "fdr-alpha" = "0.05"
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

  if (is.null(parsed[["counts-path"]]) || is.null(parsed[["labels-path"]])) {
    usage()
    stop("--counts-path and --labels-path are required.")
  }

  for (key in names(defaults)) {
    if (is.null(parsed[[key]])) {
      parsed[[key]] <- defaults[[key]]
    }
  }
  parsed
}

normalise_tcga_id <- function(x, prefix_len) {
  y <- toupper(trimws(as.character(x)))
  y <- gsub("\\.", "-", y)
  substr(y, 1, prefix_len)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  counts_path <- args[["counts-path"]]
  labels_path <- args[["labels-path"]]
  out_dir <- args[["output-dir"]]
  id_prefix_length <- as.integer(args[["id-prefix-length"]])
  contrast_case <- args[["contrast-case"]]
  contrast_control <- args[["contrast-control"]]
  min_count <- as.integer(args[["min-count"]])
  min_sample_frac <- as.numeric(args[["min-sample-frac"]])
  fdr_alpha <- as.numeric(args[["fdr-alpha"]])

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  labels <- fread(labels_path)
  required_cols <- c("sample", "group")
  missing_cols <- setdiff(required_cols, names(labels))
  if (length(missing_cols) > 0) {
    stop("Missing required columns in labels file: ", paste(missing_cols, collapse = ", "))
  }

  counts_dt <- fread(counts_path)
  gene_col <- names(counts_dt)[[1]]
  mat <- as.matrix(counts_dt[, -1, with = FALSE])
  rownames(mat) <- as.character(counts_dt[[gene_col]])
  storage.mode(mat) <- "integer"

  sample_norm <- normalise_tcga_id(colnames(mat), id_prefix_length)
  map_dt <- data.table(sample = sample_norm, count_col = colnames(mat))
  setorder(map_dt, sample, count_col)
  map_pick <- map_dt[, .SD[1], by = sample]

  labels <- merge(labels, map_pick, by = "sample", all = FALSE)
  setorder(labels, count_col)
  if (nrow(labels) < 4) {
    stop("Too few matched samples for limma.")
  }

  design_cols <- c("age_at_diagnosis", "gender", "ajcc_pathologic_stage")
  available_design_cols <- design_cols[design_cols %in% names(labels)]
  covar_missing <- setdiff(design_cols, available_design_cols)
  if (length(covar_missing) > 0) {
    warning("Missing covariate columns in labels file: ", paste(covar_missing, collapse = ", "))
  }

  labels <- labels[!is.na(group)]
  labels[, group := factor(group)]
  labels <- labels[group %in% c(contrast_control, contrast_case)]
  labels[, group := factor(group, levels = c(contrast_control, contrast_case))]

  if ("age_at_diagnosis" %in% available_design_cols) {
    labels[, age_at_diagnosis := as.numeric(age_at_diagnosis)]
  }
  if ("gender" %in% available_design_cols) {
    labels[, gender := factor(gender)]
  }
  if ("ajcc_pathologic_stage" %in% available_design_cols) {
    labels[, ajcc_pathologic_stage := factor(make.names(ajcc_pathologic_stage))]
  }

  keep_rows <- complete.cases(labels[, c("group", available_design_cols), with = FALSE])
  labels <- labels[keep_rows]
  if (nrow(labels) < 4) {
    stop("Too few complete-case samples for limma.")
  }

  sample_cols <- labels$count_col
  mat <- mat[, sample_cols, drop = FALSE]

  min_required <- ceiling(min_sample_frac * ncol(mat))
  keep_genes <- rowSums(mat >= min_count) >= min_required
  mat <- mat[keep_genes, , drop = FALSE]

  dge <- DGEList(counts = mat)
  dge <- calcNormFactors(dge)

  form_terms <- c(available_design_cols, "group")
  form <- as.formula(paste("~", paste(form_terms, collapse = " + ")))
  design <- model.matrix(form, data = as.data.frame(labels))

  v <- voom(dge, design, plot = FALSE)
  fit <- lmFit(v, design)
  fit <- eBayes(fit)

  coef_name <- paste0("group", contrast_case)
  tt <- topTable(fit, coef = coef_name, number = Inf, sort.by = "P")
  out <- as.data.table(tt, keep.rownames = "gene")
  setnames(out, old = c("P.Value", "adj.P.Val"), new = c("pvalue", "padj"))
  setorder(out, pvalue)

  all_path <- file.path(out_dir, "limma_results_all.csv")
  sig_path <- file.path(out_dir, "limma_results_significant.csv")
  summary_path <- file.path(out_dir, "limma_run_summary.txt")

  fwrite(out, all_path)
  fwrite(out[!is.na(padj) & padj <= fdr_alpha], sig_path)

  summary_lines <- c(
    paste0("counts_path: ", counts_path),
    paste0("labels_path: ", labels_path),
    paste0("samples_used: ", nrow(labels)),
    paste0("group_counts: ", paste(names(table(labels$group)), as.integer(table(labels$group)), sep = "=", collapse = ", ")),
    paste0("genes_tested: ", nrow(out)),
    paste0("min_pvalue: ", signif(min(out$pvalue, na.rm = TRUE), 6)),
    paste0("min_padj: ", signif(min(out$padj, na.rm = TRUE), 6)),
    paste0("significant_genes_fdr_", fdr_alpha, ": ", sum(!is.na(out$padj) & out$padj <= fdr_alpha))
  )
  writeLines(summary_lines, summary_path)

  cat("Wrote:\n")
  cat("- ", all_path, "\n", sep = "")
  cat("- ", sig_path, "\n", sep = "")
  cat("- ", summary_path, "\n", sep = "")
}

main()
