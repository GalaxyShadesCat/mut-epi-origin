#!/usr/bin/env Rscript

#' Partition limma hits into immune-associated versus non-immune candidates.
#'
#' This script recreates the step-5 limma-voom model used in the follow-up run,
#' annotates significant genes with immune evidence, and reruns limma including
#' an expression-derived immune score covariate for sensitivity testing.

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(limma)
  library(msigdbr)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

normalise_tcga_id <- function(x, prefix_len = 15L) {
  y <- toupper(trimws(as.character(x)))
  y <- gsub("\\.", "-", y)
  substr(y, 1, prefix_len)
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

derive_sample_scores <- function(results_path) {
  dt <- fread(results_path)
  dt <- dt[
    track_strategy == "counts_raw" &
      as.numeric(counts_raw_bin) == 500000
  ]
  dt[, sample_raw := vapply(selected_sample_ids, first_sample_from_selected, character(1))]
  dt <- dt[!is.na(sample_raw) & sample_raw != ""]

  normal_col <- "spearman_r_linear_resid_foxa2_normal_pos_mean_weighted"
  abnormal_col <- "spearman_r_linear_resid_foxa2_abnormal_zero_mean_weighted"
  dt[[normal_col]] <- -1.0 * as.numeric(dt[[normal_col]])
  dt[[abnormal_col]] <- -1.0 * as.numeric(dt[[abnormal_col]])

  agg <- dt[
    ,
    .(
      normal_score = mean(get(normal_col), na.rm = TRUE),
      abnormal_score = mean(get(abnormal_col), na.rm = TRUE)
    ),
    by = .(sample = normalise_tcga_id(sample_raw))
  ]
  agg[, best_cell_state := ifelse(abnormal_score > normal_score, "foxa2_abnormal_zero", "foxa2_normal_pos")]
  agg
}

load_model_data <- function(counts_path, metadata_path, sample_scores) {
  counts_dt <- fread(counts_path)
  gene_col <- names(counts_dt)[[1]]
  genes <- as.character(counts_dt[[gene_col]])
  mat <- as.matrix(counts_dt[, -1, with = FALSE])
  storage.mode(mat) <- "numeric"
  rownames(mat) <- genes

  sample_norm <- normalise_tcga_id(colnames(mat))
  groups <- split(seq_along(sample_norm), sample_norm)
  collapsed <- lapply(groups, function(idx) {
    if (length(idx) == 1) {
      mat[, idx]
    } else {
      rowSums(mat[, idx, drop = FALSE])
    }
  })
  mat <- do.call(cbind, collapsed)
  colnames(mat) <- names(groups)

  meta <- fread(metadata_path)
  meta <- meta[, .(tumour_sample_submitter_id, age_at_diagnosis, gender, ajcc_pathologic_stage)]
  meta[, sample := normalise_tcga_id(tumour_sample_submitter_id)]
  meta <- meta[!duplicated(sample)]
  meta <- meta[, .(sample, age_at_diagnosis, gender, ajcc_pathologic_stage)]

  cd <- data.table(sample = colnames(mat))
  cd <- merge(cd, sample_scores, by = "sample", all.x = TRUE)
  cd <- merge(cd, meta, by = "sample", all.x = TRUE)

  for (nm in c("age_at_diagnosis", "gender", "ajcc_pathologic_stage")) {
    cd[[nm]] <- vapply(cd[[nm]], normalise_missing_token, character(1))
  }
  cd[, age_at_diagnosis := as.numeric(age_at_diagnosis)]
  keep <- complete.cases(cd[, .(age_at_diagnosis, gender, ajcc_pathologic_stage, best_cell_state)])
  cd <- cd[keep]
  mat <- mat[, cd$sample, drop = FALSE]

  keep_genes <- rowSums(mat >= 5) >= ceiling(0.05 * ncol(mat))
  mat <- mat[keep_genes, , drop = FALSE]

  cd[, gender := factor(gender)]
  cd[, ajcc_pathologic_stage := factor(make.names(ajcc_pathologic_stage))]
  cd[, best_cell_state := factor(best_cell_state, levels = c("foxa2_normal_pos", "foxa2_abnormal_zero"))]
  cd[, age_scaled := as.numeric(scale(age_at_diagnosis))]

  list(counts = mat, col_data = cd)
}

run_limma <- function(counts, cd, include_immune_score = FALSE, immune_score = NULL) {
  dge <- DGEList(counts = counts)
  dge <- calcNormFactors(dge)

  design_df <- as.data.frame(cd)
  if (include_immune_score) {
    if (!is.null(names(immune_score)) && all(cd$sample %in% names(immune_score))) {
      design_df$immune_score <- as.numeric(immune_score[cd$sample])
    } else {
      design_df$immune_score <- as.numeric(immune_score)
    }
    if (length(design_df$immune_score) != nrow(design_df)) {
      stop("Immune-score length does not match sample count in design.")
    }
    design <- model.matrix(~ age_scaled + gender + ajcc_pathologic_stage + immune_score + best_cell_state, data = design_df)
  } else {
    design <- model.matrix(~ age_scaled + gender + ajcc_pathologic_stage + best_cell_state, data = design_df)
  }

  v <- voom(dge, design, plot = FALSE)
  fit <- lmFit(v, design)
  fit <- eBayes(fit)
  coef_name <- "best_cell_statefoxa2_abnormal_zero"
  tt <- topTable(fit, coef = coef_name, number = Inf, sort.by = "P")
  tab <- as.data.table(tt, keep.rownames = "gene")
  setnames(tab, old = c("P.Value", "adj.P.Val"), new = c("pvalue", "padj"))
  list(tab = tab, logcpm = v$E)
}

build_immune_score <- function(logcpm_matrix) {
  immune_hallmarks <- c(
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_IL2_STAT5_SIGNALING"
  )

  hallmark <- as.data.table(msigdbr(species = "Homo sapiens", collection = "H"))
  hallmark <- hallmark[gs_name %in% immune_hallmarks]
  immune_symbols <- unique(hallmark$gene_symbol)

  ensembl <- sub("\\..*$", "", rownames(logcpm_matrix))
  symbols <- mapIds(
    org.Hs.eg.db,
    keys = ensembl,
    keytype = "ENSEMBL",
    column = "SYMBOL",
    multiVals = "first"
  )
  keep <- !is.na(symbols) & symbols %in% immune_symbols
  if (!any(keep)) {
    stop("No immune hallmark genes overlapped for immune-score construction.")
  }
  z <- t(scale(t(logcpm_matrix[keep, , drop = FALSE])))
  colMeans(z, na.rm = TRUE)
}

annotate_hits <- function(limma_tab) {
  dt <- copy(limma_tab)
  dt[, ensembl := sub("\\..*$", "", gene)]
  dt[, symbol := as.character(mapIds(
    org.Hs.eg.db,
    keys = ensembl,
    keytype = "ENSEMBL",
    column = "SYMBOL",
    multiVals = "first"
  ))]

  immune_marker_regex <- paste(
    c("^IG[HKL][VDJMCAG]", "^TR[ABDG][VJCD]", "^HLA-", "^CD[0-9]", "^MS4A1$", "^PTPRC$",
      "^ICOS$", "^TNFSF15$", "^CXCL", "^CCL", "^IFIT", "^IFI", "^LST1$", "^TYROBP$"),
    collapse = "|"
  )

  hallmark <- as.data.table(msigdbr(species = "Homo sapiens", collection = "H"))
  immune_hallmarks <- c(
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_IL2_STAT5_SIGNALING"
  )
  immune_hallmark_symbols <- unique(hallmark[gs_name %in% immune_hallmarks]$gene_symbol)

  dt[, immune_marker_pattern := grepl(immune_marker_regex, symbol, perl = TRUE)]
  dt[, in_immune_hallmark := symbol %in% immune_hallmark_symbols]
  dt[, likely_immune_associated := immune_marker_pattern | in_immune_hallmark]
  dt[, hit_partition := fifelse(likely_immune_associated, "likely_immune_associated", "candidate_tumour_intrinsic")]
  dt
}

main <- function() {
  out_dir <- "outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/de_followups_stepwise"
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  counts_path <- "data/raw/rna/TCGA-LIHC.star_counts.tsv"
  results_path <- "outputs/experiments/lihc_foxa2_top4_all_samples_per_sample_merged/results.csv"
  metadata_path <- "data/derived/master_metadata.csv"

  sample_scores <- derive_sample_scores(results_path)
  model_data <- load_model_data(counts_path, metadata_path, sample_scores)
  counts <- model_data$counts
  cd <- model_data$col_data

  base_fit <- run_limma(counts, cd, include_immune_score = FALSE)
  base_tab <- base_fit$tab
  base_tab_annot <- annotate_hits(base_tab)
  base_sig <- base_tab_annot[!is.na(padj) & padj <= 0.05]
  fwrite(base_sig, file.path(out_dir, "step5_limma_significant_genes_annotated.csv"))

  immune_score <- build_immune_score(base_fit$logcpm)
  immune_fit <- run_limma(counts, cd, include_immune_score = TRUE, immune_score = immune_score)
  immune_tab <- immune_fit$tab
  fwrite(immune_tab, file.path(out_dir, "step5_limma_immune_adjusted_results.csv"))

  comp <- merge(
    base_tab[, .(gene, logFC_base = logFC, pvalue_base = pvalue, padj_base = padj)],
    immune_tab[, .(gene, logFC_immune_adj = logFC, pvalue_immune_adj = pvalue, padj_immune_adj = padj)],
    by = "gene",
    all = TRUE
  )
  comp <- merge(comp, base_tab_annot[, .(gene, symbol, hit_partition)], by = "gene", all.x = TRUE)
  comp[, sig_base := !is.na(padj_base) & padj_base <= 0.05]
  comp[, sig_immune_adj := !is.na(padj_immune_adj) & padj_immune_adj <= 0.05]
  comp[, delta_log10_p := -log10(pvalue_immune_adj) - (-log10(pvalue_base))]
  fwrite(comp, file.path(out_dir, "step5_limma_vs_immune_adjusted_comparison.csv"))

  summary_lines <- c(
    "limma hit partition and immune-adjustment sensitivity",
    paste0("samples_used: ", nrow(cd)),
    paste0("groups: ", paste(names(table(cd$best_cell_state)), as.integer(table(cd$best_cell_state)), sep = "=", collapse = ", ")),
    "",
    paste0("baseline_limma_significant: ", sum(base_sig$padj <= 0.05, na.rm = TRUE)),
    paste0("baseline_likely_immune_associated: ", sum(base_sig$hit_partition == "likely_immune_associated", na.rm = TRUE)),
    paste0("baseline_candidate_tumour_intrinsic: ", sum(base_sig$hit_partition == "candidate_tumour_intrinsic", na.rm = TRUE)),
    "",
    paste0("immune_adjusted_limma_significant: ", sum(!is.na(immune_tab$padj) & immune_tab$padj <= 0.05)),
    paste0("baseline_hits_retained_after_immune_adjustment: ", sum(comp$sig_base & comp$sig_immune_adj, na.rm = TRUE)),
    paste0("baseline_hits_lost_after_immune_adjustment: ", sum(comp$sig_base & !comp$sig_immune_adj, na.rm = TRUE)),
    "",
    "Top baseline significant hits:",
    paste(
      base_sig[1:min(10L, .N), paste0(symbol, " (padj=", signif(padj, 3), ", ", hit_partition, ")")],
      collapse = "; "
    )
  )
  writeLines(summary_lines, file.path(out_dir, "step5_limma_hit_partition_summary.txt"))

  message("Wrote limma partition outputs to: ", out_dir)
}

main()
