#!/usr/bin/env Rscript

#' Stepwise follow-up analyses for FOXA2-derived LIHC expression signals.
#'
#' This script runs six follow-up checks in sequence:
#' 1. DESeq2 with a continuous inferred score predictor.
#' 2. Hallmark pathway enrichment with fgsea on ranked continuous-model results.
#' 3. Continuous-model DESeq2 adjusted with surrogate variables (SVA).
#' 4. Continuous-model DESeq2 within a Hallmark-targeted gene universe.
#' 5. limma-voom sensitivity analysis for the binary-group comparison.
#' 6. Label-quality diagnostics from PCA separation and classification AUC.
#'
#' Outputs are written to one directory with a single text report.

suppressPackageStartupMessages({
  library(data.table)
  library(DESeq2)
  library(msigdbr)
  library(fgsea)
  library(sva)
  library(limma)
  library(edgeR)
  library(pROC)
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
  agg[, score_delta := abnormal_score - normal_score]
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
  cd[, score_delta := as.numeric(score_delta)]
  cd[, age_at_diagnosis := as.numeric(age_at_diagnosis)]
  keep <- complete.cases(cd[, .(age_at_diagnosis, gender, ajcc_pathologic_stage, score_delta)])
  cd <- cd[keep]

  mat <- mat[, cd$sample, drop = FALSE]
  keep_genes <- rowSums(mat >= 5) >= ceiling(0.05 * ncol(mat))
  mat <- mat[keep_genes, , drop = FALSE]

  cd[, gender := factor(gender)]
  cd[, ajcc_pathologic_stage := factor(make.names(ajcc_pathologic_stage))]
  cd[, best_cell_state := factor(best_cell_state)]
  cd[, score_delta_scaled := as.numeric(scale(score_delta))]
  cd[, age_scaled := as.numeric(scale(age_at_diagnosis))]

  list(counts = mat, col_data = cd)
}

run_continuous_deseq <- function(counts, cd, add_sv = FALSE, n_sv = NULL, gene_subset = NULL) {
  use_counts <- counts
  if (!is.null(gene_subset)) {
    use_counts <- use_counts[rownames(use_counts) %in% gene_subset, , drop = FALSE]
  }

  design_df <- data.frame(
    row.names = cd$sample,
    age_scaled = cd$age_scaled,
    gender = cd$gender,
    ajcc_pathologic_stage = cd$ajcc_pathologic_stage,
    score_delta_scaled = cd$score_delta_scaled
  )

  if (add_sv) {
    full_mod <- model.matrix(~ age_scaled + gender + ajcc_pathologic_stage + score_delta_scaled, data = design_df)
    null_mod <- model.matrix(~ age_scaled + gender + ajcc_pathologic_stage, data = design_df)
    n_sv_est <- if (is.null(n_sv)) {
      suppressWarnings(num.sv(use_counts, full_mod, method = "be"))
    } else {
      as.integer(n_sv)
    }
    n_sv_est <- max(0L, min(3L, n_sv_est))
    if (n_sv_est > 0L) {
      svobj <- sva(use_counts, mod = full_mod, mod0 = null_mod, n.sv = n_sv_est)
      for (i in seq_len(n_sv_est)) {
        design_df[[paste0("sv", i)]] <- svobj$sv[, i]
      }
    }
  }

  form_terms <- c("age_scaled", "gender", "ajcc_pathologic_stage")
  sv_terms <- grep("^sv[0-9]+$", names(design_df), value = TRUE)
  form_terms <- c(form_terms, sv_terms, "score_delta_scaled")
  design_formula <- as.formula(paste("~", paste(form_terms, collapse = " + ")))

  dds <- DESeqDataSetFromMatrix(
    countData = round(use_counts),
    colData = design_df,
    design = design_formula
  )
  dds <- DESeq(dds, fitType = "local")
  res <- results(dds, name = "score_delta_scaled")
  tab <- as.data.table(as.data.frame(res), keep.rownames = "gene")
  setorder(tab, pvalue)

  list(
    tab = tab,
    n_genes = nrow(tab),
    n_sig = sum(!is.na(tab$padj) & tab$padj <= 0.05),
    min_p = suppressWarnings(min(tab$pvalue, na.rm = TRUE)),
    min_padj = suppressWarnings(min(tab$padj, na.rm = TRUE)),
    n_sv = length(sv_terms)
  )
}

run_fgsea_hallmark <- function(res_tab) {
  rank_dt <- copy(res_tab)
  rank_dt <- rank_dt[!is.na(stat)]
  rank_dt[, ensembl := sub("\\..*$", "", gene)]
  symbols <- mapIds(
    org.Hs.eg.db,
    keys = rank_dt$ensembl,
    keytype = "ENSEMBL",
    column = "SYMBOL",
    multiVals = "first"
  )
  rank_dt[, symbol := as.character(symbols[ensembl])]
  rank_dt <- rank_dt[!is.na(symbol) & symbol != ""]
  rank_dt <- rank_dt[!duplicated(symbol)]
  stats <- rank_dt$stat
  names(stats) <- rank_dt$symbol
  stats <- sort(stats, decreasing = TRUE)

  hallmark <- msigdbr(species = "Homo sapiens", category = "H")
  pathways <- split(hallmark$gene_symbol, hallmark$gs_name)
  fg <- fgsea(pathways = pathways, stats = stats, minSize = 15, maxSize = 500)
  fg <- as.data.table(fg)
  setorder(fg, padj, pval)
  fg
}

run_limma_binary <- function(counts, cd) {
  group <- factor(cd$best_cell_state, levels = c("foxa2_normal_pos", "foxa2_abnormal_zero"))
  dge <- DGEList(counts = counts)
  dge <- calcNormFactors(dge)

  design <- model.matrix(
    ~ age_scaled + gender + ajcc_pathologic_stage + group,
    data = as.data.frame(cd)
  )
  v <- voom(dge, design, plot = FALSE)
  fit <- lmFit(v, design)
  fit <- eBayes(fit)
  tt <- topTable(fit, coef = "groupfoxa2_abnormal_zero", number = Inf, sort.by = "P")
  tab <- as.data.table(tt, keep.rownames = "gene")
  setnames(tab, old = c("P.Value", "adj.P.Val"), new = c("pvalue", "padj"))
  list(
    tab = tab,
    n_genes = nrow(tab),
    n_sig = sum(tab$padj <= 0.05, na.rm = TRUE),
    min_p = min(tab$pvalue, na.rm = TRUE),
    min_padj = min(tab$padj, na.rm = TRUE)
  )
}

run_label_quality <- function(counts, cd) {
  dds0 <- DESeqDataSetFromMatrix(
    countData = round(counts),
    colData = data.frame(row.names = cd$sample, group = cd$best_cell_state),
    design = ~ 1
  )
  vst_mat <- assay(vst(dds0, blind = TRUE))
  vars <- apply(vst_mat, 1, var)
  top <- names(sort(vars, decreasing = TRUE))[1:min(3000L, length(vars))]
  pcs <- prcomp(t(vst_mat[top, , drop = FALSE]), center = TRUE, scale. = TRUE)
  pc_df <- as.data.table(pcs$x[, 1:10, drop = FALSE])
  pc_df[, group := cd$best_cell_state]
  pc_df[, y := as.integer(group == "foxa2_abnormal_zero")]

  fit <- glm(y ~ ., data = as.data.frame(pc_df[, !"group"]), family = binomial())
  prob <- predict(fit, type = "response")
  auc <- as.numeric(pROC::auc(pc_df$y, prob))

  grp1 <- pc_df[group == "foxa2_normal_pos", .(PC1, PC2, PC3)]
  grp2 <- pc_df[group == "foxa2_abnormal_zero", .(PC1, PC2, PC3)]
  cent1 <- colMeans(grp1)
  cent2 <- colMeans(grp2)
  centroid_distance <- sqrt(sum((cent1 - cent2)^2))

  list(
    auc = auc,
    centroid_distance_pc123 = centroid_distance,
    pc1_var = summary(pcs)$importance[2, 1],
    pc2_var = summary(pcs)$importance[2, 2]
  )
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

  cont <- run_continuous_deseq(counts, cd, add_sv = FALSE)
  fwrite(cont$tab, file.path(out_dir, "step1_continuous_deseq_results.csv"))

  fg <- run_fgsea_hallmark(cont$tab)
  fwrite(fg, file.path(out_dir, "step2_fgsea_hallmark.csv"))

  cont_sva <- run_continuous_deseq(counts, cd, add_sv = TRUE)
  fwrite(cont_sva$tab, file.path(out_dir, "step3_continuous_deseq_with_sva_results.csv"))

  hallmark <- msigdbr(species = "Homo sapiens", category = "H")
  hallmark_symbols <- unique(hallmark$gene_symbol)
  ensembl_keys <- keys(org.Hs.eg.db, keytype = "ENSEMBL")
  map_dt <- data.table(
    ensembl = ensembl_keys,
    symbol = as.character(mapIds(org.Hs.eg.db, keys = ensembl_keys, keytype = "ENSEMBL", column = "SYMBOL", multiVals = "first"))
  )
  map_dt <- map_dt[!is.na(symbol) & symbol %in% hallmark_symbols]
  hallmark_ensembl <- unique(map_dt$ensembl)
  gene_subset <- sub("\\..*$", "", rownames(counts))
  keep <- hallmark_ensembl %in% gene_subset
  hallmark_ensembl <- hallmark_ensembl[keep]
  row_lookup <- data.table(gene = rownames(counts), ensembl = sub("\\..*$", "", rownames(counts)))
  gene_subset_rows <- row_lookup[ensembl %in% hallmark_ensembl]$gene

  cont_hallmark <- run_continuous_deseq(counts, cd, add_sv = FALSE, gene_subset = gene_subset_rows)
  fwrite(cont_hallmark$tab, file.path(out_dir, "step4_continuous_deseq_hallmark_universe_results.csv"))

  lim <- run_limma_binary(counts, cd)
  fwrite(lim$tab, file.path(out_dir, "step5_limma_voom_binary_results.csv"))

  lbl <- run_label_quality(counts, cd)

  report_lines <- c(
    "Stepwise follow-up report",
    paste0("samples_used: ", nrow(cd)),
    paste0("groups: ", paste(names(table(cd$best_cell_state)), as.integer(table(cd$best_cell_state)), sep = "=", collapse = ", ")),
    "",
    "Step 1: Continuous DESeq2",
    paste0("genes_tested: ", cont$n_genes),
    paste0("significant_genes_fdr_0.05: ", cont$n_sig),
    paste0("min_pvalue: ", signif(cont$min_p, 4)),
    paste0("min_padj: ", signif(cont$min_padj, 4)),
    "",
    "Step 2: Hallmark fgsea",
    paste0("pathways_tested: ", nrow(fg)),
    paste0("significant_pathways_fdr_0.05: ", sum(!is.na(fg$padj) & fg$padj <= 0.05)),
    paste0("best_pathway: ", ifelse(nrow(fg) > 0, fg$pathway[[1]], "NA")),
    paste0("best_pathway_padj: ", ifelse(nrow(fg) > 0, signif(fg$padj[[1]], 4), NA)),
    "",
    "Step 3: Continuous DESeq2 + SVA",
    paste0("surrogate_variables_used: ", cont_sva$n_sv),
    paste0("genes_tested: ", cont_sva$n_genes),
    paste0("significant_genes_fdr_0.05: ", cont_sva$n_sig),
    paste0("min_pvalue: ", signif(cont_sva$min_p, 4)),
    paste0("min_padj: ", signif(cont_sva$min_padj, 4)),
    "",
    "Step 4: Hallmark-targeted gene universe",
    paste0("genes_tested: ", cont_hallmark$n_genes),
    paste0("significant_genes_fdr_0.05: ", cont_hallmark$n_sig),
    paste0("min_pvalue: ", signif(cont_hallmark$min_p, 4)),
    paste0("min_padj: ", signif(cont_hallmark$min_padj, 4)),
    "",
    "Step 5: limma-voom binary sensitivity",
    paste0("genes_tested: ", lim$n_genes),
    paste0("significant_genes_fdr_0.05: ", lim$n_sig),
    paste0("min_pvalue: ", signif(lim$min_p, 4)),
    paste0("min_padj: ", signif(lim$min_padj, 4)),
    "",
    "Step 6: Label quality",
    paste0("pc1_variance_explained: ", signif(lbl$pc1_var, 4)),
    paste0("pc2_variance_explained: ", signif(lbl$pc2_var, 4)),
    paste0("centroid_distance_pc1_pc2_pc3: ", signif(lbl$centroid_distance_pc123, 4)),
    paste0("pc10_logistic_auc: ", signif(lbl$auc, 4))
  )

  writeLines(report_lines, file.path(out_dir, "stepwise_report.txt"))
  message("Wrote follow-up outputs to: ", out_dir)
}

main()
