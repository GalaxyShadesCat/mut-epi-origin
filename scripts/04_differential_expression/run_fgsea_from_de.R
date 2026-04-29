#!/usr/bin/env Rscript

#' Run preranked gene set enrichment from differential-expression results.
#'
#' This script converts a differential-expression table into a ranked gene
#' vector and runs fgsea on selected gene sets. It supports rank metrics based
#' on test statistic, signed significance, or fold-change weighted significance.
#'
#' Example:
#' Rscript scripts/04_differential_expression/run_fgsea_from_de.R \
#'   --de-results outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step6_deseq2_limma_gene_pool_results.csv \
#'   --rank-metric logfc_times_neglog10p \
#'   --gene-sets-source msigdb \
#'   --msigdb-collection H \
#'   --out-prefix outputs/experiments/lihc_foxa2_all_samples/de_followups_stepwise/step7_fgsea_from_deseq2_limma_pool

suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(msigdbr)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

usage <- function() {
  cat(
    paste(
      "Usage:",
      "Rscript scripts/04_differential_expression/run_fgsea_from_de.R",
      "  --de-results <path>",
      "  [--gene-col gene]",
      "  [--logfc-col log2FoldChange]",
      "  [--pvalue-col pvalue]",
      "  [--stat-col stat]",
      "  [--rank-metric stat|sign_logfc_neglog10p|logfc_times_neglog10p]",
      "  [--convert-ensembl-to-symbol true|false]",
      "  [--gene-sets-source msigdb|gmt]",
      "  [--msigdb-species \"Homo sapiens\"]",
      "  [--msigdb-collection H]",
      "  [--msigdb-subcollection <value>]",
      "  [--gmt-path <path>]",
      "  [--min-size 15]",
      "  [--max-size 500]",
      "  [--nperm-simple 10000]",
      "  [--seed 1]",
      "  [--out-prefix <prefix>]",
      sep = "\n"
    ),
    "\n"
  )
}

parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  if (length(raw) == 0L || any(raw %in% c("-h", "--help"))) {
    usage()
    quit(status = 0L)
  }

  defaults <- list(
    "gene-col" = "gene",
    "logfc-col" = "log2FoldChange",
    "pvalue-col" = "pvalue",
    "stat-col" = "stat",
    "rank-metric" = "stat",
    "convert-ensembl-to-symbol" = "true",
    "gene-sets-source" = "msigdb",
    "msigdb-species" = "Homo sapiens",
    "msigdb-collection" = "H",
    "msigdb-subcollection" = NA_character_,
    "gmt-path" = NA_character_,
    "min-size" = "15",
    "max-size" = "500",
    "nperm-simple" = "10000",
    "seed" = "1",
    "out-prefix" = NA_character_
  )

  parsed <- defaults
  i <- 1L
  while (i <= length(raw)) {
    key <- raw[[i]]
    if (!startsWith(key, "--")) {
      stop("Invalid argument: ", key)
    }
    key <- substring(key, 3L)
    if (!key %in% c(names(defaults), "de-results")) {
      stop("Unknown argument: --", key)
    }
    if (i == length(raw)) {
      stop("Missing value for --", key)
    }
    parsed[[key]] <- raw[[i + 1L]]
    i <- i + 2L
  }

  if (is.null(parsed[["de-results"]])) {
    stop("Required argument missing: --de-results")
  }
  parsed
}

to_bool <- function(x) {
  value <- tolower(trimws(as.character(x)))
  if (value %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }
  stop("Cannot parse boolean value: ", x)
}

build_rank_scores <- function(
  dt,
  gene_col,
  logfc_col,
  pvalue_col,
  stat_col,
  rank_metric
) {
  out <- dt[, .(gene_raw = as.character(get(gene_col)))]
  out <- out[!is.na(gene_raw) & gene_raw != ""]

  if (rank_metric == "stat") {
    out[, score := as.numeric(dt[[stat_col]])]
    out <- out[!is.na(score)]
    return(out)
  }

  logfc <- as.numeric(dt[[logfc_col]])
  pval <- as.numeric(dt[[pvalue_col]])
  pval[is.na(pval)] <- NA_real_
  pval[pval <= 0] <- .Machine$double.xmin

  if (rank_metric == "sign_logfc_neglog10p") {
    score <- sign(logfc) * (-log10(pval))
  } else if (rank_metric == "logfc_times_neglog10p") {
    score <- logfc * (-log10(pval))
  } else {
    stop("Unsupported rank metric: ", rank_metric)
  }

  out[, score := score]
  out <- out[!is.na(score)]
  out
}

convert_ensembl_to_symbol <- function(genes) {
  ensembl <- sub("\\..*$", "", genes)
  symbols <- mapIds(
    org.Hs.eg.db,
    keys = ensembl,
    keytype = "ENSEMBL",
    column = "SYMBOL",
    multiVals = "first"
  )
  as.character(symbols)
}

load_pathways <- function(source, species, collection, subcollection, gmt_path) {
  if (source == "msigdb") {
    if (!is.na(subcollection) && nzchar(subcollection)) {
      msig <- as.data.table(msigdbr(
        species = species,
        collection = collection,
        subcollection = subcollection
      ))
    } else {
      msig <- as.data.table(msigdbr(species = species, collection = collection))
    }
    pathways <- split(msig$gene_symbol, msig$gs_name)
    return(pathways)
  }

  if (source == "gmt") {
    if (is.na(gmt_path) || !nzchar(gmt_path)) {
      stop("--gmt-path is required when --gene-sets-source gmt")
    }
    return(gmtPathways(gmt_path))
  }

  stop("Unsupported gene set source: ", source)
}

main <- function() {
  args <- parse_args()

  de_results <- args[["de-results"]]
  gene_col <- args[["gene-col"]]
  logfc_col <- args[["logfc-col"]]
  pvalue_col <- args[["pvalue-col"]]
  stat_col <- args[["stat-col"]]
  rank_metric <- args[["rank-metric"]]
  convert_ids <- to_bool(args[["convert-ensembl-to-symbol"]])
  gene_sets_source <- args[["gene-sets-source"]]
  msigdb_species <- args[["msigdb-species"]]
  msigdb_collection <- args[["msigdb-collection"]]
  msigdb_subcollection <- args[["msigdb-subcollection"]]
  gmt_path <- args[["gmt-path"]]
  min_size <- as.integer(args[["min-size"]])
  max_size <- as.integer(args[["max-size"]])
  nperm_simple <- as.integer(args[["nperm-simple"]])
  seed <- as.integer(args[["seed"]])
  out_prefix <- args[["out-prefix"]]

  if (!file.exists(de_results)) {
    stop("DE results file not found: ", de_results)
  }

  if (is.na(out_prefix) || !nzchar(out_prefix)) {
    out_prefix <- sub("\\.[^.]+$", "", de_results)
    out_prefix <- paste0(out_prefix, "_fgsea_", rank_metric)
  }

  dt <- fread(de_results)
  required <- c(gene_col)
  if (rank_metric == "stat") {
    required <- c(required, stat_col)
  } else {
    required <- c(required, logfc_col, pvalue_col)
  }
  missing_cols <- setdiff(required, names(dt))
  if (length(missing_cols) > 0L) {
    stop("Missing required columns in DE results: ", paste(missing_cols, collapse = ", "))
  }

  ranked <- build_rank_scores(
    dt = dt,
    gene_col = gene_col,
    logfc_col = logfc_col,
    pvalue_col = pvalue_col,
    stat_col = stat_col,
    rank_metric = rank_metric
  )

  if (convert_ids) {
    ranked[, gene := convert_ensembl_to_symbol(gene_raw)]
    ranked <- ranked[!is.na(gene) & gene != ""]
  } else {
    ranked[, gene := gene_raw]
  }

  # Keep one score per gene, prioritising the strongest absolute signal.
  setorderv(ranked, cols = c("gene", "score"), order = c(1L, -1L))
  ranked <- ranked[, .SD[which.max(abs(score))], by = gene]
  setorder(ranked, -score)

  stats <- ranked$score
  names(stats) <- ranked$gene

  pathways <- load_pathways(
    source = gene_sets_source,
    species = msigdb_species,
    collection = msigdb_collection,
    subcollection = msigdb_subcollection,
    gmt_path = gmt_path
  )

  set.seed(seed)
  fg <- fgseaMultilevel(
    pathways = pathways,
    stats = stats,
    minSize = min_size,
    maxSize = max_size,
    nPermSimple = nperm_simple
  )

  fg <- as.data.table(fg)
  setorder(fg, padj, pval)

  ranked_out <- paste0(out_prefix, "_ranked_genes.csv")
  fg_out <- paste0(out_prefix, "_fgsea_results.csv")
  summary_out <- paste0(out_prefix, "_fgsea_summary.txt")

  fwrite(ranked[, .(gene, score)], ranked_out)
  fwrite(fg, fg_out)

  n_sig <- sum(!is.na(fg$padj) & fg$padj <= 0.05)
  best_pathway <- if (nrow(fg) > 0L) fg$pathway[[1]] else NA_character_
  best_padj <- if (nrow(fg) > 0L) fg$padj[[1]] else NA_real_

  summary_lines <- c(
    "Preranked fgsea summary",
    paste0("de_results: ", de_results),
    paste0("rank_metric: ", rank_metric),
    paste0("convert_ensembl_to_symbol: ", convert_ids),
    paste0("gene_sets_source: ", gene_sets_source),
    paste0("genes_ranked: ", length(stats)),
    paste0("pathways_tested: ", nrow(fg)),
    paste0("nperm_simple: ", nperm_simple),
    paste0("significant_pathways_fdr_0.05: ", n_sig),
    paste0("best_pathway: ", best_pathway),
    paste0("best_pathway_padj: ", signif(best_padj, 4))
  )
  writeLines(summary_lines, summary_out)

  message("Wrote:")
  message("- ", ranked_out)
  message("- ", fg_out)
  message("- ", summary_out)
}

main()
