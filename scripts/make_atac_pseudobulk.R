#!/usr/bin/env Rscript
# ============================================================
# scripts/make_atac_pseudobulk.R
#
# Peak-resolution bigWig tracks from ATAC peak-by-cell counts
#
# This version outputs BOTH:
#
# 1. Unfiltered condition tracks
#    - Groups: selected CellTypeSR × Condition
#    - Default selection below currently keeps hepatocytes only
#    - Example outputs:
#        Hepatocyte__AC
#        Hepatocyte__AH
#        Hepatocyte__Normal
#
# 2. FOXA2-filtered hepatocyte reference tracks
#    - Option A gating:
#        * normal reference   = Normal hepatocytes with FOXA2 > 0
#        * abnormal reference = diseased hepatocytes (AC + AH) with FOXA2 == 0
#    - Example outputs:
#        normal_FOXA2_pos
#        abnormal_FOXA2_zero
#
# Common behaviour:
# - Downsamples per group so comparable tracks use the same number of cells
# - Genome: hg38 (native)
# - Also produces hg19 bigWig via UCSC liftOver
# - RNA assay is used only for FOXA2 gating
# - ATAC assay is used to generate the pseudobulk tracks
# - Does NOT require fragment files
# - Signal is constant within each peak interval and zero elsewhere
# - Uses sparse operations end-to-end to avoid densifying the peak x cell matrix
# - hg19 liftOver can introduce overlapping intervals; we merge overlaps with bedtools
#
# Outputs:
#   data/processed/ATAC-seq/GSE281574/hg38/<track>.bigWig
#   data/processed/ATAC-seq/GSE281574/hg19/<track>.bigWig
#
# Additional QC outputs:
#   data/processed/ATAC-seq/GSE281574/qc/original_group_summary.csv
#   data/processed/ATAC-seq/GSE281574/qc/original_track_build_log.csv
#   data/processed/ATAC-seq/GSE281574/qc/foxa2_reference_cell_assignments.csv
#   data/processed/ATAC-seq/GSE281574/qc/foxa2_reference_summary.csv
#   data/processed/ATAC-seq/GSE281574/qc/foxa2_track_build_log.csv
#   data/processed/ATAC-seq/GSE281574/qc/normal_FOXA2_pos_cells.txt
#   data/processed/ATAC-seq/GSE281574/qc/abnormal_FOXA2_zero_cells.txt
#
# Stable genome resources (downloaded once, reused):
#   data/raw/reference/hg38/hg38.chrom.sizes
#   data/raw/reference/hg19/hg19.chrom.sizes
#   data/raw/reference/hg38/liftOver/hg38ToHg19.over.chain
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(Signac)
  library(Matrix)
  library(SeuratObject)
  library(GenomicRanges)
  library(IRanges)
})

# ----------------------------
# Project root (assume script is run from repo root)
# ----------------------------
PROJECT_DIR <- normalizePath(getwd(), mustWork = TRUE)

# ----------------------------
# Dataset identifier
# ----------------------------
DATASET_ID <- "GSE281574"

# ----------------------------
# Input data
# ----------------------------
RDS_PATH <- file.path(
  "data", "raw", "multiome",
  "GSE281574_Liver_Multiome_Seurat_GEO.rds"
)

# ----------------------------
# Output directories
# ----------------------------
OUT_BASE_DIR <- file.path("data", "processed", "ATAC-seq", DATASET_ID)
OUT_DIR_HG38 <- file.path(OUT_BASE_DIR, "hg38")
OUT_DIR_HG19 <- file.path(OUT_BASE_DIR, "hg19")
QC_DIR       <- file.path(OUT_BASE_DIR, "qc")

dir.create(OUT_DIR_HG38, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR_HG19, recursive = TRUE, showWarnings = FALSE)
dir.create(QC_DIR,       recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Stable genome resources
# ----------------------------
GENOME_DIR   <- file.path("data", "raw", "reference")
REF_HG38_DIR <- file.path(GENOME_DIR, "hg38")
REF_HG19_DIR <- file.path(GENOME_DIR, "hg19")
REF_LO_DIR   <- file.path(REF_HG38_DIR, "liftOver")

dir.create(REF_HG38_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REF_HG19_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REF_LO_DIR,   recursive = TRUE, showWarnings = FALSE)

HG38_SIZES_PATH <- file.path(REF_HG38_DIR, "hg38.chrom.sizes")
HG19_SIZES_PATH <- file.path(REF_HG19_DIR, "hg19.chrom.sizes")
CHAIN_PATH      <- file.path(REF_LO_DIR, "hg38ToHg19.over.chain")

# ----------------------------
# Temporary working directory
# ----------------------------
TMP_DIR <- file.path(OUT_BASE_DIR, "_tmp")
dir.create(TMP_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Configuration
# ----------------------------
set.seed(1)

# Unfiltered track settings
CELLTYPES_KEEP   <- c("Hepatocyte")
CONDITIONS_KEEP  <- c("AC", "AH", "Normal")

# Canonical naming map used by unfiltered outputs
CELLTYPE_MAP <- c(
  "T cell"        = "T_Cell",
  "Kupffer Cell"  = "Kupffer_Cell",
  "Plasma Cell"   = "Plasma_Cell"
  # Leave "Hepatocyte", "LSEC", "HSC" unchanged by not listing them
)

# FOXA2 reference settings
HEPATOCYTE_LABEL     <- "Hepatocyte"
NORMAL_CONDITION     <- "Normal"
DISEASED_CONDITIONS  <- c("AC", "AH")
FOXA2_GENE           <- "FOXA2"
NORMAL_GROUP_NAME    <- "normal_FOXA2_pos"
ABNORMAL_GROUP_NAME  <- "abnormal_FOXA2_zero"

# ----------------------------
# Logging
# ----------------------------
original_track_log_df <- data.frame(
  group_id = character(),
  n_cells_used = integer(),
  total_counts = numeric(),
  raw_mean_counts_per_cell = numeric(),
  cpm_sum = numeric(),
  nonzero_peaks = integer(),
  hg19_intervals = integer(),
  stringsAsFactors = FALSE
)

foxa2_track_log_df <- data.frame(
  group_id = character(),
  n_cells_used = integer(),
  total_counts = numeric(),
  raw_mean_counts_per_cell = numeric(),
  cpm_sum = numeric(),
  nonzero_peaks = integer(),
  hg19_intervals = integer(),
  stringsAsFactors = FALSE
)

# ----------------------------
# UCSC download URLs
# ----------------------------
HG38_SIZES_URL <- "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"
HG19_SIZES_URL <- "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes"
CHAIN_URL      <- "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz"

# ----------------------------
# Helpers
# ----------------------------
need_cmd <- function(cmd) {
  if (!nzchar(Sys.which(cmd))) {
    stop("Missing required command in PATH: ", cmd)
  }
}

download_if_missing <- function(url, dest) {
  if (!file.exists(dest)) {
    message("Downloading: ", basename(dest))
    utils::download.file(url, destfile = dest, mode = "wb", quiet = TRUE)
  }
}

safe_name <- function(x) {
  x <- gsub("[[:space:]]+", "", x)
  gsub("[^A-Za-z0-9_\\-\\.]", "", x)
}

read_chrom_sizes <- function(path) {
  df <- utils::read.table(path, sep = "\t", header = FALSE, stringsAsFactors = FALSE)
  colnames(df) <- c("chrom", "size")
  df
}

write_bedgraph <- function(df, path) {
  utils::write.table(
    df,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
}

get_layer_matrix <- function(obj, assay_name, layer_name) {
  assay_obj <- obj[[assay_name]]
  out <- tryCatch(
    SeuratObject::LayerData(assay_obj, layer = layer_name),
    error = function(e1) tryCatch(
      GetAssayData(obj, assay = assay_name, layer = layer_name),
      error = function(e2) stop(
        "Could not retrieve assay='", assay_name,
        "', layer='", layer_name, "'."
      )
    )
  )
  out
}

build_bigwig_pair <- function(group_id, cells_use, counts, peaks_gr,
                              hg38_sizes_path, hg19_sizes_path, chain_path,
                              out_dir_hg38, out_dir_hg19, tmp_dir) {

  message("\n=== ", group_id, " ===")

  fname <- group_id
  bw38 <- file.path(out_dir_hg38, paste0(fname, ".bigWig"))
  bw19 <- file.path(out_dir_hg19, paste0(fname, ".bigWig"))

  if (file.exists(bw38) && file.exists(bw19)) {
    message("Already exists, skipping build")
    return(data.frame(
      group_id = group_id,
      n_cells_used = length(cells_use),
      total_counts = NA_real_,
      raw_mean_counts_per_cell = NA_real_,
      cpm_sum = NA_real_,
      nonzero_peaks = NA_integer_,
      hg19_intervals = NA_integer_,
      stringsAsFactors = FALSE
    ))
  }

  idx <- match(cells_use, colnames(counts))
  missing_cells <- cells_use[is.na(idx)]
  if (length(missing_cells) > 0) {
    stop(
      "Selected cells missing in ATAC counts for group ",
      group_id,
      ". Missing cells: ",
      paste(missing_cells, collapse = ", ")
    )
  }
  if (length(idx) != length(cells_use)) {
    stop(
      "Cell index mismatch for group ",
      group_id,
      ": selected=",
      length(cells_use),
      ", matched=",
      length(idx)
    )
  }

  n_cells_used <- length(idx)

  w <- Matrix::sparseVector(x = rep(1, length(idx)), i = idx, length = ncol(counts))
  peak_sum_mat <- counts %*% w
  peak_sum <- as.numeric(peak_sum_mat)

  total <- sum(peak_sum)
  total_counts <- total
  if (total == 0) {
    stop("Total ATAC counts is 0 for group: ", group_id)
  }

  raw_mean_counts_per_cell <- total_counts / n_cells_used

  scale_factor <- 1e6 / total
  peak_signal <- peak_sum * scale_factor
  cpm_sum <- sum(peak_signal)

  message("Cells used: ", n_cells_used)
  message("CPM sum check: ", sprintf("%.3f", cpm_sum))

  if (!is.finite(cpm_sum) || abs(cpm_sum - 1e6) > 1) {
    stop("CPM normalisation failed for group ", group_id, ": sum = ", cpm_sum)
  }

  nz <- which(peak_signal > 0)
  nonzero_peaks <- length(nz)
  if (length(nz) == 0) {
    stop("No non-zero peaks after aggregation for group: ", group_id)
  }

  bedgraph38 <- file.path(tmp_dir, paste0(fname, ".hg38.bedGraph"))
  df38 <- data.frame(
    chrom = as.character(GenomicRanges::seqnames(peaks_gr))[nz],
    start = GenomicRanges::start(peaks_gr)[nz] - 1L,
    end   = GenomicRanges::end(peaks_gr)[nz],
    value = peak_signal[nz],
    stringsAsFactors = FALSE
  )

  tmp_unsorted <- paste0(bedgraph38, ".unsorted")
  write_bedgraph(df38, tmp_unsorted)

  system(sprintf(
    "LC_ALL=C sort -k1,1 -k2,2n %s > %s",
    shQuote(tmp_unsorted), shQuote(bedgraph38)
  ))
  file.remove(tmp_unsorted)

  system(sprintf(
    "bedGraphToBigWig %s %s %s",
    shQuote(bedgraph38), shQuote(hg38_sizes_path), shQuote(bw38)
  ))

  bedgraph19 <- file.path(tmp_dir, paste0(fname, ".hg19.bedGraph"))
  unmapped19 <- paste0(bedgraph19, ".unmapped")
  system(sprintf(
    "liftOver %s %s %s %s",
    shQuote(bedgraph38), shQuote(chain_path),
    shQuote(bedgraph19), shQuote(unmapped19)
  ))

  bedgraph19_sorted <- paste0(bedgraph19, ".sorted")
  system(sprintf(
    "LC_ALL=C sort -k1,1 -k2,2n %s > %s",
    shQuote(bedgraph19), shQuote(bedgraph19_sorted)
  ))

  bedgraph19_merged <- paste0(bedgraph19, ".merged")
  system(sprintf(
    "bedtools merge -i %s -c 4 -o max > %s",
    shQuote(bedgraph19_sorted), shQuote(bedgraph19_merged)
  ))

  system(sprintf(
    "bedGraphToBigWig %s %s %s",
    shQuote(bedgraph19_merged), shQuote(hg19_sizes_path), shQuote(bw19)
  ))

  hg19_intervals <- as.integer(sub(
    "\\s+.*$", "",
    system2("wc", c("-l", bedgraph19_merged), stdout = TRUE)
  ))

  file.remove(
    bedgraph38,
    bedgraph19,
    bedgraph19_sorted,
    bedgraph19_merged,
    unmapped19
  )

  data.frame(
    group_id = group_id,
    n_cells_used = n_cells_used,
    total_counts = total_counts,
    raw_mean_counts_per_cell = round(raw_mean_counts_per_cell, 1),
    cpm_sum = cpm_sum,
    nonzero_peaks = nonzero_peaks,
    hg19_intervals = hg19_intervals,
    stringsAsFactors = FALSE
  )
}

# ----------------------------
# Check system dependencies
# ----------------------------
for (cmd in c("bedGraphToBigWig", "liftOver", "bedtools", "sort", "gzip", "gunzip")) {
  need_cmd(cmd)
}

# ----------------------------
# Download genome resources
# ----------------------------
download_if_missing(HG38_SIZES_URL, HG38_SIZES_PATH)
download_if_missing(HG19_SIZES_URL, HG19_SIZES_PATH)

if (!file.exists(CHAIN_PATH)) {
  tmp_chain <- paste0(CHAIN_PATH, ".gz")
  download_if_missing(CHAIN_URL, tmp_chain)
  system2("gunzip", c("-f", shQuote(tmp_chain)))
}

hg38_sizes  <- read_chrom_sizes(HG38_SIZES_PATH)
hg19_sizes  <- read_chrom_sizes(HG19_SIZES_PATH)
hg38_chroms <- unique(hg38_sizes$chrom)

# ----------------------------
# Load Seurat object
# ----------------------------
if (!file.exists(RDS_PATH)) stop("RDS not found: ", RDS_PATH)

message("Loading Seurat object")
obj <- readRDS(RDS_PATH)

meta <- obj@meta.data
required_meta_cols <- c("CellTypeSR", "Condition")
if (!all(required_meta_cols %in% colnames(meta))) {
  stop("Missing required metadata columns: ",
       paste(setdiff(required_meta_cols, colnames(meta)), collapse = ", "))
}

if (!("RNA" %in% names(obj@assays))) stop("RNA assay not found in object")
if (!("ATAC" %in% names(obj@assays))) stop("ATAC assay not found in object")

# ----------------------------
# Canonicalise CellType names for unfiltered outputs
# ----------------------------
meta$CellTypeSR_canon <- ifelse(
  meta$CellTypeSR %in% names(CELLTYPE_MAP),
  unname(CELLTYPE_MAP[meta$CellTypeSR]),
  meta$CellTypeSR
)

# ----------------------------
# Prepare ATAC peaks and sparse counts once
# ----------------------------
DefaultAssay(obj) <- "ATAC"

peaks_gr <- granges(obj[["ATAC"]])
if (length(peaks_gr) == 0) stop("No peaks found in ATAC assay (granges is empty)")

keep_chr <- as.character(seqnames(peaks_gr)) %in% hg38_chroms
peaks_gr <- peaks_gr[keep_chr]
if (length(peaks_gr) == 0) stop("All peaks were filtered out (no peaks on hg38 standard chromosomes)")

counts <- get_layer_matrix(obj, assay_name = "ATAC", layer_name = "counts")

if (!inherits(counts, "sparseMatrix")) {
  stop("ATAC counts is not sparse. Refusing to coerce to dense.")
}
counts <- as(counts, "dgCMatrix")

message("ATAC counts class: ", paste(class(counts), collapse = ", "))
message("ATAC counts dim: ", paste(dim(counts), collapse = " x "))

peak_ids <- paste0(as.character(seqnames(peaks_gr)), "-", start(peaks_gr), "-", end(peaks_gr))
if (!is.null(rownames(counts)) && all(peak_ids %in% rownames(counts))) {
  counts <- counts[peak_ids, , drop = FALSE]
} else {
  if (is.null(rownames(counts))) {
    stop("ATAC counts matrix has no rownames; cannot align peaks to counts.")
  }

  rn <- rownames(counts)
  parts <- strsplit(rn, "-", fixed = TRUE)
  ok <- lengths(parts) == 3
  if (!any(ok)) {
    stop("Could not align peaks and counts. ATAC counts rownames are not in 'chr-start-end' format.")
  }

  chr <- vapply(parts[ok], `[`, character(1), 1)
  st  <- suppressWarnings(as.integer(vapply(parts[ok], `[`, character(1), 2)))
  en  <- suppressWarnings(as.integer(vapply(parts[ok], `[`, character(1), 3)))
  ok2 <- !is.na(st) & !is.na(en) & (chr %in% hg38_chroms)

  if (!any(ok2)) {
    stop("No parseable peaks on hg38 standard chromosomes found in counts rownames.")
  }

  peaks_gr <- GenomicRanges::GRanges(
    seqnames = chr[ok2],
    ranges = IRanges::IRanges(start = st[ok2], end = en[ok2])
  )
  counts <- counts[which(ok)[ok2], , drop = FALSE]
}

# ============================================================
# PART 1: UNFILTERED CONDITION TRACKS
# ============================================================
message("\n============================================================")
message("PART 1: Building unfiltered condition tracks")
message("============================================================")

meta_orig <- meta[
  meta$CellTypeSR_canon %in% CELLTYPES_KEEP & meta$Condition %in% CONDITIONS_KEEP,
  ,
  drop = FALSE
]

if (nrow(meta_orig) == 0) {
  stop("Filtering removed all cells for unfiltered tracks. Check CELLTYPES_KEEP and CONDITIONS_KEEP.")
}

tab_orig <- table(meta_orig$CellTypeSR_canon, meta_orig$Condition)

missing_orig <- which(tab_orig[, CONDITIONS_KEEP, drop = FALSE] == 0, arr.ind = TRUE)
if (nrow(missing_orig) > 0) {
  stop(
    "Some selected cell types are missing required conditions for unfiltered tracks. Table:\n",
    paste(capture.output(tab_orig[, CONDITIONS_KEEP, drop = FALSE]), collapse = "\n")
  )
}

target_n_orig <- min(tab_orig[, CONDITIONS_KEEP, drop = FALSE])

message("Selected cell types for unfiltered tracks: ", paste(CELLTYPES_KEEP, collapse = ", "))
message("Unfiltered downsampling target_n = ", target_n_orig)
print(tab_orig[, CONDITIONS_KEEP, drop = FALSE])

meta_orig$group_id <- paste(meta_orig$CellTypeSR_canon, meta_orig$Condition, sep = "__")
groups_orig <- sort(unique(meta_orig$group_id))

original_group_summary <- data.frame(
  group_id = groups_orig,
  n_cells_before_downsampling = as.integer(vapply(
    groups_orig,
    function(g) sum(meta_orig$group_id == g),
    integer(1)
  )),
  target_n_downsampled = target_n_orig,
  stringsAsFactors = FALSE
)

utils::write.csv(
  original_group_summary,
  file = file.path(QC_DIR, "original_group_summary.csv"),
  row.names = FALSE
)

for (grp in groups_orig) {
  cells_grp <- rownames(meta_orig)[meta_orig$group_id == grp]

  if (length(cells_grp) < target_n_orig) {
    stop("Group has fewer than target_n_orig cells: ", grp)
  }

  cells_use <- sample(cells_grp, size = target_n_orig, replace = FALSE)

  original_track_log_df <- rbind(
    original_track_log_df,
    build_bigwig_pair(
      group_id = grp,
      cells_use = cells_use,
      counts = counts,
      peaks_gr = peaks_gr,
      hg38_sizes_path = HG38_SIZES_PATH,
      hg19_sizes_path = HG19_SIZES_PATH,
      chain_path = CHAIN_PATH,
      out_dir_hg38 = OUT_DIR_HG38,
      out_dir_hg19 = OUT_DIR_HG19,
      tmp_dir = TMP_DIR
    )
  )
}

utils::write.csv(
  original_track_log_df,
  file = file.path(QC_DIR, "original_track_build_log.csv"),
  row.names = FALSE
)

# ============================================================
# PART 2: FOXA2-FILTERED REFERENCE TRACKS
# ============================================================
message("\n============================================================")
message("PART 2: Building FOXA2-filtered reference tracks")
message("============================================================")

hep_cells <- rownames(meta)[meta$CellTypeSR == HEPATOCYTE_LABEL]
if (length(hep_cells) == 0) {
  stop("No hepatocyte cells found using CellTypeSR == '", HEPATOCYTE_LABEL, "'")
}

meta_hep <- meta[hep_cells, , drop = FALSE]

n_hep_normal   <- sum(meta_hep$Condition == NORMAL_CONDITION)
n_hep_diseased <- sum(meta_hep$Condition %in% DISEASED_CONDITIONS)

if (n_hep_normal == 0) {
  stop("No hepatocyte cells found for normal condition: ", NORMAL_CONDITION)
}
if (n_hep_diseased == 0) {
  stop("No hepatocyte cells found for diseased conditions: ",
       paste(DISEASED_CONDITIONS, collapse = ", "))
}

rna_mat <- get_layer_matrix(obj, assay_name = "RNA", layer_name = "data")

if (!(FOXA2_GENE %in% rownames(rna_mat))) {
  stop("FOXA2 gene not found in RNA assay: ", FOXA2_GENE)
}

foxa2_vals <- as.numeric(rna_mat[FOXA2_GENE, hep_cells])
names(foxa2_vals) <- hep_cells

is_normal_hep   <- meta_hep$Condition == NORMAL_CONDITION
is_diseased_hep <- meta_hep$Condition %in% DISEASED_CONDITIONS
is_foxa2_pos    <- foxa2_vals > 0
is_foxa2_zero   <- foxa2_vals == 0

normal_cells_all    <- rownames(meta_hep)[is_normal_hep]
diseased_cells_all  <- rownames(meta_hep)[is_diseased_hep]

normal_cells_final   <- rownames(meta_hep)[is_normal_hep & is_foxa2_pos]
abnormal_cells_final <- rownames(meta_hep)[is_diseased_hep & is_foxa2_zero]

n_normal_final   <- length(normal_cells_final)
n_abnormal_final <- length(abnormal_cells_final)

if (n_normal_final == 0) {
  stop("No cells remain in ", NORMAL_GROUP_NAME, " after FOXA2 filtering")
}
if (n_abnormal_final == 0) {
  stop("No cells remain in ", ABNORMAL_GROUP_NAME, " after FOXA2 filtering")
}

target_n_foxa2 <- min(n_normal_final, n_abnormal_final)

if (target_n_foxa2 < 10) {
  stop("Too few cells remain after FOXA2 filtering. target_n_foxa2 = ", target_n_foxa2)
}

message("Hepatocyte counts before filtering:")
message("  Normal hepatocytes: ", length(normal_cells_all))
message("  Diseased hepatocytes (AC + AH): ", length(diseased_cells_all))

message("Hepatocyte counts after FOXA2 filtering:")
message("  ", NORMAL_GROUP_NAME, ": ", n_normal_final)
message("  ", ABNORMAL_GROUP_NAME, ": ", n_abnormal_final)

message("Downsampling FOXA2-filtered groups to target_n_foxa2 = ", target_n_foxa2)

normal_cells_use   <- sample(normal_cells_final, size = target_n_foxa2, replace = FALSE)
abnormal_cells_use <- sample(abnormal_cells_final, size = target_n_foxa2, replace = FALSE)

cell_assignment_df <- data.frame(
  cell_id = hep_cells,
  Condition = meta_hep$Condition,
  CellTypeSR = meta_hep$CellTypeSR,
  FOXA2_expr = foxa2_vals,
  foxa2_detected = foxa2_vals > 0,
  candidate_pool = ifelse(
    is_normal_hep, "normal",
    ifelse(is_diseased_hep, "diseased", "other")
  ),
  reference_group = ifelse(
    hep_cells %in% normal_cells_final, NORMAL_GROUP_NAME,
    ifelse(hep_cells %in% abnormal_cells_final, ABNORMAL_GROUP_NAME, "excluded")
  ),
  selected_for_track = hep_cells %in% c(normal_cells_use, abnormal_cells_use),
  stringsAsFactors = FALSE
)

utils::write.csv(
  cell_assignment_df,
  file = file.path(QC_DIR, "foxa2_reference_cell_assignments.csv"),
  row.names = FALSE
)

writeLines(normal_cells_use, con = file.path(QC_DIR, paste0(NORMAL_GROUP_NAME, "_cells.txt")))
writeLines(abnormal_cells_use, con = file.path(QC_DIR, paste0(ABNORMAL_GROUP_NAME, "_cells.txt")))

foxa2_summary_df <- data.frame(
  metric = c(
    "n_hepatocytes_total",
    "n_hepatocytes_normal",
    "n_hepatocytes_diseased",
    "n_normal_FOXA2_pos",
    "n_abnormal_FOXA2_zero",
    "target_n_downsampled",
    "foxa2_detection_rate_normal",
    "foxa2_detection_rate_diseased",
    "foxa2_mean_normal",
    "foxa2_mean_diseased"
  ),
  value = c(
    length(hep_cells),
    length(normal_cells_all),
    length(diseased_cells_all),
    n_normal_final,
    n_abnormal_final,
    target_n_foxa2,
    mean(foxa2_vals[normal_cells_all] > 0),
    mean(foxa2_vals[diseased_cells_all] > 0),
    mean(foxa2_vals[normal_cells_all]),
    mean(foxa2_vals[diseased_cells_all])
  ),
  stringsAsFactors = FALSE
)

utils::write.csv(
  foxa2_summary_df,
  file = file.path(QC_DIR, "foxa2_reference_summary.csv"),
  row.names = FALSE
)

foxa2_track_log_df <- rbind(
  foxa2_track_log_df,
  build_bigwig_pair(
    group_id = NORMAL_GROUP_NAME,
    cells_use = normal_cells_use,
    counts = counts,
    peaks_gr = peaks_gr,
    hg38_sizes_path = HG38_SIZES_PATH,
    hg19_sizes_path = HG19_SIZES_PATH,
    chain_path = CHAIN_PATH,
    out_dir_hg38 = OUT_DIR_HG38,
    out_dir_hg19 = OUT_DIR_HG19,
    tmp_dir = TMP_DIR
  )
)

foxa2_track_log_df <- rbind(
  foxa2_track_log_df,
  build_bigwig_pair(
    group_id = ABNORMAL_GROUP_NAME,
    cells_use = abnormal_cells_use,
    counts = counts,
    peaks_gr = peaks_gr,
    hg38_sizes_path = HG38_SIZES_PATH,
    hg19_sizes_path = HG19_SIZES_PATH,
    chain_path = CHAIN_PATH,
    out_dir_hg38 = OUT_DIR_HG38,
    out_dir_hg19 = OUT_DIR_HG19,
    tmp_dir = TMP_DIR
  )
)

utils::write.csv(
  foxa2_track_log_df,
  file = file.path(QC_DIR, "foxa2_track_build_log.csv"),
  row.names = FALSE
)

# ----------------------------
# Final QC output
# ----------------------------
message("\nUnfiltered group summary:")
print(original_group_summary)

message("\nUnfiltered track build log:")
print(original_track_log_df)

message("\nFOXA2 reference summary:")
print(foxa2_summary_df)

message("\nFOXA2 track build log:")
print(foxa2_track_log_df)

# ----------------------------
# Delete _tmp directory at end
# ----------------------------
if (dir.exists(TMP_DIR)) {
  unlink(TMP_DIR, recursive = TRUE, force = TRUE)
}

message("\nAll done.")
