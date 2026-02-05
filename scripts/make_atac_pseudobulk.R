#!/usr/bin/env Rscript
# ============================================================
# scripts/make_atac_pseudobulk.R
#
# Peak-resolution bigWig tracks from ATAC peak-by-cell counts
# - Groups: CellTypeSR × Condition
# - Restricts to user-selected canonical cell types + required conditions
# - Downsamples per group so all tracks use the same number of cells
# - Genome: hg38 (native)
# - Also produces hg19 bigWig via UCSC liftOver
#
# Notes:
# - Does NOT require fragment files.
# - Signal is constant within each peak interval and zero elsewhere.
# - Uses sparse operations end-to-end to avoid densifying the peak x cell matrix.
# - hg19 liftOver can introduce overlapping intervals; we merge overlaps with bedtools.
# - Prints a QC summary with per-group counts and CPM sanity checks.
#
# Outputs:
#   data/processed/ATAC-seq/GSE281574/hg38/<CellType>__<Condition>.bigWig
#   data/processed/ATAC-seq/GSE281574/hg19/<CellType>__<Condition>.bigWig
#
# Naming:
# - We canonicalise CellTypeSR via an explicit map once (no grep/heuristics later).
# - Examples:
#     "T cell"       -> "T_Cell"
#     "Kupffer Cell" -> "Kupffer_Cell"
#     "Plasma Cell"  -> "Plasma_Cell"
# - Abbreviations already all-caps are preserved (e.g. LSEC, HSC).
# - Condition is kept as-is (AC, AH, Normal).
# - Separator between CellType and Condition is "__".
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
# Dataset identifier (used for output folder organisation)
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
# Output directories (derived data)
# ----------------------------
OUT_BASE_DIR <- file.path("data", "processed", "ATAC-seq", DATASET_ID)
OUT_DIR_HG38 <- file.path(OUT_BASE_DIR, "hg38")
OUT_DIR_HG19 <- file.path(OUT_BASE_DIR, "hg19")

dir.create(OUT_DIR_HG38, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR_HG19, recursive = TRUE, showWarnings = FALSE)

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
# Temporary working directory (deleted at end)
# ----------------------------
TMP_DIR <- file.path(OUT_BASE_DIR, "_tmp")
dir.create(TMP_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Logging
# ----------------------------
log_df <- data.frame(
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

# Safe filename: keep underscores, dots, dashes; remove other punctuation/spaces
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
liver_multiome_seurat <- readRDS(RDS_PATH)
DefaultAssay(liver_multiome_seurat) <- "ATAC"

meta <- liver_multiome_seurat@meta.data
stopifnot(all(c("CellTypeSR", "Condition") %in% colnames(meta)))

# ----------------------------
# User settings: restrict cell types + enforce comparable tracks
# ----------------------------

# Cell types to include (IMPORTANT: use the canonical names produced by CELLTYPE_MAP below)
CELLTYPES_KEEP <- c("Hepatocyte")  # e.g. c("Hepatocyte") or c("Hepatocyte","HSC")

# Conditions that must be included for each kept cell type
CONDITIONS_KEEP <- c("AC", "AH", "Normal")

# Reproducible downsampling
set.seed(1)

# ----------------------------
# Canonical CellType naming (applied BEFORE group_id)
# ----------------------------
CELLTYPE_MAP <- c(
  "T cell"        = "T_Cell",
  "Kupffer Cell"  = "Kupffer_Cell",
  "Plasma Cell"   = "Plasma_Cell"
  # Leave "Hepatocyte", "LSEC", "HSC" unchanged by not listing them
)

meta$CellTypeSR_canon <- ifelse(
  meta$CellTypeSR %in% names(CELLTYPE_MAP),
  unname(CELLTYPE_MAP[meta$CellTypeSR]),
  meta$CellTypeSR
)

# ----------------------------
# Filter to selected cell types and required conditions
# ----------------------------
meta_filt <- meta[meta$CellTypeSR_canon %in% CELLTYPES_KEEP & meta$Condition %in% CONDITIONS_KEEP, , drop = FALSE]

if (nrow(meta_filt) == 0) stop("Filtering removed all cells. Check CELLTYPES_KEEP and CONDITIONS_KEEP.")

# ----------------------------
# Compute target_n so all included condition tracks use the same number of cells
# ----------------------------
tab <- table(meta_filt$CellTypeSR_canon, meta_filt$Condition)

# Ensure every selected cell type has every required condition
missing <- which(tab[, CONDITIONS_KEEP, drop = FALSE] == 0, arr.ind = TRUE)
if (nrow(missing) > 0) {
  stop("Some selected cell types are missing required conditions. Table:\n", paste(capture.output(tab[, CONDITIONS_KEEP, drop = FALSE]), collapse = "\n"))
}

# Global minimum across selected cell types and required conditions
target_n <- min(tab[, CONDITIONS_KEEP, drop = FALSE])

message("Selected cell types: ", paste(CELLTYPES_KEEP, collapse = ", "))
message("Downsampling target_n = ", target_n)
print(tab[, CONDITIONS_KEEP, drop = FALSE])

# ----------------------------
# Get peaks + peak-by-cell counts (sparse, no densification)
# ----------------------------
peaks_gr <- granges(liver_multiome_seurat[["ATAC"]])
if (length(peaks_gr) == 0) stop("No peaks found in ATAC assay (granges is empty)")

# Keep peaks on chromosomes present in hg38.chrom.sizes
keep_chr <- as.character(seqnames(peaks_gr)) %in% hg38_chroms
peaks_gr <- peaks_gr[keep_chr]
if (length(peaks_gr) == 0) stop("All peaks were filtered out (no peaks on hg38 standard chromosomes)")

# Fetch counts via Seurat v5 layers if possible; fallback to older APIs.
assay_atac <- liver_multiome_seurat[["ATAC"]]
counts <- tryCatch(
  SeuratObject::LayerData(assay_atac, layer = "counts"),
  error = function(e1) tryCatch(
    GetAssayData(liver_multiome_seurat, assay = "ATAC", layer = "counts"),
    error = function(e2) GetAssayData(liver_multiome_seurat, assay = "ATAC", slot = "counts")
  )
)

# Must remain sparse; never coerce to dense
if (!inherits(counts, "sparseMatrix")) {
  stop("ATAC counts is not sparse. Refusing to coerce to dense (would exceed memory).")
}
counts <- as(counts, "dgCMatrix")

message("ATAC counts class: ", paste(class(counts), collapse = ", "))
message("ATAC counts dim: ", paste(dim(counts), collapse = " x "))

# Align counts rows to filtered peaks if possible using common Signac peak IDs
peak_ids <- paste0(as.character(seqnames(peaks_gr)), "-", start(peaks_gr), "-", end(peaks_gr))
if (!is.null(rownames(counts)) && all(peak_ids %in% rownames(counts))) {
  counts <- counts[peak_ids, , drop = FALSE]
} else {
  # Fallback: derive peaks from counts rownames (expects chr-start-end), and filter to hg38 chroms
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

# ----------------------------
# Grouping: CellTypeSR × Condition (use canonical names)
# ----------------------------
meta_filt$group_id <- paste(meta_filt$CellTypeSR_canon, meta_filt$Condition, sep = "__")
groups <- sort(unique(meta_filt$group_id))

# ----------------------------
# Core processing loop
# ----------------------------
for (grp in groups) {

  message("\n=== ", grp, " ===")

  # grp is already "<CellType>__<Condition>" in canonical form
  parts <- strsplit(grp, "__", fixed = TRUE)[[1]]
  cell_type <- safe_name(parts[1])
  condition <- safe_name(parts[2])
  fname <- paste0(cell_type, "__", condition)

  bw38 <- file.path(OUT_DIR_HG38, paste0(fname, ".bigWig"))
  bw19 <- file.path(OUT_DIR_HG19, paste0(fname, ".bigWig"))

  if (file.exists(bw38) && file.exists(bw19)) {
    message("Already exists, skipping")
    next
  }

  cells_use <- rownames(meta_filt)[meta_filt$group_id == grp]
  if (length(cells_use) == 0) next

  # Downsample so every track uses exactly target_n cells
  if (length(cells_use) < target_n) {
    message("Group has fewer than target_n cells (unexpected after checks), skipping")
    next
  }
  cells_use <- sample(cells_use, size = target_n, replace = FALSE)
  n_cells_used <- length(cells_use)

  # Keep only cells present in the ATAC matrix
  idx <- match(cells_use, colnames(counts))
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0) {
    message("No matching cells in ATAC counts for this group, skipping")
    next
  }

  # Pseudobulk without subsetting:
  # (peaks x cells) %*% (cells x 1) -> (peaks x 1)
  w <- Matrix::sparseVector(x = rep(1, length(idx)), i = idx, length = ncol(counts))
  peak_sum_mat <- counts %*% w
  peak_sum <- as.numeric(peak_sum_mat)

  total <- sum(peak_sum)
  total_counts <- total
  if (total == 0) {
    message("Total ATAC counts is 0 for this group, skipping")
    next
  }
  raw_mean_counts_per_cell <- total_counts / n_cells_used

  # Normalise to CPM across peaks
  scale_factor <- 1e6 / total
  peak_signal <- peak_sum * scale_factor
  cpm_sum <- sum(peak_signal)
  cpm_sum_print <- sprintf("%.3f", cpm_sum)
  message("CPM sum check: ", cpm_sum_print)

  if (!is.finite(cpm_sum) || abs(cpm_sum - 1e6) > 1) {
    stop("CPM normalisation failed: sum = ", cpm_sum)
  }

  nz <- which(peak_signal > 0)
  nonzero_peaks <- length(nz)
  if (length(nz) == 0) {
    message("No non-zero peaks after aggregation, skipping")
    next
  }

  bedgraph38 <- file.path(TMP_DIR, paste0(fname, ".hg38.bedGraph"))
  df38 <- data.frame(
    chrom = as.character(GenomicRanges::seqnames(peaks_gr))[nz],
    start = GenomicRanges::start(peaks_gr)[nz] - 1L,  # bedGraph uses 0-based start
    end   = GenomicRanges::end(peaks_gr)[nz],
    value = peak_signal[nz],
    stringsAsFactors = FALSE
  )

  # Write unsorted then sort via system sort
  tmp_unsorted <- paste0(bedgraph38, ".unsorted")
  write_bedgraph(df38, tmp_unsorted)

  system(sprintf(
    "LC_ALL=C sort -k1,1 -k2,2n %s > %s",
    shQuote(tmp_unsorted), shQuote(bedgraph38)
  ))
  file.remove(tmp_unsorted)

  # hg38 bigWig
  system(sprintf(
    "bedGraphToBigWig %s %s %s",
    shQuote(bedgraph38), shQuote(HG38_SIZES_PATH), shQuote(bw38)
  ))

  # liftOver → hg19
  bedgraph19 <- file.path(TMP_DIR, paste0(fname, ".hg19.bedGraph"))
  unmapped19 <- paste0(bedgraph19, ".unmapped")
  system(sprintf(
    "liftOver %s %s %s %s",
    shQuote(bedgraph38), shQuote(CHAIN_PATH),
    shQuote(bedgraph19), shQuote(unmapped19)
  ))

  # Sort
  bedgraph19_sorted <- paste0(bedgraph19, ".sorted")
  system(sprintf(
    "LC_ALL=C sort -k1,1 -k2,2n %s > %s",
    shQuote(bedgraph19), shQuote(bedgraph19_sorted)
  ))

  # Merge overlaps (bedGraph must be non-overlapping for bedGraphToBigWig)
  # Use max to avoid inflating signal due to liftOver collisions
  bedgraph19_merged <- paste0(bedgraph19, ".merged")
  system(sprintf(
    "bedtools merge -i %s -c 4 -o max > %s",
    shQuote(bedgraph19_sorted), shQuote(bedgraph19_merged)
  ))

  # Convert merged bedGraph to bigWig
  system(sprintf(
    "bedGraphToBigWig %s %s %s",
    shQuote(bedgraph19_merged), shQuote(HG19_SIZES_PATH), shQuote(bw19)
  ))

  hg19_intervals <- as.integer(sub("\\s+.*$", "", system2("wc", c("-l", bedgraph19_merged), stdout = TRUE)))

  log_df <- rbind(
    log_df,
    data.frame(
      group_id = grp,
      n_cells_used = n_cells_used,
      total_counts = total_counts,
      raw_mean_counts_per_cell = round(raw_mean_counts_per_cell, 1),
      cpm_sum = cpm_sum,
      nonzero_peaks = nonzero_peaks,
      hg19_intervals = hg19_intervals,
      stringsAsFactors = FALSE
    )
  )

  # Cleanup intermediates (per group)
  file.remove(
    bedgraph38,
    bedgraph19,
    bedgraph19_sorted,
    bedgraph19_merged,
    unmapped19
  )
}

# ----------------------------
# Delete _tmp directory at end
# ----------------------------
if (dir.exists(TMP_DIR)) {
  unlink(TMP_DIR, recursive = TRUE, force = TRUE)
}

# ----------------------------
# Final QC output
# ----------------------------
message("\nQC summary:")
print(log_df)

message("\nAll done.")
