# TCGA SKCM Somatic Mutation Data

This directory contains the somatic mutation dataset for **Skin Cutaneous Melanoma (SKCM)** from the **TCGA PanCancer Atlas (2018)** project.

It also contains two independent **whole genome sequencing (WGS)** mutation datasets in a compact BED-like format:
1) a **UV-associated melanoma** mutation dataset (`UV_mutations.bed`), and  
2) an **ICGC WGS** mutation dataset (`ICGC_WGS_Feb20_mutations.bed`) spanning multiple projects (example shown below).

## Data source

The file `data_mutations.txt` was obtained from **cBioPortal**:

https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018

## Content

`data_mutations.txt` is a tab-delimited table in Mutation Annotation Format–style representation.  
Each row corresponds to **one somatic mutation observed in one tumour sample**.

Key fields include:

- `Tumor_Sample_Barcode`
- `Hugo_Symbol`
- `Chromosome`
- `Start_Position`, `End_Position`
- `Reference_Allele`
- `Tumor_Seq_Allele1`, `Tumor_Seq_Allele2`
- `Variant_Classification`

The cohort contains approximately **448 tumour samples** (Liu et al., 2018).  
These data are derived from **whole exome sequencing (WES)** and therefore primarily represent coding-region mutations.

---

## UV Mutation Dataset (Whole Genome Sequencing)

This directory also contains a separate mutation file generated from an independent **whole genome sequencing (WGS)** study of melanoma, focused on **UV-induced mutational processes**.

Each row in the UV mutation file represents **one somatic single-nucleotide variant (SNV)** and has the format:

```

chr1    87302   87303   TCGA-DA-A1HW   G   A   SKCM   cgg

```

Column definitions:

- `Chromosome`: Chromosome name (e.g. `chr1`)
- `Start`: 0-based genomic start coordinate
- `End`: 1-based genomic end coordinate
- `Sample_ID`: Tumour sample identifier
- `Ref`: Reference nucleotide
- `Alt`: Alternate nucleotide
- `Cancer_Type`: Cancer cohort label (`SKCM`)
- `Trinucleotide_Context`: Local trinucleotide sequence context centred on the mutated base

Because this dataset is derived from **whole genome sequencing**, it includes mutations from coding, non-coding, regulatory and intergenic regions that are not captured by the TCGA exome dataset.

The explicit annotation of trinucleotide context enables analysis of characteristic **UV mutational signatures**, which in melanoma are dominated by **C>T and complementary G>A substitutions at dipyrimidine sites**.

---

## ICGC WGS Mutation Dataset (Whole Genome Sequencing)

This directory also contains `ICGC_WGS_Feb20_mutations.bed`, a compact BED-like mutation file derived from **ICGC whole genome sequencing (WGS)** projects.  
Each row represents **one somatic SNV** in one donor/sample record, with an associated project label.

Example row format:

```

chr1    2112412  2112413  DO1000  T  C  BRCA-UK  PD3851a

```

Column definitions:

- `Chromosome`: Chromosome name (e.g. `chr1`)
- `Start`: 0-based genomic start coordinate
- `End`: 1-based genomic end coordinate
- `Donor_ID`: Donor identifier (e.g. `DO1000`)
- `Ref`: Reference nucleotide
- `Alt`: Alternate nucleotide
- `Project`: ICGC project / cohort label (e.g. `BRCA-UK`)
- `Sample_ID`: Sample identifier within the project (e.g. `PD3851a`)

Notes:
- Unlike `UV_mutations.bed`, this file does **not** include trinucleotide context as an explicit column.
- The `Project` field can include multiple cancer cohorts (for example breast cancer projects such as `BRCA-UK`).

---

## Reference genome

All genomic coordinates and trinucleotide contexts in these mutation datasets are based on the **GRCh37 (hg19)** reference genome.

## Usage note

All datasets are treated as **raw input data** and should not be modified.  
All downstream filtering and analysis outputs should be written to `data/processed/`.

The TCGA WES dataset and the WGS datasets originate from different studies and sequencing strategies and should **not be merged directly** without careful harmonisation of study design and variant calling pipelines.

## References

Liu, J., Lichtenberg, T., Hoadley, K. A., Poisson, L. M., Lazar, A. J., Cherniack, A. D., Kovatich, A. J., Benz, C. C., Levine, D. A., Lee, A. v., Omberg, L., Wolf, D. M., Shriver, C. D., Thorsson, V., Caesar-Johnson, S. J., Demchok, J. A., Felau, I., Kasapi, M., Ferguson, M. L., … Hu, H. (2018). An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. *Cell*, 173(2), 400–416.e11. https://doi.org/10.1016/j.cell.2018.02.052