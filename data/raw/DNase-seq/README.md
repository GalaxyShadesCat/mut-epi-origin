# DNase-seq accessibility reference tracks

This directory contains genome-wide **DNase-seq chromatin accessibility** signal tracks (bigWig) downloaded from the ENCODE Project.

Only tumour types with a **single, well-defined dominant cell of origin** were included.  
Heterogeneous tumours, pan-lineage groupings, cancer cell lines, and treated samples were deliberately excluded.

All local bigWig tracks are:
- **Read-depth normalised DNase-seq signal**
- Aligned to **hg19 (GRCh37)**
- Generated using ENCODE-standard DNase-seq pipelines
- Directly comparable at the genome-wide level

---

## Design principles

DNase-seq reference datasets were selected according to the following criteria:

1. **Single cell-of-origin tumours only**  
   Included tumours have a dominant, biologically accepted lineage of origin.

2. **Normal primary or progenitor cells preferred**  
   Primary cells or progenitor populations were prioritised over immortalised or tumour-derived cell lines.

3. **Developmental proximity over terminal differentiation**  
   Where applicable, progenitor-like references were chosen instead of fully differentiated cells.

---

## Tumour → DNase-seq reference mapping

| Tumour type                                           | Cell-of-origin proxy             | DNase reference                                        |
|-------------------------------------------------------|----------------------------------|--------------------------------------------------------|
| **Acute Myeloid Leukemia (AML / LAML)**               | Myeloid progenitor               | `myelProg_ENCFF201IYJ.bigWig`                          |
| **NK/T-Cell Lymphoma (NKTL)**                         | Natural killer cell              | `NKC_ENCFF448RXA.bigWig`                               |
| **Basal Cell Carcinoma (BCC)**                        | Keratinocyte (basal-layer proxy) | `kera_ENCFF597YXQ.bigWig`                              |
| **Leiomyosarcoma (LMS)**                              | Smooth muscle cell               | `SMC_ENCFF913OTC.bigWig`                               |
| **Esophageal Adenocarcinoma/Carcinoma (ESAD / ESCA)** | Oesophageal epithelial cell      | `esoEpi_ENCFF731MXF.bigWig`                            |
| **Neuroblastoma (NBL)**                               | Neuronal stem / progenitor cell  | `neuroStem_ENCFF495MWO.bigWig`                         |
| **Glioblastoma/Low-Grade Glioma (GBM / LGG)**         | Astrocyte (glial lineage)        | `astro_ENCFF227QKB.bigWig`, `astro_ENCFF727GJL.bigWig` |
| **Cutaneous Melanoma (SKCM / MELA)**                  | Melanocyte                       | `mela_ENCFF285GEW.bigWig`                              |

Notes:
- **GBM and LGG** are analysed separately but share astrocyte DNase-seq references.
- **Keratinocytes** are used as a proxy for basal keratinocytes in BCC due to lack of basal-layer–specific DNase data.
- **Fibroblast DNase** is included as a stromal skin control and is *not* used as a tumour cell-of-origin reference.

---

## Files

### `myelProg_ENCFF201IYJ.bigWig`
- **Cell type:** common myeloid progenitor (CD34⁺)
- **Tumour mapping:** AML / LAML
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR468ZXN
- **ENCODE file accession:** ENCFF201IYJ
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR468ZXN/
- https://www.encodeproject.org/files/ENCFF201IYJ/

---

### `NKC_ENCFF448RXA.bigWig`
- **Cell type:** natural killer (NK) cell
- **Tumour mapping:** NKTL
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR704HNG
- **ENCODE file accession:** ENCFF448RXA
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR704HNG/
- https://www.encodeproject.org/files/ENCFF448RXA/

---

### `kera_ENCFF597YXQ.bigWig`
- **Cell type:** foreskin keratinocyte (male newborn)
- **Tumour mapping:** BCC (proxy for basal keratinocytes)
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR035RVH
- **ENCODE file accession:** ENCFF597YXQ
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR035RVH/
- https://www.encodeproject.org/files/ENCFF597YXQ/

---

### `SMC_ENCFF913OTC.bigWig`
- **Cell type:** smooth muscle cell of the brain vasculature
- **Tumour mapping:** LMS
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR000ENG
- **ENCODE file accession:** ENCFF913OTC
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR000ENG/
- https://www.encodeproject.org/files/ENCFF913OTC/

---

### `esoEpi_ENCFF731MXF.bigWig`
- **Cell type:** epithelial cell of oesophagus
- **Tumour mapping:** ESAD / ESCA
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR000ENN
- **ENCODE file accession:** ENCFF731MXF
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR000ENN/
- https://www.encodeproject.org/files/ENCFF731MXF/

---

### `neuroStem_ENCFF495MWO.bigWig`
- **Cell type:** neuronal stem cell
- **Tumour mapping:** NBL
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR278FVO
- **ENCODE file accession:** ENCFF495MWO
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR278FVO/
- https://www.encodeproject.org/files/ENCFF495MWO/

---

### `astro_ENCFF227QKB.bigWig`
- **Cell type:** astrocyte of the hippocampus
- **Tumour mapping:** GBM / LGG
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR000ENA
- **ENCODE file accession:** ENCFF227QKB
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR000ENA/
- https://www.encodeproject.org/files/ENCFF227QKB/

---

### `astro_ENCFF727GJL.bigWig`
- **Cell type:** astrocyte of the spinal cord
- **Tumour mapping:** GBM / LGG
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR000ENB
- **ENCODE file accession:** ENCFF727GJL
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR000ENB/
- https://www.encodeproject.org/files/ENCFF727GJL/

---

### `mela_ENCFF285GEW.bigWig`
- **Cell type:** foreskin melanocyte (male newborn)
- **Tumour mapping:** SKCM
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR434OBM
- **ENCODE file accession:** ENCFF285GEW
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR434OBM/
- https://www.encodeproject.org/files/ENCFF285GEW/

---

### `fibr_ENCFF355OPU.bigWig`
- **Cell type:** foreskin fibroblast (male newborn)
- **Usage:** stromal / background skin accessibility control
- **Assay:** DNase-seq
- **ENCODE experiment:** ENCSR153LHP
- **ENCODE file accession:** ENCFF355OPU
- **Signal type:** read-depth normalised
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/experiments/ENCSR153LHP/
- https://www.encodeproject.org/files/ENCFF355OPU/
