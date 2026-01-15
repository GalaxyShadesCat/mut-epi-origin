# DNase-seq accessibility tracks (ENCODE)

This directory contains genome-wide **DNase-seq accessibility** signal tracks (bigWig) from the **ENCODE Portal**, representing human skin-related primary cell types used as targets in the mutation-vs-accessibility pipeline.

All three tracks below are **read-depth normalised signal** on **hg19 (GRCh37)**, so they are directly comparable to each other (assembly-consistent).

## Files

### `fibr_ENCFF355OPU.bigWig`
- **Cell type:** foreskin fibroblast (male newborn)
- **Assay:** DNase-seq
- **ENCODE experiment (dataset):** ENCSR153LHP
- **ENCODE file accession:** ENCFF355OPU
- **Signal / output type:** read-depth normalised signal
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/files/ENCFF355OPU/
- https://www.encodeproject.org/experiments/ENCSR153LHP/

---

### `mela_ENCFF285GEW.bigWig`
- **Cell type:** foreskin melanocyte (male newborn)
- **Assay:** DNase-seq
- **ENCODE experiment (dataset):** ENCSR434OBM
- **ENCODE file accession:** ENCFF285GEW
- **Signal / output type:** read-depth normalised signal
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/files/ENCFF285GEW/
- https://www.encodeproject.org/experiments/ENCSR434OBM/

---

### `kera_ENCFF597YXQ.bigWig`
- **Cell type:** foreskin keratinocyte (male newborn)
- **Assay:** DNase-seq
- **ENCODE experiment (dataset):** ENCSR035RVH
- **ENCODE file accession:** ENCFF597YXQ
- **Signal / output type:** read-depth normalised signal
- **Genome assembly:** hg19 (GRCh37)

ENCODE:
- https://www.encodeproject.org/files/ENCFF597YXQ/
- https://www.encodeproject.org/experiments/ENCSR035RVH/

## Pipeline mapping

```python
DNASE_MAP = {
    "mela": PROJECT_ROOT / "data" / "raw" / "DNase-seq" / "mela_ENCFF285GEW.bigWig",
    "kera": PROJECT_ROOT / "data" / "raw" / "DNase-seq" / "kera_ENCFF597YXQ.bigWig",
    "fibr": PROJECT_ROOT / "data" / "raw" / "DNase-seq" / "fibr_ENCFF355OPU.bigWig",
}
