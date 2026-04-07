#!/usr/bin/env python3
"""Build a LIHC-focused master metadata table with harmonised risk annotations.

This script merges TCGA WGS metadata shards, keeps LIHC paired tumour-normal files,
and enriches records with case-level clinical, follow-up, family history, and
annotation data from:
- data/raw/annotations/mmc1.xlsx (Table S1A)
- data/raw/annotations/TCGA.LIHC.sampleMap_LIHC_clinicalMatrix.tsv

It standardises alcohol, viral hepatitis, obesity, and fibrosis-related variables
into explicit harmonised columns and writes one sample-level CSV output. If multiple
file rows map to the same tumour sample, it keeps the largest file (deterministic
tie-break by file name then file id).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIRS = [
    BASE_DIR / "data" / "raw" / "WGS_TCGA25" / "AtoL",
    BASE_DIR / "data" / "raw" / "WGS_TCGA25" / "MtoZ",
]
ANNOTATIONS_DIR = BASE_DIR / "data" / "raw" / "annotations"
DERIVED_DIR = BASE_DIR / "data" / "derived"
OUTPUT_CSV = DERIVED_DIR / "master_metadata.csv"

MISSING_TOKENS = {
    "",
    "'--",
    "--",
    "Not Reported",
    "Not Applicable",
    "Unknown",
    "unknown",
    "not reported",
    "not applicable",
    "[Not Available]",
    "[Unknown]",
    "---",
    "nan",
}

NORMAL_SAMPLE_TYPES = {
    "Blood Derived Normal",
    "Solid Tissue Normal",
    "Bone Marrow Normal",
    "Buccal Cell Normal",
    "EBV Immortalized Normal",
    "Normal",
}

TUMOUR_SAMPLE_TYPES = {
    "Primary Tumor",
    "Recurrent Tumor",
    "Metastatic",
    "Additional - New Primary",
}


def normalise_value(value: object) -> str | None:
    """Return cleaned scalar text or None for missing tokens."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    # Some source fields use a leading apostrophe as a text marker (for example "'23.4").
    if text.startswith("'") and text.count("'") == 1:
        text = text[1:].strip()
    if text in MISSING_TOKENS:
        return None
    return text


def mode_with_tie_break(values: pd.Series) -> tuple[str | None, int]:
    """Pick most frequent non-missing value with deterministic tie-break."""
    clean = values.map(normalise_value).dropna()
    if clean.empty:
        return None, 0
    counts = clean.value_counts()
    top_count = int(counts.iloc[0])
    top_values = sorted(counts[counts == top_count].index.tolist())
    return top_values[0], len(counts)


def aggregate_case_table(
    frame: pd.DataFrame,
    case_col: str,
    value_cols: Iterable[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate case-expanded rows into one row per case."""
    rows: list[dict[str, object]] = []
    for case_id, group in frame.groupby(case_col, dropna=False):
        row: dict[str, object] = {"case_id": case_id}
        notes: list[str] = []
        for col in value_cols:
            selected, n_distinct = mode_with_tie_break(group[col])
            row[col] = selected
            if n_distinct > 1:
                notes.append(f"{prefix}:{col} has {n_distinct} values")
        row[f"{prefix}_aggregation_notes"] = " | ".join(notes) if notes else None
        rows.append(row)
    return pd.DataFrame(rows)


def deduplicate_by_key(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Deduplicate a table by key, keeping first seen row."""
    return frame.drop_duplicates(subset=[key], keep="first")


def has_any_non_missing(series: pd.Series) -> bool:
    """Return True when a series contains at least one non-missing value."""
    return bool(series.map(normalise_value).notna().any())


def select_largest_file_per_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per tumour sample, selecting the largest file per sample."""
    work = frame.copy()
    work["file_size_numeric"] = pd.to_numeric(work["file_size"], errors="coerce").fillna(-1)
    work["sample_key"] = work["tumour_sample_submitter_id"].map(normalise_value)
    work["sample_key"] = work["sample_key"].fillna(work["tumour_sample_id"].map(normalise_value))
    work = work[work["sample_key"].notna()].copy()
    work = work.sort_values(
        ["sample_key", "file_size_numeric", "file_name", "file_id"],
        ascending=[True, False, True, True],
    )
    work = work.drop_duplicates(subset=["sample_key"], keep="first").copy()
    return work.drop(columns=["sample_key", "file_size_numeric"])


def assert_unique_key(frame: pd.DataFrame, key_col: str, label: str) -> None:
    """Assert that a key column is unique after excluding missing keys."""
    work = frame.copy()
    work[key_col] = work[key_col].map(normalise_value)
    work = work[work[key_col].notna()].copy()
    if work.empty:
        return
    duplicate_mask = work.duplicated(subset=[key_col], keep=False)
    if not duplicate_mask.any():
        return
    duplicate_keys = sorted(work.loc[duplicate_mask, key_col].unique().tolist())
    preview = ", ".join(duplicate_keys[:10])
    extra = "" if len(duplicate_keys) <= 10 else f" (+{len(duplicate_keys) - 10} more)"
    raise ValueError(f"{label} key '{key_col}' is not unique for keys: {preview}{extra}")


def load_json_shards(filename: str) -> list[dict[str, object]]:
    """Load JSON objects from raw shards and deduplicate by file_id."""
    objects: list[dict[str, object]] = []
    for raw_dir in RAW_DIRS:
        path = raw_dir / filename
        if not path.exists():
            continue
        shard = json.loads(path.read_text())
        print(f"{filename} | {raw_dir}: loaded_rows={len(shard)}")
        objects.extend(shard)

    merged: dict[str, dict[str, object]] = {}
    for item in objects:
        file_id = item.get("file_id")
        if file_id is not None and file_id not in merged:
            merged[file_id] = item

    out = list(merged.values())
    print(f"{filename}: deduplicated_rows={len(out)}")
    return out


def load_tsv_shards(filename: str, dedup_key: str | None = None, dedup_exact: bool = False) -> pd.DataFrame:
    """Load TSV shards and deduplicate according to rule."""
    frames: list[pd.DataFrame] = []
    for raw_dir in RAW_DIRS:
        path = raw_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t", dtype=str)
        print(f"{filename} | {raw_dir}: loaded_rows={len(frame)}")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if dedup_key is not None and not merged.empty:
        merged = deduplicate_by_key(merged, dedup_key)
    elif dedup_exact and not merged.empty:
        merged = merged.drop_duplicates()

    print(f"{filename}: deduplicated_rows={len(merged)}")
    return merged


def build_file_aliquot_pairs(metadata: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract file-level metadata and associated aliquot rows."""
    file_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for item in metadata:
        analysis = item.get("analysis", {})
        file_rows.append(
            {
                "file_id": item.get("file_id"),
                "file_name": item.get("file_name"),
                "data_category": item.get("data_category"),
                "data_type": item.get("data_type"),
                "experimental_strategy": item.get("experimental_strategy"),
                "workflow_type": analysis.get("workflow_type"),
                "workflow_version": analysis.get("workflow_version"),
                "file_size": item.get("file_size"),
            }
        )
        for entity in item.get("associated_entities", []):
            if entity.get("entity_type") != "aliquot":
                continue
            pair_rows.append(
                {
                    "file_id": item.get("file_id"),
                    "aliquot_id": entity.get("entity_id"),
                }
            )

    return pd.DataFrame(file_rows), pd.DataFrame(pair_rows)


def split_gdc_sample_sheet(sample_sheet: pd.DataFrame) -> pd.DataFrame:
    """Split paired comma-separated sample sheet values into row-wise records."""
    rows: list[dict[str, object]] = []
    for row in sample_sheet.itertuples(index=False):
        sample_parts = [part.strip() for part in str(getattr(row, "Sample_ID")).split(",")]
        type_parts = [part.strip() for part in str(getattr(row, "Sample_Type")).split(",")]
        n = max(len(sample_parts), len(type_parts))
        for idx in range(n):
            rows.append(
                {
                    "file_id": getattr(row, "File_ID"),
                    "gdc_sample_submitter_id": sample_parts[idx] if idx < len(sample_parts) else None,
                    "gdc_sample_type": type_parts[idx] if idx < len(type_parts) else None,
                }
            )
    return pd.DataFrame(rows)


def assign_tumour_normal(group: pd.DataFrame) -> pd.Series:
    """Assign tumour and normal entities for one file from paired rows."""
    work = group.copy()
    work["sample_type_clean"] = work["sample_type"].map(normalise_value)
    work["tissue_type_clean"] = work["tissue_type"].map(normalise_value)

    work["is_normal"] = work["sample_type_clean"].isin(NORMAL_SAMPLE_TYPES) | (work["tissue_type_clean"] == "Normal")
    work["is_tumour"] = work["sample_type_clean"].isin(TUMOUR_SAMPLE_TYPES) | (work["tissue_type_clean"] == "Tumor")

    work = work.sort_values(["is_tumour", "is_normal", "sample_submitter_id"], ascending=[False, False, True])

    notes: list[str] = []
    tumour_candidates = work[work["is_tumour"]]
    normal_candidates = work[work["is_normal"]]

    tumour_row = tumour_candidates.iloc[0] if not tumour_candidates.empty else work.iloc[0]
    if tumour_candidates.empty:
        notes.append("tumour assignment fallback to first paired sample")

    normal_row = None
    if not normal_candidates.empty:
        normal_candidates = normal_candidates[normal_candidates["sample_id"] != tumour_row["sample_id"]]
        if not normal_candidates.empty:
            normal_row = normal_candidates.iloc[0]
        else:
            notes.append("normal candidate overlapped assigned tumour sample")
    else:
        notes.append("no clear normal sample")

    has_paired_normal = normal_row is not None
    tumour_normal_class = "paired_tumour_normal" if has_paired_normal else "tumour_only"

    return pd.Series(
        {
            "file_id": group.name,
            "project_id": tumour_row["project_id"],
            "case_id": tumour_row["case_id"],
            "case_submitter_id": tumour_row["case_submitter_id"],
            "tumour_sample_id": tumour_row["sample_id"],
            "tumour_sample_submitter_id": tumour_row["sample_submitter_id"],
            "normal_sample_id": normal_row["sample_id"] if normal_row is not None else None,
            "normal_sample_submitter_id": normal_row["sample_submitter_id"] if normal_row is not None else None,
            "tumour_aliquot_id": tumour_row["aliquot_id"],
            "normal_aliquot_id": normal_row["aliquot_id"] if normal_row is not None else None,
            "tumour_sample_type": tumour_row["sample_type"],
            "normal_sample_type": normal_row["sample_type"] if normal_row is not None else None,
            "tumour_tissue_type": tumour_row["tissue_type"],
            "normal_tissue_type": normal_row["tissue_type"] if normal_row is not None else None,
            "has_paired_normal": has_paired_normal,
            "tumour_normal_class": tumour_normal_class,
            "qc_notes": " | ".join(notes) if notes else None,
        }
    )


def tokenise_risk_text(value: object) -> set[str]:
    """Tokenise a risk-factor free-text field."""
    text = normalise_value(value)
    if text is None:
        return set()
    parts = [p.strip().lower() for p in str(text).split("|") if p.strip()]
    return set(parts)


def yes_no_from_value(value: object, positive_tokens: set[str]) -> str | None:
    """Convert free-text/category value to yes/no/None for a specific factor."""
    text = normalise_value(value)
    if text is None:
        return None
    t = text.lower()
    if t in {"no", "none", "no history of primary risk factors"}:
        return "no"
    if any(tok in t for tok in positive_tokens):
        return "yes"
    return None


def combine_yes_no(values: list[str | None]) -> tuple[str | None, bool]:
    """Combine source yes/no values into one label plus conflict flag."""
    has_yes = any(v == "yes" for v in values)
    has_no = any(v == "no" for v in values)
    if has_yes and has_no:
        return "yes", True
    if has_yes:
        return "yes", False
    if has_no:
        return "no", False
    return None, False


def parse_float(value: object) -> float | None:
    """Parse float safely from text value."""
    text = normalise_value(value)
    if text is None:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def bmi_class_from_value(bmi: float | None) -> str | None:
    """Map BMI numeric value to WHO international BMI classes."""
    if bmi is None:
        return None
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obesity Class I"
    if bmi < 40:
        return "Obesity Class II"
    return "Obesity Class III"


def normalise_ishak_text(value: object) -> str | None:
    """Normalise Ishak fibrosis text to canonical labels."""
    text = normalise_value(value)
    if text is None:
        return None
    t = text.lower()
    if "0" in t and "fibrosis" in t and "no" in t:
        return "0 - No Fibrosis"
    if "1,2" in t or "1-2" in t:
        return "1,2 - Portal Fibrosis"
    if "3,4" in t or "3-4" in t:
        return "3,4 - Fibrous Septa"
    if t.startswith("5") or "incomplete cirrhosis" in t:
        return "5 - Nodular Formation and Incomplete Cirrhosis"
    if t.startswith("6") or "established cirrhosis" in t:
        return "6 - Established Cirrhosis"
    return text


def ishak_to_ordinal(value: object) -> int | None:
    """Map canonical Ishak label to ordinal stage."""
    mapping = {
        "0 - No Fibrosis": 0,
        "1,2 - Portal Fibrosis": 1,
        "3,4 - Fibrous Septa": 2,
        "5 - Nodular Formation and Incomplete Cirrhosis": 3,
        "6 - Established Cirrhosis": 4,
    }
    return mapping.get(normalise_ishak_text(value))


def fibrosis_present_from_ishak(value: object) -> str | None:
    """Derive fibrosis presence from Ishak label."""
    stage = ishak_to_ordinal(value)
    if stage is None:
        return None
    return "no" if stage == 0 else "yes"


def load_annotations_harmonised() -> pd.DataFrame:
    """Load and harmonise risk annotations from mmc1 and clinicalMatrix."""
    mmc_path = ANNOTATIONS_DIR / "mmc1.xlsx"
    cm_path = ANNOTATIONS_DIR / "TCGA.LIHC.sampleMap_LIHC_clinicalMatrix.tsv"

    s1a = pd.read_excel(mmc_path, sheet_name="Table S1A - core sample set", header=3, dtype=str)
    s1a.columns = [str(c).strip() for c in s1a.columns]
    s1a = s1a[s1a["UUID"].notna()].copy()
    s1a["tumour_sample_id"] = s1a["UUID"].str.lower().str.strip()

    cm = pd.read_csv(cm_path, sep="\t", dtype=str)
    cm.columns = [str(c).strip() for c in cm.columns]
    cm["sample_barcode_norm"] = cm["bcr_sample_barcode"].astype(str).str.strip()

    # Build source-specific fields from mmc1
    mmc = s1a[[
        "tumour_sample_id",
        "Barcode",
        "history_hepato_carcinoma_risk_factor",
        "Alcoholic liver disease",
        "Hepatitis B",
        "Hepatitis C",
        "HBV_consensus",
        "HCV_consensus",
        "NAFLD",
        "viral_hepatitis_serology",
        "BMI",
        "ObesityClass1",
        "ObesityClass2",
        "Cirrhosis",
        "liver_fibrosis_ishak_score_category",
    ]].copy()
    mmc["tumour_sample_id"] = mmc["tumour_sample_id"].map(normalise_value)
    mmc = mmc[mmc["tumour_sample_id"].notna()].copy()

    # Build source-specific fields from clinicalMatrix (sample barcode key)
    cm_fields = cm[[
        "sample_barcode_norm",
        "hist_hepato_carc_fact",
        "viral_hepatitis_serology",
        "weight",
        "height",
        "fibrosis_ishak_score",
    ]].copy()
    cm_fields["sample_barcode_norm"] = cm_fields["sample_barcode_norm"].map(normalise_value)
    cm_fields = cm_fields[cm_fields["sample_barcode_norm"].notna()].copy()

    # Produce normalised risk flags in mmc1 source
    mmc["alcohol_mmc1"] = mmc["Alcoholic liver disease"].apply(
        lambda x: yes_no_from_value(x, {"alcohol"})
    )
    mmc["hbv_mmc1"] = mmc["Hepatitis B"].apply(lambda x: yes_no_from_value(x, {"hepatitis b", "hbv"}))
    mmc["hcv_mmc1"] = mmc["Hepatitis C"].apply(lambda x: yes_no_from_value(x, {"hepatitis c", "hcv"}))
    mmc["hbv_consensus_mmc1"] = mmc["HBV_consensus"].apply(
        lambda x: yes_no_from_value(x, {"hepatitis b", "hbv", "positive"})
    )
    mmc["hcv_consensus_mmc1"] = mmc["HCV_consensus"].apply(
        lambda x: yes_no_from_value(x, {"hepatitis c", "hcv", "positive"})
    )
    mmc["nafld_mmc1"] = mmc["NAFLD"].apply(lambda x: yes_no_from_value(x, {"nafld", "non-alcoholic"}))
    mmc["bmi_mmc1"] = mmc["BMI"].apply(parse_float)
    mmc["obesity_class_mmc1"] = mmc["ObesityClass1"].map(normalise_value)

    # clinicalMatrix source flags from free text
    cm_fields["alcohol_cm"] = cm_fields["hist_hepato_carc_fact"].apply(
        lambda x: "yes" if any("alcohol" in t for t in tokenise_risk_text(x)) else None
    )
    cm_fields["hbv_cm"] = cm_fields["hist_hepato_carc_fact"].apply(
        lambda x: "yes" if any(("hepatitis b" in t or "hbv" in t) for t in tokenise_risk_text(x)) else None
    )
    cm_fields["hcv_cm"] = cm_fields["hist_hepato_carc_fact"].apply(
        lambda x: "yes" if any(("hepatitis c" in t or "hcv" in t) for t in tokenise_risk_text(x)) else None
    )
    cm_fields["nafld_cm"] = cm_fields["hist_hepato_carc_fact"].apply(
        lambda x: "yes" if any(("nafld" in t or "non-alcoholic fatty liver disease" in t) for t in tokenise_risk_text(x)) else None
    )

    # Viral serology enrichment from clinicalMatrix
    cm_fields["hbv_serology_cm"] = cm_fields["viral_hepatitis_serology"].apply(
        lambda x: "yes" if x is not None and any(tok in str(x).lower() for tok in ["hbv", "hepatitis b"]) else None
    )
    cm_fields["hcv_serology_cm"] = cm_fields["viral_hepatitis_serology"].apply(
        lambda x: "yes" if x is not None and any(tok in str(x).lower() for tok in ["hcv", "hepatitis c"]) else None
    )

    cm_fields["weight_kg_cm"] = cm_fields["weight"].apply(parse_float)
    cm_fields["height_cm_cm"] = cm_fields["height"].apply(parse_float)
    cm_fields["bmi_cm"] = cm_fields.apply(
        lambda r: (r["weight_kg_cm"] / ((r["height_cm_cm"] / 100) ** 2))
        if r["weight_kg_cm"] is not None and r["height_cm_cm"] is not None and r["height_cm_cm"] > 0
        else None,
        axis=1,
    )

    # Ensure join keys are unique before merge.
    assert_unique_key(mmc, "tumour_sample_id", "mmc1 annotations")
    assert_unique_key(cm_fields, "sample_barcode_norm", "clinicalMatrix annotations")

    # return both source frames for later joins
    mmc = mmc.drop_duplicates(subset=["tumour_sample_id"])
    cm_fields = cm_fields.drop_duplicates(subset=["sample_barcode_norm"])
    mmc["fibrosis_ishak_mmc1"] = mmc["liver_fibrosis_ishak_score_category"].apply(normalise_ishak_text)
    cm_fields["fibrosis_ishak_cm"] = cm_fields["fibrosis_ishak_score"].apply(normalise_ishak_text)

    return mmc, cm_fields


def main() -> None:
    """Build LIHC master metadata with harmonised annotation variables."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_json_shards("metadata.cart.2025-03-19.json")
    gdc_sample_sheet = load_tsv_shards("gdc_sample_sheet.2025-03-19.tsv", dedup_key="File ID")
    aliquot = load_tsv_shards("aliquot.tsv", dedup_key="aliquot_id")
    analyte = load_tsv_shards("analyte.tsv", dedup_key="analyte_id")
    portion = load_tsv_shards("portion.tsv", dedup_key="portion_id")
    sample = load_tsv_shards("sample.tsv", dedup_key="sample_id")
    clinical = load_tsv_shards("clinical.tsv", dedup_exact=True)
    follow_up = load_tsv_shards("follow_up.tsv", dedup_exact=True)
    family_history = load_tsv_shards("family_history.tsv", dedup_exact=True)
    pathology = load_tsv_shards("pathology_detail.tsv", dedup_exact=True)

    gdc_sample_sheet = gdc_sample_sheet.rename(
        columns={
            "File ID": "File_ID",
            "Sample ID": "Sample_ID",
            "Sample Type": "Sample_Type",
        }
    )

    file_meta, file_aliquot_map = build_file_aliquot_pairs(metadata)
    gdc_pairs = split_gdc_sample_sheet(gdc_sample_sheet)

    biospec = file_aliquot_map.merge(
        aliquot[["aliquot_id", "analyte_id", "sample_id", "sample_submitter_id", "case_id", "case_submitter_id", "project_id"]],
        on="aliquot_id",
        how="left",
        validate="m:1",
    )
    biospec = biospec.merge(
        analyte[["analyte_id", "portion_id"]],
        on="analyte_id",
        how="left",
        validate="m:1",
    )
    biospec = biospec.merge(
        sample[["sample_id", "sample_type", "tissue_type", "tumor_descriptor"]],
        on="sample_id",
        how="left",
        validate="m:1",
    )
    biospec = biospec.merge(
        gdc_pairs[["file_id", "gdc_sample_submitter_id", "gdc_sample_type"]],
        left_on=["file_id", "sample_submitter_id"],
        right_on=["file_id", "gdc_sample_submitter_id"],
        how="left",
    )

    role_table = (
        biospec.groupby("file_id", group_keys=False)
        .apply(assign_tumour_normal, include_groups=False)
        .reset_index(drop=True)
    )

    clinical_cols = [
        "primary_diagnosis",
        "morphology",
        "tumor_grade",
        "ishak_fibrosis_score",
        "age_at_diagnosis",
        "age_at_index",
        "days_to_birth",
        "gender",
        "race",
        "ethnicity",
        "year_of_diagnosis",
        "vital_status",
        "classification_of_tumor",
        "tissue_or_organ_of_origin",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "ajcc_pathologic_stage",
        "child_pugh_classification",
        "initial_disease_status",
        "prior_treatment",
        "treatment_or_therapy",
        "treatment_type",
        "treatment_intent_type",
        "days_to_treatment_start",
    ]
    follow_up_cols = [
        "days_to_follow_up",
        "ecog_performance_status",
        "disease_response",
        "progression_or_recurrence",
        "days_to_recurrence",
        "progression_or_recurrence_type",
        "progression_or_recurrence_anatomic_site",
    ]
    family_history_cols = [
        "relative_with_cancer_history",
        "relatives_with_cancer_history_count",
        "relationship_primary_diagnosis",
    ]
    pathology_cols = [
        "additional_pathology_findings",
        "histologic_progression_type",
        "vascular_invasion_present",
        "vascular_invasion_type",
        "margin_status",
    ]

    clinical_agg = aggregate_case_table(clinical, "case_id", clinical_cols, "clinical")
    follow_up_agg = aggregate_case_table(follow_up, "case_id", follow_up_cols, "follow_up")
    family_history_agg = aggregate_case_table(family_history, "case_id", family_history_cols, "family_history")
    pathology_agg = aggregate_case_table(pathology, "case_id", pathology_cols, "pathology")

    master = file_meta.merge(role_table, on="file_id", how="left", validate="1:1")
    master = master.merge(clinical_agg[["case_id"] + clinical_cols + ["clinical_aggregation_notes"]], on="case_id", how="left")
    master = master.merge(
        follow_up_agg[["case_id"] + follow_up_cols + ["follow_up_aggregation_notes"]],
        on="case_id",
        how="left",
    )
    master = master.merge(
        family_history_agg[["case_id"] + family_history_cols + ["family_history_aggregation_notes"]],
        on="case_id",
        how="left",
    )
    master = master.merge(pathology_agg[["case_id"] + pathology_cols + ["pathology_aggregation_notes"]], on="case_id", how="left")
    master = master.rename(columns={"tumor_grade": "tumour_grade"})

    # LIHC-only focus
    master = master[master["project_id"] == "TCGA-LIHC"].copy()
    master = select_largest_file_per_sample(master)

    # Annotation harmonisation joins
    mmc, cm = load_annotations_harmonised()
    master["tumour_sample_id_norm"] = master["tumour_sample_id"].astype(str).str.lower().str.strip()
    master["tumour_sample_submitter_id_norm"] = master["tumour_sample_submitter_id"].astype(str).str.strip()

    master = master.merge(mmc, left_on="tumour_sample_id_norm", right_on="tumour_sample_id", how="left", suffixes=("", "_mmc"))
    master = master.merge(cm, left_on="tumour_sample_submitter_id_norm", right_on="sample_barcode_norm", how="left")

    # Harmonised yes/no variables with source conflict flags
    alcohol_final, alcohol_conflict = zip(
        *master.apply(
            lambda r: combine_yes_no([r.get("alcohol_mmc1"), r.get("alcohol_cm")]), axis=1
        )
    )
    hbv_final, hbv_conflict = zip(
        *master.apply(
            lambda r: combine_yes_no(
                [
                    r.get("hbv_consensus_mmc1"),
                    r.get("hbv_mmc1"),
                    r.get("hbv_cm"),
                    r.get("hbv_serology_cm"),
                ]
            ),
            axis=1,
        )
    )
    hcv_final, hcv_conflict = zip(
        *master.apply(
            lambda r: combine_yes_no(
                [
                    r.get("hcv_consensus_mmc1"),
                    r.get("hcv_mmc1"),
                    r.get("hcv_cm"),
                    r.get("hcv_serology_cm"),
                ]
            ),
            axis=1,
        )
    )
    nafld_final, nafld_conflict = zip(
        *master.apply(
            lambda r: combine_yes_no([r.get("nafld_mmc1"), r.get("nafld_cm")]), axis=1
        )
    )

    master["alcohol"] = list(alcohol_final)
    master["hbv"] = list(hbv_final)
    master["hcv"] = list(hcv_final)
    master["nafld"] = list(nafld_final)
    master["alcohol_conflict"] = list(alcohol_conflict)
    master["hbv_conflict"] = list(hbv_conflict)
    master["hcv_conflict"] = list(hcv_conflict)
    master["nafld_conflict"] = list(nafld_conflict)

    # Obesity harmonisation (calculate BMI from height/weight first; fall back to curated mmc1 BMI)
    master["bmi_mmc1"] = master["bmi_mmc1"].apply(lambda x: parse_float(x))
    master["bmi_cm"] = master["bmi_cm"].apply(lambda x: parse_float(x))
    master["bmi"] = master.apply(
        lambda r: r["bmi_cm"] if pd.notna(r["bmi_cm"]) else (r["bmi_mmc1"] if pd.notna(r["bmi_mmc1"]) else None),
        axis=1,
    )
    master["obesity_class"] = master["bmi"].apply(bmi_class_from_value)
    master["obesity_class_mmc1"] = master["obesity_class_mmc1"].map(normalise_value)
    master["cirrhosis"] = master["Cirrhosis"].apply(
        lambda x: yes_no_from_value(x, {"cirrhosis"})
    )

    # Fibrosis standardisation with a single source of truth:
    # clinical.tsv ishak_fibrosis_score is treated as canonical because it is part of
    # the TCGA clinical data model used in the metadata join path.
    master["fibrosis_ishak_clinical"] = master["ishak_fibrosis_score"].apply(normalise_ishak_text)
    master["fibrosis_ishak_source_of_truth"] = master["fibrosis_ishak_clinical"]
    master["fibrosis_ishak_cm"] = master["fibrosis_ishak_cm"].apply(normalise_ishak_text)
    master["fibrosis_ishak_mmc1"] = master["fibrosis_ishak_mmc1"].apply(normalise_ishak_text)
    master["fibrosis_ishak_ordinal"] = master["fibrosis_ishak_source_of_truth"].apply(ishak_to_ordinal)
    master["fibrosis_present"] = master["fibrosis_ishak_source_of_truth"].apply(fibrosis_present_from_ishak)
    master["fibrosis_source"] = "clinical.tsv (case-level aggregated)"
    master["fibrosis_conflict_with_cm"] = master.apply(
        lambda r: (
            normalise_value(r["fibrosis_ishak_source_of_truth"]) is not None
            and normalise_value(r["fibrosis_ishak_cm"]) is not None
            and normalise_value(r["fibrosis_ishak_source_of_truth"]) != normalise_value(r["fibrosis_ishak_cm"])
        ),
        axis=1,
    )
    master["fibrosis_conflict_with_mmc1"] = master.apply(
        lambda r: (
            normalise_value(r["fibrosis_ishak_source_of_truth"]) is not None
            and normalise_value(r["fibrosis_ishak_mmc1"]) is not None
            and normalise_value(r["fibrosis_ishak_source_of_truth"]) != normalise_value(r["fibrosis_ishak_mmc1"])
        ),
        axis=1,
    )

    # Final clean-up
    master["ishak_fibrosis_score"] = master["fibrosis_ishak_source_of_truth"]
    master = master.drop(
        columns=[
            c
            for c in [
                "tumour_sample_id_norm",
                "tumour_sample_id_mmc",
                "tumour_sample_submitter_id_norm",
                "sample_barcode_norm",
            ]
            if c in master.columns
        ]
    )

    output_columns = [
        "file_id",
        "file_name",
        "data_category",
        "data_type",
        "experimental_strategy",
        "workflow_type",
        "workflow_version",
        "project_id",
        "case_id",
        "case_submitter_id",
        "tumour_sample_id",
        "tumour_sample_submitter_id",
        "normal_sample_id",
        "normal_sample_submitter_id",
        "tumour_aliquot_id",
        "normal_aliquot_id",
        "tumour_sample_type",
        "normal_sample_type",
        "tumour_tissue_type",
        "normal_tissue_type",
        "has_paired_normal",
        "tumour_normal_class",
        "qc_notes",
        "primary_diagnosis",
        "morphology",
        "tumour_grade",
        "age_at_diagnosis",
        "age_at_index",
        "days_to_birth",
        "gender",
        "race",
        "ethnicity",
        "year_of_diagnosis",
        "vital_status",
        "classification_of_tumor",
        "tissue_or_organ_of_origin",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "ajcc_pathologic_stage",
        "child_pugh_classification",
        "initial_disease_status",
        "prior_treatment",
        "treatment_or_therapy",
        "treatment_type",
        "treatment_intent_type",
        "days_to_treatment_start",
        "days_to_follow_up",
        "ecog_performance_status",
        "disease_response",
        "progression_or_recurrence",
        "days_to_recurrence",
        "progression_or_recurrence_type",
        "progression_or_recurrence_anatomic_site",
        "relative_with_cancer_history",
        "relatives_with_cancer_history_count",
        "relationship_primary_diagnosis",
        "ishak_fibrosis_score",
        "clinical_aggregation_notes",
        "follow_up_aggregation_notes",
        "family_history_aggregation_notes",
        "additional_pathology_findings",
        "histologic_progression_type",
        "vascular_invasion_present",
        "vascular_invasion_type",
        "margin_status",
        "pathology_aggregation_notes",
        "alcohol",
        "hbv",
        "hcv",
        "nafld",
        "bmi",
        "obesity_class",
        "cirrhosis",
        "fibrosis_ishak_ordinal",
        "fibrosis_present",
    ]
    output_columns = [
        c
        for c in output_columns
        if c in master.columns and has_any_non_missing(master[c])
    ]
    master = master[output_columns].copy()

    master.to_csv(OUTPUT_CSV, index=False)

    print("\nLIHC master metadata built.")
    print(f"rows={len(master)}")
    print(f"unique_files={master['file_id'].nunique()}")
    print(f"alcohol_non_missing={(master['alcohol'].notna()).sum()}")
    print(f"hbv_non_missing={(master['hbv'].notna()).sum()}")
    print(f"hcv_non_missing={(master['hcv'].notna()).sum()}")
    print(f"bmi_non_missing={(master['bmi'].notna()).sum()}")
    print(f"fibrosis_non_missing={(master['ishak_fibrosis_score'].notna()).sum()}")
    print(f"output={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
