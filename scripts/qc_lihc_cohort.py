#!/usr/bin/env python3
"""Run QC and conservative cleaning for the LIHC master cohort metadata.

This script audits:
- data/derived/master_sample_metadata.csv

It normalises missing tokens, removes exact duplicate rows, filters to LIHC rows with
targeted phenotype completeness (fibrosis, NAFLD, and HCV variants), and writes
simplified outputs with one column per major risk attribute (alcohol, viral hepatitis,
NAFLD, obesity, fibrosis), plus explicit missingness logs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DERIVED_DIR = BASE_DIR / "data" / "derived"
INPUT_PATH = DERIVED_DIR / "master_sample_metadata.csv"

CLEANED_CSV = DERIVED_DIR / "master_sample_metadata_cleaned.csv"
FIBROSIS_CSV = DERIVED_DIR / "master_sample_metadata_lihc_fibrosis.csv"
NAFLD_CSV = DERIVED_DIR / "master_sample_metadata_lihc_nafld.csv"
HCV_CSV = DERIVED_DIR / "master_sample_metadata_lihc_hcv.csv"
ALL_CSV = DERIVED_DIR / "master_sample_metadata_lihc_all.csv"
ROWS_WITH_ISSUES_CSV = DERIVED_DIR / "master_sample_metadata_rows_with_issues.csv"
QC_REPORT = DERIVED_DIR / "master_sample_metadata_qc_report.txt"

MISSING_TOKENS = {
    "",
    "'--",
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

ALLOWED_PRIMARY_DIAGNOSES = {
    "Hepatocellular carcinoma, NOS",
    "Hepatocellular carcinoma, clear cell type",
}


def normalise_value(value: object) -> str | None:
    """Return cleaned text or None for missing tokens."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in MISSING_TOKENS:
        return None
    return text


def trim_and_normalise_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and map known missing tokens to NA."""
    cleaned = frame.copy()
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()
        cleaned[col] = cleaned[col].apply(lambda v: pd.NA if normalise_value(v) is None else normalise_value(v))
    return cleaned


def missingness_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build missingness metrics for selected variables."""
    rows: list[dict[str, object]] = []
    total = len(frame)
    for col in columns:
        if col not in frame.columns:
            rows.append(
                {
                    "column": col,
                    "non_missing": 0,
                    "missing": total,
                    "missing_pct": 100.0 if total else 0.0,
                    "note": "column_missing",
                }
            )
            continue
        series = frame[col].apply(normalise_value)
        missing = int(series.isna().sum())
        non_missing = int(total - missing)
        rows.append(
            {
                "column": col,
                "non_missing": non_missing,
                "missing": missing,
                "missing_pct": round((missing / total) * 100, 2) if total else 0.0,
                "note": None,
            }
        )
    return pd.DataFrame(rows)


def build_simplified_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep relevant columns and standardise to one column per risk attribute."""
    out = frame.copy()
    out["alcohol_status"] = out["risk_alcohol"].apply(normalise_value) if "risk_alcohol" in out.columns else pd.NA
    out["hbv_status"] = out["risk_hbv"].apply(normalise_value) if "risk_hbv" in out.columns else pd.NA
    out["hcv_status"] = out["risk_hcv"].apply(normalise_value) if "risk_hcv" in out.columns else pd.NA
    out["nafld_status"] = out["risk_nafld"].apply(normalise_value) if "risk_nafld" in out.columns else pd.NA
    out["obesity_class"] = (
        out["obesity_class_from_bmi"].apply(normalise_value) if "obesity_class_from_bmi" in out.columns else pd.NA
    )
    out["fibrosis_ishak_score"] = (
        out["fibrosis_ishak_source_of_truth"].apply(normalise_value)
        if "fibrosis_ishak_source_of_truth" in out.columns
        else out["ishak_fibrosis_score"].apply(normalise_value)
    )

    keep_cols = [
        "file_id",
        "file_name",
        "project_id",
        "case_id",
        "case_submitter_id",
        "tumour_sample_id",
        "tumour_sample_submitter_id",
        "normal_sample_id",
        "normal_sample_submitter_id",
        "primary_diagnosis",
        "tumour_sample_type",
        "normal_sample_type",
        "alcohol_status",
        "hbv_status",
        "hcv_status",
        "nafld_status",
        "obesity_class",
        "fibrosis_ishak_score",
        "fibrosis_present",
        "qc_notes",
        "row_issue_flag",
        "row_issue_notes",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()


def _lihc_tumour_base_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply shared LIHC primary tumour filters used by cohort subsets."""
    return frame[
        (frame["project_id"] == "TCGA-LIHC")
        & frame["primary_diagnosis"].apply(normalise_value).isin(ALLOWED_PRIMARY_DIAGNOSES)
        & (frame["tumour_sample_type"].apply(normalise_value) == "Primary Tumor")
    ].copy()


def main() -> None:
    """Run QC, write cleaned and fibrosis-filtered outputs, and report missingness."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(INPUT_PATH, dtype=str)
    cleaned = trim_and_normalise_missing(raw)

    exact_duplicate_mask = cleaned.duplicated(keep="first")
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    if exact_duplicate_count:
        cleaned = cleaned.loc[~exact_duplicate_mask].copy()

    issue_notes: list[str | None] = []
    for row in cleaned.itertuples(index=False):
        notes: list[str] = []

        if normalise_value(getattr(row, "project_id", None)) != "TCGA-LIHC":
            notes.append("project_id_not_TCGA-LIHC")
        if normalise_value(getattr(row, "file_id", None)) is None:
            notes.append("missing_file_id")
        if normalise_value(getattr(row, "file_name", None)) is None:
            notes.append("missing_file_name")
        if normalise_value(getattr(row, "ishak_fibrosis_score", None)) is None:
            notes.append("missing_ishak_fibrosis_score")

        issue_notes.append(" | ".join(notes) if notes else None)

    cleaned["row_issue_notes"] = issue_notes
    cleaned["row_issue_flag"] = cleaned["row_issue_notes"].notna()

    lihc_tumour_base = _lihc_tumour_base_filter(cleaned)
    all_lihc = lihc_tumour_base[
        lihc_tumour_base["risk_alcohol"].apply(normalise_value).notna()
        & lihc_tumour_base["risk_hbv"].apply(normalise_value).notna()
        & lihc_tumour_base["risk_hcv"].apply(normalise_value).notna()
        & lihc_tumour_base["risk_nafld"].apply(normalise_value).notna()
        & lihc_tumour_base["obesity_class_from_bmi"].apply(normalise_value).notna()
        & lihc_tumour_base["ishak_fibrosis_score"].apply(normalise_value).notna()
    ].copy()
    fibrosis = lihc_tumour_base[
        lihc_tumour_base["ishak_fibrosis_score"].apply(normalise_value).notna()
    ].copy()
    if "risk_nafld" in lihc_tumour_base.columns:
        nafld = lihc_tumour_base[
            lihc_tumour_base["risk_nafld"].apply(normalise_value).notna()
        ].copy()
    else:
        nafld = lihc_tumour_base.iloc[0:0].copy()
    if "risk_hcv" in lihc_tumour_base.columns:
        hcv = lihc_tumour_base[
            lihc_tumour_base["risk_hcv"].apply(normalise_value).notna()
        ].copy()
    else:
        hcv = lihc_tumour_base.iloc[0:0].copy()

    cleaned_simple = build_simplified_table(cleaned)
    all_lihc_simple = build_simplified_table(all_lihc)
    fibrosis_simple = build_simplified_table(fibrosis)
    nafld_simple = build_simplified_table(nafld)
    hcv_simple = build_simplified_table(hcv)

    risk_cols = [
        "alcohol_status",
        "hbv_status",
        "hcv_status",
        "nafld_status",
        "obesity_class",
        "fibrosis_ishak_score",
    ]

    missing_cleaned = missingness_metrics(cleaned_simple, risk_cols)
    missing_all_lihc = missingness_metrics(all_lihc_simple, risk_cols)
    missing_fibrosis = missingness_metrics(fibrosis_simple, risk_cols)
    missing_nafld = missingness_metrics(nafld_simple, risk_cols)
    missing_hcv = missingness_metrics(hcv_simple, risk_cols)

    rows_with_issues = cleaned[cleaned["row_issue_flag"]].copy()

    cleaned_simple.to_csv(CLEANED_CSV, index=False)
    all_lihc_simple.to_csv(ALL_CSV, index=False)
    fibrosis_simple.to_csv(FIBROSIS_CSV, index=False)
    nafld_simple.to_csv(NAFLD_CSV, index=False)
    hcv_simple.to_csv(HCV_CSV, index=False)
    rows_with_issues.to_csv(ROWS_WITH_ISSUES_CSV, index=False)

    with QC_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("LIHC master metadata QC report\n")
        handle.write(f"input_file\t{INPUT_PATH}\n")
        handle.write(f"rows_raw\t{len(raw)}\n")
        handle.write(f"rows_after_exact_dedup\t{len(cleaned)}\n")
        handle.write(f"exact_duplicate_rows_removed\t{exact_duplicate_count}\n")
        handle.write(f"rows_lihc_all_filtered\t{len(all_lihc)}\n")
        handle.write(f"rows_lihc_with_non_missing_fibrosis\t{len(fibrosis)}\n")
        handle.write(f"rows_lihc_with_non_missing_nafld\t{len(nafld)}\n")
        handle.write(f"rows_lihc_with_non_missing_hcv\t{len(hcv)}\n")
        handle.write(
            "primary_diagnosis_filter\t"
            + "; ".join(sorted(ALLOWED_PRIMARY_DIAGNOSES))
            + "\n"
        )
        handle.write("tumour_sample_type_filter\tPrimary Tumor\n")
        handle.write(f"rows_with_issue_flag\t{len(rows_with_issues)}\n")

        handle.write("\nMissingness summary (cleaned cohort): alcohol/virus/obesity\n")
        handle.write(missing_cleaned.to_string(index=False))
        handle.write("\n\nMissingness summary (LIHC all-filtered cohort): alcohol/virus/obesity\n")
        handle.write(missing_all_lihc.to_string(index=False))

        handle.write("\n\nMissingness summary (LIHC fibrosis-filtered cohort): alcohol/virus/obesity\n")
        handle.write(missing_fibrosis.to_string(index=False))
        handle.write("\n\nMissingness summary (LIHC NAFLD-filtered cohort): alcohol/virus/obesity\n")
        handle.write(missing_nafld.to_string(index=False))
        handle.write("\n\nMissingness summary (LIHC HCV-filtered cohort): alcohol/virus/obesity\n")
        handle.write(missing_hcv.to_string(index=False))

        handle.write("\n\nValue counts (cleaned): risk_alcohol\n")
        if "alcohol_status" in cleaned_simple.columns:
            handle.write(cleaned_simple["alcohol_status"].fillna("<NA>").value_counts(dropna=False).to_string())

        handle.write("\n\nValue counts (cleaned): hbv_status\n")
        if "hbv_status" in cleaned_simple.columns:
            handle.write(cleaned_simple["hbv_status"].fillna("<NA>").value_counts(dropna=False).to_string())

        handle.write("\n\nValue counts (cleaned): hcv_status\n")
        if "hcv_status" in cleaned_simple.columns:
            handle.write(cleaned_simple["hcv_status"].fillna("<NA>").value_counts(dropna=False).to_string())

        handle.write("\n\nValue counts (cleaned): obesity_class\n")
        if "obesity_class" in cleaned_simple.columns:
            handle.write(cleaned_simple["obesity_class"].fillna("<NA>").value_counts(dropna=False).to_string())

    print("QC completed.")
    print(f"rows_cleaned={len(cleaned)}")
    print(f"rows_all_lihc_filtered={len(all_lihc)}")
    print(f"rows_fibrosis_filtered={len(fibrosis)}")
    print(f"rows_nafld_filtered={len(nafld)}")
    print(f"rows_hcv_filtered={len(hcv)}")
    print("Missingness (LIHC all-filtered cohort):")
    print(missing_all_lihc.to_string(index=False))
    print("Missingness (LIHC fibrosis-filtered cohort):")
    print(missing_fibrosis.to_string(index=False))
    print("Missingness (LIHC NAFLD-filtered cohort):")
    print(missing_nafld.to_string(index=False))
    print("Missingness (LIHC HCV-filtered cohort):")
    print(missing_hcv.to_string(index=False))
    print("Wrote:")
    for path in [CLEANED_CSV, ALL_CSV, FIBROSIS_CSV, NAFLD_CSV, HCV_CSV, ROWS_WITH_ISSUES_CSV, QC_REPORT]:
        print(path)


if __name__ == "__main__":
    main()
