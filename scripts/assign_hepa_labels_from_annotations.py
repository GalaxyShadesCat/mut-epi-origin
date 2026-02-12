#!/usr/bin/env python3
"""Assign hepatocyte labels to tumour samples using annotations and mutation BED IDs.

Rules (in order):
1) If viral-positive -> hepatocyte_normal
2) Else if alcohol-positive:
   - TCGA:
     - Cirrhosis == Yes -> hepatocyte_ac
     - Cirrhosis == No -> hepatocyte_ah
     - Cirrhosis missing/unclear -> hepatocyte_ambiguous
   - LIRI:
     - FIBROSIS >= 4 -> hepatocyte_ac
     - FIBROSIS == 3 -> hepatocyte_ah
     - FIBROSIS 0-2 with strong alcohol -> hepatocyte_ah
     - FIBROSIS 0-1 with weak alcohol -> hepatocyte_ambiguous
     - FIBROSIS missing -> hepatocyte_ambiguous
3) Else -> hepatocyte_ambiguous

Alcohol detection:
- TCGA uses strict token matching on risk factors to avoid false positives
  such as "Non-Alcoholic Fatty Liver Disease".
- LIRI uses alcohol-strength tiers (strong vs weak) based on ALCOHOL score.

Input annotations:
- TCGA: data/raw/annotations/mmc1.xlsx (Table S1A - core sample set)
- LIRI-JP: data/raw/annotations/HCCDB18.patient.txt

Input samples:
- Mutation BED/TSV file (default: data/raw/mutations/ICGC_WGS_Feb20_mutations.LIHC_LIRI.bed)
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CELLTYPE_AC = "hepatocyte_ac"
CELLTYPE_AH = "hepatocyte_ah"
CELLTYPE_NORMAL = "hepatocyte_normal"
CELLTYPE_AMBIGUOUS = "hepatocyte_ambiguous"

TCGA_SAMPLE_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-.+")
LIRI_SAMPLE_RE = re.compile(r"^RK\d{2,4}(?:_.+)?$")


@dataclass(frozen=True)
class AssignmentRules:
    """Rule configuration for annotation-to-hepatocyte label assignment."""

    liri_ac_threshold: int
    liri_alcohol_strong_threshold: int


def _clean_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_int(value: object) -> int | None:
    text = _clean_str(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _tcga_short_barcode(sample_id: str) -> str:
    parts = sample_id.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return sample_id


def _is_sample_id(text: str) -> bool:
    return bool(TCGA_SAMPLE_RE.match(text) or LIRI_SAMPLE_RE.match(text))


def detect_sample_column(mutations_bed: Path, probe_rows: int = 2000) -> int:
    counts: dict[int, int] = {}
    seen_rows = 0

    with mutations_bed.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            seen_rows += 1
            for idx, value in enumerate(row):
                v = value.strip()
                if _is_sample_id(v):
                    counts[idx] = counts.get(idx, 0) + 1
            if seen_rows >= probe_rows:
                break

    if not counts:
        raise ValueError(
            "Could not detect a sample-ID column in the mutation file. "
            "Expected TCGA-* or RK* sample IDs."
        )

    return max(counts, key=counts.get)


def extract_unique_samples(mutations_bed: Path) -> list[str]:
    sample_col = detect_sample_column(mutations_bed)
    sample_ids: set[str] = set()

    with mutations_bed.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if sample_col >= len(row):
                continue
            sample_id = row[sample_col].strip()
            if _is_sample_id(sample_id):
                sample_ids.add(sample_id)

    if not sample_ids:
        raise ValueError("No sample IDs were extracted from mutation file.")

    return sorted(sample_ids)


def infer_tumour_type(sample_id: str) -> str:
    if sample_id.startswith("TCGA-"):
        return "LIHC"
    if sample_id.startswith("RK"):
        return "LIRI"
    return "UNKNOWN"


def load_tcga_annotations(mmc1_path: Path) -> pd.DataFrame:
    tcga = pd.read_excel(
        mmc1_path,
        sheet_name="Table S1A - core sample set",
        header=3,
    )
    needed_cols = [
        "Barcode",
        "Alcoholic liver disease",
        "Hepatitis B",
        "Hepatitis C",
        "HBV_consensus",
        "HCV_consensus",
        "Cirrhosis",
        "history_hepato_carcinoma_risk_factor",
    ]
    missing = [c for c in needed_cols if c not in tcga.columns]
    if missing:
        raise ValueError(f"Missing expected TCGA columns: {missing}")

    tcga = tcga[needed_cols].copy()
    tcga["tcga_key"] = tcga["Barcode"].astype(str).str.strip()
    tcga = tcga.drop_duplicates(subset=["tcga_key"])
    return tcga.set_index("tcga_key", drop=False)


def load_liri_annotations(liri_path: Path) -> pd.DataFrame:
    liri_raw = pd.read_csv(liri_path, sep="\t")
    if "PATIENT_ID" not in liri_raw.columns:
        raise ValueError("LIRI annotation file is missing PATIENT_ID column.")

    liri = liri_raw.set_index("PATIENT_ID").T
    needed_cols = ["PATIENT1", "VIRUS", "ALCOHOL", "FIBROSIS"]
    missing = [c for c in needed_cols if c not in liri.columns]
    if missing:
        raise ValueError(f"Missing expected LIRI fields: {missing}")

    liri = liri[needed_cols].copy()
    liri["liri_key"] = liri["PATIENT1"].astype(str).str.strip()
    liri = liri.drop_duplicates(subset=["liri_key"])
    return liri.set_index("liri_key", drop=False)


def classify_tcga(row: pd.Series, rules: AssignmentRules) -> str:
    hbv_cons = _clean_str(row.get("HBV_consensus", "")).lower()
    hcv_cons = _clean_str(row.get("HCV_consensus", "")).lower()
    hep_b = _clean_str(row.get("Hepatitis B", "")).lower()
    hep_c = _clean_str(row.get("Hepatitis C", "")).lower()

    viral_positive = (
        hbv_cons == "pos"
        or hcv_cons == "pos"
        or "hepatitis b" in hep_b
        or "hepatitis c" in hep_c
    )

    alcoholic_liver_disease = _clean_str(row.get("Alcoholic liver disease", "")).lower()
    risk_factor = _clean_str(row.get("history_hepato_carcinoma_risk_factor", ""))
    risk_tokens = {token.strip().lower() for token in risk_factor.split("|") if token.strip()}
    alcohol_positive = (
        alcoholic_liver_disease == "alcohol"
        or "alcohol consumption" in risk_tokens
    )

    if viral_positive:
        return CELLTYPE_NORMAL

    if alcohol_positive:
        cirrhosis = _clean_str(row.get("Cirrhosis", "")).lower()
        if cirrhosis == "yes":
            return CELLTYPE_AC
        if cirrhosis == "no":
            return CELLTYPE_AH
        return CELLTYPE_AMBIGUOUS
    return CELLTYPE_AMBIGUOUS


def classify_liri(row: pd.Series, rules: AssignmentRules) -> str:
    virus = _clean_str(row.get("VIRUS", "")).replace(" ", "").upper()
    viral_positive = bool(virus) and virus != "NBNC"

    alcohol = _parse_int(row.get("ALCOHOL", ""))
    alcohol_positive = alcohol is not None and alcohol > 0
    alcohol_strong = alcohol is not None and alcohol >= rules.liri_alcohol_strong_threshold
    alcohol_weak = alcohol_positive and not alcohol_strong

    if viral_positive:
        return CELLTYPE_NORMAL

    if alcohol_positive:
        fibrosis = _parse_int(row.get("FIBROSIS", ""))
        if fibrosis is None:
            return CELLTYPE_AMBIGUOUS
        # In this file, 4 is treated as cirrhosis-level fibrosis proxy.
        if fibrosis >= rules.liri_ac_threshold:
            return CELLTYPE_AC
        if fibrosis == 3:
            return CELLTYPE_AH
        if alcohol_weak and fibrosis <= 1:
            return CELLTYPE_AMBIGUOUS
        return CELLTYPE_AH
    return CELLTYPE_AMBIGUOUS


def build_label_assignment(
    sample_ids: list[str],
    tcga_ann: pd.DataFrame,
    liri_ann: pd.DataFrame,
    rules: AssignmentRules,
) -> tuple[pd.DataFrame, list[str]]:
    labels: list[str] = []
    missing_annotation_samples: list[str] = []

    for sample_id in sample_ids:
        tumour_type = infer_tumour_type(sample_id)
        label = CELLTYPE_AMBIGUOUS

        if tumour_type == "LIHC":
            tcga_key = _tcga_short_barcode(sample_id)
            if tcga_key in tcga_ann.index:
                label = classify_tcga(tcga_ann.loc[tcga_key], rules)
            else:
                missing_annotation_samples.append(sample_id)
        elif tumour_type == "LIRI":
            liri_key = sample_id.split("_")[0]
            if liri_key in liri_ann.index:
                label = classify_liri(liri_ann.loc[liri_key], rules)
            else:
                missing_annotation_samples.append(sample_id)
        else:
            missing_annotation_samples.append(sample_id)

        labels.append(label)

    out_df = pd.DataFrame({"sample_id": sample_ids, "cell_type_label": labels})
    return out_df, sorted(set(missing_annotation_samples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/raw/annotations"),
        help="Directory containing mmc1.xlsx and HCCDB18.patient.txt",
    )
    parser.add_argument(
        "--mutations-bed",
        type=Path,
        default=Path("data/raw/mutations/ICGC_WGS_Feb20_mutations.LIHC_LIRI.bed"),
        help="Mutation BED/TSV file used to extract tumour sample IDs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/hepa_labels/hepa_labels_from_annotations.csv"),
        help="Output CSV path (columns: sample_id, cell_type_label)",
    )
    parser.add_argument(
        "--liri-ac-threshold",
        type=int,
        default=4,
        help="LIRI fibrosis threshold for hepatocyte_ac (default: 4).",
    )
    parser.add_argument(
        "--liri-alcohol-strong-threshold",
        type=int,
        default=2,
        help=(
            "LIRI ALCOHOL threshold for strong alcohol evidence. "
            "Scores >0 but below this are treated as weak alcohol."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = AssignmentRules(
        liri_ac_threshold=args.liri_ac_threshold,
        liri_alcohol_strong_threshold=args.liri_alcohol_strong_threshold,
    )

    tcga_path = args.annotations_dir / "mmc1.xlsx"
    liri_path = args.annotations_dir / "HCCDB18.patient.txt"

    sample_ids = extract_unique_samples(args.mutations_bed)
    tcga_ann = load_tcga_annotations(tcga_path)
    liri_ann = load_liri_annotations(liri_path)

    assigned_labels, missing = build_label_assignment(sample_ids, tcga_ann, liri_ann, rules)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    assigned_labels.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(assigned_labels)} sample labels to: {args.output_csv}")
    print("Rule settings:")
    print(f"- liri_ac_threshold: {rules.liri_ac_threshold}")
    print("- liri_fibrosis3_label: hepatocyte_ah (fixed)")
    print(
        "- liri_alcohol_strength: "
        f"strong if ALCOHOL >= {rules.liri_alcohol_strong_threshold}"
    )
    print(
        "- liri_weak_low_fibrosis_policy (ALCOHOL>0 and < strong threshold, "
        "FIBROSIS<=1): hepatocyte_ambiguous (fixed)"
    )
    print("- tcga_cirrhosis_unclear_label: hepatocyte_ambiguous (fixed)")
    print(assigned_labels["cell_type_label"].value_counts().to_string())
    if missing:
        print("\nSamples without matched annotation (defaulted to hepatocyte_ambiguous):")
        for sample in missing:
            print(f"- {sample}")


if __name__ == "__main__":
    main()
