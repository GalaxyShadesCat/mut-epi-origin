#!/usr/bin/env python3
"""Run Kaplan-Meier survival analysis for inferred FOXA2 label groups.

This script reads a label + clinical CSV, derives overall-survival time/event,
optionally enriches rows with selected `mmc1.xlsx` clinical covariates, and
writes reproducible Kaplan-Meier and log-rank outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=(
            "outputs/experiments/"
            "lihc_foxa2_top4_all_samples_per_sample_merged/"
            "labels_with_clinical_counts.csv"
        ),
        help="Input CSV with predicted_label and survival-related columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/experiments/"
            "lihc_foxa2_top4_all_samples_per_sample_merged/"
            "survival_counts_raw500k_spearman_r_linear_resid"
        ),
        help="Output directory for survival analysis artefacts.",
    )
    parser.add_argument(
        "--label-col",
        default="predicted_label",
        help="Column containing inferred label groups.",
    )
    parser.add_argument(
        "--sample-col",
        default="sample_id",
        help="Column containing sample identifiers.",
    )
    parser.add_argument(
        "--vital-col",
        default="vital_status",
        help="Vital status column (DECEASED/LIVING).",
    )
    parser.add_argument(
        "--death-days-col",
        default="days_to_death",
        help="Days-to-death column.",
    )
    parser.add_argument(
        "--followup-days-col",
        default="days_to_last_followup",
        help="Days-to-last-follow-up column.",
    )
    parser.add_argument(
        "--mmc1-path",
        default="data/raw/annotations/mmc1.xlsx",
        help="Path to mmc1 workbook for optional covariate enrichment.",
    )
    parser.add_argument(
        "--disable-mmc1-enrichment",
        action="store_true",
        help="Disable enrichment with mmc1 covariates.",
    )
    return parser.parse_args()


def normalise_id(values: pd.Series, prefix_length: int = 15) -> pd.Series:
    """Normalise sample IDs to upper-case TCGA prefixes."""
    normalised = values.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return normalised.str.slice(0, prefix_length)


def clean_text(series: pd.Series) -> pd.Series:
    """Trim text and map blank tokens to missing."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "[Not Available]": pd.NA})
    return cleaned


def kaplan_meier_table(times: np.ndarray, events: np.ndarray, label: str) -> pd.DataFrame:
    """Return a Kaplan-Meier step table for one group."""
    event_times = np.sort(np.unique(times[events == 1]))
    n_total = len(times)
    surv = 1.0
    rows: list[dict[str, float | int | str]] = [
        {
            "label": label,
            "time_days": 0.0,
            "n_at_risk": int(n_total),
            "n_events": 0,
            "survival_prob": 1.0,
        }
    ]
    for t in event_times:
        at_risk = int(np.sum(times >= t))
        n_events = int(np.sum((times == t) & (events == 1)))
        if at_risk > 0:
            surv *= (1.0 - (n_events / at_risk))
        rows.append(
            {
                "label": label,
                "time_days": float(t),
                "n_at_risk": at_risk,
                "n_events": n_events,
                "survival_prob": float(surv),
            }
        )
    return pd.DataFrame(rows)


def logrank_test(time: np.ndarray, event: np.ndarray, group_binary: np.ndarray) -> tuple[float, float]:
    """Compute two-group log-rank statistic and p-value."""
    unique_event_times = np.sort(np.unique(time[event == 1]))
    oe_sum = 0.0
    var_sum = 0.0
    for t in unique_event_times:
        at_risk = time >= t
        n = int(np.sum(at_risk))
        if n <= 1:
            continue
        group1 = group_binary == 1
        n1 = int(np.sum(at_risk & group1))
        n0 = n - n1
        d_all = int(np.sum((time == t) & (event == 1)))
        d1 = int(np.sum((time == t) & (event == 1) & group1))
        if d_all == 0:
            continue
        e1 = d_all * (n1 / n)
        variance = (n1 * n0 * d_all * (n - d_all)) / (n * n * (n - 1))
        oe_sum += (d1 - e1)
        var_sum += variance

    if var_sum <= 0:
        return float("nan"), float("nan")
    chi_sq = (oe_sum * oe_sum) / var_sum
    p_value = chi2.sf(chi_sq, df=1)
    return float(chi_sq), float(p_value)


def median_survival_from_km(km: pd.DataFrame) -> float:
    """Return median survival time from KM step table."""
    below = km[km["survival_prob"] <= 0.5]
    if below.empty:
        return float("nan")
    return float(below["time_days"].iloc[0])


def load_mmc1_covariates(mmc1_path: Path) -> pd.DataFrame:
    """Load selected covariates from mmc1 Table S1A."""
    mmc1 = pd.read_excel(mmc1_path, sheet_name="Table S1A - core sample set", header=3)
    keep_cols = [
        "Barcode",
        "gender",
        "pathologic_stage",
        "neoplasm_histologic_grade",
        "BMI",
    ]
    mmc1 = mmc1[keep_cols].copy()
    mmc1["sample_id"] = normalise_id(mmc1["Barcode"])
    mmc1 = mmc1.rename(
        columns={
            "gender": "mmc1_gender",
            "pathologic_stage": "mmc1_pathologic_stage",
            "neoplasm_histologic_grade": "mmc1_histologic_grade",
            "BMI": "mmc1_bmi",
        }
    )
    for col in ["mmc1_gender", "mmc1_pathologic_stage", "mmc1_histologic_grade", "mmc1_bmi"]:
        mmc1[col] = clean_text(mmc1[col])
    mmc1 = mmc1.drop_duplicates(subset=["sample_id"])
    return mmc1


def main() -> None:
    """Run survival analysis and write outputs."""
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, dtype=str)
    required_cols = [
        args.sample_col,
        args.label_col,
        args.vital_col,
        args.death_days_col,
        args.followup_days_col,
    ]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    frame["sample_id"] = normalise_id(frame[args.sample_col])
    frame["predicted_label"] = clean_text(frame[args.label_col])
    frame["vital_status"] = clean_text(frame[args.vital_col]).str.upper()

    days_to_death = pd.to_numeric(clean_text(frame[args.death_days_col]), errors="coerce")
    days_to_followup = pd.to_numeric(clean_text(frame[args.followup_days_col]), errors="coerce")

    frame["event"] = (frame["vital_status"] == "DECEASED").astype(float)
    frame.loc[frame["vital_status"].isna(), "event"] = np.nan
    frame["time_days"] = days_to_death.fillna(days_to_followup)

    analysis = frame[["sample_id", "predicted_label", "time_days", "event"]].copy()
    analysis = analysis.dropna(subset=["sample_id", "predicted_label", "time_days", "event"])
    analysis = analysis.drop_duplicates(subset=["sample_id"])

    if not args.disable_mmc1_enrichment:
        mmc1_path = Path(args.mmc1_path)
        if mmc1_path.exists():
            mmc1 = load_mmc1_covariates(mmc1_path)
            analysis = analysis.merge(mmc1, on="sample_id", how="left")

    labels = sorted(analysis["predicted_label"].unique())
    if len(labels) != 2:
        raise ValueError(f"Expected exactly two label groups, found: {labels}")

    label0, label1 = labels
    analysis["group_binary"] = (analysis["predicted_label"] == label1).astype(int)

    km_tables = []
    for label in labels:
        sub = analysis[analysis["predicted_label"] == label]
        km = kaplan_meier_table(
            times=sub["time_days"].to_numpy(dtype=float),
            events=sub["event"].to_numpy(dtype=int),
            label=label,
        )
        km_tables.append(km)
    km_all = pd.concat(km_tables, ignore_index=True)

    chi_sq, p_value = logrank_test(
        time=analysis["time_days"].to_numpy(dtype=float),
        event=analysis["event"].to_numpy(dtype=int),
        group_binary=analysis["group_binary"].to_numpy(dtype=int),
    )

    cox_model = PHReg(
        endog=analysis["time_days"].to_numpy(dtype=float),
        exog=analysis[["group_binary"]].to_numpy(dtype=float),
        status=analysis["event"].to_numpy(dtype=int),
    )
    cox_result = cox_model.fit()
    coef = float(cox_result.params[0])
    se = float(cox_result.bse[0])
    hr = math.exp(coef)
    hr_ci_low = math.exp(coef - (1.96 * se))
    hr_ci_high = math.exp(coef + (1.96 * se))
    cox_p = float(cox_result.pvalues[0])

    input_out = output_dir / "survival_input_cleaned.csv"
    km_out = output_dir / "kaplan_meier_curve_table.csv"
    logrank_out = output_dir / "logrank_test.csv"
    cox_out = output_dir / "cox_unadjusted.csv"
    summary_out = output_dir / "summary.txt"
    plot_out = output_dir / "kaplan_meier_plot.png"

    analysis.to_csv(input_out, index=False)
    km_all.to_csv(km_out, index=False)
    pd.DataFrame(
        [{"chi_square": chi_sq, "df": 1, "p_value": p_value, "group_reference": label0, "group_contrast": label1}]
    ).to_csv(logrank_out, index=False)
    pd.DataFrame(
        [
            {
                "contrast": f"{label1} vs {label0}",
                "coef_log_hr": coef,
                "se": se,
                "hazard_ratio": hr,
                "hr_ci95_low": hr_ci_low,
                "hr_ci95_high": hr_ci_high,
                "p_value": cox_p,
            }
        ]
    ).to_csv(cox_out, index=False)

    plt.figure(figsize=(8, 6))
    colours = ["#1f77b4", "#d62728"]
    for idx, label in enumerate(labels):
        sub = km_all[km_all["label"] == label]
        plt.step(sub["time_days"], sub["survival_prob"], where="post", label=label, color=colours[idx])
    plt.xlabel("Time (days)")
    plt.ylabel("Overall survival probability")
    plt.title("Kaplan-Meier by predicted_label")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out, dpi=180)
    plt.close()

    label_counts = analysis["predicted_label"].value_counts().to_dict()
    event_counts = analysis.groupby("predicted_label")["event"].sum().to_dict()
    cens_counts = (
        analysis.groupby("predicted_label")
        .apply(lambda g: int((g["event"] == 0).sum()), include_groups=False)
        .to_dict()
    )

    medians = {}
    for label in labels:
        km = km_all[km_all["label"] == label]
        medians[label] = median_survival_from_km(km)

    mmc1_coverage_lines = []
    for col in ["mmc1_gender", "mmc1_pathologic_stage", "mmc1_histologic_grade", "mmc1_bmi"]:
        if col in analysis.columns:
            mmc1_coverage_lines.append(f"{col}_non_missing: {int(analysis[col].notna().sum())}")

    summary_lines = [
        f"input_csv: {input_path}",
        f"samples_used: {len(analysis)}",
        f"labels: {label0}, {label1}",
        f"{label0}_n: {label_counts.get(label0, 0)}",
        f"{label1}_n: {label_counts.get(label1, 0)}",
        f"{label0}_events: {int(event_counts.get(label0, 0))}",
        f"{label1}_events: {int(event_counts.get(label1, 0))}",
        f"{label0}_censored: {int(cens_counts.get(label0, 0))}",
        f"{label1}_censored: {int(cens_counts.get(label1, 0))}",
        f"{label0}_median_days: {medians[label0]}",
        f"{label1}_median_days: {medians[label1]}",
        f"logrank_chi_square: {chi_sq}",
        f"logrank_p_value: {p_value}",
        f"cox_contrast: {label1} vs {label0}",
        f"cox_hazard_ratio: {hr}",
        f"cox_p_value: {cox_p}",
    ] + mmc1_coverage_lines
    summary_out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {input_out}")
    print(f"Wrote: {km_out}")
    print(f"Wrote: {logrank_out}")
    print(f"Wrote: {cox_out}")
    print(f"Wrote: {plot_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
