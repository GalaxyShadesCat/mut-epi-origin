"""
analysis_helpers.py

Notebook utilities for loading and summarising run outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_results(out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    return pd.read_csv(out_dir / "results.csv")


def top_configs(
    df: pd.DataFrame,
    metric: str = "best_celltype_linear_resid_value",
    n: int = 10,
    most_negative: bool = True,
) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame.")
    order = df[metric].sort_values(ascending=not most_negative)
    idx = order.index[:n]
    return df.loc[idx].copy()


def summarise_best_celltype(
    df: pd.DataFrame,
    index: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Summarise best_celltype_linear_resid counts by configuration.
    """
    if "best_celltype_linear_resid" not in df.columns:
        raise ValueError("Column 'best_celltype_linear_resid' not found in DataFrame.")

    if index is None:
        index = ["sample_size_k", "track_strategy", "covariates", "bin_size", "include_trinuc"]

    sub = df[index + ["best_celltype_linear_resid"]].copy()
    return pd.pivot_table(
        sub,
        index=index,
        columns="best_celltype_linear_resid",
        values="best_celltype_linear_resid",
        aggfunc="count",
        fill_value=0,
    )
