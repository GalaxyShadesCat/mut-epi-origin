"""
stats_utils.py

Small numeric helpers for NaN-safe stats.
"""

from __future__ import annotations

import numpy as np


def zscore_nan(x: np.ndarray) -> np.ndarray:
    """
    Z-score using finite values only; preserve NaNs in output.
    Returns all-NaN if <3 finite points or zero variance.
    """
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return out
    vals = x[mask].astype(float)
    mean = vals.mean()
    std = vals.std()
    if std == 0 or not np.isfinite(std):
        return out
    out[mask] = (vals - mean) / std
    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted mean ignoring NaNs in values and non-positive weights.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    v = values[mask]
    w = weights[mask]
    return float(np.sum(v * w) / np.sum(w))
