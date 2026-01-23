"""Scoring and residualization helpers."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from scripts.stats_utils import zscore_nan


def pearsonr_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = np.sqrt((xx ** 2).sum()) * np.sqrt((yy ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((xx * yy).sum() / denom)


def linear_residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualise y on X using the least squares with intercept.
    Rows with NaNs in y or X are ignored; residuals returned with NaN where dropped.
    """
    n = len(y)
    resid = np.full(n, np.nan, dtype=float)
    if X.size == 0:
        return y.copy()

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 5:
        return resid

    yy = y[mask]
    XX = X[mask]

    A = np.column_stack([np.ones(len(yy)), XX])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    yhat = A @ beta
    resid[mask] = yy - yhat
    return resid


def rf_residualise(y: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    """
    Fit RandomForestRegressor: y ~ X, return residuals.
    """
    n = len(y)
    resid = np.full(n, np.nan, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return resid

    yy = y[mask]
    XX = X[mask]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    rf.fit(XX, yy)
    yhat = rf.predict(XX)
    resid[mask] = yy - yhat
    return resid


def standardise_matrix(X: np.ndarray) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=float)
    for i in range(X.shape[1]):
        out[:, i] = zscore_nan(X[:, i])
    return out


def rf_feature_analysis(
    y: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, Dict[str, float], float]:
    """
    Fit RF for y ~ X and return (perm_importances, sign_corr, impurity_importances, rf_r2, ridge_coef, ridge_r2).
    """
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names must align with X columns")

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 20:
        return {}, {}, {}, float("nan"), {}, float("nan")

    yy = y[mask]
    XX = X[mask]

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    rf.fit(XX, yy)
    rf_r2 = float(rf.score(XX, yy))
    impurity_importances = {name: float(val) for name, val in zip(feature_names, rf.feature_importances_)}

    perm = permutation_importance(rf, XX, yy, n_repeats=10, random_state=seed, n_jobs=-1)
    perm_importances = {name: float(val) for name, val in zip(feature_names, perm.importances_mean)}

    sign_corr = {name: float(pearsonr_nan(XX[:, i], yy)) for i, name in enumerate(feature_names)}

    ridge_coef: Dict[str, float] = {}
    ridge_r2 = float("nan")
    Xz = standardise_matrix(XX)
    yz = zscore_nan(yy)
    mask2 = np.isfinite(yz) & np.all(np.isfinite(Xz), axis=1)
    if mask2.sum() >= 20:
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xz[mask2], yz[mask2])
        ridge_coef = {name: float(val) for name, val in zip(feature_names, ridge.coef_)}
        ridge_r2 = float(ridge.score(Xz[mask2], yz[mask2]))

    return perm_importances, sign_corr, impurity_importances, rf_r2, ridge_coef, ridge_r2


def best_and_margin(values: Dict[str, float]) -> Tuple[Optional[str], float, float]:
    valid = [(k, v) for k, v in values.items() if np.isfinite(v)]
    if not valid:
        return None, float("nan"), float("nan")
    valid.sort(key=lambda kv: kv[1])
    best_k, best_v = valid[0]
    if len(valid) < 2:
        return best_k, float(best_v), float("nan")
    second_v = valid[1][1]
    margin = float(second_v - best_v)
    return best_k, float(best_v), margin


def aggregate_dict_column(df: pd.DataFrame, col: str, weight_col: str) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    for raw, w in zip(df[col].fillna("{}"), df[weight_col].fillna(0)):
        if not np.isfinite(w) or w <= 0:
            continue
        try:
            d = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            d = {}
        for k, v in d.items():
            if not np.isfinite(v):
                continue
            sums[k] = sums.get(k, 0.0) + float(v) * float(w)
            weights[k] = weights.get(k, 0.0) + float(w)
    return {k: sums[k] / weights[k] for k in sums if weights.get(k, 0.0) > 0}
