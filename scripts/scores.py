"""
scores.py

Local correlation scoring for binned genomic tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from scripts.stats_utils import weighted_mean, zscore_nan


@dataclass(frozen=True)
class LocalScoreResult:
    total: np.ndarray
    shape: np.ndarray
    slope: np.ndarray
    shape_corr: np.ndarray
    slope_corr: np.ndarray
    weights: np.ndarray
    global_score: float
    negative_corr_fraction: float


def _ensure_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _nan_convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="same")
    weights = np.convolve(np.isfinite(x).astype(float), kernel, mode="same")
    out = np.full_like(values, np.nan, dtype=float)
    mask = weights > 0
    out[mask] = values[mask] / weights[mask]
    return out


def _gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive for gaussian smoothing.")
    if radius is None:
        radius = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_track(x: np.ndarray, method: str, param: float | int | None) -> np.ndarray:
    if method == "none":
        return x.copy()
    if method == "moving_average":
        if param is None:
            raise ValueError("moving_average requires a window size.")
        window = int(param)
        if window < 1:
            raise ValueError("moving_average window must be >= 1.")
        kernel = np.ones(window, dtype=float)
        kernel /= kernel.sum()
        return _nan_convolve(x, kernel)
    if method == "gaussian":
        if param is None:
            raise ValueError("gaussian smoothing requires sigma.")
        kernel = _gaussian_kernel(float(param))
        return _nan_convolve(x, kernel)
    raise ValueError(f"Unknown smoothing method: {method}")


def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return x.copy()
    if transform == "log1p":
        if np.nanmin(x) < 0:
            raise ValueError("log1p transform requires non-negative values.")
        out = x.copy()
        mask = np.isfinite(out)
        out[mask] = np.log1p(out[mask])
        return out
    raise ValueError(f"Unknown transform: {transform}")


def _zscore(x: np.ndarray) -> np.ndarray:
    return zscore_nan(x.astype(float))


def _corr_windows_pearson(xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    if not np.isnan(xw).any() and not np.isnan(yw).any():
        x_mean = xw.mean(axis=1)
        y_mean = yw.mean(axis=1)
        xx = xw - x_mean[:, None]
        yy = yw - y_mean[:, None]
        denom = np.sqrt((xx ** 2).sum(axis=1) * (yy ** 2).sum(axis=1))
        num = (xx * yy).sum(axis=1)
        out = np.full(len(denom), np.nan, dtype=float)
        mask = denom > 0
        out[mask] = num[mask] / denom[mask]
        return out
    out = np.full(xw.shape[0], np.nan, dtype=float)
    for i in range(xw.shape[0]):
        mask = np.isfinite(xw[i]) & np.isfinite(yw[i])
        if mask.sum() < 3:
            continue
        xx = xw[i, mask] - np.mean(xw[i, mask])
        yy = yw[i, mask] - np.mean(yw[i, mask])
        denom = np.sqrt((xx ** 2).sum()) * np.sqrt((yy ** 2).sum())
        if denom == 0:
            continue
        out[i] = float((xx * yy).sum() / denom)
    return out


def _corr_windows_spearman(xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    out = np.full(xw.shape[0], np.nan, dtype=float)
    for i in range(xw.shape[0]):
        mask = np.isfinite(xw[i]) & np.isfinite(yw[i])
        if mask.sum() < 3:
            continue
        rx = rankdata(xw[i, mask])
        ry = rankdata(yw[i, mask])
        xx = rx - rx.mean()
        yy = ry - ry.mean()
        denom = np.sqrt((xx ** 2).sum()) * np.sqrt((yy ** 2).sum())
        if denom == 0:
            continue
        out[i] = float((xx * yy).sum() / denom)
    return out


def _corr_windows(
        xw: np.ndarray,
        yw: np.ndarray,
        method: str,
) -> np.ndarray:
    if method == "pearson":
        return _corr_windows_pearson(xw, yw)
    if method == "spearman":
        return _corr_windows_spearman(xw, yw)
    raise ValueError(f"Unknown correlation method: {method}")


def _window_std(xw: np.ndarray) -> np.ndarray:
    if not np.isnan(xw).any():
        return xw.std(axis=1)
    return np.nanstd(xw, axis=1)


def _map_corr(corr: np.ndarray) -> np.ndarray:
    return corr.copy()


def _normalize_weights(weights: Iterable[float]) -> tuple[float, float]:
    a, b = weights
    total = a + b
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return a / total, b / total


def compute_local_scores(
        M: np.ndarray,
        D: np.ndarray,
        *,
        w: int,
        corr_type: str = "pearson",
        smoothing: str = "none",
        smooth_param: float | int | None = None,
        transform: str = "none",
        zscore: bool = False,
        weights: tuple[float, float] = (0.7, 0.3),
) -> LocalScoreResult:
    """
    Compute per-bin correlation scores for two 1D tracks.
    More negative values indicate stronger local anti-correlation.
    Returns NaN for bins where windows are invalid or flat.
    """
    M = _ensure_1d(M, "M")
    D = _ensure_1d(D, "D")
    if len(M) != len(D):
        raise ValueError("M and D must have the same length.")
    if w < 1:
        raise ValueError("w must be >= 1.")
    n = len(M)
    if n < 2 * w + 1:
        empty = np.full(n, np.nan, dtype=float)
        return LocalScoreResult(
            total=empty,
            shape=empty,
            slope=empty,
            shape_corr=empty,
            slope_corr=empty,
            weights=empty,
            global_score=float("nan"),
            negative_corr_fraction=float("nan"),
        )

    M_proc = apply_transform(M, transform)
    D_proc = apply_transform(D, transform)
    if zscore:
        M_proc = _zscore(M_proc)
        D_proc = _zscore(D_proc)

    M_smooth = smooth_track(M_proc, smoothing, smooth_param)
    D_smooth = smooth_track(D_proc, smoothing, smooth_param)

    window = 2 * w + 1
    m_windows = np.lib.stride_tricks.sliding_window_view(M_smooth, window)
    d_windows = np.lib.stride_tricks.sliding_window_view(D_smooth, window)
    shape_corr = _corr_windows(m_windows, d_windows, corr_type)
    shape = _map_corr(shape_corr)

    dM = np.diff(M_smooth)
    dD = np.diff(D_smooth)
    slope_windows = 2 * w
    dm_windows = np.lib.stride_tricks.sliding_window_view(dM, slope_windows)
    dd_windows = np.lib.stride_tricks.sliding_window_view(dD, slope_windows)
    slope_corr = _corr_windows(dm_windows, dd_windows, corr_type)
    slope = _map_corr(slope_corr)

    std_m = _window_std(m_windows)
    std_d = _window_std(d_windows)
    weights_windows = std_m * std_d

    a, b = _normalize_weights(weights)
    total_window = np.full(len(shape), np.nan, dtype=float)
    valid = np.isfinite(shape) & np.isfinite(slope)
    total_window[valid] = a * shape[valid] + b * slope[valid]

    total = np.full(n, np.nan, dtype=float)
    shape_full = np.full(n, np.nan, dtype=float)
    slope_full = np.full(n, np.nan, dtype=float)
    shape_corr_full = np.full(n, np.nan, dtype=float)
    slope_corr_full = np.full(n, np.nan, dtype=float)
    weights_full = np.full(n, np.nan, dtype=float)
    idx_start = w
    idx_end = n - w
    total[idx_start:idx_end] = total_window
    shape_full[idx_start:idx_end] = shape
    slope_full[idx_start:idx_end] = slope
    shape_corr_full[idx_start:idx_end] = shape_corr
    slope_corr_full[idx_start:idx_end] = slope_corr
    weights_full[idx_start:idx_end] = weights_windows

    global_score = weighted_mean(total_window, weights_windows)
    negative_corr_fraction = float(np.mean(shape_corr[np.isfinite(shape_corr)] < 0)) if np.isfinite(
        shape_corr).any() else float("nan")

    return LocalScoreResult(
        total=total,
        shape=shape_full,
        slope=slope_full,
        shape_corr=shape_corr_full,
        slope_corr=slope_corr_full,
        weights=weights_full,
        global_score=global_score,
        negative_corr_fraction=negative_corr_fraction,
    )


def compute_scores_by_chrom(
        tracks: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        w: int,
        corr_type: str = "pearson",
        smoothing: str = "none",
        smooth_param: float | int | None = None,
        transform: str = "none",
        zscore: bool = False,
        weights: tuple[float, float] = (0.7, 0.3),
) -> tuple[dict[str, LocalScoreResult], float]:
    results: dict[str, LocalScoreResult] = {}
    totals = []
    weights_all = []
    for chrom, (M, D) in tracks.items():
        res = compute_local_scores(
            M,
            D,
            w=w,
            corr_type=corr_type,
            smoothing=smoothing,
            smooth_param=smooth_param,
            transform=transform,
            zscore=zscore,
            weights=weights,
        )
        results[chrom] = res
        totals.append(res.total[np.isfinite(res.total)])
        weights_all.append(res.weights[np.isfinite(res.total)])
    if totals:
        total_concat = np.concatenate(totals) if totals else np.array([], dtype=float)
        weights_concat = np.concatenate(weights_all) if weights_all else np.array([], dtype=float)
        global_score = weighted_mean(total_concat, weights_concat) if len(total_concat) else float("nan")
    else:
        global_score = float("nan")
    return results, global_score
