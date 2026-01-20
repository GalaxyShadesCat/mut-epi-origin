"""
tracks.py

Mutation track strategies (the three tracks in your plot):

1) counts_raw: raw binned mutation counts (step track)
2) counts_gauss: binned counts Gaussian-smoothed in bin space
3) inv_dist_gauss: inverse distance to nearest mutation (at bin centres) + smoothing
4) exp_decay: sum of exponential (Laplace) decays from all mutations (bp space)
5) exp_decay_adaptive: exponential decay with locally adaptive bandwidth (bp space)

All tracks return one value per bin.
"""

from __future__ import annotations

from typing import Dict, Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d


def mutations_to_bin_counts(mut_positions: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    if mut_positions.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    idx = np.searchsorted(bin_edges, mut_positions, side="right") - 1
    idx = idx[(idx >= 0) & (idx < len(bin_edges) - 1)]
    counts = np.bincount(idx, minlength=len(bin_edges) - 1).astype(float)
    return counts


def track_counts_raw(mut_positions: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return mutations_to_bin_counts(mut_positions, bin_edges)


def track_counts_gauss(mut_positions: np.ndarray, bin_edges: np.ndarray, sigma_bins: float) -> np.ndarray:
    raw = mutations_to_bin_counts(mut_positions, bin_edges)
    return gaussian_filter1d(raw, sigma=sigma_bins, mode="constant")


def nearest_mutation_distance(points: np.ndarray, mut_positions: np.ndarray, max_distance_bp: int) -> np.ndarray:
    """
    For each point in `points`, compute distance to nearest mutation, clipped at max_distance_bp.
    points and mut_positions must be sorted arrays of ints (positions in bp).
    """
    dists = np.full(points.shape, fill_value=float(max_distance_bp), dtype=float)
    if mut_positions.size == 0 or points.size == 0:
        return dists

    idx = np.searchsorted(mut_positions, points)
    right_ok = idx < mut_positions.size
    d_right = np.full(points.shape, float(max_distance_bp), dtype=float)
    d_right[right_ok] = mut_positions[idx[right_ok]] - points[right_ok]

    left_ok = idx > 0
    d_left = np.full(points.shape, float(max_distance_bp), dtype=float)
    d_left[left_ok] = points[left_ok] - mut_positions[idx[left_ok] - 1]

    dists = np.minimum(d_left, d_right)
    dists = np.clip(dists, 0, float(max_distance_bp))
    return dists


def track_inv_dist_gauss(
        mut_positions: np.ndarray,
        bin_centres: np.ndarray,
        sigma_bins: float,
        max_distance_bp: int,
        eps: float = 1.0,
) -> np.ndarray:
    dists = nearest_mutation_distance(bin_centres, mut_positions, max_distance_bp=max_distance_bp)
    raw = 1.0 / (dists + eps)
    smooth = gaussian_filter1d(raw, sigma=sigma_bins, mode="constant")
    return smooth


def track_exp_decay(
        mut_positions: np.ndarray,
        bin_centres: np.ndarray,
        decay_bp: float,
        max_distance_bp: int,
) -> np.ndarray:
    if mut_positions.size == 0 or bin_centres.size == 0:
        return np.zeros(bin_centres.shape, dtype=float)
    decay_bp = float(decay_bp)
    max_distance_bp = int(max_distance_bp)
    out = np.zeros(bin_centres.shape, dtype=float)
    left = 0
    right = 0
    n_mut = mut_positions.size
    inv_decay = 1.0 / decay_bp
    for i, centre in enumerate(bin_centres):
        min_pos = centre - max_distance_bp
        max_pos = centre + max_distance_bp
        while left < n_mut and mut_positions[left] < min_pos:
            left += 1
        if right < left:
            right = left
        while right < n_mut and mut_positions[right] <= max_pos:
            right += 1
        if right > left:
            dists = np.abs(mut_positions[left:right] - centre).astype(float)
            out[i] = np.exp(-dists * inv_decay).sum()
    return out


def kth_nearest_distance(
        points: np.ndarray,
        mut_positions: np.ndarray,
        k: int,
        max_distance_bp: int,
) -> np.ndarray:
    dists = np.full(points.shape, fill_value=float(max_distance_bp), dtype=float)
    if mut_positions.size == 0 or points.size == 0:
        return dists
    n_mut = mut_positions.size
    k = int(max(1, k))
    for i, point in enumerate(points):
        idx = np.searchsorted(mut_positions, point)
        left_start = max(0, idx - k)
        right_end = min(n_mut, idx + k)
        candidates = mut_positions[left_start:right_end]
        if candidates.size == 0:
            continue
        dist_sorted = np.sort(np.abs(candidates - point))
        if dist_sorted.size >= k:
            dists[i] = min(float(max_distance_bp), float(dist_sorted[k - 1]))
        else:
            dists[i] = min(float(max_distance_bp), float(dist_sorted[-1]))
    return dists


def track_exp_decay_adaptive(
        mut_positions: np.ndarray,
        bin_centres: np.ndarray,
        k_nearest: int,
        min_bandwidth_bp: float,
        max_distance_bp: int,
) -> np.ndarray:
    if mut_positions.size == 0 or bin_centres.size == 0:
        return np.zeros(bin_centres.shape, dtype=float)
    max_distance_bp = int(max_distance_bp)
    bandwidths = kth_nearest_distance(
        bin_centres,
        mut_positions,
        k=int(k_nearest),
        max_distance_bp=max_distance_bp,
    )
    bandwidths = np.maximum(bandwidths, float(min_bandwidth_bp))
    out = np.zeros(bin_centres.shape, dtype=float)
    left = 0
    right = 0
    n_mut = mut_positions.size
    for i, centre in enumerate(bin_centres):
        min_pos = centre - max_distance_bp
        max_pos = centre + max_distance_bp
        while left < n_mut and mut_positions[left] < min_pos:
            left += 1
        if right < left:
            right = left
        while right < n_mut and mut_positions[right] <= max_pos:
            right += 1
        if right > left:
            dists = np.abs(mut_positions[left:right] - centre).astype(float)
            out[i] = np.exp(-dists / bandwidths[i]).sum()
    return out


# Registry to keep pipeline clean
TRACK_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "counts_raw": track_counts_raw,
    "counts_gauss": track_counts_gauss,
    "inv_dist_gauss": track_inv_dist_gauss,
    "exp_decay": track_exp_decay,
    "exp_decay_adaptive": track_exp_decay_adaptive,
}
