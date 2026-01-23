"""Grid configuration helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def sigma_to_bins(sigma: float, bin_size: int, units: str) -> float:
    if units == "bins":
        return float(sigma)
    if units == "bp":
        return float(sigma) / float(bin_size)
    raise ValueError(f"Unsupported sigma_units: {units}. Use bins or bp.")


def _range_like(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Range step must be positive.")
    if end < start:
        raise ValueError("Range end must be >= start.")
    values = np.arange(start, end + step * 0.5, step, dtype=float)
    return values.tolist()


def _looks_like_range_spec(seq: Sequence[float | int]) -> bool:
    if len(seq) != 3:
        return False
    start, end, step = seq
    try:
        start_f = float(start)
        end_f = float(end)
        step_f = float(step)
    except (TypeError, ValueError):
        return False
    if not np.isfinite([start_f, end_f, step_f]).all():
        return False
    if step_f <= 0:
        return False
    if end_f < start_f:
        return False
    if step_f > (end_f - start_f):
        return False
    return True


def expand_grid_values(
    values: Union[str, int, float, Sequence[int | float]],
    *,
    name: str,
    cast: type,
) -> List[Union[int, float]]:
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1]
            tokens = [t.strip() for t in inner.split(",") if t.strip()]
            if len(tokens) != 3:
                raise ValueError(f"{name} range spec must have three values: [start,end,step].")
            start, end, step = (float(t) for t in tokens)
            return [cast(v) for v in _range_like(start, end, step)]
        if ":" in raw:
            parts = [p.strip() for p in raw.split(":") if p.strip()]
            if len(parts) == 3:
                start, end, step = (float(p) for p in parts)
                return [cast(v) for v in _range_like(start, end, step)]
        return [cast(x) for x in raw.split(",") if x.strip()]

    if isinstance(values, (int, float, np.integer, np.floating)):
        return [cast(values)]

    seq = list(values)
    if not seq:
        return []
    if _looks_like_range_spec(seq):
        start, end, step = (float(x) for x in seq)
        return [cast(v) for v in _range_like(start, end, step)]
    return [cast(v) for v in seq]


def _normalize_downsample_values(
    values: Optional[Union[str, int, Sequence[int]]],
) -> List[Optional[int]]:
    if values is None:
        return [None]
    if isinstance(values, str) and values.strip().lower() in {"none", "null"}:
        return [None]
    expanded = expand_grid_values(values, name="downsample", cast=int)
    if not expanded:
        return [None]
    return [int(v) for v in expanded]


def _prefixed_track_params(
    *,
    track_strategy: str,
    bin_size: int,
    counts_sigma_bins_run: float,
    inv_sigma_bins_run: float,
    max_distance_bp_run: int,
    exp_decay_bp_run: float,
    exp_max_distance_bp_run: int,
    adaptive_k_run: int,
    adaptive_min_bandwidth_bp_run: float,
    adaptive_max_distance_bp_run: int,
    sigma_units: str,
) -> Dict[str, Any]:
    prefix = track_strategy
    params: Dict[str, Any] = {f"{prefix}_bin": int(bin_size)}
    if track_strategy == "counts_gauss":
        params[f"{prefix}_sigma_bins"] = float(counts_sigma_bins_run)
        params[f"{prefix}_sigma_units"] = str(sigma_units)
    elif track_strategy == "inv_dist_gauss":
        params[f"{prefix}_sigma_bins"] = float(inv_sigma_bins_run)
        params[f"{prefix}_max_distance_bp"] = int(max_distance_bp_run)
        params[f"{prefix}_sigma_units"] = str(sigma_units)
    elif track_strategy == "exp_decay":
        params[f"{prefix}_decay_bp"] = float(exp_decay_bp_run)
        params[f"{prefix}_max_distance_bp"] = int(exp_max_distance_bp_run)
    elif track_strategy == "exp_decay_adaptive":
        params[f"{prefix}_k"] = int(adaptive_k_run)
        params[f"{prefix}_min_bandwidth_bp"] = float(adaptive_min_bandwidth_bp_run)
        params[f"{prefix}_max_distance_bp"] = int(adaptive_max_distance_bp_run)
    return params
