"""Sample selection and downsampling helpers."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from scripts.sample_selector import (
    COLNAMES,
    count_mutations_per_sample,
    count_mutations_per_sample_multi,
    extract_mutations_for_samples,
    infer_sample_order,
)


def _prepare_non_overlapping_plan(
    mut_path: str | Path | Sequence[str | Path],
    k: Optional[int],
    repeats: int,
    seed: int,
    chunksize: int,
    allowed_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if k is not None and k < 0:
        raise ValueError("k must be >= 0")

    if isinstance(mut_path, (list, tuple)):
        paths = [Path(p) for p in mut_path]
        global_keys: List[str] = []
        for p in paths:
            sel = infer_sample_order(p, seed=seed, chunksize=chunksize)
            uniq = list(dict.fromkeys(sel.ordered_samples))
            global_keys.extend([f"{p.stem}::{sid}" for sid in uniq])

        rng = random.Random(seed)
        rng.shuffle(global_keys)
        if allowed_keys is not None:
            global_keys = [key for key in global_keys if key in allowed_keys]
        if k is not None and repeats * k > len(global_keys):
            raise ValueError(
                "Non-overlapping repeats require repeats * k <= total samples "
                f"({repeats} * {k} > {len(global_keys)})."
            )
        return {"paths": paths, "ordered_keys": global_keys}

    sel = infer_sample_order(mut_path, seed=seed, chunksize=chunksize)
    ordered = sel.ordered_samples
    if allowed_keys is not None:
        ordered = [s for s in ordered if s in allowed_keys]
    if k is not None and repeats * k > len(ordered):
        raise ValueError(
            "Non-overlapping repeats require repeats * k <= total samples "
            f"({repeats} * {k} > {len(ordered)})."
        )
    return {"paths": None, "ordered_keys": ordered}


def _select_non_overlapping_samples(
    plan: Dict[str, Any],
    mut_path: str | Path | Sequence[str | Path],
    rep: int,
    k: Optional[int],
    chunksize: int,
    tumour_filter: Optional[Sequence[str]],
) -> Tuple[List[str], pd.DataFrame, int, int]:
    ordered_keys = plan["ordered_keys"]
    if k is None:
        start = 0
        end = len(ordered_keys)
    else:
        start = rep * k
        end = start + k
    if plan["paths"] is None:
        chosen = ordered_keys[start:end]
        muts = extract_mutations_for_samples(
            mut_path,
            chosen,
            chunksize=chunksize,
            tumour_filter=tumour_filter,
        )
        return chosen, muts, start, end

    chosen_keys = ordered_keys[start:end]
    paths = plan["paths"]
    chosen_by_file: Dict[Path, List[str]] = {p: [] for p in paths}
    for key in chosen_keys:
        stem, sid = key.split("::", 1)
        matched = [p for p in paths if p.stem == stem]
        for p in matched:
            chosen_by_file[p].append(sid)

    kept_dfs: List[pd.DataFrame] = []
    for p in paths:
        sub = extract_mutations_for_samples(
            p,
            chosen_by_file.get(p, []),
            chunksize=chunksize,
            tumour_filter=tumour_filter,
        )
        if not sub.empty:
            kept_dfs.append(sub)

    muts = pd.concat(kept_dfs, ignore_index=True) if kept_dfs else pd.DataFrame(columns=COLNAMES)
    return chosen_keys, muts, start, end


def _eligible_sample_keys(
    mut_path: str | Path | Sequence[str | Path],
    min_mutations: int,
    chunksize: int,
    tumour_filter: Optional[Sequence[str]],
) -> Set[str]:
    if min_mutations <= 0:
        raise ValueError("min_mutations must be > 0")
    if isinstance(mut_path, (list, tuple)):
        counts = count_mutations_per_sample_multi(
            mut_path,
            chunksize=chunksize,
            tumour_filter=tumour_filter,
        )
    else:
        counts = count_mutations_per_sample(
            mut_path,
            chunksize=chunksize,
            tumour_filter=tumour_filter,
        )
    return {key for key, n in counts.items() if int(n) >= min_mutations}


def _allocate_stratified_counts(
    chrom_counts: Dict[str, int], target_n: int
) -> Dict[str, int]:
    total = sum(chrom_counts.values())
    if target_n > total:
        raise ValueError(
            f"Cannot downsample {target_n} mutations from only {total} rows."
        )

    floor_alloc: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for chrom, count in chrom_counts.items():
        ideal = (count / total) * target_n
        floor_val = int(math.floor(ideal))
        floor_alloc[chrom] = min(floor_val, count)
        remainders.append((ideal - floor_val, chrom))

    alloc = dict(floor_alloc)
    remaining = target_n - sum(alloc.values())
    remainders.sort(key=lambda x: (-x[0], x[1]))

    idx = 0
    while remaining > 0 and remainders:
        _, chrom = remainders[idx]
        if alloc[chrom] < chrom_counts[chrom]:
            alloc[chrom] += 1
            remaining -= 1
        idx = (idx + 1) % len(remainders)

    if remaining > 0:
        rem_lookup = {chrom: rem for rem, chrom in remainders}
        while remaining > 0:
            candidates = [
                (rem_lookup.get(chrom, 0.0), chrom)
                for chrom, count in chrom_counts.items()
                if alloc[chrom] < count
            ]
            if not candidates:
                break
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, chrom = candidates[0]
            alloc[chrom] += 1
            remaining -= 1

    return alloc


def _downsample_mutations_df(
    muts_df: pd.DataFrame, target_n: int, seed: int
) -> pd.DataFrame:
    if target_n <= 0:
        raise ValueError("downsample target must be > 0")
    if len(muts_df) <= target_n:
        return muts_df

    chrom_counts = (
        muts_df["Chromosome"].dropna().astype(str).value_counts().to_dict()
    )
    alloc = _allocate_stratified_counts(chrom_counts, target_n)
    rng = np.random.default_rng(seed)

    indices: List[int] = []
    for chrom, sub in muts_df.groupby("Chromosome", sort=False):
        need = alloc.get(chrom, 0)
        if need <= 0:
            continue
        idx = sub.index.to_numpy()
        if need >= len(idx):
            indices.extend(idx.tolist())
        else:
            indices.extend(rng.choice(idx, size=need, replace=False).tolist())

    rng.shuffle(indices)
    return muts_df.loc[indices].reset_index(drop=True)


def _unique_nonempty(values: Sequence[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for val in values:
        s = str(val).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
