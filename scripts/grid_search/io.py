"""I/O helpers for grid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from scripts.dnase_map import DnaseCellTypeMap


def _resolve_relative_map(data: Dict[str, str | Path], base: Path) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for key, value in data.items():
        if not isinstance(value, (str, Path)):
            raise ValueError("Map entries must be file paths.")
        path_val = Path(value)
        if not path_val.is_absolute():
            path_val = base / path_val
        resolved[key] = path_val
    return resolved


def _load_dnase_map_path(
    path: Path,
    *,
    track_key: str | Sequence[str] | None = None,
    label: str = "DNase",
) -> tuple[Dict[str, str | Path], Optional[DnaseCellTypeMap]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "entries" in data:
            cell_map = DnaseCellTypeMap.from_json(path, track_key=track_key)
            return cell_map.as_dnase_bigwigs(), cell_map
        if all(isinstance(v, (str, Path)) for v in data.values()):
            return _resolve_relative_map(data, path.parent), None
    if isinstance(data, list):
        cell_map = DnaseCellTypeMap.from_json(path, track_key=track_key)
        return cell_map.as_dnase_bigwigs(), cell_map
    raise ValueError(
        f"{label} map file must be a dict of celltype->path or a celltype mapping list."
    )
