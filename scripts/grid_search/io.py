"""I/O helpers for grid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from scripts.dnase_map import DnaseCellTypeMap


def _load_dnase_map_path(path: Path) -> tuple[Dict[str, str | Path], Optional[DnaseCellTypeMap]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "entries" in data:
            cell_map = DnaseCellTypeMap.from_json(path)
            return cell_map.as_dnase_bigwigs(), cell_map
        if all(isinstance(v, str) for v in data.values()):
            return data, None
    if isinstance(data, list):
        cell_map = DnaseCellTypeMap.from_json(path)
        return cell_map.as_dnase_bigwigs(), cell_map
    raise ValueError("DNase map file must be a dict of celltype->path or a celltype mapping list.")
