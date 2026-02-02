"""
dnase_map.py

Load and query a cell-type -> DNase-seq path mapping with associated tumour
filters. Supports list- or dict-style JSON mappings (including an "entries"
wrapper) and provides helpers to resolve aliases and infer acceptable cell
types for tumour codes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def _norm_key(value: str) -> str:
    return str(value).strip().lower()


def _norm_tumour(value: str) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


DEFAULT_MAP_PATH = Path("data/raw/DNase-seq/celltype_dnase_map.json")


@dataclass(frozen=True)
class DnaseCellTypeEntry:
    key: str
    name: str
    dnase_path: Path
    tumour_types: tuple[str, ...]
    aliases: tuple[str, ...] = ()


class DnaseCellTypeMap:
    def __init__(self, entries: Iterable[DnaseCellTypeEntry]) -> None:
        self._entries = list(entries)
        self._by_key: dict[str, DnaseCellTypeEntry] = {}
        self._aliases: dict[str, DnaseCellTypeEntry] = {}
        for entry in self._entries:
            key = _norm_key(entry.key)
            if not key:
                raise ValueError("Cell type key cannot be empty.")
            if key in self._by_key:
                raise ValueError(f"Duplicate cell type key: {entry.key}")
            self._by_key[key] = entry
            self._register_alias(entry.key, entry)
            self._register_alias(entry.name, entry)
            for alias in entry.aliases:
                self._register_alias(alias, entry)

    def _register_alias(self, alias: str, entry: DnaseCellTypeEntry) -> None:
        if not alias:
            return
        norm = _norm_key(alias)
        if not norm:
            return
        existing = self._aliases.get(norm)
        if existing and existing.key != entry.key:
            raise ValueError(
                f"Alias '{alias}' is ambiguous between '{existing.key}' and '{entry.key}'."
            )
        self._aliases[norm] = entry

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        track_key: str | Sequence[str] | None = None,
    ) -> "DnaseCellTypeMap":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict) and "entries" in data:
            entries_raw = data["entries"]
        elif isinstance(data, dict):
            entries_raw = [{"key": k, **v} for k, v in data.items()]
        elif isinstance(data, list):
            entries_raw = data
        else:
            raise ValueError("Mapping JSON must be a list or dict.")

        base = Path(project_root) if project_root else path.parent
        if track_key is None:
            track_keys: List[str] = ["dnase_path"]
        elif isinstance(track_key, str):
            track_keys = [track_key]
        else:
            track_keys = list(track_key)
        track_keys = track_keys + ["path", "file", "filename"]
        entries: List[DnaseCellTypeEntry] = []
        for row in entries_raw:
            key = str(row.get("key") or "").strip()
            name = str(row.get("name") or row.get("cell_type") or key).strip()
            dnase_raw = None
            for key_name in track_keys:
                dnase_raw = row.get(key_name)
                if dnase_raw:
                    break
            if not dnase_raw:
                expected = ", ".join(track_keys)
                raise ValueError(
                    f"Missing track path for cell type '{key or name}'. Expected one of: {expected}."
                )
            dnase_path = Path(dnase_raw)
            if not dnase_path.is_absolute():
                dnase_path = base / dnase_path
            tumour_types_raw = row.get("tumour_types") or row.get("tumor_types") or []
            aliases_raw = row.get("alias") or row.get("aliases") or []
            if isinstance(aliases_raw, str):
                aliases_list: Sequence[str] = [aliases_raw]
            else:
                aliases_list = aliases_raw
            entries.append(
                DnaseCellTypeEntry(
                    key=key,
                    name=name,
                    dnase_path=dnase_path,
                    tumour_types=tuple(str(t).strip() for t in tumour_types_raw if str(t).strip()),
                    aliases=tuple(str(a).strip() for a in aliases_list if str(a).strip()),
                )
            )
        return cls(entries)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "DnaseCellTypeMap":
        root = Path(project_root)
        return cls.from_json(root / DEFAULT_MAP_PATH, project_root=root)

    def resolve(self, cell_type: str) -> DnaseCellTypeEntry:
        key = _norm_key(cell_type)
        if not key:
            raise KeyError("Cell type cannot be empty.")
        entry = self._aliases.get(key)
        if not entry:
            known = ", ".join(sorted(self._by_key))
            raise KeyError(f"Unknown cell type '{cell_type}'. Known keys: {known}")
        return entry

    def dnase_path(self, cell_type: str) -> Path:
        return self.resolve(cell_type).dnase_path

    def tumour_types(self, cell_type: str) -> tuple[str, ...]:
        return self.resolve(cell_type).tumour_types

    def as_dnase_bigwigs(self) -> Mapping[str, Path]:
        return {entry.key: entry.dnase_path for entry in self._entries}

    def tumour_filter(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for entry in self._entries:
            for tumour in entry.tumour_types:
                norm = _norm_tumour(tumour)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                out.append(tumour)
        return out


    def infer_correct_celltypes(self, tumour_types: Sequence[str]) -> List[str]:
        if not tumour_types:
            return []
        tumour_map: dict[str, set[str]] = {}
        for entry in self._entries:
            for tumour in entry.tumour_types:
                norm = _norm_tumour(tumour)
                if not norm:
                    continue
                tumour_map.setdefault(norm, set()).add(entry.key)

        inferred: set[str] = set()
        for tumour in tumour_types:
            norm = _norm_tumour(tumour)
            if not norm:
                return []
            celltypes = tumour_map.get(norm)
            if not celltypes:
                return []
            inferred.update(celltypes)

        return sorted(inferred)

    def keys(self) -> Sequence[str]:
        return [entry.key for entry in self._entries]

    def entries(self) -> Sequence[DnaseCellTypeEntry]:
        return list(self._entries)
