from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_TAXONOMY_PATH = Path(__file__).with_name("aml_family_taxonomy.json")


@lru_cache(maxsize=1)
def load_aml_family_taxonomy() -> dict[str, Any]:
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


def active_family_id(family_id: str) -> str:
    family_id = str(family_id or "UNKNOWN")
    override = (load_aml_family_taxonomy().get("family_overrides") or {}).get(family_id) or {}
    return str(override.get("legacy_alias_for") or family_id)


@lru_cache(maxsize=1)
def _family_index() -> dict[str, dict[str, Any]]:
    taxonomy = load_aml_family_taxonomy()
    parents = taxonomy.get("parents") or {}
    overrides = taxonomy.get("family_overrides") or {}
    index: dict[str, dict[str, Any]] = {}
    for parent_id, parent in parents.items():
        for family_id in parent.get("children") or []:
            family_id = str(family_id)
            row = index.setdefault(
                family_id,
                {
                    "family_id": family_id,
                    "taxonomy_parent_id": str(parent_id),
                    "taxonomy_parent_label": str(parent.get("label") or parent_id),
                    "taxonomy_recoverability": str(parent.get("recoverability") or "unknown"),
                    "taxonomy_evidence_axes": [str(item) for item in parent.get("evidence_axes") or []],
                    "taxonomy_secondary_parent_ids": [],
                    "ambiguity_boundary": "motion_geometry",
                },
            )
            if row["taxonomy_parent_id"] != str(parent_id):
                row.setdefault("taxonomy_secondary_parent_ids", []).append(str(parent_id))
    for family_id, override in overrides.items():
        row = index.setdefault(
            str(family_id),
            {
                "family_id": str(family_id),
                "taxonomy_parent_id": str((taxonomy.get("default_family") or {}).get("parent_id") or "UNKNOWN_OR_FALLBACK"),
                "taxonomy_parent_label": "unknown or fallback",
                "taxonomy_recoverability": str((taxonomy.get("default_family") or {}).get("recoverability") or "unknown"),
                "taxonomy_evidence_axes": [],
                "taxonomy_secondary_parent_ids": [],
                "ambiguity_boundary": str((taxonomy.get("default_family") or {}).get("ambiguity_boundary") or "unknown_family"),
            },
        )
        row["taxonomy_secondary_parent_ids"] = sorted(
            set(row.get("taxonomy_secondary_parent_ids") or []) | {str(item) for item in override.get("secondary_parents") or []}
        )
        if override.get("ambiguity_boundary"):
            row["ambiguity_boundary"] = str(override["ambiguity_boundary"])
        if override.get("legacy_alias_for"):
            target = active_family_id(str(family_id))
            target_row = index.get(target)
            if target_row:
                row.update(
                    {
                        "taxonomy_parent_id": target_row.get("taxonomy_parent_id"),
                        "taxonomy_parent_label": target_row.get("taxonomy_parent_label"),
                        "taxonomy_recoverability": target_row.get("taxonomy_recoverability"),
                        "taxonomy_evidence_axes": list(target_row.get("taxonomy_evidence_axes") or []),
                        "taxonomy_secondary_parent_ids": sorted(
                            set(target_row.get("taxonomy_secondary_parent_ids") or [])
                            | set(row.get("taxonomy_secondary_parent_ids") or [])
                        ),
                        "ambiguity_boundary": target_row.get("ambiguity_boundary"),
                        "legacy_alias_for": target,
                    }
                )
    return index


def family_taxonomy_metadata(family_id: str) -> dict[str, Any]:
    family_id = str(family_id or "UNKNOWN")
    index = _family_index()
    if family_id in index:
        return dict(index[family_id])
    default = load_aml_family_taxonomy().get("default_family") or {}
    return {
        "family_id": family_id,
        "taxonomy_parent_id": str(default.get("parent_id") or "UNKNOWN_OR_FALLBACK"),
        "taxonomy_parent_label": "unknown or fallback",
        "taxonomy_recoverability": str(default.get("recoverability") or "unknown"),
        "taxonomy_evidence_axes": [],
        "taxonomy_secondary_parent_ids": [],
        "ambiguity_boundary": str(default.get("ambiguity_boundary") or "unknown_family"),
    }


def taxonomy_parent_for_family(family_id: str) -> str:
    return str(family_taxonomy_metadata(family_id).get("taxonomy_parent_id") or "UNKNOWN_OR_FALLBACK")
