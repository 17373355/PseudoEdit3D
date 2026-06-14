from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .aml_family_taxonomy import active_family_id


_PROTO_REGISTRY_PATH = Path(__file__).with_name("aml_proto_registry.json")


@lru_cache(maxsize=1)
def proto_registry() -> dict[str, Any]:
    return json.loads(_PROTO_REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_set(*path: str) -> set[str]:
    value: Any = proto_registry()
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    if isinstance(value, dict):
        return {str(item) for item in value.keys()}
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def registry_map(*path: str) -> dict[str, Any]:
    value: Any = proto_registry()
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def legacy_proto_aliases() -> dict[str, str]:
    return {str(key): str(value) for key, value in registry_map("legacy_aliases").items()}


def active_proto_id(proto_id: str) -> str:
    return legacy_proto_aliases().get(str(proto_id), active_family_id(str(proto_id)))


def proto_in_group(proto_id: str, *path: str) -> bool:
    return active_proto_id(proto_id) in registry_set(*path)
