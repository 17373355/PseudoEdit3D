from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_PATTERN_TREE_PATH = Path(__file__).with_name("aml_pattern_tree.json")
_MISSING = object()


@dataclass(frozen=True)
class PatternMatch:
    node_id: str
    family_id: str | None
    node: dict[str, Any]
    path: list[str]
    depth: int
    score: float
    prototype: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def pattern_tree() -> dict[str, Any]:
    return json.loads(_PATTERN_TREE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def pattern_nodes() -> dict[str, dict[str, Any]]:
    nodes = pattern_tree().get("nodes") or []
    return {str(node["node_id"]): dict(node) for node in nodes if isinstance(node, dict) and node.get("node_id")}


def primary_selection_order() -> list[str]:
    return [str(item) for item in pattern_tree().get("primary_selection_order") or []]


def composed_selection_order() -> list[str]:
    return [str(item) for item in pattern_tree().get("composed_selection_order") or []]


def sparse_action_selection_order() -> list[str]:
    return [str(item) for item in pattern_tree().get("sparse_action_selection_order") or []]


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    node_id = str(node.get("node_id", ""))
    metadata = {
        "pattern_node_id": node_id,
        "pattern_path": pattern_path(node_id),
    }
    if node.get("taxonomy_parent_id"):
        metadata["pattern_taxonomy_parent_id"] = str(node["taxonomy_parent_id"])
    return metadata


def pattern_path(node_id: str) -> list[str]:
    nodes = pattern_nodes()
    path: list[str] = []
    current = str(node_id)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        node = nodes.get(current)
        if not node:
            break
        path.append(current)
        parent = node.get("parent_id")
        if parent is None:
            break
        current = str(parent)
    return list(reversed(path))


@lru_cache(maxsize=1)
def family_pattern_nodes() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for node in pattern_nodes().values():
        family_id = node.get("family_id")
        if family_id:
            out.setdefault(str(family_id), []).append(dict(node))
    return out


def pattern_node_for_family(family_id: str, *, preferred_node_types: tuple[str, ...] | None = None) -> dict[str, Any] | None:
    nodes = family_pattern_nodes().get(str(family_id), [])
    if not nodes:
        return None
    if preferred_node_types:
        preferred = {str(item) for item in preferred_node_types}
        for node in nodes:
            if str(node.get("node_type", "")) in preferred:
                return dict(node)
    return dict(nodes[0])


def action_pattern_metadata_for_family(family_id: str, *, preferred_node_types: tuple[str, ...] | None = None) -> dict[str, Any]:
    node = pattern_node_for_family(family_id, preferred_node_types=preferred_node_types)
    return _node_metadata(node) if node else {}


@lru_cache(maxsize=1)
def event_proxy_map() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for node in pattern_nodes().values():
        if str(node.get("node_type", "")) != "event_proxy":
            continue
        proxy = node.get("event_proxy") or {}
        super_family = str(proxy.get("super_family", ""))
        cluster_id = str(proxy.get("cluster_id", ""))
        if not super_family or not cluster_id:
            continue
        out[(super_family, cluster_id)] = dict(node)
    return out


def event_proxy_for_event(event: dict[str, Any]) -> dict[str, Any] | None:
    key = (str(event.get("super_family", "")), str(event.get("cluster_id", "")))
    node = event_proxy_map().get(key)
    return dict(node) if node else None


def event_proxy_action_fields(node: dict[str, Any]) -> tuple[str, str, str, float]:
    proxy = node.get("event_proxy") or {}
    return (
        str(node.get("family_id", "")),
        str(proxy.get("name_hint") or node.get("family_id") or ""),
        str(proxy.get("primary_direction", "unknown")),
        float(proxy.get("base_confidence", 0.0) or 0.0),
    )


def action_pattern_metadata_for_node(node: dict[str, Any]) -> dict[str, Any]:
    return _node_metadata(node)


def ctx_path(ctx: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = ctx
    for part in str(path).split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def cast_spec_value(value: Any, cast: str | None, default: Any = None) -> Any:
    if value is _MISSING or value is None:
        return default
    if cast == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if cast == "float":
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return default
    if cast == "list":
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return list(default or [])
        return [value]
    if cast == "str":
        return str(value)
    if cast == "bool":
        return bool(value)
    return value


def condition_matches(condition: dict[str, Any], ctx: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(condition_matches(dict(child), ctx) for child in condition.get("all") or [])
    if "any" in condition:
        return any(condition_matches(dict(child), ctx) for child in condition.get("any") or [])
    if "not" in condition:
        child = condition.get("not")
        return not (isinstance(child, dict) and condition_matches(child, ctx))

    actual = ctx_path(ctx, str(condition.get("field", "")), _MISSING)
    op = str(condition.get("op", "eq"))
    expected = condition.get("value")
    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not bool(actual)
    if actual is _MISSING:
        return False
    if op == "eq":
        return actual == expected
    if op == "neq":
        return actual != expected
    if op == "in":
        return actual in set(expected or [])
    if op == "not_in":
        return actual not in set(expected or [])
    if op == "gt":
        return numeric(actual) > numeric(expected)
    if op == "gte":
        return numeric(actual) >= numeric(expected)
    if op == "lt":
        return numeric(actual) < numeric(expected)
    if op == "lte":
        return numeric(actual) <= numeric(expected)
    raise ValueError(f"unsupported pattern-tree condition op: {op}")


def node_matches(node: dict[str, Any], ctx: dict[str, Any]) -> bool:
    return all(condition_matches(dict(condition), ctx) for condition in node.get("match") or [])


def resolve_spec_value(node: Any, ctx: dict[str, Any]) -> Any:
    if not isinstance(node, dict):
        return node
    if "cases" in node:
        for case in node.get("cases") or []:
            when = case.get("when")
            if isinstance(when, dict) and condition_matches(when, ctx):
                return resolve_spec_value(case.get("then"), ctx)
        return resolve_spec_value(node.get("default"), ctx)
    if "value" in node:
        return node["value"]
    if "path" in node:
        default = node.get("default")
        value = ctx_path(ctx, str(node["path"]), _MISSING)
        return cast_spec_value(value, node.get("cast"), default)
    if "len_path" in node:
        value = ctx_path(ctx, str(node["len_path"]), [])
        return len(value or [])
    if "template" in node:
        args = {
            str(key): resolve_spec_value(arg_node, ctx)
            for key, arg_node in (node.get("args") or {}).items()
        }
        return str(node["template"]).format(**args)
    if "base" in node and "add_min" in node:
        add = node.get("add_min") or {}
        raw = ctx_path(ctx, str(add.get("path", "")), 0.0)
        return float(node["base"]) + min(float(add.get("cap", 0.0)), float(add.get("scale", 1.0)) * numeric(raw))
    return dict(node)


def prototype_from_node_outputs(node: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    outputs = node.get("outputs") or {}
    prototype = {
        str(key): resolve_spec_value(value, ctx)
        for key, value in outputs.items()
    }
    prototype.setdefault("prototype_id", node.get("family_id"))
    prototype["pattern_node_id"] = str(node.get("node_id", ""))
    prototype["pattern_path"] = pattern_path(str(node.get("node_id", "")))
    if node.get("taxonomy_parent_id"):
        prototype["pattern_taxonomy_parent_id"] = str(node["taxonomy_parent_id"])
    return prototype


def match_pattern_tree(ctx: dict[str, Any], *, node_type: str | None = None) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    for node in pattern_nodes().values():
        if node_type is not None and str(node.get("node_type", "")) != node_type:
            continue
        if not node.get("match"):
            continue
        if not node_matches(node, ctx):
            continue
        node_id = str(node["node_id"])
        path = pattern_path(node_id)
        matches.append(
            PatternMatch(
                node_id=node_id,
                family_id=str(node["family_id"]) if node.get("family_id") else None,
                node=dict(node),
                path=path,
                depth=len(path) - 1,
                score=float(node.get("priority_score", len(path))),
            )
        )
    return matches


def select_primary_pattern_match(ctx: dict[str, Any]) -> PatternMatch | None:
    nodes = pattern_nodes()
    for node_id in primary_selection_order():
        node = nodes.get(node_id)
        if not node or not node_matches(node, ctx):
            continue
        path = pattern_path(node_id)
        return PatternMatch(
            node_id=node_id,
            family_id=str(node["family_id"]) if node.get("family_id") else None,
            node=dict(node),
            path=path,
            depth=len(path) - 1,
            score=float(node.get("priority_score", len(path))),
        )
    return None


def select_composed_pattern_match(ctx: dict[str, Any]) -> PatternMatch | None:
    return _select_ordered_pattern_match(ctx, node_type="composed_candidate", ordered_node_ids=composed_selection_order())


def select_sparse_pattern_match(ctx: dict[str, Any]) -> PatternMatch | None:
    return _select_ordered_pattern_match(ctx, node_type="sparse_candidate", ordered_node_ids=sparse_action_selection_order())


def _select_ordered_pattern_match(
    ctx: dict[str, Any],
    *,
    node_type: str,
    ordered_node_ids: list[str],
) -> PatternMatch | None:
    nodes = pattern_nodes()
    selected = set(ordered_node_ids)
    ordered_ids = list(ordered_node_ids) + [
        node_id
        for node_id, node in nodes.items()
        if str(node.get("node_type", "")) == node_type and node_id not in selected
    ]
    for node_id in ordered_ids:
        node = nodes.get(node_id)
        if not node or str(node.get("node_type", "")) != node_type or not node.get("match"):
            continue
        if not node_matches(node, ctx):
            continue
        path = pattern_path(node_id)
        return PatternMatch(
            node_id=node_id,
            family_id=str(node["family_id"]) if node.get("family_id") else None,
            node=dict(node),
            path=path,
            depth=len(path) - 1,
            score=float(node.get("priority_score", len(path))),
        )
    return None
