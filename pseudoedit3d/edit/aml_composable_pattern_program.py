from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROGRAM_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "aml_regression_testset_v2"
    / "aml_composable_pattern_program_v0"
    / "aml_composable_pattern_program.json"
)


@lru_cache(maxsize=4)
def load_composable_pattern_program(path: str | Path | None = None) -> dict[str, Any]:
    program_path = Path(path) if path is not None else DEFAULT_PROGRAM_PATH
    return json.loads(program_path.read_text(encoding="utf-8"))


def program_nodes(program: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("program_node_id") or ""): dict(node)
        for node in program.get("nodes") or []
        if node.get("program_node_id")
    }


def child_node_ids(program: dict[str, Any], program_node_id: str) -> list[str]:
    out: list[str] = []
    for edge in program.get("edges") or []:
        if str(edge.get("parent_program_node_id") or "") == str(program_node_id):
            child = str(edge.get("child_program_node_id") or "")
            if child:
                out.append(child)
    return out


def child_nodes(program: dict[str, Any], program_node_id: str) -> list[dict[str, Any]]:
    nodes = program_nodes(program)
    return [nodes[node_id] for node_id in child_node_ids(program, program_node_id) if node_id in nodes]


def condition_vocabulary(
    program: dict[str, Any],
    *,
    scopes: Iterable[str] | None = None,
    review_statuses: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    scope_set = {str(item) for item in scopes} if scopes is not None else None
    status_set = {str(item) for item in review_statuses} if review_statuses is not None else None
    out: list[dict[str, Any]] = []
    for condition in program.get("condition_vocabulary") or []:
        if scope_set is not None and str(condition.get("scope") or "") not in scope_set:
            continue
        if status_set is not None and str(condition.get("review_status") or "") not in status_set:
            continue
        out.append(dict(condition))
    return out


def _as_set(values: Iterable[Any] | None) -> set[str]:
    return {str(value) for value in values or [] if str(value)}


def _signature_score(signature: dict[str, Any], evidence: dict[str, set[str]]) -> tuple[float, dict[str, Any]]:
    required_channels = _as_set(signature.get("required_channels"))
    required_zones = _as_set(signature.get("required_zones"))
    required_clusters = _as_set(signature.get("required_cluster_ids"))
    required_families = _as_set(signature.get("required_event_families"))

    channel_hits = sorted(required_channels & evidence["channels"])
    zone_hits = sorted(required_zones & evidence["zones"])
    cluster_hits = sorted(required_clusters & evidence["cluster_ids"])
    family_hits = sorted(required_families & evidence["event_families"])

    min_channel_overlap = int(signature.get("min_channel_overlap") or 0)
    min_cluster_overlap = int(signature.get("min_cluster_overlap") or 0)
    min_event_family_overlap = int(signature.get("min_event_family_overlap") or 0)
    if len(channel_hits) < min_channel_overlap:
        return 0.0, {"failed": "channel_overlap", "channel_hits": channel_hits}
    if len(cluster_hits) < min_cluster_overlap:
        return 0.0, {"failed": "cluster_overlap", "cluster_hits": cluster_hits}
    if len(family_hits) < min_event_family_overlap:
        return 0.0, {"failed": "event_family_overlap", "event_family_hits": family_hits}

    channel_score = len(channel_hits) / max(1, len(required_channels))
    zone_score = len(zone_hits) / max(1, len(required_zones)) if required_zones else 0.0
    cluster_score = len(cluster_hits) / max(1, len(required_clusters)) if required_clusters else 0.0
    family_score = len(family_hits) / max(1, len(required_families)) if required_families else 0.0
    score = (0.35 * channel_score) + (0.15 * zone_score) + (0.35 * cluster_score) + (0.15 * family_score)
    return score, {
        "channel_hits": channel_hits,
        "zone_hits": zone_hits,
        "cluster_hits": cluster_hits,
        "event_family_hits": family_hits,
        "channel_score": round(channel_score, 4),
        "zone_score": round(zone_score, 4),
        "cluster_score": round(cluster_score, 4),
        "event_family_score": round(family_score, 4),
    }


def search_program_nodes(
    program: dict[str, Any],
    *,
    channels: Iterable[Any] | None = None,
    zones: Iterable[Any] | None = None,
    cluster_ids: Iterable[Any] | None = None,
    event_families: Iterable[Any] | None = None,
    semantic_levels: Iterable[Any] | None = None,
    node_kinds: Iterable[Any] | None = ("structure_group", "composition_family"),
    top_k: int = 20,
    min_score: float = 0.20,
) -> list[dict[str, Any]]:
    evidence = {
        "channels": _as_set(channels),
        "zones": _as_set(zones),
        "cluster_ids": _as_set(cluster_ids),
        "event_families": _as_set(event_families),
    }
    level_filter = _as_set(semantic_levels)
    kind_filter = _as_set(node_kinds) if node_kinds is not None else set()
    candidates: list[dict[str, Any]] = []
    for node in program.get("nodes") or []:
        if kind_filter and str(node.get("program_node_kind") or "") not in kind_filter:
            continue
        if level_filter and str(node.get("semantic_level") or "") not in level_filter:
            continue
        signature = node.get("match_signature") or {}
        if not signature:
            continue
        score, detail = _signature_score(signature, evidence)
        if score < min_score:
            continue
        candidates.append(
            {
                "program_node_id": node.get("program_node_id"),
                "program_node_kind": node.get("program_node_kind"),
                "motion_structure_label": node.get("motion_structure_label"),
                "semantic_level": node.get("semantic_level"),
                "edit_scope": node.get("edit_scope"),
                "composition_policy": node.get("composition_policy"),
                "review_status": node.get("review_status"),
                "scope": node.get("scope"),
                "score": round(score, 4),
                "match_detail": detail,
                "condition_entry": node.get("condition_entry"),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row.get("score") or 0.0),
            str(row.get("semantic_level") or ""),
            str(row.get("program_node_id") or ""),
        )
    )
    return candidates[: max(0, int(top_k))]


def condition_by_program_node(program: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for condition in program.get("condition_vocabulary") or []:
        node_id = str(condition.get("program_node_id") or "")
        if node_id:
            out[node_id] = dict(condition)
    return out


def edit_handles_for_condition(condition: dict[str, Any]) -> list[str]:
    return [str(item.get("name") or "") for item in condition.get("edit_handles") or [] if item.get("name")]


def summarize_program(program: dict[str, Any]) -> dict[str, Any]:
    summary = dict(program.get("summary") or {})
    summary["root_count"] = sum(1 for node in program.get("nodes") or [] if node.get("program_node_kind") == "program_root")
    return summary
