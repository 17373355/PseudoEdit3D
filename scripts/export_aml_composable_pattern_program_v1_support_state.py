"""Export a v1 support-state AML composable pattern program.

This script materializes the reviewed support-state forest into the same
program-facing shape used by the older v0 composable program:

    program nodes
    condition vocabulary
    match signatures
    search index
    readable tree

It is an offline bridge. It does not change the runtime matcher or the legacy
coarse signature logic.

Typical use:
    python scripts/export_aml_composable_pattern_program_v1_support_state.py

Smoke test:
    python scripts/export_aml_composable_pattern_program_v1_support_state.py --self-test
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REVIEWED_FOREST = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_reviewed_draft/"
    "aml_pattern_forest_v1_reviewed_draft.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_reviewed_draft"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_id(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "unnamed"


def _child_index(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child:
            out[parent].append(child)
    return out


def _ids(values: Any) -> list[str]:
    out: list[str] = []
    for item in values or []:
        if isinstance(item, dict):
            value = str(item.get("id") or "")
        else:
            value = str(item or "")
        if value:
            out.append(value)
    return sorted(set(out))


def _counted_ids(values: Any) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for item in values or []:
        if isinstance(item, dict):
            value = str(item.get("id") or "")
            try:
                count = int(item.get("count") or 1)
            except (TypeError, ValueError):
                count = 1
        else:
            value = str(item or "")
            count = 1
        if not value:
            continue
        if value in seen:
            seen[value] = max(seen[value], count)
            continue
        seen[value] = count
        out.append((value, count))
    return [(value, seen[value]) for value, _ in out]


def _stable_ids(values: Any, *, min_ratio: float, max_items: int, min_count: int = 1) -> list[str]:
    counted = _counted_ids(values)
    if not counted:
        return []
    max_count = max(count for _, count in counted)
    threshold = max(float(min_count), float(max_count) * float(min_ratio))
    kept = [value for value, count in counted if float(count) >= threshold]
    if not kept:
        kept = [counted[0][0]]
    return sorted(set(kept[: max(1, int(max_items))]))


def _motion_summary(node: dict[str, Any]) -> dict[str, Any]:
    return dict(node.get("motion_summary") or {})


def _axis_tokens(node: dict[str, Any]) -> dict[str, list[str]]:
    motion = _motion_summary(node)
    status = str(node.get("status") or "")
    channel_ratio = 0.30 if status == "accepted" else 0.45
    role_ratio = 0.35 if status == "accepted" else 0.50
    geometry_ratio = 0.30 if status == "accepted" else 0.50
    channels = _stable_ids(motion.get("channels"), min_ratio=channel_ratio, max_items=8)
    zones = _stable_ids(motion.get("zones"), min_ratio=channel_ratio, max_items=8)
    canonical_roles = _stable_ids(motion.get("canonical_role_items"), min_ratio=role_ratio, max_items=8)
    geometry_roles = _stable_ids(motion.get("geometry_clusters"), min_ratio=geometry_ratio, max_items=10)
    cluster_ids: list[str] = []
    event_families: list[str] = []
    for role in geometry_roles:
        if "/" in role:
            event_families.append(role.split("/", 1)[0])
            cluster_ids.append(role.rsplit("/", 1)[-1])
        elif ":" in role:
            event_families.append(role.split(":", 1)[0])
        else:
            event_families.append(role)
    return {
        "channels": channels,
        "zones": zones,
        "canonical_roles": canonical_roles,
        "geometry_roles": geometry_roles,
        "cluster_ids": sorted(set(cluster_ids)),
        "event_families": sorted(set(event_families)),
    }


def _semantic_level(node: dict[str, Any]) -> str:
    kind = str(node.get("node_kind") or "")
    status = str(node.get("status") or "")
    if kind == "root":
        return "taxonomy_root"
    if kind == "source_closure_candidate":
        return "source_evidence"
    if status == "accepted":
        return "whole_body_pattern"
    if status == "component":
        return "component"
    if status == "split_required":
        return "split_required_candidate"
    if status == "pending_closure":
        return "closure_required_candidate"
    return "review_candidate"


def _edit_scope(node: dict[str, Any]) -> str:
    status = str(node.get("status") or "")
    axes = _axis_tokens(node)
    channels = set(axes["channels"])
    if status == "accepted":
        return "whole_body"
    if channels & {"whole_body_support", "acrobatics_or_inversion", "whole_body_state", "whole_body_vertical"}:
        if channels & {"left_arm", "right_arm", "bimanual"} and channels & {"left_leg", "right_leg"}:
            return "whole_body"
        return "root_or_body"
    if channels & {"left_arm", "right_arm", "bimanual"} and channels & {"left_leg", "right_leg"}:
        return "multi_part"
    if channels & {"left_arm", "right_arm", "bimanual"}:
        return "upper_body_or_arm"
    if channels & {"left_leg", "right_leg"}:
        return "lower_body_or_leg"
    return "local"


def _composition_policy(node: dict[str, Any]) -> str:
    status = str(node.get("status") or "")
    if status == "accepted":
        return "bind_as_reviewed_full_pattern"
    if status == "component":
        return "bind_as_reusable_component"
    if status == "split_required":
        return "requires_structural_split_before_promotion"
    if status == "pending_closure":
        return "requires_composition_closure_before_promotion"
    if status.startswith("source_"):
        return "source_evidence_only"
    return "review_before_binding"


def _match_signature(node: dict[str, Any]) -> dict[str, Any]:
    axes = _axis_tokens(node)
    kind = str(node.get("node_kind") or "")
    status = str(node.get("status") or "")
    if kind == "root":
        min_channels = 0
    elif status == "accepted":
        min_channels = max(2, min(4, len(axes["channels"])))
    else:
        min_channels = max(1, min(2, len(axes["channels"])))
    if kind == "root":
        min_clusters = 0
    elif status == "accepted":
        min_clusters = min(2, len(axes["cluster_ids"]))
    else:
        min_clusters = 1 if axes["cluster_ids"] else 0
    min_families = 1 if axes["event_families"] else 0
    if status == "accepted" and axes["event_families"]:
        min_families = min(2, len(axes["event_families"]))
    return {
        "match_version": "aml_composable_match_signature_v1_support_state",
        "required_channels": axes["channels"],
        "required_zones": axes["zones"],
        "required_cluster_ids": axes["cluster_ids"],
        "required_event_families": axes["event_families"],
        "geometry_roles": axes["geometry_roles"],
        "canonical_roles": axes["canonical_roles"],
        "min_channel_overlap": min_channels,
        "min_cluster_overlap": min_clusters,
        "min_event_family_overlap": min_families,
        "temporal_relation": "coactivation_or_nested_span",
        "support_hint": {
            "support_cases_max": int((node.get("evidence") or {}).get("support_cases_max") or 0),
            "support_cases_sum": int((node.get("evidence") or {}).get("support_cases_sum") or 0),
            "source_candidate_count": int((node.get("evidence") or {}).get("source_candidate_count") or 0),
        },
    }


def _edit_handles(node: dict[str, Any]) -> list[dict[str, str]]:
    text = " ".join(
        [
            str(node.get("accepted_name") or ""),
            " ".join(_axis_tokens(node)["channels"]),
            " ".join(_axis_tokens(node)["canonical_roles"]),
            " ".join(_axis_tokens(node)["geometry_roles"]),
        ]
    ).lower()
    handles: list[dict[str, str]] = [
        {"name": "span", "type": "temporal", "description": "start/end frame interval"},
        {"name": "count", "type": "numeric", "description": "repetition or phase count when available"},
    ]
    if "vertical" in text or "level" in text or "support" in text:
        handles.append({"name": "body_level", "type": "numeric", "description": "body height, support, or low-body state"})
    if "arm" in text or "bimanual" in text:
        handles.append({"name": "arm_height", "type": "numeric", "description": "hand or arm elevation"})
        handles.append({"name": "arm_symmetry", "type": "categorical", "description": "left/right/bilateral arm role"})
    if "leg" in text:
        handles.append({"name": "leg_extension", "type": "numeric", "description": "leg extension or lateral spread amount"})
        handles.append({"name": "leg_side", "type": "categorical", "description": "left/right/bilateral leg role"})
    if "inverted" in text or "acrobatics" in text:
        handles.append({"name": "inversion_support", "type": "numeric", "description": "strength/duration of inverted support evidence"})
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for handle in handles:
        if handle["name"] in seen:
            continue
        seen.add(handle["name"])
        out.append(handle)
    return out


def _condition_id(node: dict[str, Any]) -> str:
    # Use the reviewed family node id as the stable condition id. The human
    # label remains in `motion_structure_label`; condition ids must stay unique
    # because batch exporters commonly build a vocabulary from this field.
    return "AMLV1_" + _safe_id(str(node.get("node_id") or "pattern")).upper()


def _condition_entry(node: dict[str, Any], program_node_id: str) -> dict[str, Any]:
    axes = _axis_tokens(node)
    status = str(node.get("status") or "")
    weight = 1.0 if status == "accepted" else 0.0
    evidence = dict(node.get("evidence") or {})
    return {
        "condition_id": _condition_id(node),
        "condition_entry_id": "AMLV1NODE_" + _safe_id(program_node_id).upper(),
        "program_node_id": program_node_id,
        "motion_structure_label": str(node.get("accepted_name") or node.get("node_id") or ""),
        "scope": str(node.get("scope") or ""),
        "review_status": status,
        "review_decision": str(node.get("review_decision") or ""),
        "condition_weight_default": weight,
        "promotion_policy": _composition_policy(node),
        "channels": axes["channels"],
        "zones": axes["zones"],
        "geometry_roles": axes["geometry_roles"],
        "canonical_roles": axes["canonical_roles"],
        "support_cases_max": int(evidence.get("support_cases_max") or 0),
        "support_cases_sum": int(evidence.get("support_cases_sum") or 0),
        "source_candidate_count": int(evidence.get("source_candidate_count") or 0),
        "caption_name_candidates": list(node.get("language_aliases") or []),
        "semantic_level": _semantic_level(node),
        "edit_scope": _edit_scope(node),
        "composition_policy": _composition_policy(node),
        "match_signature": _match_signature(node),
        "edit_handles": _edit_handles(node),
        "review_image_path": str(node.get("review_image_path") or ""),
        "review_notes": str(node.get("review_notes") or ""),
    }


def _program_node_id(source_node_id: str, source_kind: str) -> str:
    prefix = {
        "root": "program_root",
        "pattern_family_candidate": "program_family",
        "source_closure_candidate": "program_source",
    }.get(source_kind, "program_node")
    return f"{prefix}_{_safe_id(source_node_id)}"


def _convert_node(node: dict[str, Any], children: list[str]) -> dict[str, Any]:
    source_kind = str(node.get("node_kind") or "")
    program_id = _program_node_id(str(node.get("node_id") or ""), source_kind)
    converted = {
        "program_node_id": program_id,
        "source_node_id": str(node.get("node_id") or ""),
        "source_node_kind": source_kind,
        "program_node_kind": {
            "root": "program_root",
            "pattern_family_candidate": "pattern_family",
            "source_closure_candidate": "source_evidence",
        }.get(source_kind, "unknown"),
        "display_name": node.get("accepted_name"),
        "motion_structure_label": node.get("accepted_name"),
        "review_status": node.get("status"),
        "review_decision": node.get("review_decision"),
        "scope": node.get("scope"),
        "semantic_level": _semantic_level(node),
        "edit_scope": _edit_scope(node),
        "composition_policy": _composition_policy(node),
        "match_signature": _match_signature(node),
        "children": children,
    }
    for key in [
        "accepted_name",
        "language_aliases",
        "description",
        "evidence",
        "motion_summary",
        "review_notes",
        "review_image_path",
        "review_example_case_ids",
        "source_examples",
    ]:
        if key in node:
            converted[key] = node[key]
    if source_kind == "pattern_family_candidate":
        converted["condition_entry"] = _condition_entry(node, program_id)
    return converted


def _build_search_index(program_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    axes = {
        "channels": defaultdict(list),
        "zones": defaultdict(list),
        "cluster_ids": defaultdict(list),
        "event_families": defaultdict(list),
        "geometry_roles": defaultdict(list),
        "canonical_roles": defaultdict(list),
        "semantic_levels": defaultdict(list),
        "review_statuses": defaultdict(list),
    }
    for node in program_nodes:
        node_id = str(node.get("program_node_id") or "")
        signature = node.get("match_signature") or {}
        for value in signature.get("required_channels") or []:
            axes["channels"][str(value)].append(node_id)
        for value in signature.get("required_zones") or []:
            axes["zones"][str(value)].append(node_id)
        for value in signature.get("required_cluster_ids") or []:
            axes["cluster_ids"][str(value)].append(node_id)
        for value in signature.get("required_event_families") or []:
            axes["event_families"][str(value)].append(node_id)
        for value in signature.get("geometry_roles") or []:
            axes["geometry_roles"][str(value)].append(node_id)
        for value in signature.get("canonical_roles") or []:
            axes["canonical_roles"][str(value)].append(node_id)
        level = str(node.get("semantic_level") or "")
        if level:
            axes["semantic_levels"][level].append(node_id)
        status = str(node.get("review_status") or "")
        if status:
            axes["review_statuses"][status].append(node_id)
    return {
        "index_version": "aml_composable_search_index_v1_support_state",
        "axes": {
            axis: {key: sorted(set(values)) for key, values in sorted(axis_values.items())}
            for axis, axis_values in axes.items()
        },
    }


def build_program(forest: dict[str, Any]) -> dict[str, Any]:
    source_nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    source_children = _child_index(list(forest.get("edges") or []))
    source_to_program = {
        source_id: _program_node_id(source_id, str(node.get("node_kind") or ""))
        for source_id, node in source_nodes.items()
    }

    program_nodes: list[dict[str, Any]] = []
    for source_id, node in source_nodes.items():
        child_ids = [source_to_program[child_id] for child_id in source_children.get(source_id, []) if child_id in source_to_program]
        program_nodes.append(_convert_node(node, child_ids))

    edges: list[dict[str, str]] = []
    for edge in forest.get("edges") or []:
        parent = source_to_program.get(str(edge.get("parent_node_id") or ""))
        child = source_to_program.get(str(edge.get("child_node_id") or ""))
        if parent and child:
            edges.append(
                {
                    "parent_program_node_id": parent,
                    "child_program_node_id": child,
                    "edge_type": str(edge.get("edge_type") or "program_edge"),
                }
            )

    condition_entries = [
        dict(node["condition_entry"])
        for node in program_nodes
        if isinstance(node.get("condition_entry"), dict)
    ]
    condition_entries.sort(
        key=lambda row: (
            -float(row.get("condition_weight_default") or 0.0),
            str(row.get("review_status") or ""),
            str(row.get("condition_id") or ""),
            str(row.get("program_node_id") or ""),
        )
    )

    return {
        "schema_version": "aml_composable_pattern_program_v1_support_state_reviewed_draft",
        "runtime_policy": "reviewed support-state AML vocabulary draft; only accepted nodes should train as positive labels",
        "source_forest_schema": forest.get("schema_version"),
        "source_forest_summary": forest.get("summary") or {},
        "summary": {
            "program_node_count": len(program_nodes),
            "edge_count": len(edges),
            "condition_entry_count": len(condition_entries),
            "node_kind_counts": dict(sorted(Counter(str(node.get("program_node_kind") or "") for node in program_nodes).items())),
            "review_status_counts": dict(sorted(Counter(str(node.get("review_status") or "") for node in program_nodes).items())),
            "semantic_level_counts": dict(sorted(Counter(str(node.get("semantic_level") or "") for node in program_nodes).items())),
            "condition_scope_counts": dict(sorted(Counter(str(row.get("scope") or "") for row in condition_entries).items())),
            "positive_condition_count": sum(1 for row in condition_entries if float(row.get("condition_weight_default") or 0.0) > 0.0),
        },
        "nodes": sorted(program_nodes, key=lambda row: str(row.get("program_node_id") or "")),
        "edges": edges,
        "condition_vocabulary": condition_entries,
        "search_index": _build_search_index(program_nodes),
        "source_to_program_node_id": source_to_program,
        "program_node_index": {
            str(node.get("program_node_id")): idx
            for idx, node in enumerate(sorted(program_nodes, key=lambda row: str(row.get("program_node_id") or "")))
        },
    }


def compact_program(program: dict[str, Any]) -> dict[str, Any]:
    compact_nodes: list[dict[str, Any]] = []
    for node in program.get("nodes") or []:
        if node.get("program_node_kind") == "source_evidence":
            continue
        compact_nodes.append(
            {
                key: node.get(key)
                for key in [
                    "program_node_id",
                    "source_node_id",
                    "program_node_kind",
                    "display_name",
                    "motion_structure_label",
                    "review_status",
                    "review_decision",
                    "scope",
                    "semantic_level",
                    "edit_scope",
                    "composition_policy",
                    "match_signature",
                    "children",
                    "condition_entry",
                ]
                if key in node
            }
        )
    return {
        "schema_version": program.get("schema_version"),
        "runtime_policy": program.get("runtime_policy"),
        "source_forest_schema": program.get("source_forest_schema"),
        "source_forest_summary": program.get("source_forest_summary"),
        "summary": program.get("summary"),
        "nodes": compact_nodes,
        "edges": program.get("edges") or [],
        "condition_vocabulary": program.get("condition_vocabulary") or [],
        "search_index": program.get("search_index") or {},
    }


def write_tree(path: Path, program: dict[str, Any]) -> None:
    nodes = {str(node.get("program_node_id") or ""): node for node in program.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in program.get("edges") or []:
        child = nodes.get(str(edge.get("child_program_node_id") or ""))
        if child:
            children[str(edge.get("parent_program_node_id") or "")].append(child)

    summary = program.get("summary") or {}
    lines = [
        "# AML Composable Pattern Program v1 Support-State Reviewed Draft",
        "",
        f"nodes={summary.get('program_node_count')} edges={summary.get('edge_count')} "
        f"conditions={summary.get('condition_entry_count')} positives={summary.get('positive_condition_count')}",
        "",
    ]
    roots = [node for node in nodes.values() if node.get("program_node_kind") == "program_root"]
    for root in sorted(roots, key=lambda row: str(row.get("program_node_id") or "")):
        families = sorted(children.get(str(root.get("program_node_id")), []), key=lambda row: str(row.get("program_node_id") or ""))
        lines.append(f"- {root.get('display_name')} [{len(families)} families]")
        for family in families:
            condition = family.get("condition_entry") or {}
            aliases = ", ".join(f"{item.get('id')}:{item.get('count')}" for item in (condition.get("caption_name_candidates") or [])[:4])
            handles = ", ".join(str(item.get("name")) for item in condition.get("edit_handles") or [])
            lines.append(
                f"  - {family.get('program_node_id')} | status={family.get('review_status')} "
                f"decision={family.get('review_decision')} level={family.get('semantic_level')} "
                f"edit={family.get('edit_scope')} condition={condition.get('condition_id')} "
                f"weight={condition.get('condition_weight_default')} aliases={aliases or 'none'}"
            )
            lines.append(f"    handles: {handles}")
            note = str(family.get("review_notes") or "")
            if note:
                lines.append(f"    note: {note}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, program: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "aml_composable_pattern_program.json", program)
    _write_json(output_dir / "aml_composable_pattern_program_compact.json", compact_program(program))
    _write_json(output_dir / "aml_condition_vocabulary.json", program.get("condition_vocabulary") or [])
    _write_json(output_dir / "aml_composable_pattern_search_index.json", program.get("search_index") or {})
    _write_json(output_dir / "summary.json", program.get("summary") or {})
    write_tree(output_dir / "aml_composable_pattern_program_tree.txt", program)


def run_self_test() -> None:
    tiny = {
        "schema_version": "tiny_reviewed",
        "summary": {},
        "nodes": [
            {"node_id": "accepted_full_patterns", "node_kind": "root", "status": "root", "scope": "forest_root", "accepted_name": "accepted"},
            {
                "node_id": "family_test",
                "node_kind": "pattern_family_candidate",
                "status": "accepted",
                "scope": "full_pattern",
                "accepted_name": "cartwheel",
                "language_aliases": [{"id": "cartwheel", "count": 10}],
                "evidence": {"source_candidate_count": 1, "support_cases_max": 10, "support_cases_sum": 10},
                "motion_summary": {
                    "channels": [{"id": "whole_body_support"}, {"id": "acrobatics_or_inversion"}],
                    "zones": [{"id": "inversion"}],
                    "canonical_role_items": [{"id": "whole_body_support:inverted_support"}],
                    "geometry_clusters": [
                        {"id": "WHOLE_BODY_SUPPORT/WB_SUPPORT_INVERTED"},
                        {"id": "WHOLE_BODY_ACROBATICS/WB_CARTWHEEL_CANDIDATE"},
                    ],
                },
            },
        ],
        "edges": [{"parent_node_id": "accepted_full_patterns", "child_node_id": "family_test", "edge_type": "test"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        program = build_program(tiny)
        assert program["summary"]["condition_entry_count"] == 1
        assert program["summary"]["positive_condition_count"] == 1
        condition = program["condition_vocabulary"][0]
        assert condition["condition_id"] == "AMLV1_FAMILY_TEST"
        assert condition["motion_structure_label"] == "cartwheel"
        assert "inversion_support" in [item["name"] for item in condition["edit_handles"]]
        write_outputs(Path(tmp), program)
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-forest", type=Path, default=DEFAULT_REVIEWED_FOREST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    forest = _read_json(args.reviewed_forest)
    program = build_program(forest)
    program["source_forest_path"] = str(args.reviewed_forest)
    write_outputs(args.output_dir, program)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "summary": program["summary"]}, indent=2))


if __name__ == "__main__":
    main()
