"""Build a dense AML motion pattern candidate forest.

This is the broad candidate view. It preserves global Motion-BPE candidates
instead of only the reviewed/promoted nodes in `aml_pattern_forest_v0`.

Example:
    python scripts/build_aml_pattern_forest_dense_v0.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/motion_pattern_forest_candidates.json")
DEFAULT_REVIEWED = Path("outputs/aml_regression_testset_v2/aml_pattern_forest_v0/aml_pattern_forest_compact.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_forest_candidates_v0_dense")


ROOTS = [
    ("dense_coordination_candidates", "coordination_candidates"),
    ("dense_channel_sequence_candidates", "channel_sequence_candidates"),
    ("dense_component_candidates", "component_candidates"),
    ("dense_named_candidates", "named_candidates"),
    ("dense_diagnostic_candidates", "diagnostic_candidates"),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_id(text: str) -> str:
    out = []
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


def _top_aliases(node: dict[str, Any]) -> list[dict[str, Any]]:
    naming = node.get("naming_diagnostics") or {}
    if naming.get("top_caption_aliases"):
        return naming.get("top_caption_aliases") or []
    alias = naming.get("top_caption_alias")
    if alias:
        return [{"id": alias, "count": int((node.get("support_cases") or node.get("support_cases_sum") or 0) or 0)}]
    return []


def _top_alias(node: dict[str, Any]) -> str:
    aliases = _top_aliases(node)
    return str(aliases[0].get("id") or "") if aliases else ""


def _motion_definition(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("node_kind") == "motif_family":
        return node.get("motion_definition") or {}
    return {
        "operator": node.get("operator"),
        "required_channels": [item.get("id") for item in node.get("channels") or [] if item.get("id")],
        "required_relation_types": [item.get("id") for item in node.get("relation_profile") or [] if item.get("id")],
        "required_geometry_clusters": [item.get("id") for item in node.get("top_geometry_clusters") or [] if item.get("id")],
        "top_channels": node.get("channels") or [],
        "top_relation_types": node.get("relation_profile") or [],
        "top_geometry_clusters": node.get("top_geometry_clusters") or [],
    }


def _support(node: dict[str, Any]) -> int:
    return int((node.get("support_cases_sum") or node.get("support_cases") or 0) or 0)


def _family_status(node: dict[str, Any]) -> tuple[str, str, str]:
    motion = _motion_definition(node)
    operator = str(motion.get("operator") or node.get("operator") or "")
    channels = motion.get("required_channels") or []
    relations = motion.get("required_relation_types") or []
    alias = _top_alias(node)
    support = _support(node)
    if node.get("status") == "diagnostic_family" or support < 80:
        return "diagnostic_candidate", "diagnostic_candidate", "dense_diagnostic_candidates"
    if alias:
        return "named_candidate", "candidate_with_caption_name", "dense_named_candidates"
    if operator == "COORDINATION_MERGE" or len(channels) >= 2 or relations:
        return "coordination_candidate", "multi_channel_coordination_candidate", "dense_coordination_candidates"
    if operator == "SEQ_CHANNEL_MERGE":
        if len(channels) == 1:
            return "component_candidate", "single_channel_component_candidate", "dense_component_candidates"
        return "sequence_candidate", "channel_sequence_candidate", "dense_channel_sequence_candidates"
    return "diagnostic_candidate", "diagnostic_candidate", "dense_diagnostic_candidates"


def _reviewed_index(reviewed: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in reviewed.get("nodes") or []:
        if node.get("node_kind") != "pattern_node":
            continue
        status = str(node.get("status") or "")
        for item in node.get("source_symbols") or []:
            symbol = str(item.get("symbol") or "")
            if symbol:
                out[symbol] = status
    return out


def build_dense(
    source: dict[str, Any],
    reviewed: dict[str, Any],
    *,
    source_path: Path = DEFAULT_SOURCE,
    reviewed_path: Path = DEFAULT_REVIEWED,
) -> dict[str, Any]:
    reviewed_by_symbol = _reviewed_index(reviewed)
    source_nodes = source.get("nodes") or []
    source_edges = source.get("edges") or []
    source_node_by_id = {str(node.get("node_id") or ""): node for node in source_nodes}

    root_nodes = [
        {
            "node_id": root_id,
            "node_kind": "dense_root",
            "status": "root",
            "scope": "candidate_group",
            "display_name": name,
        }
        for root_id, name in ROOTS
    ]

    nodes: list[dict[str, Any]] = list(root_nodes)
    edges: list[dict[str, Any]] = []
    family_parent: dict[str, str] = {}

    for node in source_nodes:
        if node.get("node_kind") != "motif_family":
            continue
        status, scope, root_id = _family_status(node)
        motion = _motion_definition(node)
        aliases = _top_aliases(node)
        new_id = "dense_" + str(node.get("node_id"))
        family_parent[str(node.get("node_id"))] = new_id
        nodes.append(
            {
                "node_id": new_id,
                "source_node_id": node.get("node_id"),
                "node_kind": "dense_motif_family",
                "status": status,
                "scope": scope,
                "display_name": _top_alias(node) or str(node.get("motion_family_key") or ""),
                "support_cases_sum": node.get("support_cases_sum"),
                "occurrences_sum": node.get("occurrences_sum"),
                "motion_definition": motion,
                "naming_diagnostics": {
                    "top_caption_aliases": aliases,
                    "policy": "diagnostic naming only",
                },
            }
        )
        edges.append({"parent_node_id": root_id, "child_node_id": new_id, "edge_type": "dense_root_member"})

    for edge in source_edges:
        parent_source = str(edge.get("parent_node_id") or "")
        child_source = str(edge.get("child_node_id") or "")
        parent = family_parent.get(parent_source)
        source_child = source_node_by_id.get(child_source)
        if not parent or not source_child:
            continue
        motif_id = str(source_child.get("motif_id") or "")
        reviewed_status = reviewed_by_symbol.get(motif_id, "")
        status = "reviewed_" + reviewed_status if reviewed_status else str(source_child.get("status") or "candidate")
        new_child_id = "dense_" + child_source
        nodes.append(
            {
                "node_id": new_child_id,
                "source_node_id": source_child.get("node_id"),
                "node_kind": "dense_motif_leaf",
                "status": status,
                "scope": "motif_leaf",
                "motif_id": motif_id,
                "operator": source_child.get("operator"),
                "support_cases": source_child.get("support_cases"),
                "occurrences": source_child.get("occurrences"),
                "motion_definition": _motion_definition(source_child),
                "naming_diagnostics": source_child.get("naming_diagnostics") or {},
            }
        )
        edges.append({"parent_node_id": parent, "child_node_id": new_child_id, "edge_type": "dense_family_member"})

    # Deduplicate leaves defensively.
    deduped = []
    seen = set()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if node_id in seen:
            continue
        deduped.append(node)
        seen.add(node_id)

    status_counts = Counter(str(node.get("status") or "") for node in deduped)
    kind_counts = Counter(str(node.get("node_kind") or "") for node in deduped)
    scope_counts = Counter(str(node.get("scope") or "") for node in deduped)
    return {
        "schema_version": "aml_pattern_forest_candidates_v0_dense",
        "runtime_policy": "dense offline candidate forest; not the reviewed runtime vocabulary",
        "source_candidate_forest": str(source_path),
        "source_reviewed_forest": str(reviewed_path),
        "summary": {
            "node_count": len(deduped),
            "edge_count": len(edges),
            "root_count": len(ROOTS),
            "family_count": sum(1 for node in deduped if node.get("node_kind") == "dense_motif_family"),
            "leaf_count": sum(1 for node in deduped if node.get("node_kind") == "dense_motif_leaf"),
            "status_counts": dict(sorted(status_counts.items())),
            "kind_counts": dict(sorted(kind_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
        },
        "nodes": deduped,
        "edges": edges,
    }


def write_tree(path: Path, forest: dict[str, Any], max_children: int) -> None:
    node_by_id = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = {}
    child_ids = set()
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child in node_by_id:
            children.setdefault(parent, []).append(node_by_id[child])
            child_ids.add(child)
    roots = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "dense_root"]
    lines = ["# Dense AML Pattern Candidate Forest v0", ""]
    summary = forest.get("summary") or {}
    lines.append(
        f"nodes={summary.get('node_count')} families={summary.get('family_count')} "
        f"leaves={summary.get('leaf_count')} edges={summary.get('edge_count')}"
    )
    lines.append("")
    for root in roots:
        root_children = sorted(children.get(str(root.get("node_id")), []), key=lambda n: -_support(n))
        lines.append(f"- {root.get('display_name')} [{len(root_children)} families]")
        for family in root_children[:max_children]:
            motion = family.get("motion_definition") or {}
            aliases = family.get("naming_diagnostics", {}).get("top_caption_aliases") or []
            alias = aliases[0]["id"] if aliases else ""
            fam_children = sorted(children.get(str(family.get("node_id")), []), key=lambda n: -_support(n))
            lines.append(
                f"  - {family.get('source_node_id')} [{family.get('status')}] support={family.get('support_cases_sum')} "
                f"motifs={len(fam_children)} op={motion.get('operator')} ch={motion.get('required_channels')} "
                f"geo={motion.get('required_geometry_clusters')} alias={alias}"
            )
            for leaf in fam_children[: min(3, max_children)]:
                lines.append(
                    f"    - {leaf.get('motif_id')} [{leaf.get('status')}] support={leaf.get('support_cases')} "
                    f"alias={(leaf.get('naming_diagnostics') or {}).get('top_caption_alias') or ''}"
                )
        if len(root_children) > max_children:
            lines.append(f"  - ... {len(root_children) - max_children} more families")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any]) -> None:
    lines = ["# Dense AML Pattern Candidate Forest v0", ""]
    lines.append("This is the broad candidate pool from full-HML3D Motion-BPE. It is not the reviewed AML vocabulary.")
    lines.append("")
    summary = forest.get("summary") or {}
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## How To Use")
    lines.append("")
    lines.append("- `coordination_candidate`: inspect first for full-pattern promotion.")
    lines.append("- `candidate_with_caption_name`: useful for naming, but caption name is not enough for promotion.")
    lines.append("- `single_channel_component_candidate`: usually component library, not full action.")
    lines.append("- `diagnostic_candidate`: keep as evidence, do not promote without stronger support.")
    lines.append("- leaves marked `reviewed_*` are already connected to the conservative reviewed v0 tree.")
    lines.append("")
    lines.append("## Top Families")
    lines.append("")
    families = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "dense_motif_family"]
    for family in sorted(families, key=lambda n: -_support(n))[:40]:
        motion = family.get("motion_definition") or {}
        aliases = family.get("naming_diagnostics", {}).get("top_caption_aliases") or []
        alias_text = ", ".join(f"{a['id']}:{a['count']}" for a in aliases[:4])
        lines.append(f"### {family.get('source_node_id')}")
        lines.append("")
        lines.append(f"- status: `{family.get('status')}`")
        lines.append(f"- scope: `{family.get('scope')}`")
        lines.append(f"- support: `{family.get('support_cases_sum')}`")
        lines.append(f"- operator: `{motion.get('operator')}`")
        lines.append(f"- channels: `{motion.get('required_channels')}`")
        lines.append(f"- geometry: `{motion.get('required_geometry_clusters')}`")
        lines.append(f"- aliases: {alias_text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source)
    reviewed_path = Path(args.reviewed)
    source = _read_json(source_path)
    reviewed = _read_json(reviewed_path) if reviewed_path.exists() else {"nodes": []}
    forest = build_dense(source, reviewed, source_path=source_path, reviewed_path=reviewed_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "aml_pattern_forest_candidates_dense.json", forest)
    _write_json(output_dir / "summary.json", forest.get("summary") or {})
    write_tree(output_dir / "aml_pattern_forest_candidates_dense_tree.txt", forest, int(args.max_tree_children))
    write_report(output_dir / "aml_pattern_forest_candidates_dense_review.md", forest)
    return forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tree-children", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    forest = run(parse_args())
    print(json.dumps(forest.get("summary") or {}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
