from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = Path("outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/coordination_pattern_forest_loose_v1")

STATUS_RANK = {
    "promote_named_coordination_candidate": 0,
    "review_structural_coordination_candidate": 1,
    "review_named_low_support_candidate": 2,
    "diagnostic_coordination_motif": 3,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
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
    compact = "".join(out).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "unnamed"


def _family_key(candidate: dict[str, Any]) -> str:
    naming = candidate.get("naming_diagnostics") or {}
    motion = candidate.get("motion_definition") or {}
    alias = str(naming.get("top_caption_alias") or "")
    status = str(candidate.get("status") or "")
    if status == "promote_named_coordination_candidate" and alias:
        return "named:" + alias
    channels = "+".join(motion.get("required_channels") or []) or "unknown_channels"
    geometry = "+".join(motion.get("required_geometry_clusters") or []) or "unknown_geometry"
    return "structural:" + channels + "|" + geometry


def _node_status(candidates: list[dict[str, Any]]) -> str:
    return min((str(item.get("status") or "diagnostic_coordination_motif") for item in candidates), key=lambda x: STATUS_RANK.get(x, 99))


def _top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _candidate_leaf(candidate: dict[str, Any]) -> dict[str, Any]:
    motion = candidate.get("motion_definition") or {}
    naming = candidate.get("naming_diagnostics") or {}
    support = candidate.get("support") or {}
    return {
        "node_id": "coord_leaf_" + _safe_id(str(candidate.get("source_motif_id") or candidate.get("candidate_id") or "")),
        "node_kind": "coordination_motif_leaf",
        "status": candidate.get("status"),
        "candidate_id": candidate.get("candidate_id"),
        "source_motif_id": candidate.get("source_motif_id"),
        "support": support,
        "motion_definition": motion,
        "naming_diagnostics": naming,
        "review_examples": candidate.get("review_examples") or [],
    }


def build_forest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") or []
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(_family_key(candidate), []).append(candidate)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for idx, (key, rows) in enumerate(
        sorted(groups.items(), key=lambda item: (STATUS_RANK.get(_node_status(item[1]), 99), -sum(int((r.get("support") or {}).get("support_cases") or 0) for r in item[1]), item[0])),
        start=1,
    ):
        alias_counter: Counter[str] = Counter()
        channel_counter: Counter[str] = Counter()
        geometry_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        support_sum = 0
        occurrence_sum = 0
        for row in rows:
            status_counter[str(row.get("status") or "")] += 1
            support_sum += int((row.get("support") or {}).get("support_cases") or 0)
            occurrence_sum += int((row.get("support") or {}).get("occurrences") or 0)
            naming = row.get("naming_diagnostics") or {}
            alias = str(naming.get("top_caption_alias") or "")
            if alias:
                alias_counter[alias] += int((row.get("support") or {}).get("support_cases") or 0)
            motion = row.get("motion_definition") or {}
            channel_counter.update(str(item) for item in motion.get("required_channels") or [])
            geometry_counter.update(str(item) for item in motion.get("required_geometry_clusters") or [])
        family_node_id = f"coord_family_{idx:04d}"
        top_alias = alias_counter.most_common(1)[0][0] if alias_counter else ""
        node_kind = "named_coordination_family" if key.startswith("named:") else "structural_coordination_family"
        nodes.append(
            {
                "node_id": family_node_id,
                "node_kind": node_kind,
                "status": _node_status(rows),
                "family_key": key,
                "display_name": top_alias or key.replace("structural:", ""),
                "support": {
                    "candidate_count": len(rows),
                    "support_cases_sum": support_sum,
                    "occurrences_sum": occurrence_sum,
                },
                "motion_definition": {
                    "required_channels": sorted(channel_counter),
                    "required_geometry_clusters": sorted(geometry_counter),
                    "top_channels": _top_counter(channel_counter),
                    "top_geometry_clusters": _top_counter(geometry_counter),
                },
                "naming_diagnostics": {
                    "top_caption_aliases": _top_counter(alias_counter),
                    "policy": "diagnostic only; named families are still motion-derived coordination nodes",
                },
                "status_counts": dict(sorted(status_counter.items())),
            }
        )
        for row in rows:
            leaf = _candidate_leaf(row)
            nodes.append(leaf)
            edges.append(
                {
                    "parent_node_id": family_node_id,
                    "child_node_id": leaf["node_id"],
                    "edge_type": "coordination_family_member",
                    "policy": "offline candidate forest edge from promotion queue grouping",
                }
            )

    status_counts = Counter(str(node.get("status") or "") for node in nodes)
    family_status_counts = Counter(
        str(node.get("status") or "")
        for node in nodes
        if str(node.get("node_kind") or "").endswith("coordination_family")
    )
    leaf_status_counts = Counter(
        str(node.get("status") or "")
        for node in nodes
        if str(node.get("node_kind") or "") == "coordination_motif_leaf"
    )
    return {
        "schema_version": "coordination_pattern_forest_v1",
        "source_candidates": payload.get("source_motif_audit"),
        "runtime_policy": "offline review forest only; not the AML runtime tree",
        "summary": {
            "family_count": len(groups),
            "leaf_count": len(candidates),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "status_counts": dict(sorted(status_counts.items())),
            "family_status_counts": dict(sorted(family_status_counts.items())),
            "leaf_status_counts": dict(sorted(leaf_status_counts.items())),
        },
        "nodes": nodes,
        "edges": edges,
    }


def write_tree(path: Path, forest: dict[str, Any]) -> None:
    children: dict[str, list[dict[str, Any]]] = {}
    node_by_id = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    child_ids = set()
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child:
            children.setdefault(parent, []).append(node_by_id[child])
            child_ids.add(child)
    roots = [node for node in forest.get("nodes") or [] if str(node.get("node_id") or "") not in child_ids]
    roots.sort(key=lambda node: (STATUS_RANK.get(str(node.get("status") or ""), 99), -int((node.get("support") or {}).get("support_cases_sum") or 0), str(node.get("node_id") or "")))
    lines = ["# Coordination Pattern Forest", ""]
    summary = forest.get("summary") or {}
    lines.append(f"families: {summary.get('family_count')}  leaves: {summary.get('leaf_count')}  nodes: {summary.get('node_count')}  edges: {summary.get('edge_count')}")
    lines.append("")
    for root in roots:
        support = root.get("support") or {}
        lines.append(f"- {root.get('node_id')} [{root.get('status')}] {root.get('display_name')} support={support.get('support_cases_sum')}")
        for child in sorted(children.get(str(root.get("node_id")), []), key=lambda node: -int((node.get("support") or {}).get("support_cases") or 0)):
            naming = child.get("naming_diagnostics") or {}
            csupport = child.get("support") or {}
            lines.append(f"  - {child.get('source_motif_id')} [{child.get('status')}] support={csupport.get('support_cases')} alias={naming.get('top_caption_alias') or ''} purity={naming.get('caption_alias_purity')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any]) -> None:
    nodes = forest.get("nodes") or []
    node_by_id = {str(node.get("node_id") or ""): node for node in nodes}
    children: dict[str, list[dict[str, Any]]] = {}
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child and child in node_by_id:
            children.setdefault(parent, []).append(node_by_id[child])
    family_nodes = [node for node in nodes if str(node.get("node_kind") or "").endswith("coordination_family")]
    lines = ["# Coordination Pattern Forest Review", "", "Offline review forest only; not the AML runtime tree.", ""]
    summary = forest.get("summary") or {}
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    for family in family_nodes:
        support = family.get("support") or {}
        naming = family.get("naming_diagnostics") or {}
        motion = family.get("motion_definition") or {}
        lines.append(f"## {family.get('node_id')} `{family.get('display_name')}`")
        lines.append("")
        lines.append(f"- status: `{family.get('status')}`")
        lines.append(f"- kind: `{family.get('node_kind')}`")
        lines.append(f"- support cases sum: `{support.get('support_cases_sum')}`")
        lines.append(f"- required channels: `{motion.get('required_channels')}`")
        lines.append(f"- required geometry: `{motion.get('required_geometry_clusters')}`")
        aliases = ', '.join(f"{row['id']}:{row['count']}" for row in naming.get("top_caption_aliases", [])[:8])
        lines.append(f"- aliases: {aliases}")
        lines.append("")
        for child in sorted(children.get(str(family.get("node_id")), []), key=lambda node: -int((node.get("support") or {}).get("support_cases") or 0))[:5]:
            child_support = child.get("support") or {}
            child_naming = child.get("naming_diagnostics") or {}
            lines.append(f"### {child.get('source_motif_id')}")
            lines.append("")
            lines.append(f"- status: `{child.get('status')}`")
            lines.append(f"- support cases: `{child_support.get('support_cases')}`")
            lines.append(f"- top alias: `{child_naming.get('top_caption_alias') or ''}`")
            lines.append(f"- alias purity: `{child_naming.get('caption_alias_purity')}`")
            lines.append("- examples:")
            for example in (child.get("review_examples") or [])[:5]:
                caption = str(example.get("caption") or "").replace("\n", " ")
                lines.append(f"  - `{example.get('case_id')}` span `{example.get('span')}`: {caption}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline coordination pattern forest from promotion candidates.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forest = build_forest(_read_json(Path(args.candidates)))
    _write_json(output_dir / "coordination_pattern_forest.json", forest)
    write_tree(output_dir / "coordination_pattern_forest_tree.txt", forest)
    write_report(output_dir / "coordination_pattern_forest_review.md", forest)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "coordination_pattern_forest_summary_v1",
            "source_candidates": str(args.candidates),
            **(forest.get("summary") or {}),
            "outputs": {
                "forest": str(output_dir / "coordination_pattern_forest.json"),
                "tree": str(output_dir / "coordination_pattern_forest_tree.txt"),
                "review": str(output_dir / "coordination_pattern_forest_review.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    print(output_dir / "summary.json")
    print(output_dir / "coordination_pattern_forest_tree.txt")


if __name__ == "__main__":
    main()
