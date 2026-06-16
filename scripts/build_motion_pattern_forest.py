from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_DRAFT = Path("outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json")
DEFAULT_SPLIT_PROPOSALS = Path("outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json")
DEFAULT_MANUAL_SPLIT_REVIEW = Path("outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_pattern_forest_v1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _node_kind(status: str) -> str:
    if status == "promoted_candidate":
        return "pattern_candidate"
    if status == "split_required":
        return "variation_parent"
    if status == "structural_component":
        return "component"
    if status == "diagnostic_only":
        return "diagnostic"
    if status == "split_child_promoted":
        return "pattern_variation_candidate"
    if status == "split_child_pending":
        return "pending_variation"
    return "review_required"


def _draft_node_to_forest_node(node: dict[str, Any]) -> dict[str, Any]:
    status = str(node.get("status") or "")
    return {
        "node_id": node.get("draft_node_id"),
        "source_node_id": node.get("source_motion_node_id"),
        "node_kind": _node_kind(status),
        "status": status,
        "structural_role": node.get("structural_role"),
        "motion_family_key": node.get("motion_family_key"),
        "support": node.get("support") or {},
        "motion_evidence": {
            "source_motif_ids": node.get("source_motif_ids") or [],
            "required_geometry_clusters": node.get("motion_evidence", {}).get("required_geometry_clusters") or [],
            "top_geometry_clusters": node.get("motion_evidence", {}).get("top_geometry_clusters") or [],
        },
        "naming": node.get("naming") or {},
        "runtime_policy": "offline forest node; not runtime AML",
    }


def _review_decisions(manual_review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("proposal_node_id") or ""): row for row in manual_review.get("decisions") or []}


def _proposal_children(split_proposals: dict[str, Any], manual_review: dict[str, Any]) -> list[dict[str, Any]]:
    review_map = _review_decisions(manual_review)
    out: list[dict[str, Any]] = []
    for proposal in split_proposals.get("split_proposals") or []:
        for child in proposal.get("split_child_candidates") or []:
            child_id = str(child.get("proposal_node_id") or "")
            review = review_map.get(child_id, {})
            decision = str(review.get("decision") or "review_pending")
            status = "split_child_promoted" if decision == "promote_candidate" else "split_child_pending"
            axis = child.get("structural_axis") or {}
            support = child.get("support") or {}
            out.append(
                {
                    "node_id": child_id,
                    "source_node_id": child_id,
                    "parent_node_id": child.get("parent_node_id"),
                    "node_kind": _node_kind(status),
                    "status": status,
                    "manual_decision": decision,
                    "structural_role": "variation_candidate" if status == "split_child_promoted" else "pending_variation",
                    "motion_family_key": " + ".join(child.get("motion_evidence", {}).get("required_geometry_clusters") or []),
                    "support": {
                        "support_cases": support.get("support_cases"),
                        "support_ratio": support.get("support_ratio"),
                        "parent_occurrence_count": support.get("parent_occurrence_count"),
                    },
                    "motion_evidence": {
                        "source_motif_ids": child.get("motion_evidence", {}).get("source_motif_ids") or [],
                        "required_geometry_clusters": child.get("motion_evidence", {}).get("required_geometry_clusters") or [],
                        "top_cooccurring_clusters": child.get("motion_evidence", {}).get("top_cooccurring_clusters") or [],
                    },
                    "structural_axis": axis,
                    "example_refs": child.get("example_refs") or [],
                    "runtime_policy": "offline split child; not runtime AML",
                }
            )
    return out


def _edges(draft: dict[str, Any], children: list[dict[str, Any]], *, include_pending_children: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for link in draft.get("component_links") or []:
        out.append(
            {
                "parent_node_id": link.get("parent_node_id"),
                "child_node_id": link.get("component_node_id"),
                "edge_type": "component",
                "confidence": link.get("geometry_jaccard"),
                "policy": link.get("policy"),
            }
        )
    for child in children:
        if child.get("status") == "split_child_pending" and not include_pending_children:
            continue
        out.append(
            {
                "parent_node_id": child.get("parent_node_id"),
                "child_node_id": child.get("node_id"),
                "edge_type": "variation",
                "confidence": child.get("support", {}).get("support_ratio"),
                "policy": "split-child edge accepted from manual split review seed" if child.get("status") == "split_child_promoted" else "pending split-child edge",
            }
        )
    return out


def _tree_stats(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    parent_by_child: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if not parent or not child:
            continue
        children_by_parent[parent].append(child)
        parent_by_child[child].append(parent)

    roots = sorted(node_id for node_id in nodes if node_id not in parent_by_child)
    depths: dict[str, int] = {}
    for root in roots:
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        seen: set[str] = set()
        while queue:
            node_id, depth = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            depths[node_id] = max(depths.get(node_id, 0), depth)
            for child in children_by_parent.get(node_id, []):
                queue.append((child, depth + 1))

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "root_count": len(roots),
        "roots": roots,
        "max_depth": max(depths.values(), default=0),
        "node_kind_counts": dict(sorted(Counter(str(node.get("node_kind") or "") for node in nodes.values()).items())),
        "status_counts": dict(sorted(Counter(str(node.get("status") or "") for node in nodes.values()).items())),
        "edge_type_counts": dict(sorted(Counter(str(edge.get("edge_type") or "") for edge in edges).items())),
    }


def build_forest(
    draft: dict[str, Any],
    split_proposals: dict[str, Any],
    manual_review: dict[str, Any],
    *,
    include_pending_children: bool,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    for draft_node in draft.get("draft_nodes") or []:
        node = _draft_node_to_forest_node(draft_node)
        nodes[str(node["node_id"])] = node

    split_children = _proposal_children(split_proposals, manual_review)
    for child in split_children:
        if child.get("status") == "split_child_pending" and not include_pending_children:
            continue
        nodes[str(child["node_id"])] = child

    edges = _edges(draft, split_children, include_pending_children=include_pending_children)
    edges = [edge for edge in edges if str(edge.get("parent_node_id") or "") in nodes and str(edge.get("child_node_id") or "") in nodes]
    stats = _tree_stats(nodes, edges)
    return {
        "schema_version": "motion_pattern_forest_v1",
        "runtime_policy": "offline forest draft only; not the runtime AML tree",
        "forest_policy": "multiple roots are allowed; action patterns are represented as a forest of components, candidates, and variations",
        "summary": {
            **stats,
            "include_pending_children": include_pending_children,
        },
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": sorted(edges, key=lambda item: (str(item.get("parent_node_id")), str(item.get("edge_type")), str(item.get("child_node_id")))),
    }


def _children_by_parent(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        out[str(edge.get("parent_node_id") or "")].append(edge)
    for rows in out.values():
        rows.sort(key=lambda item: (str(item.get("edge_type") or ""), str(item.get("child_node_id") or "")))
    return out


def _node_label(node: dict[str, Any]) -> str:
    clusters = node.get("motion_evidence", {}).get("required_geometry_clusters") or []
    compact = " + ".join(str(item).split("/", 1)[-1] for item in clusters[:3])
    if len(clusters) > 3:
        compact += " + ..."
    support = node.get("support") or {}
    support_text = support.get("support_cases") or support.get("candidate_support_cases") or ""
    return f"{node.get('node_id')} [{node.get('node_kind')}] support={support_text} :: {compact}"


def write_tree_view(path: Path, forest: dict[str, Any]) -> None:
    nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children_by_parent = _children_by_parent(forest.get("edges") or [])
    roots = forest.get("summary", {}).get("roots") or []
    lines: list[str] = []
    for root in roots:
        stack: list[tuple[str, int, str]] = [(root, 0, "root")]
        seen: set[str] = set()
        while stack:
            node_id, depth, edge_type = stack.pop()
            node = nodes.get(node_id)
            if not node:
                continue
            prefix = "  " * depth
            edge_prefix = "" if depth == 0 else f"({edge_type}) "
            lines.append(f"{prefix}- {edge_prefix}{_node_label(node)}")
            if node_id in seen:
                lines.append(f"{prefix}  - cycle_or_shared_reference_skipped")
                continue
            seen.add(node_id)
            child_edges = children_by_parent.get(node_id, [])
            for edge in reversed(child_edges):
                stack.append((str(edge.get("child_node_id") or ""), depth + 1, str(edge.get("edge_type") or "")))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any]) -> None:
    nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children_by_parent = _children_by_parent(forest.get("edges") or [])
    lines: list[str] = []
    lines.append("# Motion Pattern Forest V1")
    lines.append("")
    lines.append("This is an offline forest draft. Multiple roots are expected.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (forest.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Roots")
    for root in forest.get("summary", {}).get("roots") or []:
        node = nodes.get(str(root), {})
        lines.append("")
        lines.append(f"### {root}")
        lines.append("")
        lines.append(f"- kind: `{node.get('node_kind')}`")
        lines.append(f"- status: `{node.get('status')}`")
        lines.append(f"- role: `{node.get('structural_role')}`")
        lines.append(f"- geometry: `{', '.join(node.get('motion_evidence', {}).get('required_geometry_clusters') or [])}`")
        child_edges = children_by_parent.get(str(root), [])
        if child_edges:
            lines.append("")
            lines.append("| edge | child | child kind | support | geometry |")
            lines.append("| --- | --- | --- | ---: | --- |")
            for edge in child_edges:
                child = nodes.get(str(edge.get("child_node_id") or ""), {})
                support = child.get("support", {}).get("support_cases") or child.get("support", {}).get("candidate_support_cases") or ""
                geometry = "<br>".join(child.get("motion_evidence", {}).get("required_geometry_clusters") or [])
                lines.append(
                    f"| `{edge.get('edge_type')}` | `{child.get('node_id')}` | `{child.get('node_kind')}` | {support} | {geometry} |"
                )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline motion pattern forest from reviewed candidates and split children.")
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--split-proposals", default=str(DEFAULT_SPLIT_PROPOSALS))
    parser.add_argument("--manual-split-review", default=str(DEFAULT_MANUAL_SPLIT_REVIEW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-pending-children", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forest = build_forest(
        _read_json(Path(args.draft)),
        _read_json(Path(args.split_proposals)),
        _read_json(Path(args.manual_split_review)),
        include_pending_children=bool(args.include_pending_children),
    )
    forest["source"] = {
        "draft": str(args.draft),
        "split_proposals": str(args.split_proposals),
        "manual_split_review": str(args.manual_split_review),
    }
    _write_json(output_dir / "motion_pattern_forest.json", forest)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_pattern_forest_summary_v1",
            **forest["summary"],
            "source": forest["source"],
        },
    )
    write_tree_view(output_dir / "motion_pattern_forest_tree.txt", forest)
    write_report(output_dir / "motion_pattern_forest.md", forest)
    print(output_dir)


if __name__ == "__main__":
    main()
