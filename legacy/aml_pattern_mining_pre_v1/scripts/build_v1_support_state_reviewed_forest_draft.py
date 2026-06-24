"""Build a reviewed draft forest from support-state review decisions.

This is still an offline artifact. It converts editable review decisions into
a cleaner tree shape:

    accepted full patterns
    reusable components
    split-required candidates
    closure-required candidates

It does not change runtime AML matching logic. Promotion is only as reliable as
the reviewed decision JSON used as input.

Typical use:
    python scripts/build_v1_support_state_reviewed_forest_draft.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_FOREST = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_draft/"
    "aml_pattern_forest_v1_draft.json"
)
DEFAULT_DECISIONS = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_review_decisions_draft/"
    "review_decisions_draft.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_reviewed_draft"
)


ROOTS = [
    {
        "node_id": "accepted_full_patterns",
        "accepted_name": "accepted full motion patterns",
        "description": "Reviewed motion structures that may enter AML as full pattern vocabulary candidates.",
    },
    {
        "node_id": "reusable_components",
        "accepted_name": "reusable motion components",
        "description": "Motion structures that are useful as components but should not be exposed as complete action names.",
    },
    {
        "node_id": "split_required",
        "accepted_name": "split-required pattern candidates",
        "description": "Candidates with useful evidence that need a cleaner structural split before promotion.",
    },
    {
        "node_id": "closure_required",
        "accepted_name": "closure-required pattern candidates",
        "description": "Near-pattern candidates that need one or more stable roles to close the structure.",
    },
]


DECISION_TO_ROOT = {
    "promote": "accepted_full_patterns",
    "downgrade_to_component": "reusable_components",
    "split": "split_required",
    "split_axis_confirmed": "split_required",
    "needs_closure": "closure_required",
    "merge_with_review_candidate": "closure_required",
}

DECISION_TO_STATUS = {
    "promote": "accepted",
    "downgrade_to_component": "component",
    "split": "split_required",
    "split_axis_confirmed": "split_required",
    "needs_closure": "pending_closure",
    "merge_with_review_candidate": "pending_closure",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _node_map(forest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}


def _child_ids(forest: dict[str, Any]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child:
            children[parent].append(child)
    return children


def _family_decisions(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("decisions") or []:
        family_id = str(item.get("family_id") or "")
        if family_id:
            out[family_id] = item
    return out


def _copy_family_node(family: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    node = dict(family)
    draft_decision = str(decision.get("decision") or "pending")
    node["review_decision"] = draft_decision
    node["status"] = DECISION_TO_STATUS.get(draft_decision, "review_pending")
    node["review_notes"] = str(decision.get("notes") or "")
    node["review_image_path"] = str(decision.get("image_path") or "")
    node["review_example_case_ids"] = list(decision.get("example_case_ids") or [])
    if draft_decision == "promote":
        node["scope"] = "full_pattern"
    elif draft_decision == "downgrade_to_component":
        node["scope"] = "component"
    return node


def _copy_source_node(source: dict[str, Any]) -> dict[str, Any]:
    node = dict(source)
    node["status"] = f"source_{source.get('status') or 'review'}"
    return node


def build_reviewed_forest(source_forest: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
    nodes_by_id = _node_map(source_forest)
    children = _child_ids(source_forest)
    decision_by_family = _family_decisions(decisions)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for root in ROOTS:
        nodes.append(
            {
                "node_id": root["node_id"],
                "node_kind": "root",
                "status": "root",
                "scope": "forest_root",
                "accepted_name": root["accepted_name"],
                "description": root["description"],
            }
        )

    for family_id, decision in sorted(decision_by_family.items()):
        family = nodes_by_id.get(family_id)
        if not family:
            continue
        draft_decision = str(decision.get("decision") or "pending")
        root_id = DECISION_TO_ROOT.get(draft_decision)
        if not root_id:
            continue

        family_node = _copy_family_node(family, decision)
        nodes.append(family_node)
        edges.append(
            {
                "parent_node_id": root_id,
                "child_node_id": family_node["node_id"],
                "edge_type": "review_root_to_family",
            }
        )

        for child_id in children.get(family_id, []):
            child = nodes_by_id.get(child_id)
            if not child:
                continue
            child_node = _copy_source_node(child)
            nodes.append(child_node)
            edges.append(
                {
                    "parent_node_id": family_node["node_id"],
                    "child_node_id": child_node["node_id"],
                    "edge_type": "family_to_source_candidate",
                }
            )

    summary = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "root_count": sum(1 for node in nodes if node.get("node_kind") == "root"),
        "family_count": sum(1 for node in nodes if node.get("node_kind") == "pattern_family_candidate"),
        "source_candidate_node_count": sum(1 for node in nodes if node.get("node_kind") == "source_closure_candidate"),
        "status_counts": dict(sorted(Counter(str(node.get("status") or "") for node in nodes).items())),
        "scope_counts": dict(sorted(Counter(str(node.get("scope") or "") for node in nodes).items())),
        "decision_counts": dict(sorted(Counter(str(node.get("review_decision") or "") for node in nodes if node.get("review_decision")).items())),
    }
    return {
        "schema_version": "aml_pattern_forest_v1_support_state_reviewed_draft",
        "runtime_policy": "offline reviewed draft only; do not use as final runtime logic without user acceptance",
        "source": {
            "source_forest_schema": source_forest.get("schema_version"),
            "source_decision_schema": decisions.get("schema_version"),
            "source_forest": str(DEFAULT_SOURCE_FOREST),
            "source_decisions": str(DEFAULT_DECISIONS),
        },
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
    }


def compact_forest(forest: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node in forest.get("nodes") or []:
        compact = {
            key: node.get(key)
            for key in [
                "node_id",
                "node_kind",
                "status",
                "scope",
                "accepted_name",
                "review_decision",
                "review_notes",
                "language_aliases",
                "evidence",
                "motion_summary",
            ]
            if key in node
        }
        if node.get("node_kind") == "source_closure_candidate":
            compact["source_example_ids"] = [row.get("case_id") for row in node.get("source_examples") or []]
        nodes.append(compact)
    return {
        "schema_version": f"{forest.get('schema_version')}_compact",
        "runtime_policy": forest.get("runtime_policy"),
        "source": forest.get("source"),
        "summary": forest.get("summary"),
        "nodes": nodes,
        "edges": forest.get("edges") or [],
    }


def _children(forest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_id = _node_map(forest)
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        child = by_id.get(str(edge.get("child_node_id") or ""))
        if child:
            children[str(edge.get("parent_node_id") or "")].append(child)
    return children


def write_tree(path: Path, forest: dict[str, Any]) -> None:
    children = _children(forest)
    summary = forest.get("summary") or {}
    lines = [
        "# AML Pattern Forest v1 Support-State Reviewed Draft",
        "",
        "nodes={node_count} roots={root_count} families={family_count} source_candidates={source_candidate_node_count} edges={edge_count}".format(
            **summary
        ),
        "",
    ]
    roots = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "root"]
    for root in roots:
        families = sorted(children.get(str(root.get("node_id")), []), key=lambda item: str(item.get("node_id")))
        lines.append(f"- {root.get('node_id')} [root] {root.get('accepted_name')} ({len(families)} families)")
        for family in families:
            evidence = family.get("evidence") or {}
            aliases = ", ".join(f"{item.get('id')}:{item.get('count')}" for item in (family.get("language_aliases") or [])[:4])
            lines.append(
                "  - {node_id} [{status}] {name} decision={decision} scope={scope} variants={variants} support_max={support} aliases={aliases}".format(
                    node_id=family.get("node_id"),
                    status=family.get("status"),
                    name=family.get("accepted_name"),
                    decision=family.get("review_decision"),
                    scope=family.get("scope"),
                    variants=evidence.get("source_candidate_count"),
                    support=evidence.get("support_cases_max"),
                    aliases=aliases or "none",
                )
            )
            note = str(family.get("review_notes") or "")
            if note:
                lines.append(f"    note: {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any]) -> None:
    lines = [
        "# AML Pattern Forest v1 Support-State Reviewed Draft",
        "",
        "This is the reviewed draft produced from the editable decision JSON.",
        "",
        "## Summary",
        "",
    ]
    for key, value in (forest.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Families", ""])

    children = _children(forest)
    for root in [node for node in forest.get("nodes") or [] if node.get("node_kind") == "root"]:
        families = sorted(children.get(str(root.get("node_id")), []), key=lambda item: str(item.get("node_id")))
        if not families:
            continue
        lines.extend([f"### {root.get('accepted_name')}", ""])
        for family in families:
            evidence = family.get("evidence") or {}
            roles = ", ".join(
                str(item.get("id") if isinstance(item, dict) else item)
                for item in (family.get("motion_summary") or {}).get("canonical_role_items") or []
            )
            lines.extend(
                [
                    f"- `{family.get('node_id')}`",
                    f"  - status: `{family.get('status')}`; decision: `{family.get('review_decision')}`; scope: `{family.get('scope')}`",
                    f"  - name: `{family.get('accepted_name')}`",
                    f"  - support max: `{evidence.get('support_cases_max')}`; source candidates: `{evidence.get('source_candidate_count')}`",
                    f"  - roles: `{roles}`",
                    f"  - note: {family.get('review_notes')}",
                    f"  - review image: `{family.get('review_image_path')}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-forest", type=Path, default=DEFAULT_SOURCE_FOREST)
    parser.add_argument("--review-decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_forest = _read_json(args.source_forest)
    decisions = _read_json(args.review_decisions)
    forest = build_reviewed_forest(source_forest, decisions)
    forest["source"]["source_forest"] = str(args.source_forest)
    forest["source"]["source_decisions"] = str(args.review_decisions)

    _write_json(output_dir / "aml_pattern_forest_v1_reviewed_draft.json", forest)
    _write_json(output_dir / "aml_pattern_forest_v1_reviewed_draft_compact.json", compact_forest(forest))
    write_tree(output_dir / "aml_pattern_forest_v1_reviewed_draft_tree.txt", forest)
    write_report(output_dir / "aml_pattern_forest_v1_reviewed_draft_review.md", forest)
    _write_json(output_dir / "summary.json", forest["summary"])

    print(json.dumps({"ok": True, "summary": forest["summary"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
