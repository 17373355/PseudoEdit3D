"""Build a reviewable pattern-forest draft from v4 coord-role closure output.

This script is an offline review step:

1. Read `composition_closure_candidates.json` from the v4 coord-role closure
   audit.
2. Keep all `promote_review` rows plus a small alias-indexed set of hard
   review rows.
3. Group them into root -> family -> source-candidate nodes.

Caption aliases are used only for naming and review grouping. Motion evidence
comes from `canonical_role_items`, channels, zones, geometry clusters, support,
and promotion blockers.

Typical use:
    python scripts/build_v4_closure_pattern_forest_draft.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLOSURE = Path(
    "outputs/aml_regression_testset_v2/"
    "hml3d_multichannel_motion_bpe_v4_coord_role_full_closure_review_v0/"
    "composition_closure_candidates.json"
)
DEFAULT_ALIAS_SIDECAR = Path("pseudoedit3d/edit/aml_semantic_alias_sidecar.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_forest_v1_from_v4_closure_draft")


ROOTS = [
    {
        "node_id": "coordination_patterns",
        "accepted_name": "coordinated whole-body patterns",
        "description": "Upper/lower/vertical coordination that may become complete action patterns after review.",
    },
    {
        "node_id": "acrobatics_inversion",
        "accepted_name": "inversion and acrobatics",
        "description": "Inverted or acrobatic body configurations with limb/posture evidence.",
    },
    {
        "node_id": "posture_transitions",
        "accepted_name": "posture and level transitions",
        "description": "Sit, stand, kneel, low-body, or level-change transition candidates.",
    },
    {
        "node_id": "floor_prone_or_mime",
        "accepted_name": "floor-prone or mime-like patterns",
        "description": "Floor/prone/swim/fly-like conflicts that need support-state split before promotion.",
    },
    {
        "node_id": "strike_guard_coordination",
        "accepted_name": "strike or guard coordination",
        "description": "Martial/strike-like coordination candidates that require stricter arm-leg semantics.",
    },
    {
        "node_id": "dance_cheer_coordination",
        "accepted_name": "dance and cheer coordination",
        "description": "Rhythmic expressive coordination candidates that are often composition families.",
    },
    {
        "node_id": "object_activity_proxies",
        "accepted_name": "object-activity proxy candidates",
        "description": "Object-heavy labels whose motion evidence is only a proxy without object observables.",
    },
    {
        "node_id": "environment_transition_candidates",
        "accepted_name": "environment transition candidates",
        "description": "Duck, climb, step-over, or obstacle-like labels requiring environment context.",
    },
    {
        "node_id": "component_library",
        "accepted_name": "reusable motion components",
        "description": "Reusable upper/lower/body components, not complete action names by themselves.",
    },
    {
        "node_id": "generic_composition_review",
        "accepted_name": "generic composition review",
        "description": "Useful structures that do not yet have a stable semantic scope.",
    },
]


ALIAS_ROOT_HINTS = {
    "jumping_jack": "coordination_patterns",
    "cartwheel": "acrobatics_inversion",
    "swim_like_motion": "floor_prone_or_mime",
    "fly_like_motion": "floor_prone_or_mime",
    "sit_down": "posture_transitions",
    "sit_down_stand_up": "posture_transitions",
    "kneel_or_fall_to_knees": "posture_transitions",
    "martial_arts": "strike_guard_coordination",
    "cheer_dance": "dance_cheer_coordination",
    "ballet_dance": "dance_cheer_coordination",
    "basketball_dribble": "object_activity_proxies",
    "tennis_like": "object_activity_proxies",
    "dumbbell_lift": "object_activity_proxies",
    "duck_under_obstacle": "environment_transition_candidates",
    "climb_over_obstacle": "environment_transition_candidates",
    "jump_rope": "coordination_patterns",
    "arm_circle_or_windmill": "component_library",
    "hand_to_face_or_ear": "component_library",
}


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


def _top_alias(row: dict[str, Any]) -> dict[str, Any]:
    aliases = row.get("caption_aliases") or []
    if aliases:
        return dict(aliases[0])
    return {"id": "unnamed", "count": 0}


def _alias_labels(alias_sidecar: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for rule in alias_sidecar.get("rules") or []:
        alias_id = str(rule.get("alias_id") or "")
        if alias_id:
            labels[alias_id] = str(rule.get("label") or alias_id.replace("_", " "))
    return labels


def _role_name(item: str) -> str:
    return item.split(":", 1)[1] if ":" in item else item


def _root_for_row(row: dict[str, Any]) -> str:
    alias_id = str(_top_alias(row).get("id") or "")
    blockers = {str(item) for item in row.get("promotion_blockers") or []}
    scope = str(row.get("composition_scope") or "")
    zones = {str(item) for item in row.get("zones") or []}
    roles = {_role_name(str(item)) for item in row.get("canonical_role_items") or []}

    if "top_alias_motion_scope_conflict" in blockers and alias_id in {"swim_like_motion", "fly_like_motion"}:
        return "floor_prone_or_mime"
    if alias_id in ALIAS_ROOT_HINTS:
        return ALIAS_ROOT_HINTS[alias_id]
    if scope == "floor_prone_or_mime_candidate":
        return "floor_prone_or_mime"
    if scope == "inversion_acrobatic_candidate":
        return "acrobatics_inversion"
    if {"upper", "lower"}.issubset(zones) and ("vertical" in zones or "posture" in zones):
        return "coordination_patterns"
    if roles & {"body_level_cycle", "body_level_down", "body_level_up", "body_low_posture", "vertical_low_body_transition"}:
        return "posture_transitions"
    if zones == {"upper"} or zones == {"lower"}:
        return "component_library"
    return "generic_composition_review"


def _family_status(rows: list[dict[str, Any]]) -> str:
    blockers = Counter(str(item) for row in rows for item in row.get("promotion_blockers") or [])
    recommendations = Counter(str(row.get("recommendation") or "") for row in rows)
    if recommendations.get("promote_review") and not blockers:
        return "review_candidate"
    if blockers.get("top_alias_motion_scope_conflict"):
        return "split_required"
    if blockers.get("suppressed_discriminative_roles"):
        return "composition_needs_closure"
    if recommendations.get("composition_review"):
        return "composition_review"
    return "review_required"


def _family_scope(root_id: str, rows: list[dict[str, Any]]) -> str:
    status = _family_status(rows)
    scopes = Counter(str(row.get("composition_scope") or "") for row in rows)
    dominant_scope = scopes.most_common(1)[0][0] if scopes else "unknown"
    if root_id in {"coordination_patterns", "acrobatics_inversion"} and status == "review_candidate":
        return "full_pattern_candidate"
    if root_id == "posture_transitions":
        return "transition_pattern_candidate"
    if root_id == "floor_prone_or_mime":
        return "floor_or_prone_pattern_candidate"
    if root_id in {"object_activity_proxies", "environment_transition_candidates"}:
        return "proxy_or_context_candidate"
    if root_id == "component_library":
        return "component"
    return dominant_scope or "composition_candidate"


def _family_key(row: dict[str, Any]) -> tuple[str, str]:
    root_id = _root_for_row(row)
    alias_id = str(_top_alias(row).get("id") or "unnamed")
    blockers = {str(item) for item in row.get("promotion_blockers") or []}
    recommendation = str(row.get("recommendation") or "")
    if recommendation == "promote_review":
        disposition = "promote_review"
    elif "top_alias_motion_scope_conflict" in blockers:
        disposition = "scope_conflict"
    elif "suppressed_discriminative_roles" in blockers:
        disposition = "needs_closure"
    else:
        disposition = "composition_review"
    scope = str(row.get("composition_scope") or "unknown")
    return root_id, _safe_id(f"{alias_id}_{scope}_{disposition}")


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows_per_alias: int,
    min_alias_count: int,
    include_conflicts: bool,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        alias = _top_alias(row)
        alias_id = str(alias.get("id") or "")
        alias_count = int(alias.get("count") or 0)
        blockers = {str(item) for item in row.get("promotion_blockers") or []}
        if row.get("recommendation") == "promote_review":
            selected[candidate_id] = row
        if include_conflicts and "top_alias_motion_scope_conflict" in blockers:
            selected[candidate_id] = row
        if alias_id and alias_count >= min_alias_count:
            by_alias[alias_id].append(row)

    def row_key(row: dict[str, Any]) -> tuple[int, int, float, int, int]:
        recommendation_order = 0 if row.get("recommendation") == "promote_review" else 1
        blocker_count = len(row.get("promotion_blockers") or [])
        alias_count = int(_top_alias(row).get("count") or 0)
        return (
            recommendation_order,
            blocker_count,
            -float(row.get("score") or 0.0),
            -alias_count,
            -int(row.get("support_cases") or 0),
        )

    for alias_rows in by_alias.values():
        for row in sorted(alias_rows, key=row_key)[:max_rows_per_alias]:
            selected[str(row.get("candidate_id") or "")] = row

    return sorted(
        selected.values(),
        key=lambda row: (
            _root_for_row(row),
            _family_key(row)[1],
            int(row.get("rank") or 999999),
        ),
    )


def _count_items(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get(key) or []:
            counter[str(item)] += 1
    return [{"id": item, "count": int(count)} for item, count in counter.most_common()]


def _count_aliases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get("caption_aliases") or []:
            counter[str(item.get("id") or "")] += int(item.get("count") or 0)
    if "" in counter:
        del counter[""]
    return [{"id": item, "count": int(count)} for item, count in counter.most_common()]


def _count_geometry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get("geometry_clusters") or []:
            counter[str(item.get("id") or "")] += int(item.get("count") or 0)
    if "" in counter:
        del counter[""]
    return [{"id": item, "count": int(count)} for item, count in counter.most_common(24)]


def _source_candidate_node(row: dict[str, Any], alias_labels: dict[str, str]) -> dict[str, Any]:
    top = _top_alias(row)
    alias_id = str(top.get("id") or "unnamed")
    return {
        "node_id": f"source_{row.get('candidate_id')}",
        "node_kind": "source_closure_candidate",
        "status": str(row.get("recommendation") or "review_required"),
        "scope": str(row.get("composition_scope") or "unknown"),
        "accepted_name": alias_labels.get(alias_id, alias_id.replace("_", " ")),
        "language_aliases": list(row.get("caption_aliases") or []),
        "description": str(row.get("reason") or ""),
        "evidence": {
            "candidate_id": row.get("candidate_id"),
            "rank": row.get("rank"),
            "support_cases": row.get("support_cases"),
            "occurrences": row.get("occurrences"),
            "score": row.get("score"),
            "specificity_bucket": row.get("specificity_bucket"),
            "name_purity": row.get("name_purity") or {},
            "promotion_blockers": row.get("promotion_blockers") or [],
            "suppressed_discriminative_roles": row.get("suppressed_discriminative_roles") or [],
            "is_near_closed": row.get("is_near_closed"),
        },
        "motion_summary": {
            "channels": list(row.get("channels") or []),
            "zones": list(row.get("zones") or []),
            "canonical_role_items": list(row.get("canonical_role_items") or []),
            "geometry_clusters": list(row.get("geometry_clusters") or []),
            "discriminative_role_coverage": list(row.get("discriminative_role_coverage") or []),
        },
        "source_examples": list(row.get("examples") or []),
    }


def _unique_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = (str(example.get("case_id") or ""), str(example.get("caption") or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(example)
        if len(rows) >= limit:
            break
    return rows


def _family_node(
    family_id: str,
    root_id: str,
    rows: list[dict[str, Any]],
    alias_labels: dict[str, str],
) -> dict[str, Any]:
    aliases = _count_aliases(rows)
    top_alias = aliases[0]["id"] if aliases else "unnamed"
    label = alias_labels.get(top_alias, top_alias.replace("_", " "))
    status = _family_status(rows)
    blocker_counts = Counter(str(item) for row in rows for item in row.get("promotion_blockers") or [])
    recommendation_counts = Counter(str(row.get("recommendation") or "") for row in rows)
    supports = [int(row.get("support_cases") or 0) for row in rows]
    return {
        "node_id": f"family_{family_id}",
        "node_kind": "pattern_family_candidate",
        "status": status,
        "scope": _family_scope(root_id, rows),
        "accepted_name": label,
        "language_aliases": aliases[:12],
        "description": "Motion-derived family grouped from v4 coord-role closure candidates; visual review is still required before acceptance.",
        "evidence": {
            "source_candidate_count": len(rows),
            "support_cases_max": max(supports) if supports else 0,
            "support_cases_sum": sum(supports),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "promotion_blocker_counts": dict(sorted(blocker_counts.items())),
            "specificity_bucket_counts": dict(sorted(Counter(str(row.get("specificity_bucket") or "") for row in rows).items())),
        },
        "motion_summary": {
            "channels": _count_items(rows, "channels"),
            "zones": _count_items(rows, "zones"),
            "canonical_role_items": _count_items(rows, "canonical_role_items"),
            "geometry_clusters": _count_geometry(rows),
        },
    }


def build_forest(
    closure_payload: dict[str, Any],
    alias_sidecar: dict[str, Any],
    *,
    max_rows_per_alias: int,
    min_alias_count: int,
    include_conflicts: bool,
) -> dict[str, Any]:
    alias_labels = _alias_labels(alias_sidecar)
    selected_rows = _select_rows(
        list(closure_payload.get("candidates") or []),
        max_rows_per_alias=max_rows_per_alias,
        min_alias_count=min_alias_count,
        include_conflicts=include_conflicts,
    )

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

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[_family_key(row)].append(row)

    for (root_id, family_id), rows in sorted(grouped.items()):
        family = _family_node(family_id, root_id, rows, alias_labels)
        nodes.append(family)
        edges.append(
            {
                "parent_node_id": root_id,
                "child_node_id": family["node_id"],
                "edge_type": "root_to_family",
            }
        )
        for row in sorted(rows, key=lambda item: int(item.get("rank") or 999999)):
            child = _source_candidate_node(row, alias_labels)
            nodes.append(child)
            edges.append(
                {
                    "parent_node_id": family["node_id"],
                    "child_node_id": child["node_id"],
                    "edge_type": "family_to_source_candidate",
                }
            )

    status_counts = Counter(str(node.get("status") or "") for node in nodes)
    scope_counts = Counter(str(node.get("scope") or "") for node in nodes)
    return {
        "schema_version": "aml_pattern_forest_v1_from_v4_closure_draft",
        "runtime_policy": "offline review only; do not use as final AML runtime matching logic",
        "source": {
            "closure_schema": closure_payload.get("schema_version"),
            "closure_summary": closure_payload.get("summary") or {},
            "alias_sidecar_version": alias_sidecar.get("version"),
        },
        "summary": {
            "selected_source_candidate_count": len(selected_rows),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "root_count": sum(1 for node in nodes if node.get("node_kind") == "root"),
            "family_count": sum(1 for node in nodes if node.get("node_kind") == "pattern_family_candidate"),
            "source_candidate_node_count": sum(1 for node in nodes if node.get("node_kind") == "source_closure_candidate"),
            "status_counts": dict(sorted(status_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
        },
        "nodes": nodes,
        "edges": edges,
    }


def compact_forest(forest: dict[str, Any]) -> dict[str, Any]:
    compact_nodes: list[dict[str, Any]] = []
    for node in forest.get("nodes") or []:
        compact = {
            key: node.get(key)
            for key in [
                "node_id",
                "node_kind",
                "status",
                "scope",
                "accepted_name",
                "language_aliases",
                "description",
                "evidence",
                "motion_summary",
            ]
            if key in node
        }
        if node.get("node_kind") == "source_closure_candidate":
            compact["source_example_ids"] = [row.get("case_id") for row in node.get("source_examples") or []]
        compact_nodes.append(compact)
    return {
        "schema_version": f"{forest.get('schema_version')}_compact",
        "runtime_policy": forest.get("runtime_policy"),
        "source": forest.get("source") or {},
        "summary": forest.get("summary") or {},
        "nodes": compact_nodes,
        "edges": forest.get("edges") or [],
    }


def _child_index(forest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    node_by_id = {str(node.get("node_id")): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        child = node_by_id.get(str(edge.get("child_node_id") or ""))
        if child:
            children[str(edge.get("parent_node_id") or "")].append(child)
    return children


def write_tree(path: Path, forest: dict[str, Any], *, max_source_children: int) -> None:
    children = _child_index(forest)
    lines = ["# AML Pattern Forest v1 From v4 Closure Draft", ""]
    summary = forest.get("summary") or {}
    lines.append(
        "nodes={node_count} roots={root_count} families={family_count} source_candidates={source_candidate_node_count} edges={edge_count}".format(
            **summary
        )
    )
    lines.append("")
    roots = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "root"]
    for root in roots:
        root_children = sorted(children.get(str(root.get("node_id")), []), key=lambda row: str(row.get("node_id")))
        lines.append(f"- {root.get('node_id')} [root] {root.get('accepted_name')} ({len(root_children)} families)")
        for family in root_children:
            evidence = family.get("evidence") or {}
            aliases = ", ".join(f"{a['id']}:{a['count']}" for a in (family.get("language_aliases") or [])[:4])
            lines.append(
                "  - {node_id} [{status}] {name} scope={scope} variants={variants} support_max={support} aliases={aliases}".format(
                    node_id=family.get("node_id"),
                    status=family.get("status"),
                    name=family.get("accepted_name"),
                    scope=family.get("scope"),
                    variants=evidence.get("source_candidate_count"),
                    support=evidence.get("support_cases_max"),
                    aliases=aliases or "none",
                )
            )
            for child in sorted(children.get(str(family.get("node_id")), []), key=lambda row: str(row.get("node_id")))[:max_source_children]:
                ce = child.get("evidence") or {}
                roles = ", ".join(str(item) for item in (child.get("motion_summary") or {}).get("canonical_role_items") or [])
                blockers = ", ".join(str(item) for item in ce.get("promotion_blockers") or [])
                lines.append(
                    f"    - {ce.get('candidate_id')} [{child.get('status')}] support={ce.get('support_cases')} blockers={blockers or 'none'} roles={roles}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any], *, examples_per_candidate: int) -> None:
    children = _child_index(forest)
    lines = [
        "# AML Pattern Forest v1 From v4 Closure Draft",
        "",
        "This is a review artifact generated from v4 coord-role composition closure candidates.",
        "Caption aliases name/group nodes, but the tree structure is backed by motion role items and geometry evidence.",
        "",
        "## Summary",
        "",
    ]
    for key, value in (forest.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    roots = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "root"]
    for root in roots:
        root_children = sorted(children.get(str(root.get("node_id")), []), key=lambda row: str(row.get("node_id")))
        if not root_children:
            continue
        lines.extend([f"## {root.get('accepted_name')}", "", root.get("description") or "", ""])
        for family in root_children:
            evidence = family.get("evidence") or {}
            motion = family.get("motion_summary") or {}
            aliases = ", ".join(f"{a['id']}:{a['count']}" for a in (family.get("language_aliases") or [])[:8])
            roles = ", ".join(item["id"] for item in (motion.get("canonical_role_items") or [])[:12])
            blockers = evidence.get("promotion_blocker_counts") or {}
            lines.extend(
                [
                    f"### {family.get('node_id')}",
                    "",
                    f"- status: `{family.get('status')}`",
                    f"- scope: `{family.get('scope')}`",
                    f"- name candidate: `{family.get('accepted_name')}`",
                    f"- source candidates: `{evidence.get('source_candidate_count')}`; support max: `{evidence.get('support_cases_max')}`",
                    f"- aliases: {aliases or 'none'}",
                    f"- blockers: `{blockers}`",
                    f"- roles: `{roles}`",
                    "",
                    "| source candidate | recommendation | support | blockers | example captions |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            for child in sorted(children.get(str(family.get("node_id")), []), key=lambda row: str(row.get("node_id"))):
                ce = child.get("evidence") or {}
                examples = _unique_examples(child.get("source_examples") or [], examples_per_candidate)
                caption_rows: list[str] = []
                for ex in examples:
                    caption = str(ex.get("caption") or "").replace("|", "\\|")
                    caption_rows.append(f"`{ex.get('case_id')}` {caption}")
                captions = "<br>".join(caption_rows)
                lines.append(
                    "| `{cid}` | `{rec}` | {support} | `{blockers}` | {captions} |".format(
                        cid=ce.get("candidate_id"),
                        rec=child.get("status"),
                        support=ce.get("support_cases"),
                        blockers=", ".join(str(item) for item in ce.get("promotion_blockers") or []) or "none",
                        captions=captions or "",
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-candidates", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--alias-sidecar", type=Path, default=DEFAULT_ALIAS_SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows-per-alias", type=int, default=5)
    parser.add_argument("--min-alias-count", type=int, default=8)
    parser.add_argument("--examples-per-candidate", type=int, default=3)
    parser.add_argument("--max-source-children-in-tree", type=int, default=8)
    parser.add_argument("--no-conflicts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forest = build_forest(
        _read_json(args.closure_candidates),
        _read_json(args.alias_sidecar),
        max_rows_per_alias=int(args.max_rows_per_alias),
        min_alias_count=int(args.min_alias_count),
        include_conflicts=not bool(args.no_conflicts),
    )
    forest["source"]["closure_candidates"] = str(args.closure_candidates)
    forest["source"]["alias_sidecar"] = str(args.alias_sidecar)
    _write_json(output_dir / "aml_pattern_forest_v1_draft.json", forest)
    _write_json(output_dir / "aml_pattern_forest_v1_draft_compact.json", compact_forest(forest))
    _write_json(output_dir / "summary.json", forest.get("summary") or {})
    write_tree(output_dir / "aml_pattern_forest_v1_draft_tree.txt", forest, max_source_children=int(args.max_source_children_in_tree))
    write_report(output_dir / "aml_pattern_forest_v1_draft_review.md", forest, examples_per_candidate=int(args.examples_per_candidate))
    print(json.dumps(forest.get("summary") or {}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
