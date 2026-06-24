from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DRAFT = Path("outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json")
DEFAULT_SPLIT_PLAN = Path("outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_split_proposals_v1")


RELATION_PRIORITY = {
    "overlap": 0,
    "after": 1,
    "before": 2,
}

STRUCTURAL_CONTEXT_BUCKETS = {
    "leg_context",
    "locomotion_context",
    "vertical_context",
    "rotation_context",
    "state_context",
    "other_context",
    "posture_context",
}

MODIFIER_CONTEXT_BUCKETS = {
    "arm_context",
    "torso_context",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _context_bucket(cluster: str) -> str:
    if cluster.startswith("WHOLE_BODY_LOCOMOTION/"):
        return "locomotion_context"
    if cluster.startswith("WHOLE_BODY_VERTICAL/"):
        return "vertical_context"
    if cluster.startswith("WHOLE_BODY_ROTATION/"):
        return "rotation_context"
    if cluster.startswith("WHOLE_BODY_POSTURE/"):
        return "posture_context"
    if cluster.startswith("WHOLE_BODY_STATE/"):
        return "state_context"
    if cluster.startswith("TORSO_"):
        return "torso_context"
    if cluster.startswith("LEFT_LEG_ACTION/") or cluster.startswith("RIGHT_LEG_ACTION/"):
        return "leg_context"
    if cluster.startswith("LEFT_ARM_") or cluster.startswith("RIGHT_ARM_") or cluster.startswith("BIMANUAL_"):
        return "arm_context"
    return "other_context"


def _source_buckets(node: dict[str, Any]) -> set[str]:
    clusters = node.get("motion_evidence", {}).get("required_geometry_clusters") or []
    return {_context_bucket(str(cluster)) for cluster in clusters if str(cluster)}


def _axis_role(axis: dict[str, Any], source_buckets: set[str]) -> str:
    bucket = str(axis.get("context_bucket") or "")
    if bucket in source_buckets:
        return "source_subtype_axis"
    if bucket in STRUCTURAL_CONTEXT_BUCKETS:
        return "structural_context_axis"
    if bucket in MODIFIER_CONTEXT_BUCKETS:
        if source_buckets & MODIFIER_CONTEXT_BUCKETS:
            return "source_modifier_subtype_axis"
        return "modifier_context_axis"
    return "diagnostic_context_axis"


def _axis_score(axis: dict[str, Any], source_buckets: set[str]) -> tuple[int, int, int, int, str]:
    role = _axis_role(axis, source_buckets)
    role_rank = {
        "source_subtype_axis": 0,
        "source_modifier_subtype_axis": 1,
        "structural_context_axis": 2,
        "modifier_context_axis": 4,
        "diagnostic_context_axis": 5,
    }.get(role, 9)
    relation = str(axis.get("relation") or "")
    relation_rank = RELATION_PRIORITY.get(relation, 9)
    groups = axis.get("groups") or []
    max_support = max((int(group.get("support_cases") or 0) for group in groups), default=0)
    group_count = int(axis.get("group_count") or 0)
    return (role_rank, relation_rank, -max_support, -group_count, str(axis.get("axis_id") or ""))


def _select_axes(
    axes: list[dict[str, Any]],
    source_buckets: set[str],
    *,
    min_child_support: int,
    max_axes_per_node: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for axis in axes:
        groups = axis.get("groups") or []
        strong_groups = [group for group in groups if int(group.get("support_cases") or 0) >= min_child_support]
        axis = {**axis, "axis_role": _axis_role(axis, source_buckets), "eligible_group_count": len(strong_groups)}
        if not strong_groups:
            deferred.append({**axis, "defer_reason": "no_group_reaches_min_child_support"})
            continue
        if axis["axis_role"] in {"modifier_context_axis", "diagnostic_context_axis"}:
            deferred.append({**axis, "defer_reason": "modifier_or_diagnostic_axis"})
            continue
        eligible.append(axis)

    selected_by_bucket: dict[str, dict[str, Any]] = {}
    alternatives: list[dict[str, Any]] = []
    for axis in sorted(eligible, key=lambda item: _axis_score(item, source_buckets)):
        bucket = str(axis.get("context_bucket") or "")
        if bucket not in selected_by_bucket:
            selected_by_bucket[bucket] = axis
        else:
            alternatives.append({**axis, "defer_reason": "lower_ranked_axis_for_same_context_bucket"})

    selected = sorted(selected_by_bucket.values(), key=lambda item: _axis_score(item, source_buckets))[:max_axes_per_node]
    selected_ids = {str(axis.get("axis_id") or "") for axis in selected}
    overflow = [
        {**axis, "defer_reason": "beyond_max_axes_per_node"}
        for axis in selected_by_bucket.values()
        if str(axis.get("axis_id") or "") not in selected_ids
    ]
    return selected, sorted(deferred + alternatives + overflow, key=lambda item: str(item.get("axis_id") or ""))


def _top_clusters(group: dict[str, Any]) -> list[dict[str, Any]]:
    merged = Counter()
    for key in ["top_overlap_clusters", "top_after_clusters", "top_before_clusters"]:
        for item in group.get(key) or []:
            cluster_id = str(item.get("id") or "")
            if cluster_id:
                merged[cluster_id] += int(item.get("count") or 0)
    return [{"id": key, "count": int(value)} for key, value in merged.most_common(12)]


def _example_refs(group: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in (group.get("example_occurrences") or [])[:limit]:
        refs.append(
            {
                "case_id": row.get("case_id"),
                "span": row.get("span"),
                "motif_symbol": row.get("motif_symbol"),
            }
        )
    return refs


def _readiness(support: int, total: int) -> str:
    ratio = support / max(total, 1)
    if support >= 30 and ratio >= 0.12:
        return "review_for_promotion"
    if support >= 12:
        return "review_as_minor_split"
    return "low_support_diagnostic"


def _child_candidates_for_axis(
    source_node: dict[str, Any],
    plan: dict[str, Any],
    axis: dict[str, Any],
    *,
    min_child_support: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_node_id = str(plan.get("source_node_id") or "")
    total_occurrences = int(plan.get("occurrence_count") or 0)
    base_clusters = source_node.get("motion_evidence", {}).get("required_geometry_clusters") or []
    for index, group in enumerate(axis.get("groups") or [], start=1):
        support = int(group.get("support_cases") or 0)
        if support < min_child_support:
            continue
        cluster_id = str(group.get("cluster_id") or "")
        child_id = f"{source_node_id}__{str(axis.get('context_bucket') or 'context').replace('_context', '')}_{index:02d}"
        required_clusters = list(dict.fromkeys([*base_clusters, cluster_id]))
        out.append(
            {
                "proposal_node_id": child_id,
                "parent_node_id": source_node_id,
                "status": "split_child_candidate",
                "promotion_readiness": _readiness(support, total_occurrences),
                "structural_axis": {
                    "axis_id": axis.get("axis_id"),
                    "axis_role": axis.get("axis_role"),
                    "relation": axis.get("relation"),
                    "context_bucket": axis.get("context_bucket"),
                    "cluster_id": cluster_id,
                },
                "support": {
                    "support_cases": support,
                    "occurrences": int(group.get("occurrences") or 0),
                    "parent_occurrence_count": total_occurrences,
                    "support_ratio": round(support / max(total_occurrences, 1), 4),
                },
                "motion_evidence": {
                    "source_motif_ids": plan.get("source_motif_ids") or [],
                    "required_geometry_clusters": required_clusters,
                    "top_cooccurring_clusters": _top_clusters(group),
                },
                "example_refs": _example_refs(group),
                "diagnostics": {
                    "caption_alias_diagnostics": group.get("caption_alias_diagnostics") or [],
                    "language_policy": "diagnostic only; not used for proposal structure",
                },
                "runtime_policy": "offline proposal only; not inserted into runtime AML tree",
            }
        )
    return out


def build_proposals(
    draft: dict[str, Any],
    split_plan: dict[str, Any],
    *,
    min_child_support: int,
    max_axes_per_node: int,
) -> dict[str, Any]:
    draft_nodes = {str(node.get("draft_node_id") or ""): node for node in draft.get("draft_nodes") or []}
    split_proposals: list[dict[str, Any]] = []
    skipped_plans: list[dict[str, Any]] = []
    for plan in split_plan.get("split_plans") or []:
        source_node_id = str(plan.get("source_node_id") or "")
        source_node = draft_nodes.get(source_node_id)
        if not source_node:
            skipped_plans.append({"source_node_id": source_node_id, "reason": "missing_source_node_in_draft"})
            continue
        source_buckets = _source_buckets(source_node)
        axes = plan.get("candidate_axis_groups") or []
        selected_axes, deferred_axes = _select_axes(
            axes,
            source_buckets,
            min_child_support=min_child_support,
            max_axes_per_node=max_axes_per_node,
        )
        child_candidates: list[dict[str, Any]] = []
        for axis in selected_axes:
            child_candidates.extend(
                _child_candidates_for_axis(
                    source_node,
                    plan,
                    axis,
                    min_child_support=min_child_support,
                )
            )
        split_proposals.append(
            {
                "source_node_id": source_node_id,
                "source_node_status": source_node.get("status"),
                "source_structural_role": source_node.get("structural_role"),
                "source_buckets": sorted(source_buckets),
                "source_motif_ids": plan.get("source_motif_ids") or [],
                "parent_required_geometry_clusters": source_node.get("motion_evidence", {}).get("required_geometry_clusters") or [],
                "occurrence_count": plan.get("occurrence_count"),
                "unique_case_count": plan.get("unique_case_count"),
                "selected_axes": [
                    {
                        "axis_id": axis.get("axis_id"),
                        "axis_role": axis.get("axis_role"),
                        "relation": axis.get("relation"),
                        "context_bucket": axis.get("context_bucket"),
                        "group_count": axis.get("group_count"),
                        "eligible_group_count": axis.get("eligible_group_count"),
                    }
                    for axis in selected_axes
                ],
                "deferred_axes": [
                    {
                        "axis_id": axis.get("axis_id"),
                        "axis_role": axis.get("axis_role"),
                        "relation": axis.get("relation"),
                        "context_bucket": axis.get("context_bucket"),
                        "group_count": axis.get("group_count"),
                        "eligible_group_count": axis.get("eligible_group_count"),
                        "defer_reason": axis.get("defer_reason"),
                    }
                    for axis in deferred_axes
                ],
                "split_child_candidates": child_candidates,
                "recommendation": (
                    "review_child_candidates_for_tree_insertion"
                    if child_candidates
                    else "keep_parent_as_unsplit_component"
                ),
                "policy": "children are proposed from selected motion axes only; text labels remain diagnostics",
            }
        )

    status_counts = Counter()
    axis_counts = Counter()
    child_count = 0
    for proposal in split_proposals:
        child_count += len(proposal.get("split_child_candidates") or [])
        for child in proposal.get("split_child_candidates") or []:
            status_counts[str(child.get("promotion_readiness") or "")] += 1
            axis_counts[str(child.get("structural_axis", {}).get("context_bucket") or "")] += 1

    return {
        "schema_version": "motion_split_proposals_v1",
        "runtime_policy": "offline proposal only; generated from motion split axes, not from action-name rules",
        "summary": {
            "source_split_plan_count": len(split_plan.get("split_plans") or []),
            "proposal_count": len(split_proposals),
            "split_child_candidate_count": child_count,
            "min_child_support": min_child_support,
            "max_axes_per_node": max_axes_per_node,
            "promotion_readiness_counts": dict(sorted(status_counts.items())),
            "child_context_bucket_counts": dict(sorted(axis_counts.items())),
            "skipped_plans": skipped_plans,
        },
        "split_proposals": split_proposals,
    }


def promotion_queue(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for proposal in payload.get("split_proposals") or []:
        for child in proposal.get("split_child_candidates") or []:
            readiness = str(child.get("promotion_readiness") or "")
            if readiness == "low_support_diagnostic":
                continue
            rows.append(
                {
                    "proposal_node_id": child.get("proposal_node_id"),
                    "parent_node_id": child.get("parent_node_id"),
                    "promotion_readiness": readiness,
                    "structural_axis": child.get("structural_axis") or {},
                    "support": child.get("support") or {},
                    "required_geometry_clusters": child.get("motion_evidence", {}).get("required_geometry_clusters") or [],
                    "top_cooccurring_clusters": child.get("motion_evidence", {}).get("top_cooccurring_clusters") or [],
                    "example_refs": child.get("example_refs") or [],
                    "policy": "compact queue excludes caption aliases and text labels",
                }
            )
    rows.sort(
        key=lambda item: (
            0 if item["promotion_readiness"] == "review_for_promotion" else 1,
            -int(item.get("support", {}).get("support_cases") or 0),
            str(item.get("proposal_node_id") or ""),
        )
    )
    return {
        "schema_version": "motion_split_promotion_queue_v1",
        "runtime_policy": "offline review queue only; no language diagnostics included",
        "summary": {
            "queue_count": len(rows),
            "readiness_counts": dict(sorted(Counter(str(item["promotion_readiness"]) for item in rows).items())),
        },
        "queue": rows,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Motion Split Proposals V1")
    lines.append("")
    lines.append("This is an offline proposal layer. It converts motion-context split axes into child-node candidates.")
    lines.append("Text labels are diagnostic only and are not used to select axes or children.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (payload.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    for proposal in payload.get("split_proposals") or []:
        lines.append("")
        lines.append(f"## {proposal['source_node_id']}")
        lines.append("")
        lines.append(f"- source role: `{proposal.get('source_structural_role')}`")
        lines.append(f"- source buckets: `{', '.join(proposal.get('source_buckets') or [])}`")
        lines.append(f"- occurrences: `{proposal.get('occurrence_count')}`")
        lines.append(f"- recommendation: `{proposal.get('recommendation')}`")
        lines.append("")
        lines.append("### Selected Axes")
        lines.append("")
        if not proposal.get("selected_axes"):
            lines.append("No selected axes.")
        else:
            lines.append("| axis | role | group count | eligible groups |")
            lines.append("| --- | --- | ---: | ---: |")
            for axis in proposal.get("selected_axes") or []:
                lines.append(
                    f"| `{axis['axis_id']}` | `{axis['axis_role']}` | {axis['group_count']} | {axis['eligible_group_count']} |"
                )
        lines.append("")
        lines.append("### Child Candidates")
        lines.append("")
        if not proposal.get("split_child_candidates"):
            lines.append("No child candidates.")
        else:
            lines.append("| child | readiness | axis | support | required geometry |")
            lines.append("| --- | --- | --- | ---: | --- |")
            for child in proposal.get("split_child_candidates") or []:
                axis = child.get("structural_axis") or {}
                support = child.get("support") or {}
                geometry = "<br>".join(child.get("motion_evidence", {}).get("required_geometry_clusters") or [])
                lines.append(
                    "| `{child}` | `{ready}` | `{axis}` | {support} | {geometry} |".format(
                        child=child.get("proposal_node_id"),
                        ready=child.get("promotion_readiness"),
                        axis=f"{axis.get('axis_id')}::{axis.get('cluster_id')}",
                        support=support.get("support_cases"),
                        geometry=geometry,
                    )
                )
        lines.append("")
        lines.append("### Deferred Axes")
        lines.append("")
        if not proposal.get("deferred_axes"):
            lines.append("No deferred axes.")
        else:
            lines.append("| axis | role | reason |")
            lines.append("| --- | --- | --- |")
            for axis in proposal.get("deferred_axes") or []:
                lines.append(f"| `{axis['axis_id']}` | `{axis['axis_role']}` | `{axis['defer_reason']}` |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline child-node proposals from motion split axes.")
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--split-plan", default=str(DEFAULT_SPLIT_PLAN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-child-support", type=int, default=5)
    parser.add_argument("--max-axes-per-node", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_proposals(
        _read_json(Path(args.draft)),
        _read_json(Path(args.split_plan)),
        min_child_support=int(args.min_child_support),
        max_axes_per_node=int(args.max_axes_per_node),
    )
    payload["source"] = {
        "draft": str(args.draft),
        "split_plan": str(args.split_plan),
    }
    _write_json(output_dir / "motion_split_proposals.json", payload)
    queue = promotion_queue(payload)
    _write_json(output_dir / "motion_split_promotion_queue.json", queue)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_split_proposals_summary_v1",
            **payload["summary"],
            "promotion_queue": str(output_dir / "motion_split_promotion_queue.json"),
            "source": payload["source"],
        },
    )
    write_report(output_dir / "motion_split_proposals.md", payload)
    print(output_dir)


if __name__ == "__main__":
    main()
