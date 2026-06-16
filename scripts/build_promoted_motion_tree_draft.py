from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = Path("outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidates.json")
DEFAULT_MANUAL_REVIEW = Path("outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/manual_review_v1.json")
DEFAULT_NAMING_LAYER = Path("outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1")


PROMOTION_STATUS_BY_DECISION = {
    "promote": "promoted_candidate",
    "promote_candidate": "promoted_candidate",
    "rename_or_reframe": "structural_component",
    "downgrade": "structural_component",
    "keep_diagnostic": "diagnostic_only",
    "split": "split_required",
    "merge": "merge_required",
}

COMPONENT_ROLES = {
    "upper_body_component",
    "transition_component",
    "reusable_component",
    "ambiguous_low_body_component_family",
    "diagnostic_bilateral_arm_motion",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _top_items(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return items[:limit] if isinstance(items, list) else []


def _primary_label(labels: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for label in labels:
        status = str(label.get("status") or "")
        label_type = str(label.get("label_type") or "")
        if label_type == "caption_alias":
            score = float(label.get("confidence") or 0.0)
        elif label_type == "text_phrase":
            score = float(label.get("score") or 0.0)
        else:
            score = 0.15
        if status.startswith("strong"):
            score += 0.25
        ranked.append((score, label))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], str(item[1].get("label") or "")))[1]


def _node_label_summary(label_row: dict[str, Any] | None) -> dict[str, Any]:
    if not label_row:
        return {
            "primary_label": None,
            "candidate_labels": [],
            "top_phrase_alignments": [],
        }
    labels = list(label_row.get("candidate_labels") or [])
    primary = _primary_label(labels)
    return {
        "primary_label": primary,
        "candidate_labels": _top_items(labels, 12),
        "top_phrase_alignments": _top_items(label_row.get("top_phrase_alignments") or [], 12),
        "node_case_support": label_row.get("node_case_support"),
    }


def _geometry_overlap(a: list[str], b: list[str]) -> float:
    set_a = {str(item) for item in a if str(item)}
    set_b = {str(item) for item in b if str(item)}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _source_motifs(node: dict[str, Any]) -> list[str]:
    return [str(row.get("motif_id") or "") for row in node.get("source_motifs") or [] if str(row.get("motif_id") or "")]


def _node_record(
    candidate: dict[str, Any],
    review: dict[str, Any],
    label_summary: dict[str, Any],
) -> dict[str, Any]:
    decision = str(review.get("decision") or "unreviewed")
    structural_role = str(review.get("recommended_structural_role") or "unspecified")
    status = PROMOTION_STATUS_BY_DECISION.get(decision, "review_required")
    if structural_role in COMPONENT_ROLES and status == "review_required":
        status = "structural_component"
    return {
        "draft_node_id": str(candidate.get("node_id") or ""),
        "source_motion_node_id": str(candidate.get("node_id") or ""),
        "status": status,
        "manual_decision": decision,
        "structural_role": structural_role,
        "motion_family_key": candidate.get("motion_family_key"),
        "motion_evidence": candidate.get("motion_evidence") or {},
        "source_motif_ids": _source_motifs(candidate),
        "support": {
            "candidate_support_cases": candidate.get("support_cases"),
            "candidate_support_count_policy": candidate.get("support_count_policy"),
            "unique_node_case_support_from_naming_layer": label_summary.get("node_case_support"),
        },
        "naming": label_summary,
        "review": {
            "language_hint": review.get("language_hint"),
            "review_note": review.get("review_note"),
            "next_action": review.get("next_action"),
        },
        "runtime_policy": "draft_only_not_runtime_tree",
    }


def _component_links(nodes: list[dict[str, Any]], *, min_overlap: float) -> list[dict[str, Any]]:
    promoted = [node for node in nodes if node["status"] == "promoted_candidate"]
    components = [node for node in nodes if node["status"] == "structural_component"]
    links: list[dict[str, Any]] = []
    for comp in components:
        comp_clusters = comp.get("motion_evidence", {}).get("required_geometry_clusters") or []
        for parent in promoted:
            parent_clusters = parent.get("motion_evidence", {}).get("required_geometry_clusters") or []
            overlap = _geometry_overlap(comp_clusters, parent_clusters)
            if overlap >= min_overlap:
                links.append(
                    {
                        "parent_node_id": parent["draft_node_id"],
                        "component_node_id": comp["draft_node_id"],
                        "geometry_jaccard": round(overlap, 4),
                        "relationship": "candidate_component_of_promoted_pattern",
                        "policy": "relationship proposed only from motion-geometry overlap; shared language labels are diagnostics, not structure",
                    }
                )
    return sorted(links, key=lambda item: (str(item["parent_node_id"]), -float(item["geometry_jaccard"]), str(item["component_node_id"])))


def _split_requests(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if node["status"] != "split_required":
            continue
        phrases = node.get("naming", {}).get("top_phrase_alignments") or []
        phrase_terms = [str(item.get("normalized_phrase") or "") for item in phrases[:12] if str(item.get("normalized_phrase") or "")]
        cluster_ids = node.get("motion_evidence", {}).get("required_geometry_clusters") or []
        out.append(
            {
                "source_node_id": node["draft_node_id"],
                "structural_role": node["structural_role"],
                "required_geometry_clusters": cluster_ids,
                "source_motif_ids": node.get("source_motif_ids") or [],
                "language_terms_for_diagnosis": phrase_terms,
                "review_next_action": node.get("review", {}).get("next_action"),
                "recommended_method": "rerun motif-family split using motion subclusters, transition context, and phrase diagnostics; do not split solely by text label",
            }
        )
    return out


def _naming_conflicts(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        for label in node.get("naming", {}).get("candidate_labels") or []:
            if str(label.get("status") or "").startswith("strong"):
                key = str(label.get("label") or "").lower()
                if key:
                    by_label[key].append(node)
    conflicts: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        statuses = {str(row.get("status") or "") for row in rows}
        if len(rows) <= 1 or len(statuses) <= 1:
            continue
        conflicts.append(
            {
                "label": label,
                "node_ids": [row["draft_node_id"] for row in rows],
                "node_statuses": sorted(statuses),
                "policy": "same strong language label maps to different motion dispositions; trust motion review for promotion",
            }
        )
    return conflicts


def build_draft(
    candidates: dict[str, Any],
    manual_review: dict[str, Any],
    naming_layer: dict[str, Any],
    *,
    component_overlap_threshold: float,
) -> dict[str, Any]:
    candidate_map = {str(row.get("node_id") or ""): row for row in candidates.get("candidate_nodes") or []}
    review_map = {str(row.get("motion_node_id") or ""): row for row in manual_review.get("decisions") or []}
    label_map = {str(row.get("motion_node_id") or ""): row for row in naming_layer.get("motion_node_labels") or []}

    draft_nodes: list[dict[str, Any]] = []
    missing_review: list[str] = []
    for node_id, candidate in sorted(candidate_map.items()):
        review = review_map.get(node_id)
        if not review:
            missing_review.append(node_id)
            review = {"decision": "unreviewed", "recommended_structural_role": "unspecified"}
        label_summary = _node_label_summary(label_map.get(node_id))
        draft_nodes.append(_node_record(candidate, review, label_summary))

    status_counts = Counter(str(row["status"]) for row in draft_nodes)
    draft = {
        "schema_version": "promoted_motion_tree_draft_v1",
        "runtime_policy": "offline draft only; generated from generic artifact joins, not from hard-coded node/action cases",
        "source": {
            "candidate_nodes": str(DEFAULT_CANDIDATES),
            "manual_review": str(DEFAULT_MANUAL_REVIEW),
            "naming_layer": str(DEFAULT_NAMING_LAYER),
        },
        "summary": {
            "candidate_node_count": len(candidate_map),
            "reviewed_node_count": len(review_map),
            "draft_node_count": len(draft_nodes),
            "status_counts": dict(sorted(status_counts.items())),
            "missing_review_node_ids": missing_review,
        },
        "draft_nodes": draft_nodes,
        "component_links": _component_links(draft_nodes, min_overlap=component_overlap_threshold),
        "split_requests": _split_requests(draft_nodes),
        "naming_conflicts": _naming_conflicts(draft_nodes),
    }
    return draft


def write_report(path: Path, draft: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Promoted Motion Tree Draft V1")
    lines.append("")
    lines.append("This is an offline draft generated from candidate nodes, manual review decisions, and the language naming layer.")
    lines.append("It is not the runtime AML tree.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (draft.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Draft Nodes")
    lines.append("")
    lines.append("| node | status | role | primary label | geometry | manual decision |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for node in draft.get("draft_nodes") or []:
        primary = node.get("naming", {}).get("primary_label") or {}
        label = primary.get("label") or ""
        clusters = "<br>".join(node.get("motion_evidence", {}).get("required_geometry_clusters") or [])
        lines.append(
            "| `{node_id}` | `{status}` | `{role}` | `{label}` | {clusters} | `{decision}` |".format(
                node_id=node["draft_node_id"],
                status=node["status"],
                role=node["structural_role"],
                label=label,
                clusters=clusters or "none",
                decision=node["manual_decision"],
            )
        )
    lines.append("")
    lines.append("## Component Links")
    lines.append("")
    lines.append("Component links are proposed only from motion-geometry overlap. Shared language labels are reported under naming conflicts, not used as structure.")
    if not draft.get("component_links"):
        lines.append("No component links proposed.")
    else:
        lines.append("")
        lines.append("| parent | component | geometry jaccard |")
        lines.append("| --- | --- | ---: |")
        for link in draft.get("component_links") or []:
            lines.append(
                f"| `{link['parent_node_id']}` | `{link['component_node_id']}` | {link['geometry_jaccard']} |"
            )
    lines.append("")
    lines.append("## Split Requests")
    if not draft.get("split_requests"):
        lines.append("")
        lines.append("No split requests.")
    else:
        for req in draft.get("split_requests") or []:
            lines.append("")
            lines.append(f"### {req['source_node_id']}")
            lines.append("")
            lines.append(f"- role: `{req['structural_role']}`")
            lines.append(f"- geometry: `{', '.join(req['required_geometry_clusters'])}`")
            lines.append(f"- source motifs: `{', '.join(req['source_motif_ids'])}`")
            lines.append(f"- language diagnostics: `{', '.join(req['language_terms_for_diagnosis'][:10])}`")
            lines.append(f"- method: {req['recommended_method']}")
    lines.append("")
    lines.append("## Naming Conflicts")
    if not draft.get("naming_conflicts"):
        lines.append("")
        lines.append("No naming conflicts.")
    else:
        for conflict in draft.get("naming_conflicts") or []:
            lines.append(
                f"- `{conflict['label']}` maps to nodes `{', '.join(conflict['node_ids'])}` with statuses `{', '.join(conflict['node_statuses'])}`"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline promoted motion-tree draft from reviewed motion candidates.")
    parser.add_argument("--candidate-nodes", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--manual-review", default=str(DEFAULT_MANUAL_REVIEW))
    parser.add_argument("--naming-layer", default=str(DEFAULT_NAMING_LAYER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--component-overlap-threshold", type=float, default=0.45)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draft = build_draft(
        _read_json(Path(args.candidate_nodes)),
        _read_json(Path(args.manual_review)),
        _read_json(Path(args.naming_layer)),
        component_overlap_threshold=float(args.component_overlap_threshold),
    )
    draft["source"] = {
        "candidate_nodes": str(args.candidate_nodes),
        "manual_review": str(args.manual_review),
        "naming_layer": str(args.naming_layer),
    }
    _write_json(output_dir / "promoted_motion_tree_draft.json", draft)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "promoted_motion_tree_draft_summary_v1",
            **draft["summary"],
            "source": draft["source"],
        },
    )
    write_report(output_dir / "promoted_motion_tree_draft.md", draft)
    print(output_dir)


if __name__ == "__main__":
    main()
