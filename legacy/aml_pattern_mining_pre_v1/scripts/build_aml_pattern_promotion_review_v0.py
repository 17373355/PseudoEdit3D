"""Build a review table for promoting dense AML pattern candidates.

This script is offline bookkeeping. It does not create runtime rules and does
not decide final action names. It turns the dense Motion-BPE candidate forest
into a compact table that can be manually checked before editing the reviewed
AML forest policy.

Example:
    python scripts/build_aml_pattern_promotion_review_v0.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DENSE_FOREST = Path(
    "outputs/aml_regression_testset_v2/aml_pattern_forest_candidates_v0_dense/"
    "aml_pattern_forest_candidates_dense.json"
)
DEFAULT_MOTIF_AUDIT = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/"
    "motif_audit.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0"
)


CSV_FIELDS = [
    "priority_rank",
    "family_id",
    "dense_status",
    "recommendation",
    "review_bucket",
    "support_cases_sum",
    "motif_count",
    "operator",
    "channels",
    "relations",
    "geometry_clusters",
    "caption_aliases",
    "review_reason",
]


ALIAS_PROMOTION_COUNT = 20


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _support(node: dict[str, Any]) -> int:
    return int((node.get("support_cases_sum") or node.get("support_cases") or 0) or 0)


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("id") or "") for row in rows if row.get("id")]


def _top_aliases(family: dict[str, Any], leaves: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in (family.get("naming_diagnostics") or {}).get("top_caption_aliases") or []:
        alias = str(row.get("id") or "")
        if alias:
            counter[alias] += int(row.get("count") or 0)
    for leaf in leaves:
        motif = audit_by_id.get(str(leaf.get("motif_id") or ""))
        if not motif:
            continue
        for row in motif.get("top_caption_aliases") or []:
            alias = str(row.get("id") or "")
            if alias:
                counter[alias] += int(row.get("count") or 0)
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(8)]


def _examples(leaves: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for leaf in sorted(leaves, key=lambda node: -_support(node)):
        motif_id = str(leaf.get("motif_id") or "")
        motif = audit_by_id.get(motif_id)
        if not motif:
            continue
        for item in motif.get("example_occurrences") or []:
            examples.append(
                {
                    "motif_id": motif_id,
                    "case_id": item.get("case_id"),
                    "span": item.get("span"),
                    "caption": item.get("caption"),
                }
            )
            if len(examples) >= max_examples:
                return examples
    return examples


def _leaf_status_counts(leaves: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(leaf.get("status") or "") for leaf in leaves).items()))


def _recommend(
    family: dict[str, Any],
    leaves: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
) -> tuple[str, str, str]:
    status = str(family.get("status") or "")
    motion = family.get("motion_definition") or {}
    operator = str(motion.get("operator") or "")
    channels = list(motion.get("required_channels") or [])
    relations = list(motion.get("required_relation_types") or [])
    geometry = list(motion.get("required_geometry_clusters") or [])
    support = _support(family)
    leaf_statuses = {str(leaf.get("status") or "") for leaf in leaves}

    if any(item.startswith("reviewed_accepted") for item in leaf_statuses):
        return (
            "already_reviewed",
            "accepted_reference",
            "at least one source motif is already linked to the reviewed v0 forest",
        )

    has_relation = operator == "COORDINATION_MERGE" or bool(relations)
    has_multi_channel = len(channels) >= 2
    has_vertical = any("whole_body_vertical" == channel for channel in channels)
    has_upper = any(channel in {"left_arm", "right_arm", "bimanual"} for channel in channels)
    has_lower = any(channel in {"left_leg", "right_leg", "whole_body_state"} for channel in channels)

    top_alias_count = int(aliases[0].get("count") or 0) if aliases else 0

    if status == "coordination_candidate" or (has_relation and has_multi_channel):
        if support >= 40 and has_vertical and has_upper and top_alias_count >= ALIAS_PROMOTION_COUNT:
            return (
                "promote_review",
                "full_or_composed_pattern_candidate",
                "multi-channel vertical and upper-body coordination has strong caption-name concentration; inspect examples before promotion",
            )
        if support >= 80:
            return (
                "composition_review",
                "composition_candidate_needs_purity",
                "multi-channel coordination is frequent, but examples/names are too diffuse for direct promotion",
            )
        return (
            "diagnostic_keep",
            "low_support_coordination",
            "coordination evidence exists but support is below the v0 review threshold",
        )

    if status == "named_candidate":
        return (
            "name_only_review",
            "caption_name_alignment_candidate",
            "caption aliases are informative, but the motion structure is not enough for a full action node",
        )

    if status == "component_candidate":
        if support >= 80:
            return (
                "component_review",
                "component_library_candidate",
                "single-channel or local sequence appears frequently and should be reviewed as a reusable component",
            )
        return (
            "diagnostic_keep",
            "low_support_component",
            "component-like structure exists but support is low",
        )

    if any("CANDIDATE" in item for item in geometry) and support >= 40:
        return (
            "component_review",
            "observable_candidate_review",
            "geometry cluster is explicitly marked candidate; inspect whether it is a real reusable node",
        )

    return (
        "diagnostic_keep",
        "diagnostic_or_noisy_candidate",
        "keep as mining evidence; do not promote without stronger structure or cleaner examples",
    )


def _priority(row: dict[str, Any]) -> tuple[int, int]:
    recommendation_order = {
        "already_reviewed": 0,
        "promote_review": 1,
        "composition_review": 2,
        "component_review": 3,
        "name_only_review": 4,
        "diagnostic_keep": 5,
    }
    return (recommendation_order.get(str(row.get("recommendation")), 9), -int(row.get("support_cases_sum") or 0))


def build_review_table(
    dense_forest: dict[str, Any],
    motif_audit: dict[str, Any],
    max_examples: int,
    *,
    dense_forest_path: Path = DEFAULT_DENSE_FOREST,
    motif_audit_path: Path = DEFAULT_MOTIF_AUDIT,
) -> dict[str, Any]:
    node_by_id = {str(node.get("node_id") or ""): node for node in dense_forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = {}
    for edge in dense_forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child in node_by_id:
            children.setdefault(parent, []).append(node_by_id[child])

    audit_by_id = {str(row.get("motif_id") or ""): row for row in motif_audit.get("motifs") or []}
    rows: list[dict[str, Any]] = []
    families = [
        node for node in dense_forest.get("nodes") or []
        if node.get("node_kind") == "dense_motif_family"
    ]
    for family in families:
        leaves = children.get(str(family.get("node_id") or ""), [])
        motion = family.get("motion_definition") or {}
        aliases = _top_aliases(family, leaves, audit_by_id)
        recommendation, review_bucket, reason = _recommend(family, leaves, aliases)
        rows.append(
            {
                "family_id": family.get("source_node_id"),
                "dense_node_id": family.get("node_id"),
                "dense_status": family.get("status"),
                "scope": family.get("scope"),
                "recommendation": recommendation,
                "review_bucket": review_bucket,
                "review_reason": reason,
                "support_cases_sum": _support(family),
                "occurrences_sum": int(family.get("occurrences_sum") or 0),
                "motif_count": len(leaves),
                "operator": motion.get("operator"),
                "channels": list(motion.get("required_channels") or []),
                "relations": list(motion.get("required_relation_types") or []),
                "geometry_clusters": list(motion.get("required_geometry_clusters") or []),
                "caption_aliases": aliases,
                "leaf_status_counts": _leaf_status_counts(leaves),
                "top_source_motifs": [
                    {
                        "motif_id": leaf.get("motif_id"),
                        "status": leaf.get("status"),
                        "support_cases": leaf.get("support_cases"),
                    }
                    for leaf in sorted(leaves, key=lambda node: -_support(node))[:8]
                ],
                "examples": _examples(leaves, audit_by_id, max_examples),
            }
        )

    rows = sorted(rows, key=_priority)
    for idx, row in enumerate(rows, start=1):
        row["priority_rank"] = idx

    recommendation_counts = Counter(str(row.get("recommendation") or "") for row in rows)
    bucket_counts = Counter(str(row.get("review_bucket") or "") for row in rows)
    status_counts = Counter(str(row.get("dense_status") or "") for row in rows)
    return {
        "schema_version": "aml_pattern_promotion_review_v0",
        "runtime_policy": "offline review table only; not runtime AML matching logic",
        "source_dense_forest": str(dense_forest_path),
        "source_motif_audit": str(motif_audit_path),
        "summary": {
            "family_count": len(rows),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "review_bucket_counts": dict(sorted(bucket_counts.items())),
            "dense_status_counts": dict(sorted(status_counts.items())),
        },
        "review_rows": rows,
    }


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(f"{item.get('id')}:{item.get('count')}" for item in value)
        return "; ".join(str(item) for item in value)
    return str(value or "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def write_markdown(path: Path, payload: dict[str, Any], max_rows: int) -> None:
    rows = payload.get("review_rows") or []
    lines = ["# AML Pattern Promotion Review v0", ""]
    lines.append("This table ranks dense Motion-BPE families for manual promotion review.")
    lines.append("It is an offline review artifact, not runtime matching logic.")
    lines.append("")
    summary = payload.get("summary") or {}
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- family_count: `{summary.get('family_count')}`")
    lines.append(f"- recommendation_counts: `{summary.get('recommendation_counts')}`")
    lines.append(f"- dense_status_counts: `{summary.get('dense_status_counts')}`")
    lines.append("")
    lines.append("## Review Labels")
    lines.append("")
    lines.append("- `promote_review`: inspect as a possible full or composed pattern.")
    lines.append("- `composition_review`: inspect as a reusable composition; not ready for promotion.")
    lines.append("- `component_review`: inspect as a reusable component, not a full action name.")
    lines.append("- `name_only_review`: caption names are useful, but structure is insufficient.")
    lines.append("- `diagnostic_keep`: keep as evidence; do not promote in v0.")
    lines.append("- `already_reviewed`: already linked to the reviewed v0 forest.")
    lines.append("")
    lines.append("## Priority Rows")
    lines.append("")
    for row in rows[:max_rows]:
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("caption_aliases") or [] if item.get("id"))
        lines.append(f"### {row.get('priority_rank')}. {row.get('family_id')}")
        lines.append("")
        lines.append(f"- recommendation: `{row.get('recommendation')}` / `{row.get('review_bucket')}`")
        lines.append(f"- dense_status: `{row.get('dense_status')}`")
        lines.append(f"- support: `{row.get('support_cases_sum')}`; motifs: `{row.get('motif_count')}`")
        lines.append(f"- operator: `{row.get('operator')}`")
        lines.append(f"- channels: `{row.get('channels')}`")
        lines.append(f"- relations: `{row.get('relations')}`")
        lines.append(f"- geometry: `{row.get('geometry_clusters')}`")
        lines.append(f"- caption aliases: {aliases or 'none'}")
        lines.append(f"- reason: {row.get('review_reason')}")
        examples = row.get("examples") or []
        if examples:
            lines.append("- examples:")
            for example in examples[:3]:
                lines.append(
                    f"  - `{example.get('case_id')}` span={example.get('span')} "
                    f"motif={example.get('motif_id')}: {example.get('caption')}"
                )
        lines.append("")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows in JSON/CSV.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dense_forest_path = Path(args.dense_forest)
    motif_audit_path = Path(args.motif_audit)
    dense_forest = _read_json(dense_forest_path)
    motif_audit = _read_json(motif_audit_path)
    payload = build_review_table(
        dense_forest,
        motif_audit,
        int(args.max_examples),
        dense_forest_path=dense_forest_path,
        motif_audit_path=motif_audit_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "promotion_review_table.json", payload)
    _write_json(output_dir / "summary.json", payload.get("summary") or {})
    write_csv(output_dir / "promotion_review_table.csv", payload.get("review_rows") or [])
    write_markdown(output_dir / "promotion_review_table.md", payload, int(args.max_markdown_rows))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-forest", type=Path, default=DEFAULT_DENSE_FOREST)
    parser.add_argument("--motif-audit", type=Path, default=DEFAULT_MOTIF_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-markdown-rows", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload.get("summary") or {}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
