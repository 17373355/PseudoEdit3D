from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DRAFT = Path("outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl")
DEFAULT_NAMING_LAYER = Path("outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_split_planner_v1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _clusters(token: dict[str, Any]) -> list[str]:
    return [str(item) for item in token.get("clusters") or [] if str(item)]


def _span(token: dict[str, Any]) -> tuple[int, int]:
    span = token.get("span") or [0, 0]
    return int(span[0]), int(span[1])


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _symbol_cluster(symbol: str) -> str:
    return symbol.split("|", 1)[0]


def _context_bucket(cluster: str) -> str:
    if cluster.startswith("WHOLE_BODY_LOCOMOTION/"):
        return "locomotion_context"
    if cluster.startswith("WHOLE_BODY_VERTICAL/"):
        return "vertical_context"
    if cluster.startswith("TORSO_"):
        return "torso_context"
    if cluster.startswith("LEFT_LEG_ACTION/") or cluster.startswith("RIGHT_LEG_ACTION/"):
        return "leg_context"
    if cluster.startswith("LEFT_ARM_") or cluster.startswith("RIGHT_ARM_") or cluster.startswith("BIMANUAL_"):
        return "arm_context"
    if cluster.startswith("WHOLE_BODY_ROTATION/"):
        return "rotation_context"
    if cluster.startswith("WHOLE_BODY_STATE/"):
        return "state_context"
    return "other_context"


def _top(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _occurrences_for_motifs(sequence_rows: list[dict[str, Any]], motif_ids: set[str], *, neighbor_window: int) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for row in sequence_rows:
        toks = row.get("bpe_tokens") or []
        for idx, tok in enumerate(toks):
            if str(tok.get("symbol") or "") not in motif_ids:
                continue
            target_span = _span(tok)
            before: Counter[str] = Counter()
            after: Counter[str] = Counter()
            overlap_counter: Counter[str] = Counter()
            bucket_counter: Counter[str] = Counter()
            for jdx, ctx in enumerate(toks):
                if jdx == idx:
                    continue
                ctx_clusters = _clusters(ctx) or [_symbol_cluster(str(ctx.get("symbol") or ""))]
                ctx_span = _span(ctx)
                relation = ""
                if jdx < idx and idx - jdx <= neighbor_window:
                    relation = "before"
                    before.update(ctx_clusters)
                elif jdx > idx and jdx - idx <= neighbor_window:
                    relation = "after"
                    after.update(ctx_clusters)
                if _overlap(target_span, ctx_span) > 0:
                    overlap_counter.update(ctx_clusters)
                    relation = relation or "overlap"
                if relation:
                    for cluster in ctx_clusters:
                        bucket_counter[_context_bucket(cluster)] += 1
            target_clusters = _clusters(tok)
            signature_parts = []
            for name, counter in [
                ("before", before),
                ("overlap", overlap_counter),
                ("after", after),
            ]:
                top_ids = [item["id"] for item in _top(counter, 2)]
                if top_ids:
                    signature_parts.append(f"{name}=" + "+".join(top_ids))
            signature = " | ".join(signature_parts) or "no_salient_context"
            occurrences.append(
                {
                    "case_id": str(row.get("case_id") or ""),
                    "caption": str(row.get("caption") or ""),
                    "caption_alias_ids": row.get("caption_alias_ids") or [],
                    "motif_symbol": tok.get("symbol"),
                    "span": list(target_span),
                    "target_clusters": target_clusters,
                    "context_signature": signature,
                    "before_clusters": _top(before, 8),
                    "overlap_clusters": _top(overlap_counter, 8),
                    "after_clusters": _top(after, 8),
                    "context_buckets": _top(bucket_counter, 8),
                }
            )
    return occurrences


def _group_occurrences(occurrences: list[dict[str, Any]], *, min_group_support: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occ in occurrences:
        groups[str(occ.get("context_signature") or "no_salient_context")].append(occ)
    out: list[dict[str, Any]] = []
    for signature, rows in groups.items():
        if len(rows) < min_group_support:
            continue
        before = Counter()
        overlap = Counter()
        after = Counter()
        aliases = Counter()
        buckets = Counter()
        for row in rows:
            before.update({item["id"]: int(item["count"]) for item in row.get("before_clusters") or []})
            overlap.update({item["id"]: int(item["count"]) for item in row.get("overlap_clusters") or []})
            after.update({item["id"]: int(item["count"]) for item in row.get("after_clusters") or []})
            aliases.update(str(item) for item in row.get("caption_alias_ids") or [])
            buckets.update({item["id"]: int(item["count"]) for item in row.get("context_buckets") or []})
        out.append(
            {
                "context_signature": signature,
                "support_cases": len({row["case_id"] for row in rows}),
                "occurrences": len(rows),
                "top_before_clusters": _top(before, 8),
                "top_overlap_clusters": _top(overlap, 8),
                "top_after_clusters": _top(after, 8),
                "top_context_buckets": _top(buckets, 8),
                "caption_alias_diagnostics": _top(aliases, 8),
                "example_occurrences": rows[:8],
                "structural_policy": "candidate split group is defined by motion context signature; caption aliases are diagnostics only",
            }
        )
    return sorted(out, key=lambda item: (-int(item["support_cases"]), str(item["context_signature"])))


def _summarize_occurrence_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = Counter()
    overlap = Counter()
    after = Counter()
    aliases = Counter()
    buckets = Counter()
    for row in rows:
        before.update({item["id"]: int(item["count"]) for item in row.get("before_clusters") or []})
        overlap.update({item["id"]: int(item["count"]) for item in row.get("overlap_clusters") or []})
        after.update({item["id"]: int(item["count"]) for item in row.get("after_clusters") or []})
        aliases.update(str(item) for item in row.get("caption_alias_ids") or [])
        buckets.update({item["id"]: int(item["count"]) for item in row.get("context_buckets") or []})
    return {
        "support_cases": len({row["case_id"] for row in rows}),
        "occurrences": len(rows),
        "top_before_clusters": _top(before, 8),
        "top_overlap_clusters": _top(overlap, 8),
        "top_after_clusters": _top(after, 8),
        "top_context_buckets": _top(buckets, 8),
        "caption_alias_diagnostics": _top(aliases, 8),
        "example_occurrences": rows[:8],
    }


def _axis_group_candidates(occurrences: list[dict[str, Any]], *, min_group_support: int) -> list[dict[str, Any]]:
    relation_keys = [
        ("before", "before_clusters"),
        ("overlap", "overlap_clusters"),
        ("after", "after_clusters"),
    ]
    axis_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for occ in occurrences:
        for relation, key in relation_keys:
            seen: set[tuple[str, str, str]] = set()
            for item in occ.get(key) or []:
                cluster_id = str(item.get("id") or "")
                if not cluster_id:
                    continue
                axis_key = (relation, _context_bucket(cluster_id), cluster_id)
                if axis_key in seen:
                    continue
                axis_rows[axis_key].append(occ)
                seen.add(axis_key)

    by_axis: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (relation, bucket, cluster_id), rows in axis_rows.items():
        support = len({row["case_id"] for row in rows})
        if support < min_group_support:
            continue
        summary = _summarize_occurrence_group(rows)
        by_axis[(relation, bucket)].append(
            {
                "cluster_id": cluster_id,
                **summary,
                "structural_policy": "axis group is defined by motion relation, context bucket, and cluster id; caption aliases are diagnostics only",
            }
        )

    axes: list[dict[str, Any]] = []
    for (relation, bucket), groups in by_axis.items():
        groups = sorted(groups, key=lambda item: (-int(item["support_cases"]), str(item["cluster_id"])))
        axes.append(
            {
                "axis_id": f"{relation}/{bucket}",
                "relation": relation,
                "context_bucket": bucket,
                "group_count": len(groups),
                "total_group_support": sum(int(item["support_cases"]) for item in groups),
                "exclusive_partition": False,
                "recommended_use": "split_axis_candidate" if len(groups) >= 2 else "context_marker",
                "groups": groups[:12],
            }
        )
    return sorted(
        axes,
        key=lambda item: (
            0 if item["recommended_use"] == "split_axis_candidate" else 1,
            -int(item["group_count"]),
            str(item["axis_id"]),
        ),
    )


def _axis_summary(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axis_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for occ in occurrences:
        for key, out_key in [
            ("before_clusters", "before"),
            ("overlap_clusters", "overlap"),
            ("after_clusters", "after"),
            ("context_buckets", "bucket"),
        ]:
            for item in occ.get(key) or []:
                axis_counters[out_key][str(item["id"])] += 1
    return [
        {
            "axis": axis,
            "top_values": _top(counter, 16),
            "candidate_use": "split_axis_candidate" if len(counter) >= 2 else "weak_axis",
        }
        for axis, counter in sorted(axis_counters.items())
    ]


def _language_diagnostics_for_node(naming_layer: dict[str, Any], node_id: str) -> dict[str, Any]:
    for row in naming_layer.get("motion_node_labels") or []:
        if str(row.get("motion_node_id") or "") == node_id:
            return {
                "top_phrase_alignments": row.get("top_phrase_alignments") or [],
                "candidate_labels": row.get("candidate_labels") or [],
                "policy": "language diagnostics only; do not define split groups from these labels",
            }
    return {"top_phrase_alignments": [], "candidate_labels": [], "policy": "missing language diagnostics"}


def build_plan(
    draft: dict[str, Any],
    sequence_rows: list[dict[str, Any]],
    naming_layer: dict[str, Any],
    *,
    neighbor_window: int,
    min_group_support: int,
) -> dict[str, Any]:
    split_requests = draft.get("split_requests") or []
    plans: list[dict[str, Any]] = []
    for req in split_requests:
        node_id = str(req.get("source_node_id") or "")
        motif_ids = {str(item) for item in req.get("source_motif_ids") or [] if str(item)}
        occurrences = _occurrences_for_motifs(sequence_rows, motif_ids, neighbor_window=neighbor_window)
        groups = _group_occurrences(occurrences, min_group_support=min_group_support)
        axis_groups = _axis_group_candidates(occurrences, min_group_support=min_group_support)
        plans.append(
            {
                "source_node_id": node_id,
                "source_motif_ids": sorted(motif_ids),
                "source_structural_role": req.get("structural_role"),
                "required_geometry_clusters": req.get("required_geometry_clusters") or [],
                "occurrence_count": len(occurrences),
                "unique_case_count": len({row["case_id"] for row in occurrences}),
                "split_axes": _axis_summary(occurrences),
                "candidate_axis_groups": axis_groups,
                "candidate_split_groups": groups,
                "language_diagnostics": _language_diagnostics_for_node(naming_layer, node_id),
                "recommendation": (
                    "inspect_motion_axis_groups_before_split"
                    if axis_groups
                    else "inspect_candidate_groups_before_promotion"
                    if groups
                    else "insufficient_motion_context_groups"
                ),
                "policy": "split plan is motion-context driven; language is diagnostic only",
            }
        )
    return {
        "schema_version": "motion_split_planner_v1",
        "runtime_policy": "offline split planning only; no runtime AML tree mutation",
        "summary": {
            "split_request_count": len(split_requests),
            "planned_node_count": len(plans),
            "neighbor_window": neighbor_window,
            "min_group_support": min_group_support,
        },
        "split_plans": plans,
    }


def compact_axis_summary(payload: dict[str, Any], *, max_axes: int = 8, max_groups: int = 6) -> dict[str, Any]:
    node_summaries: list[dict[str, Any]] = []
    for plan in payload.get("split_plans") or []:
        axes: list[dict[str, Any]] = []
        for axis in plan.get("candidate_axis_groups") or []:
            if axis.get("recommended_use") != "split_axis_candidate":
                continue
            groups = [
                {
                    "cluster_id": group.get("cluster_id"),
                    "support_cases": group.get("support_cases"),
                    "occurrences": group.get("occurrences"),
                }
                for group in (axis.get("groups") or [])[:max_groups]
            ]
            axes.append(
                {
                    "axis_id": axis.get("axis_id"),
                    "relation": axis.get("relation"),
                    "context_bucket": axis.get("context_bucket"),
                    "group_count": axis.get("group_count"),
                    "top_groups": groups,
                }
            )
        node_summaries.append(
            {
                "source_node_id": plan.get("source_node_id"),
                "source_motif_ids": plan.get("source_motif_ids") or [],
                "occurrence_count": plan.get("occurrence_count"),
                "unique_case_count": plan.get("unique_case_count"),
                "recommendation": plan.get("recommendation"),
                "top_motion_split_axes": axes[:max_axes],
                "language_policy": "language diagnostics are excluded from compact structural summary",
            }
        )
    return {
        "schema_version": "motion_split_axis_summary_v1",
        "runtime_policy": payload.get("runtime_policy"),
        "summary": payload.get("summary") or {},
        "node_summaries": node_summaries,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Motion Split Planner V1")
    lines.append("")
    lines.append("This report proposes split axes for draft nodes marked `split_required`.")
    lines.append("Split groups are based on motion context signatures only; language is diagnostic.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (payload.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    for plan in payload.get("split_plans") or []:
        lines.append("")
        lines.append(f"## {plan['source_node_id']}")
        lines.append("")
        lines.append(f"- source motifs: `{', '.join(plan['source_motif_ids'])}`")
        lines.append(f"- occurrences: `{plan['occurrence_count']}`")
        lines.append(f"- unique cases: `{plan['unique_case_count']}`")
        lines.append(f"- recommendation: `{plan['recommendation']}`")
        lines.append("")
        lines.append("### Split Axes")
        lines.append("")
        lines.append("| axis | candidate use | top values |")
        lines.append("| --- | --- | --- |")
        for axis in plan.get("split_axes") or []:
            values = ", ".join(f"{item['id']}:{item['count']}" for item in axis.get("top_values", [])[:8])
            lines.append(f"| `{axis['axis']}` | `{axis['candidate_use']}` | {values} |")
        lines.append("")
        lines.append("### Candidate Split Groups")
        if not plan.get("candidate_split_groups"):
            lines.append("")
            lines.append("No groups passed the support threshold.")
        else:
            lines.append("")
            lines.append("| group | support | top buckets | top overlap | caption diagnostics |")
            lines.append("| --- | ---: | --- | --- | --- |")
            for group in plan.get("candidate_split_groups") or []:
                buckets = ", ".join(f"{item['id']}:{item['count']}" for item in group.get("top_context_buckets", [])[:4])
                overlap = ", ".join(f"{item['id']}:{item['count']}" for item in group.get("top_overlap_clusters", [])[:4])
                aliases = ", ".join(f"{item['id']}:{item['count']}" for item in group.get("caption_alias_diagnostics", [])[:4])
                lines.append(
                    f"| `{group['context_signature']}` | {group['support_cases']} | {buckets} | {overlap} | {aliases} |"
                )
        lines.append("")
        lines.append("### Motion Context Axis Groups")
        if not plan.get("candidate_axis_groups"):
            lines.append("")
            lines.append("No axis groups passed the support threshold.")
        else:
            for axis in plan.get("candidate_axis_groups") or []:
                lines.append("")
                lines.append(
                    f"#### `{axis['axis_id']}` ({axis['recommended_use']}, groups={axis['group_count']})"
                )
                lines.append("")
                lines.append("| cluster | support | top overlap | top after | caption diagnostics |")
                lines.append("| --- | ---: | --- | --- | --- |")
                for group in axis.get("groups") or []:
                    overlap = ", ".join(
                        f"{item['id']}:{item['count']}" for item in group.get("top_overlap_clusters", [])[:4]
                    )
                    after = ", ".join(
                        f"{item['id']}:{item['count']}" for item in group.get("top_after_clusters", [])[:4]
                    )
                    aliases = ", ".join(
                        f"{item['id']}:{item['count']}" for item in group.get("caption_alias_diagnostics", [])[:4]
                    )
                    lines.append(f"| `{group['cluster_id']}` | {group['support_cases']} | {overlap} | {after} | {aliases} |")
        lines.append("")
        lines.append("### Language Diagnostics")
        for phrase in (plan.get("language_diagnostics") or {}).get("top_phrase_alignments", [])[:10]:
            lines.append(
                f"- `{phrase.get('normalized_phrase')}` overlap={phrase.get('case_overlap')} score={phrase.get('score')}"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan motion-derived splits for draft tree nodes marked split_required.")
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--naming-layer", default=str(DEFAULT_NAMING_LAYER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--min-group-support", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_plan(
        _read_json(Path(args.draft)),
        _read_jsonl(Path(args.bpe_sequences)),
        _read_json(Path(args.naming_layer)),
        neighbor_window=int(args.neighbor_window),
        min_group_support=int(args.min_group_support),
    )
    payload["source"] = {
        "draft": str(args.draft),
        "bpe_sequences": str(args.bpe_sequences),
        "naming_layer": str(args.naming_layer),
    }
    _write_json(output_dir / "motion_split_plan.json", payload)
    _write_json(output_dir / "motion_split_axis_summary.json", compact_axis_summary(payload))
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "motion_split_planner_summary_v1",
            **payload["summary"],
            "compact_axis_summary": str(output_dir / "motion_split_axis_summary.json"),
            "source": payload["source"],
        },
    )
    write_report(output_dir / "motion_split_plan.md", payload)
    print(output_dir)


if __name__ == "__main__":
    main()
