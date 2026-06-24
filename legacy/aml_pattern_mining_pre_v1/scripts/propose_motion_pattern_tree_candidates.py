from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _cluster_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _cluster_count(item: dict[str, Any]) -> int:
    try:
        return int(item.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _required_clusters(
    clusters: list[dict[str, Any]],
    *,
    min_relative_count: float,
    max_clusters: int,
) -> list[str]:
    if not clusters:
        return []
    top_count = max(_cluster_count(item) for item in clusters)
    cutoff = max(1, int(round(top_count * min_relative_count)))
    kept = [
        _cluster_id(item)
        for item in sorted(clusters, key=lambda item: (-_cluster_count(item), _cluster_id(item)))
        if _cluster_id(item) and _cluster_count(item) >= cutoff
    ]
    if not kept:
        kept = [_cluster_id(max(clusters, key=_cluster_count))]
    return sorted(kept[:max_clusters])


def _family_key(clusters: list[str]) -> str:
    if not clusters:
        return "UNSPECIFIED_GEOMETRY"
    return " + ".join(clusters)


def _super_family(cluster_id: str) -> str:
    return cluster_id.split("/", 1)[0] if "/" in cluster_id else cluster_id


def _top_counter(counter: Counter[str], limit: int = 8) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _counter_from_items(items: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        key = str(item.get("id") or "")
        if key and not key.startswith("__"):
            counter[key] += int(item.get("count") or 0)
    return counter


def _dominant_status(row: dict[str, Any]) -> str:
    statuses = row.get("top_tree_statuses") or []
    if not statuses:
        return "unknown"
    return str(statuses[0].get("id") or "unknown")


def _dominant_cluster_ids(row: dict[str, Any], limit: int = 4) -> list[str]:
    return [
        str(item.get("id") or "")
        for item in (row.get("top_geometry_clusters") or [])[:limit]
        if str(item.get("id") or "")
    ]


def _is_generic_or_context_like(row: dict[str, Any]) -> bool:
    family = str(row.get("top_tree_family") or "")
    clusters = " ".join(_dominant_cluster_ids(row, limit=6))
    generic_family_markers = (
        "TRANSLATING_GAIT",
        "IN_PLACE_GAIT",
        "ROTATION_DOMINANT",
        "TURN_SEGMENT",
        "RECOVERY_STEP_SEGMENT",
        "TERMINAL_STILL",
        "STATIC_OR_SUBTLE_STATE_PROXY",
        "WHOLE_BODY_VERTICAL_MOTION_PROXY",
        "BIMANUAL_ARM_RAISE_SPREAD_PROXY",
    )
    if family in generic_family_markers:
        return True
    generic_cluster_markers = (
        "LOCO_",
        "TURN_",
        "VERT_",
        "REPEAT_LOCO",
        "REPEAT_ALT_LOCO",
    )
    return any(marker in clusters for marker in generic_cluster_markers)


def _motif_tier(row: dict[str, Any], *, stable_alias_ids: set[str]) -> str:
    motif_id = str(row.get("motif_id") or "")
    support = int(row.get("support_cases") or 0)
    alias_purity = float(row.get("caption_alias_purity") or 0.0)
    tree_purity = float(row.get("tree_family_purity") or 0.0)
    if motif_id in stable_alias_ids:
        return "named_motion_candidate"
    if support >= 80 and tree_purity >= 0.85 and not _is_generic_or_context_like(row):
        return "motion_stable_unnamed"
    if support >= 80 and tree_purity >= 0.70:
        return "legacy_aligned_diagnostic"
    if alias_purity > 0.0:
        return "language_weak_diagnostic"
    return "generic_or_low_purity"


def _compact_motif_row(row: dict[str, Any], tier: str) -> dict[str, Any]:
    return {
        "motif_id": row.get("motif_id"),
        "tier": tier,
        "support_cases": int(row.get("support_cases") or 0),
        "occurrences": int(row.get("occurrences") or 0),
        "caption_alias_purity": row.get("caption_alias_purity"),
        "top_caption_alias": row.get("top_caption_alias") or "",
        "tree_family_purity": row.get("tree_family_purity"),
        "top_tree_family": row.get("top_tree_family") or "",
        "dominant_status": _dominant_status(row),
        "top_geometry_clusters": row.get("top_geometry_clusters") or [],
        "top_caption_aliases": row.get("top_caption_aliases") or [],
        "top_caption_keywords": row.get("top_caption_keywords") or [],
        "parents": row.get("parents") or [],
        "example_occurrences": row.get("example_occurrences") or [],
    }


def _weighted_mean(rows: list[dict[str, Any]], key: str, weight_key: str = "support_cases") -> float:
    total_weight = 0
    total = 0.0
    for row in rows:
        weight = int(row.get(weight_key) or 0)
        total += float(row.get(key) or 0.0) * weight
        total_weight += weight
    return total / max(1, total_weight)


def _example_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for row in rows:
        for example in row.get("example_occurrences") or []:
            span = example.get("span") or []
            span_key = tuple(int(item) for item in span[:2]) if len(span) >= 2 else (-1, -1)
            key = (str(example.get("case_id") or ""), span_key)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "case_id": example.get("case_id"),
                    "span": example.get("span"),
                    "event_indices": example.get("event_indices"),
                    "caption": example.get("caption"),
                    "caption_alias_ids": example.get("caption_alias_ids") or [],
                    "linked_families": example.get("linked_families") or [],
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _node_recommendation(node: dict[str, Any]) -> str:
    support = int(node["support_cases"])
    motif_count = len(node["source_motifs"])
    alias_purity = float(node["naming_evidence"]["weighted_caption_alias_purity"])
    required_clusters = node["motion_evidence"]["required_geometry_clusters"]
    if support >= 120 and len(required_clusters) >= 2 and alias_purity >= 0.50:
        return "inspect_as_composite_motion_tree_node"
    if support >= 80 and motif_count >= 2 and alias_purity >= 0.45:
        return "inspect_as_motif_family_node"
    if support >= 40 and alias_purity >= 0.55:
        return "inspect_as_named_leaf_or_component"
    return "keep_as_diagnostic_candidate_until_motion_split_improves"


def build_nodes(
    candidates: list[dict[str, Any]],
    motif_rows_by_id: dict[str, dict[str, Any]],
    *,
    min_relative_cluster_count: float,
    max_required_clusters: int,
    examples_per_node: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_clusters: dict[str, list[str]] = {}
    for candidate in candidates:
        motif_id = str(candidate.get("motif_id") or "")
        motif_row = motif_rows_by_id.get(motif_id, {})
        top_clusters = motif_row.get("top_geometry_clusters") or candidate.get("top_geometry_clusters") or []
        required = _required_clusters(
            top_clusters,
            min_relative_count=min_relative_cluster_count,
            max_clusters=max_required_clusters,
        )
        key = _family_key(required)
        merged = {**motif_row, **candidate, "required_geometry_clusters": required}
        grouped[key].append(merged)
        group_clusters[key] = required

    nodes: list[dict[str, Any]] = []
    for idx, (key, rows) in enumerate(sorted(grouped.items(), key=lambda item: (-sum(int(r.get("support_cases") or 0) for r in item[1]), item[0])), start=1):
        required = group_clusters[key]
        alias_counter: Counter[str] = Counter()
        old_family_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        geometry_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        base_symbol_counter: Counter[str] = Counter()
        support_cases = 0
        occurrence_count = 0
        source_motifs: list[dict[str, Any]] = []
        for row in rows:
            support = int(row.get("support_cases") or 0)
            support_cases += support
            occurrence_count += int(row.get("occurrences") or support)
            alias = str(row.get("stable_caption_alias") or row.get("top_caption_alias") or "")
            if alias:
                alias_counter[alias] += support
            for item in row.get("top_caption_aliases") or []:
                alias_id = str(item.get("id") or "")
                if alias_id and not alias_id.startswith("__"):
                    alias_counter[alias_id] += int(item.get("count") or 0)
            family = str(row.get("top_tree_family") or "")
            if family:
                old_family_counter[family] += support
            for item in row.get("top_tree_families") or []:
                family_id = str(item.get("id") or "")
                if family_id and not family_id.startswith("__"):
                    old_family_counter[family_id] += int(item.get("count") or 0)
            for item in row.get("top_tree_statuses") or []:
                status_id = str(item.get("id") or "")
                if status_id and not status_id.startswith("__"):
                    status_counter[status_id] += int(item.get("count") or 0)
            for item in row.get("top_geometry_clusters") or []:
                cluster = str(item.get("id") or "")
                if cluster:
                    geometry_counter[cluster] += int(item.get("count") or 0)
            for item in row.get("top_caption_keywords") or []:
                keyword = str(item.get("id") or "")
                if keyword and not keyword.startswith("__"):
                    keyword_counter[keyword] += int(item.get("count") or 0)
            for item in row.get("top_base_symbols") or []:
                symbol = str(item.get("id") or "")
                if symbol:
                    base_symbol_counter[symbol] += int(item.get("count") or 0)
            source_motifs.append(
                {
                    "motif_id": row.get("motif_id"),
                    "support_cases": support,
                    "occurrences": int(row.get("occurrences") or support),
                    "caption_alias": alias,
                    "caption_alias_purity": row.get("caption_alias_purity"),
                    "top_tree_family": row.get("top_tree_family"),
                    "tree_family_purity": row.get("tree_family_purity"),
                    "parents": row.get("parents") or [],
                }
            )

        node = {
            "node_id": f"motion_node_{idx:04d}",
            "schema_version": "motion_pattern_tree_candidate_node_v1",
            "status": "offline_candidate",
            "runtime_policy": "do_not_use_as_runtime_tree_without_manual_promotion",
            "motion_family_key": key,
            "motion_evidence": {
                "required_geometry_clusters": required,
                "dominant_super_families": sorted({_super_family(cluster) for cluster in required if cluster}),
                "top_geometry_clusters": _top_counter(geometry_counter, 12),
                "top_base_symbols": _top_counter(base_symbol_counter, 12),
            },
            "naming_evidence": {
                "top_caption_aliases": _top_counter(alias_counter, 8),
                "top_caption_keywords": _top_counter(keyword_counter, 8),
                "weighted_caption_alias_purity": round(_weighted_mean(rows, "caption_alias_purity"), 4),
                "policy": "language names this motion candidate only after motion evidence is accepted",
            },
            "legacy_alignment": {
                "top_old_tree_families": _top_counter(old_family_counter, 8),
                "top_old_tree_statuses": _top_counter(status_counter, 8),
                "weighted_old_tree_family_purity": round(_weighted_mean(rows, "tree_family_purity"), 4),
                "policy": "old tree is an evaluator and bootstrap reference, not authority",
            },
            "support_cases": support_cases,
            "support_count_policy": "sum_of_source_motif_supports; cases may overlap across motifs",
            "occurrences": occurrence_count,
            "occurrence_count_policy": "sum_of_source_motif_occurrences",
            "source_motifs": sorted(source_motifs, key=lambda item: (-int(item["support_cases"]), str(item["motif_id"]))),
            "example_occurrences": _example_rows(rows, examples_per_node),
        }
        node["recommendation"] = _node_recommendation(node)
        nodes.append(node)
    return nodes


def build_motif_tiers(
    motif_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    examples_per_tier: int,
) -> dict[str, Any]:
    stable_alias_ids = {str(row.get("motif_id") or "") for row in candidates}
    tiered_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_compact_rows: list[dict[str, Any]] = []
    for row in motif_rows:
        tier = _motif_tier(row, stable_alias_ids=stable_alias_ids)
        compact = _compact_motif_row(row, tier)
        tiered_rows[tier].append(compact)
        all_compact_rows.append(compact)

    tier_order = [
        "named_motion_candidate",
        "motion_stable_unnamed",
        "legacy_aligned_diagnostic",
        "language_weak_diagnostic",
        "generic_or_low_purity",
    ]
    summaries: list[dict[str, Any]] = []
    for tier in tier_order:
        rows = sorted(
            tiered_rows.get(tier, []),
            key=lambda item: (
                -int(item["support_cases"]),
                -float(item.get("tree_family_purity") or 0.0),
                str(item["motif_id"]),
            ),
        )
        alias_counter: Counter[str] = Counter()
        family_counter: Counter[str] = Counter()
        cluster_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        for row in rows:
            alias_counter.update(_counter_from_items(row.get("top_caption_aliases") or []))
            keyword_counter.update(_counter_from_items(row.get("top_caption_keywords") or []))
            family = str(row.get("top_tree_family") or "")
            if family:
                family_counter[family] += int(row.get("support_cases") or 0)
            status = str(row.get("dominant_status") or "")
            if status:
                status_counter[status] += int(row.get("support_cases") or 0)
            cluster_counter.update(_counter_from_items(row.get("top_geometry_clusters") or []))
        summaries.append(
            {
                "tier": tier,
                "motif_count": len(rows),
                "support_cases_sum": sum(int(row.get("support_cases") or 0) for row in rows),
                "support_count_policy": "sum_of_motif_supports; cases may overlap across motifs",
                "top_caption_aliases": _top_counter(alias_counter, 8),
                "top_caption_keywords": _top_counter(keyword_counter, 8),
                "top_old_tree_families": _top_counter(family_counter, 8),
                "top_geometry_clusters": _top_counter(cluster_counter, 12),
                "top_tree_statuses": _top_counter(status_counter, 8),
                "top_motifs": rows[:examples_per_tier],
            }
        )
    return {
        "schema_version": "motion_bpe_motif_tiers_v1",
        "tier_policy": {
            "named_motion_candidate": "stable caption alias candidate already selected by the audit",
            "motion_stable_unnamed": "high-support motif with strong old-tree/geometry alignment, excluding generic locomotion/context patterns",
            "legacy_aligned_diagnostic": "high-support motif aligned to old tree but likely generic, context-only, or too broad for promotion",
            "language_weak_diagnostic": "motif has weak language signal but not enough purity for naming",
            "generic_or_low_purity": "remaining motif; still kept for corpus diagnostics",
        },
        "summary": summaries,
        "motifs": sorted(
            all_compact_rows,
            key=lambda item: (
                tier_order.index(str(item["tier"])) if str(item["tier"]) in tier_order else 999,
                -int(item["support_cases"]),
                str(item["motif_id"]),
            ),
        ),
    }


def write_report(path: Path, summary: dict[str, Any], nodes: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Motion-Corpus Pattern Tree Candidates")
    lines.append("")
    lines.append("This report is offline evidence for inducing a motion-derived pattern tree.")
    lines.append("It does not change the runtime AML tree.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        if key == "source":
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Candidate Nodes")
    lines.append("")
    lines.append("`support` is the sum of source motif supports. Cases may overlap when a node has multiple source motifs.")
    lines.append("")
    lines.append("| node | support | motifs | required geometry | naming evidence | old-tree alignment | recommendation |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- |")
    for node in nodes:
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in node["naming_evidence"]["top_caption_aliases"][:3]) or "unnamed"
        old = ", ".join(f"{item['id']}:{item['count']}" for item in node["legacy_alignment"]["top_old_tree_families"][:3]) or "unlinked"
        clusters = "<br>".join(node["motion_evidence"]["required_geometry_clusters"]) or "unspecified"
        lines.append(
            "| {node_id} | {support} | {motifs} | {clusters} | {aliases} | {old} | {rec} |".format(
                node_id=node["node_id"],
                support=node["support_cases"],
                motifs=len(node["source_motifs"]),
                clusters=clusters,
                aliases=aliases,
                old=old,
                rec=node["recommendation"],
            )
        )
    lines.append("")
    lines.append("## Node Details")
    for node in nodes:
        lines.append("")
        lines.append(f"### {node['node_id']}")
        lines.append("")
        lines.append(f"- motion family key: `{node['motion_family_key']}`")
        lines.append(f"- recommendation: `{node['recommendation']}`")
        lines.append(f"- support cases: `{node['support_cases']}`")
        lines.append(f"- support count policy: `{node['support_count_policy']}`")
        lines.append(f"- occurrences: `{node['occurrences']}`")
        lines.append(f"- weighted caption-alias purity: `{node['naming_evidence']['weighted_caption_alias_purity']}`")
        lines.append(f"- weighted old-tree family purity: `{node['legacy_alignment']['weighted_old_tree_family_purity']}`")
        lines.append("- source motifs:")
        for motif in node["source_motifs"]:
            lines.append(
                "  - `{motif_id}` support={support} alias=`{alias}` alias_purity={alias_purity} old_family=`{family}`".format(
                    motif_id=motif["motif_id"],
                    support=motif["support_cases"],
                    alias=motif.get("caption_alias") or "",
                    alias_purity=motif.get("caption_alias_purity"),
                    family=motif.get("top_tree_family") or "",
                )
            )
        if node["example_occurrences"]:
            lines.append("- examples:")
            for example in node["example_occurrences"][:5]:
                caption = str(example.get("caption") or "").replace("|", "\\|")
                lines.append(
                    f"  - case `{example.get('case_id')}`, span `{example.get('span')}`, caption: {caption}"
                )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_motif_tier_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Motion-BPE Motif Tiers")
    lines.append("")
    lines.append("This report explains where all learned BPE motifs went after the conservative named-node filter.")
    lines.append("It is diagnostic only and does not change the runtime AML tree.")
    lines.append("")
    lines.append("## Tier Policy")
    lines.append("")
    for tier, policy in (payload.get("tier_policy") or {}).items():
        lines.append(f"- `{tier}`: {policy}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| tier | motifs | support sum | top old-tree families | top geometry clusters | top language aliases |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for row in payload.get("summary") or []:
        families = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("top_old_tree_families", [])[:3]) or "none"
        clusters = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("top_geometry_clusters", [])[:3]) or "none"
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("top_caption_aliases", [])[:3]) or "none"
        lines.append(
            f"| `{row['tier']}` | {row['motif_count']} | {row['support_cases_sum']} | {families} | {clusters} | {aliases} |"
        )
    lines.append("")
    lines.append("## Top Motifs By Tier")
    for row in payload.get("summary") or []:
        lines.append("")
        lines.append(f"### {row['tier']}")
        lines.append("")
        lines.append("| motif | support | old-tree family | tree purity | alias | alias purity | geometry |")
        lines.append("| --- | ---: | --- | ---: | --- | ---: | --- |")
        for motif in row.get("top_motifs") or []:
            clusters = "<br>".join(_dominant_cluster_ids(motif, limit=3)) or "none"
            lines.append(
                "| `{motif}` | {support} | `{family}` | {tree_purity} | `{alias}` | {alias_purity} | {clusters} |".format(
                    motif=motif.get("motif_id"),
                    support=motif.get("support_cases"),
                    family=motif.get("top_tree_family") or "",
                    tree_purity=motif.get("tree_family_purity"),
                    alias=motif.get("top_caption_alias") or "",
                    alias_purity=motif.get("caption_alias_purity"),
                    clusters=clusters,
                )
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose offline motion-pattern-tree candidates from Layer3 event-BPE audit artifacts.")
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-relative-cluster-count", type=float, default=0.50)
    parser.add_argument("--max-required-clusters", type=int, default=4)
    parser.add_argument("--examples-per-node", type=int, default=8)
    parser.add_argument("--examples-per-tier", type=int, default=20)
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_payload = _read_json(audit_dir / "bpe_phrase_to_pattern_tree_candidates.json")
    motif_payload = _read_json(audit_dir / "bpe_motif_audit.json")
    summary = dict(candidate_payload.get("summary") or motif_payload.get("summary") or {})
    candidates = list(candidate_payload.get("candidates") or [])
    motif_rows = list(motif_payload.get("motifs") or [])
    motif_rows_by_id = {str(row.get("motif_id") or ""): row for row in motif_rows}

    nodes = build_nodes(
        candidates,
        motif_rows_by_id,
        min_relative_cluster_count=float(args.min_relative_cluster_count),
        max_required_clusters=int(args.max_required_clusters),
        examples_per_node=int(args.examples_per_node),
    )
    motif_tiers = build_motif_tiers(
        motif_rows,
        candidates,
        examples_per_tier=int(args.examples_per_tier),
    )

    output_summary = {
        "schema_version": "motion_pattern_tree_candidate_summary_v1",
        "source": {
            "audit_dir": str(audit_dir),
            "candidate_source": str(audit_dir / "bpe_phrase_to_pattern_tree_candidates.json"),
            "motif_audit_source": str(audit_dir / "bpe_motif_audit.json"),
        },
        "input_records": summary.get("num_records"),
        "input_original_token_count": summary.get("original_token_count"),
        "input_bpe_token_count": summary.get("bpe_token_count"),
        "input_compression_ratio": summary.get("compression_ratio"),
        "input_num_merges": summary.get("num_merges"),
        "input_stable_candidate_count": len(candidates),
        "candidate_node_count": len(nodes),
        "motif_count": len(motif_rows),
        "motif_tier_counts": {
            row["tier"]: row["motif_count"]
            for row in motif_tiers.get("summary", [])
        },
        "runtime_policy": "offline proposal only; no runtime AML tree mutation",
        "principle": "motion clusters and motion-BPE decide structure; text aliases and WordNet only name accepted motion nodes",
    }
    artifact = {
        "schema_version": "motion_pattern_tree_candidates_v1",
        "summary": output_summary,
        "candidate_nodes": nodes,
    }
    _write_json(output_dir / "summary.json", output_summary)
    _write_json(output_dir / "motion_pattern_tree_candidates.json", artifact)
    _write_json(output_dir / "motion_bpe_motif_tiers.json", motif_tiers)
    write_report(output_dir / "motion_pattern_tree_candidate_report.md", output_summary, nodes)
    write_motif_tier_report(output_dir / "motion_bpe_motif_tiers.md", motif_tiers)
    print(output_dir)


if __name__ == "__main__":
    main()
