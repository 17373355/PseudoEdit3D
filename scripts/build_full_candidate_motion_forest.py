from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_MOTIF_TIERS = Path("outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json")
DEFAULT_CASE_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/full_candidate_motion_forest_v1")


DEFAULT_STRUCTURAL_TIERS = [
    "named_motion_candidate",
    "motion_stable_unnamed",
]

STRUCTURAL_TIER_SET = set(DEFAULT_STRUCTURAL_TIERS)
LEGACY_DIAGNOSTIC_TIER = "legacy_aligned_diagnostic"


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


def _safe_motif_id(motif_id: str) -> str:
    return "motif_" + motif_id.strip("<>").replace("/", "_").replace("|", "_")


def _top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _counter_from_items(items: list[dict[str, Any]], *, skip_no_marker: bool = True) -> Counter[str]:
    out: Counter[str] = Counter()
    for item in items:
        key = str(item.get("id") or "")
        if not key:
            continue
        if skip_no_marker and key.startswith("__"):
            continue
        out[key] += int(item.get("count") or 0)
    return out


def _family_status(tier_counter: Counter[str]) -> str:
    structural_count = sum(count for tier, count in tier_counter.items() if tier in STRUCTURAL_TIER_SET)
    diagnostic_count = sum(count for tier, count in tier_counter.items() if tier not in STRUCTURAL_TIER_SET)
    if structural_count and diagnostic_count:
        return "mixed_family"
    if structural_count:
        return "candidate_family"
    return "diagnostic_family"


def _load_case_coverage(path: Path, motif_ids: set[str]) -> tuple[dict[str, set[str]], int]:
    coverage: dict[str, set[str]] = {motif_id: set() for motif_id in motif_ids}
    total_cases = 0
    if not path.exists():
        return coverage, total_cases
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_cases += 1
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            seen_in_case: set[str] = set()
            for token in row.get("bpe_tokens") or []:
                symbol = str(token.get("symbol") or "")
                if symbol in motif_ids:
                    seen_in_case.add(symbol)
            for motif_id in seen_in_case:
                coverage[motif_id].add(case_id)
    return coverage, total_cases


def _tier_list(args: argparse.Namespace) -> list[str]:
    if args.tiers:
        return [item.strip() for item in args.tiers.split(",") if item.strip()]
    tiers = list(DEFAULT_STRUCTURAL_TIERS)
    if args.include_legacy_diagnostic:
        tiers.append(LEGACY_DIAGNOSTIC_TIER)
    return tiers


def _motif_node(
    motif: dict[str, Any],
    required_clusters: list[str],
    *,
    coverage_cases: set[str],
    total_case_count: int,
) -> dict[str, Any]:
    motif_id = str(motif.get("motif_id") or "")
    return {
        "node_id": _safe_motif_id(motif_id),
        "motif_id": motif_id,
        "node_kind": "motif_leaf",
        "tier": motif.get("tier"),
        "status": "candidate" if motif.get("tier") in STRUCTURAL_TIER_SET else "diagnostic",
        "support": {
            "support_cases_reported": int(motif.get("support_cases") or 0),
            "occurrences_reported": int(motif.get("occurrences") or 0),
            "unique_case_coverage": len(coverage_cases),
            "coverage_ratio": round(len(coverage_cases) / max(total_case_count, 1), 6),
        },
        "motion_evidence": {
            "required_geometry_clusters": required_clusters,
            "top_geometry_clusters": motif.get("top_geometry_clusters") or [],
            "parents": motif.get("parents") or [],
            "dominant_super_families": sorted({_super_family(cluster) for cluster in required_clusters}),
        },
        "naming_diagnostics": {
            "top_caption_alias": motif.get("top_caption_alias") or "",
            "caption_alias_purity": motif.get("caption_alias_purity"),
            "top_caption_aliases": motif.get("top_caption_aliases") or [],
            "top_caption_keywords": motif.get("top_caption_keywords") or [],
            "policy": "diagnostic only; not used to create forest edges",
        },
        "legacy_diagnostics": {
            "top_tree_family": motif.get("top_tree_family") or "",
            "tree_family_purity": motif.get("tree_family_purity"),
            "dominant_status": motif.get("dominant_status") or "",
            "policy": "old tree is diagnostic only",
        },
        "example_occurrences": motif.get("example_occurrences") or [],
    }


def _family_node(
    family_id: str,
    family_key: str,
    required_clusters: list[str],
    motifs: list[dict[str, Any]],
    motif_case_coverage: dict[str, set[str]],
    *,
    total_case_count: int,
) -> dict[str, Any]:
    alias_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    old_family_counter: Counter[str] = Counter()
    cluster_counter: Counter[str] = Counter()
    tier_counter: Counter[str] = Counter()
    case_ids: set[str] = set()
    support_sum = 0
    occurrence_sum = 0
    for motif in motifs:
        motif_id = str(motif.get("motif_id") or "")
        tier_counter[str(motif.get("tier") or "")] += 1
        support_sum += int(motif.get("support_cases") or 0)
        occurrence_sum += int(motif.get("occurrences") or 0)
        case_ids.update(motif_case_coverage.get(motif_id, set()))
        alias_counter.update(_counter_from_items(motif.get("top_caption_aliases") or []))
        keyword_counter.update(_counter_from_items(motif.get("top_caption_keywords") or []))
        old_family = str(motif.get("top_tree_family") or "")
        if old_family:
            old_family_counter[old_family] += int(motif.get("support_cases") or 0)
        cluster_counter.update(_counter_from_items(motif.get("top_geometry_clusters") or [], skip_no_marker=False))
    return {
        "node_id": family_id,
        "node_kind": "geometry_family",
        "status": _family_status(tier_counter),
        "motion_family_key": family_key,
        "tier_counts": dict(sorted(tier_counter.items())),
        "support": {
            "motif_count": len(motifs),
            "support_cases_sum": support_sum,
            "occurrences_sum": occurrence_sum,
            "unique_case_coverage": len(case_ids),
            "coverage_ratio": round(len(case_ids) / max(total_case_count, 1), 6),
            "support_count_policy": "support_cases_sum can double-count cases across motifs; unique_case_coverage scans BPE sequences",
        },
        "motion_evidence": {
            "required_geometry_clusters": required_clusters,
            "dominant_super_families": sorted({_super_family(cluster) for cluster in required_clusters}),
            "top_geometry_clusters": _top_counter(cluster_counter, 12),
        },
        "naming_diagnostics": {
            "top_caption_aliases": _top_counter(alias_counter, 10),
            "top_caption_keywords": _top_counter(keyword_counter, 10),
            "policy": "diagnostic only; not used to create forest edges",
        },
        "legacy_diagnostics": {
            "top_tree_families": _top_counter(old_family_counter, 10),
            "policy": "old tree is diagnostic only",
        },
    }


def build_forest(
    motif_tiers: dict[str, Any],
    *,
    selected_tiers: list[str],
    case_bpe_sequences: Path,
    min_relative_cluster_count: float,
    max_required_clusters: int,
) -> dict[str, Any]:
    motifs = [motif for motif in motif_tiers.get("motifs") or [] if str(motif.get("tier") or "") in selected_tiers]
    motif_ids = {str(motif.get("motif_id") or "") for motif in motifs if str(motif.get("motif_id") or "")}
    motif_case_coverage, total_case_count = _load_case_coverage(case_bpe_sequences, motif_ids)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_required: dict[str, list[str]] = {}
    for motif in motifs:
        required = _required_clusters(
            motif.get("top_geometry_clusters") or [],
            min_relative_count=min_relative_cluster_count,
            max_clusters=max_required_clusters,
        )
        key = _family_key(required)
        grouped[key].append(motif)
        group_required[key] = required

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -sum(int(motif.get("support_cases") or 0) for motif in item[1]),
            item[0],
        ),
    )
    family_id_by_key: dict[str, str] = {
        key: f"geometry_family_{idx:04d}"
        for idx, (key, _) in enumerate(sorted_groups, start=1)
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for key, rows in sorted_groups:
        family_id = family_id_by_key[key]
        rows = sorted(rows, key=lambda motif: (-int(motif.get("support_cases") or 0), str(motif.get("motif_id") or "")))
        nodes.append(
            _family_node(
                family_id,
                key,
                group_required[key],
                rows,
                motif_case_coverage,
                total_case_count=total_case_count,
            )
        )
        for motif in rows:
            motif_id = str(motif.get("motif_id") or "")
            nodes.append(
                _motif_node(
                    motif,
                    group_required[key],
                    coverage_cases=motif_case_coverage.get(motif_id, set()),
                    total_case_count=total_case_count,
                )
            )
            edges.append(
                {
                    "parent_node_id": family_id,
                    "child_node_id": _safe_motif_id(motif_id),
                    "edge_type": "motif_member",
                    "policy": "family membership is determined by required geometry cluster set",
                }
            )

    family_count = len(sorted_groups)
    motif_count = len(motifs)
    covered_cases = set()
    for cases in motif_case_coverage.values():
        covered_cases.update(cases)
    tier_counts = Counter(str(motif.get("tier") or "") for motif in motifs)
    return {
        "schema_version": "full_candidate_motion_forest_v1",
        "runtime_policy": "offline candidate forest only; not the runtime AML tree",
        "forest_policy": "geometry-family roots with BPE motif leaves; multiple roots are expected",
        "summary": {
            "selected_tiers": selected_tiers,
            "source_motif_count": len(motif_tiers.get("motifs") or []),
            "included_motif_count": motif_count,
            "geometry_family_count": family_count,
            "node_count": family_count + motif_count,
            "edge_count": len(edges),
            "root_count": family_count,
            "max_depth": 1 if edges else 0,
            "total_case_count": total_case_count,
            "unique_case_coverage": len(covered_cases),
            "coverage_ratio": round(len(covered_cases) / max(total_case_count, 1), 6),
            "tier_counts": dict(sorted(tier_counts.items())),
            "support_count_policy": "coverage scans case_bpe_sequences; support sums may double-count cases",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _children_by_parent(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        out[str(edge.get("parent_node_id") or "")].append(str(edge.get("child_node_id") or ""))
    for children in out.values():
        children.sort()
    return out


def write_tree_view(path: Path, forest: dict[str, Any]) -> None:
    nodes = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children_by_parent = _children_by_parent(forest.get("edges") or [])
    family_nodes = [
        node for node in forest.get("nodes") or []
        if node.get("node_kind") == "geometry_family"
    ]
    family_nodes.sort(key=lambda node: (-int(node.get("support", {}).get("support_cases_sum") or 0), str(node.get("node_id") or "")))
    lines: list[str] = []
    for family in family_nodes:
        support = family.get("support", {})
        clusters = " + ".join(str(item).split("/", 1)[-1] for item in family.get("motion_evidence", {}).get("required_geometry_clusters") or [])
        lines.append(
            f"- {family['node_id']} [{family.get('status')}] motifs={support.get('motif_count')} coverage={support.get('unique_case_coverage')} support_sum={support.get('support_cases_sum')} :: {clusters}"
        )
        child_ids = children_by_parent.get(str(family.get("node_id") or ""), [])
        child_nodes = [nodes[child_id] for child_id in child_ids if child_id in nodes]
        child_nodes.sort(key=lambda node: (-int(node.get("support", {}).get("support_cases_reported") or 0), str(node.get("node_id") or "")))
        for child in child_nodes[:12]:
            child_support = child.get("support", {})
            alias = child.get("naming_diagnostics", {}).get("top_caption_alias") or ""
            legacy = child.get("legacy_diagnostics", {}).get("top_tree_family") or ""
            lines.append(
                f"  - {child.get('motif_id')} [{child.get('tier')}] support={child_support.get('support_cases_reported')} coverage={child_support.get('unique_case_coverage')} alias={alias or 'none'} old={legacy or 'none'}"
            )
        if len(child_nodes) > 12:
            lines.append(f"  - ... {len(child_nodes) - 12} more motif leaves")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, forest: dict[str, Any]) -> None:
    fields = [
        "node_id",
        "node_kind",
        "tier",
        "status",
        "support_cases",
        "unique_case_coverage",
        "coverage_ratio",
        "motif_count",
        "required_geometry_clusters",
        "top_caption_alias",
        "top_tree_family",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for node in forest.get("nodes") or []:
            support = node.get("support") or {}
            geometry = node.get("motion_evidence", {}).get("required_geometry_clusters") or []
            writer.writerow(
                {
                    "node_id": node.get("node_id"),
                    "node_kind": node.get("node_kind"),
                    "tier": node.get("tier") or "",
                    "status": node.get("status"),
                    "support_cases": support.get("support_cases_reported") or support.get("support_cases_sum") or "",
                    "unique_case_coverage": support.get("unique_case_coverage") or "",
                    "coverage_ratio": support.get("coverage_ratio") or "",
                    "motif_count": support.get("motif_count") or "",
                    "required_geometry_clusters": "; ".join(geometry),
                    "top_caption_alias": node.get("naming_diagnostics", {}).get("top_caption_alias") or "",
                    "top_tree_family": node.get("legacy_diagnostics", {}).get("top_tree_family") or "",
                }
            )


def write_report(path: Path, forest: dict[str, Any]) -> None:
    nodes = forest.get("nodes") or []
    family_nodes = [node for node in nodes if node.get("node_kind") == "geometry_family"]
    family_nodes.sort(key=lambda node: (-int(node.get("support", {}).get("support_cases_sum") or 0), str(node.get("node_id") or "")))
    lines: list[str] = []
    lines.append("# Full Candidate Motion Forest V1")
    lines.append("")
    lines.append("This is an offline forest of geometry-family roots and BPE motif leaves.")
    lines.append("Caption aliases and old-tree families are diagnostics only.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (forest.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Geometry Families")
    lines.append("")
    lines.append("| family | status | motifs | coverage | support sum | geometry | top aliases | old-tree families |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for family in family_nodes:
        support = family.get("support") or {}
        geometry = "<br>".join(family.get("motion_evidence", {}).get("required_geometry_clusters") or [])
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in family.get("naming_diagnostics", {}).get("top_caption_aliases", [])[:4])
        legacy = ", ".join(f"{item['id']}:{item['count']}" for item in family.get("legacy_diagnostics", {}).get("top_tree_families", [])[:4])
        lines.append(
            f"| `{family.get('node_id')}` | `{family.get('status')}` | {support.get('motif_count')} | {support.get('unique_case_coverage')} | {support.get('support_cases_sum')} | {geometry} | {aliases} | {legacy} |"
        )
    lines.append("")
    lines.append("## Top Motif Leaves")
    lines.append("")
    motif_nodes = [node for node in nodes if node.get("node_kind") == "motif_leaf"]
    motif_nodes.sort(key=lambda node: (-int(node.get("support", {}).get("support_cases_reported") or 0), str(node.get("node_id") or "")))
    lines.append("| motif | tier | support | coverage | geometry | alias | old-tree family |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
    for node in motif_nodes[:80]:
        support = node.get("support") or {}
        geometry = "<br>".join(node.get("motion_evidence", {}).get("required_geometry_clusters") or [])
        alias = node.get("naming_diagnostics", {}).get("top_caption_alias") or ""
        legacy = node.get("legacy_diagnostics", {}).get("top_tree_family") or ""
        lines.append(
            f"| `{node.get('motif_id')}` | `{node.get('tier')}` | {support.get('support_cases_reported')} | {support.get('unique_case_coverage')} | {geometry} | `{alias}` | `{legacy}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full offline candidate motion forest from all Motion-BPE motif tiers.")
    parser.add_argument("--motif-tiers", default=str(DEFAULT_MOTIF_TIERS))
    parser.add_argument("--case-bpe-sequences", default=str(DEFAULT_CASE_BPE_SEQUENCES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tiers", default="")
    parser.add_argument("--include-legacy-diagnostic", action="store_true")
    parser.add_argument("--min-relative-cluster-count", type=float, default=0.50)
    parser.add_argument("--max-required-clusters", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tiers = _tier_list(args)
    forest = build_forest(
        _read_json(Path(args.motif_tiers)),
        selected_tiers=selected_tiers,
        case_bpe_sequences=Path(args.case_bpe_sequences),
        min_relative_cluster_count=float(args.min_relative_cluster_count),
        max_required_clusters=int(args.max_required_clusters),
    )
    forest["source"] = {
        "motif_tiers": str(args.motif_tiers),
        "case_bpe_sequences": str(args.case_bpe_sequences),
    }
    _write_json(output_dir / "full_candidate_motion_forest.json", forest)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "full_candidate_motion_forest_summary_v1",
            **forest["summary"],
            "source": forest["source"],
        },
    )
    write_tree_view(output_dir / "full_candidate_motion_forest_tree.txt", forest)
    write_report(output_dir / "full_candidate_motion_forest.md", forest)
    write_csv(output_dir / "full_candidate_motion_forest_nodes.csv", forest)
    print(output_dir)


if __name__ == "__main__":
    main()
