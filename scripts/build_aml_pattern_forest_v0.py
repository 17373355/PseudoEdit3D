"""Build the first reviewed AML motion pattern forest.

The script is offline artifact generation only. It reads a review policy JSON
that decides which audited motion symbols may enter the v0 forest; Python only
collects evidence and writes reviewable outputs.

Example:
    python scripts/build_aml_pattern_forest_v0.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_POLICY = Path("configs/aml_pattern_forest_v0_review_policy.json")
DEFAULT_MANUAL_AUDITS = Path("outputs/aml_regression_testset_v2/manual_text_target_audits_v0")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_forest_v0")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "unnamed"


def _variant_metric(variant: dict[str, Any], key: str) -> float:
    metrics = variant.get("metrics") or {}
    value = metrics.get(key)
    if value is None and key == "precision":
        value = metrics.get("candidate_precision")
    if value is None and key == "recall":
        value = metrics.get("candidate_recall")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _variant_support_cases(variant: dict[str, Any]) -> int:
    support = variant.get("support") or {}
    metrics = variant.get("metrics") or {}
    for key in ("support_cases", "predicted_case_count"):
        value = support.get(key, metrics.get(key))
        if value is not None:
            return int(value or 0)
    return 0


def _selector_matches(variant: dict[str, Any], spec: dict[str, Any]) -> bool:
    statuses = set(str(item) for item in spec.get("include_statuses") or [])
    contains = [str(item) for item in spec.get("symbol_contains_any") or []]
    equals = [str(item) for item in spec.get("symbol_equals_any") or []]
    if not statuses and not contains and not equals:
        return False
    if statuses and str(variant.get("status") or "") not in statuses:
        return False
    symbol = str(variant.get("symbol") or "")
    if contains and not any(part in symbol for part in contains):
        return False
    if equals and symbol not in equals:
        return False
    return True


def _load_target_variants(manual_audits: Path, target: str) -> list[dict[str, Any]]:
    path = manual_audits / target / "pattern_family_proposal.json"
    if not path.exists():
        return []
    return list((_read_json(path).get("variants") or []))


def _collect_variants(manual_audits: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for target in spec.get("source_targets") or []:
        for variant in _load_target_variants(manual_audits, str(target)):
            if _selector_matches(variant, spec):
                item = dict(variant)
                item["source_target"] = str(target)
                variants.append(item)
    return variants


def _motion_summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    channel_counter: Counter[str] = Counter()
    geometry_counter: Counter[str] = Counter()
    unit_counter: Counter[str] = Counter()
    for variant in variants:
        signature = variant.get("motion_signature") or {}
        for key in signature.get("channels") or []:
            channel_counter[str(key)] += 1
        for key in signature.get("geometry_clusters") or []:
            geometry_counter[str(key)] += 1
        for row in signature.get("unit_types") or []:
            unit_counter[str(row.get("id") or "")] += int(row.get("count") or 1)
        for row in signature.get("top_channels") or []:
            channel_counter[str(row.get("id") or "")] += int(row.get("count") or 0)
        for row in signature.get("top_geometry_clusters") or []:
            geometry_counter[str(row.get("id") or "")] += int(row.get("count") or 0)
    for counter in (channel_counter, geometry_counter, unit_counter):
        if "" in counter:
            del counter[""]
    return {
        "channels": [{"id": key, "count": int(value)} for key, value in channel_counter.most_common(12)],
        "geometry_clusters": [{"id": key, "count": int(value)} for key, value in geometry_counter.most_common(12)],
        "unit_types": [{"id": key, "count": int(value)} for key, value in unit_counter.most_common(8)],
    }


def _evidence_summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    if not variants:
        return {
            "variant_count": 0,
            "support_cases_max": 0,
            "support_cases_sum": 0,
            "max_precision": 0.0,
            "max_recall": 0.0,
            "source_targets": [],
            "status_counts": {},
        }
    status_counter = Counter(str(v.get("status") or "") for v in variants)
    targets = sorted({str(v.get("source_target") or "") for v in variants if v.get("source_target")})
    supports = [_variant_support_cases(v) for v in variants]
    precisions = [_variant_metric(v, "precision") for v in variants]
    recalls = [_variant_metric(v, "recall") for v in variants]
    return {
        "variant_count": len(variants),
        "support_cases_max": max(supports) if supports else 0,
        "support_cases_sum": sum(supports),
        "max_precision": round(max(precisions), 6) if precisions else 0.0,
        "max_recall": round(max(recalls), 6) if recalls else 0.0,
        "source_targets": targets,
        "status_counts": dict(sorted(status_counter.items())),
    }


def _variant_ref(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_target": variant.get("source_target"),
        "variant_id": variant.get("variant_id"),
        "status": variant.get("status"),
        "symbol": variant.get("symbol"),
        "support": variant.get("support") or {},
        "metrics": variant.get("metrics") or {},
        "motion_signature": variant.get("motion_signature") or {},
        "examples": variant.get("examples") or [],
    }


def build_forest(policy: dict[str, Any], manual_audits: Path) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for root in policy.get("roots") or []:
        nodes.append(
            {
                "node_id": root["node_id"],
                "node_kind": "root",
                "status": "root",
                "scope": "forest_root",
                "accepted_name": root["name"],
                "description": root.get("description", ""),
                "children_policy": "offline reviewed AML pattern forest root",
            }
        )

    for spec in policy.get("nodes") or []:
        variants = _collect_variants(manual_audits, spec)
        node_id = str(spec["node_id"])
        parent_id = str(spec.get("parent_node_id") or "")
        nodes.append(
            {
                "node_id": node_id,
                "node_kind": "pattern_node",
                "status": spec.get("status"),
                "scope": spec.get("scope"),
                "accepted_name": spec.get("accepted_name"),
                "language_aliases": spec.get("language_aliases") or [],
                "description": spec.get("description", ""),
                "source_targets": spec.get("source_targets") or [],
                "review_selectors": {
                    "include_statuses": spec.get("include_statuses") or [],
                    "symbol_contains_any": spec.get("symbol_contains_any") or [],
                    "symbol_equals_any": spec.get("symbol_equals_any") or [],
                },
                "evidence": _evidence_summary(variants),
                "motion_summary": _motion_summary(variants),
                "source_variants": [_variant_ref(v) for v in variants],
            }
        )
        if parent_id:
            edges.append(
                {
                    "parent_node_id": parent_id,
                    "child_node_id": node_id,
                    "edge_type": "reviewed_pattern_forest_edge",
                }
            )

    status_counts = Counter(str(node.get("status") or "") for node in nodes)
    scope_counts = Counter(str(node.get("scope") or "") for node in nodes)
    return {
        "schema_version": "aml_pattern_forest_v0",
        "runtime_policy": "offline reviewed motion pattern forest; do not use as case-specific runtime rules",
        "source_policy": str(DEFAULT_POLICY),
        "source_manual_audits": str(manual_audits),
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "root_count": sum(1 for node in nodes if node.get("node_kind") == "root"),
            "pattern_node_count": sum(1 for node in nodes if node.get("node_kind") == "pattern_node"),
            "status_counts": dict(sorted(status_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
        },
        "nodes": nodes,
        "edges": edges,
    }


def compact_forest(forest: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for node in forest.get("nodes") or []:
        compact = {
            "node_id": node.get("node_id"),
            "node_kind": node.get("node_kind"),
            "status": node.get("status"),
            "scope": node.get("scope"),
            "accepted_name": node.get("accepted_name"),
            "language_aliases": node.get("language_aliases") or [],
            "description": node.get("description", ""),
        }
        if node.get("node_kind") == "pattern_node":
            compact["source_targets"] = node.get("source_targets") or []
            compact["evidence"] = node.get("evidence") or {}
            compact["motion_summary"] = node.get("motion_summary") or {}
            compact["source_symbols"] = [
                {
                    "source_target": item.get("source_target"),
                    "status": item.get("status"),
                    "symbol": item.get("symbol"),
                    "support": item.get("support") or {},
                    "metrics": item.get("metrics") or {},
                }
                for item in node.get("source_variants") or []
            ]
        nodes.append(compact)
    return {
        "schema_version": "aml_pattern_forest_v0_compact",
        "runtime_policy": forest.get("runtime_policy"),
        "source_policy": forest.get("source_policy"),
        "source_manual_audits": forest.get("source_manual_audits"),
        "summary": forest.get("summary") or {},
        "nodes": nodes,
        "edges": forest.get("edges") or [],
    }


def write_tree(path: Path, forest: dict[str, Any]) -> None:
    node_by_id = {str(node.get("node_id")): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = {}
    child_ids = set()
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if parent and child and child in node_by_id:
            children.setdefault(parent, []).append(node_by_id[child])
            child_ids.add(child)
    roots = [node for node in forest.get("nodes") or [] if str(node.get("node_id")) not in child_ids]
    lines = ["# AML Pattern Forest v0", ""]
    summary = forest.get("summary") or {}
    lines.append(f"nodes={summary.get('node_count')} roots={summary.get('root_count')} pattern_nodes={summary.get('pattern_node_count')} edges={summary.get('edge_count')}")
    lines.append("")
    for root in roots:
        lines.append(f"- {root.get('node_id')} [{root.get('status')}] {root.get('accepted_name')}")
        for child in sorted(children.get(str(root.get("node_id")), []), key=lambda n: str(n.get("node_id"))):
            evidence = child.get("evidence") or {}
            lines.append(
                f"  - {child.get('node_id')} [{child.get('status')}] "
                f"{child.get('accepted_name')} scope={child.get('scope')} "
                f"variants={evidence.get('variant_count')} max_precision={evidence.get('max_precision')} max_recall={evidence.get('max_recall')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any]) -> None:
    lines = ["# AML Pattern Forest v0 Review", ""]
    lines.append("Offline reviewed motion pattern forest. This is not runtime matching logic.")
    lines.append("")
    summary = forest.get("summary") or {}
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    for node in forest.get("nodes") or []:
        if node.get("node_kind") != "pattern_node":
            continue
        evidence = node.get("evidence") or {}
        motion = node.get("motion_summary") or {}
        aliases = ", ".join(node.get("language_aliases") or [])
        lines.append(f"## {node.get('node_id')}")
        lines.append("")
        lines.append(f"- status: `{node.get('status')}`")
        lines.append(f"- scope: `{node.get('scope')}`")
        lines.append(f"- accepted name: `{node.get('accepted_name')}`")
        lines.append(f"- aliases: {aliases}")
        lines.append(f"- source targets: `{evidence.get('source_targets')}`")
        lines.append(f"- variants: `{evidence.get('variant_count')}`")
        lines.append(f"- max precision / recall: `{evidence.get('max_precision')}` / `{evidence.get('max_recall')}`")
        lines.append(f"- channels: `{motion.get('channels')}`")
        lines.append(f"- geometry: `{motion.get('geometry_clusters')}`")
        lines.append("")
        lines.append(node.get("description") or "")
        lines.append("")
        for variant in (node.get("source_variants") or [])[:4]:
            lines.append(f"- source `{variant.get('source_target')}` `{variant.get('status')}` `{variant.get('symbol')}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = _read_json(Path(args.policy))
    forest = build_forest(policy, Path(args.manual_audits))
    _write_json(output_dir / "aml_pattern_forest.json", forest)
    _write_json(output_dir / "aml_pattern_forest_compact.json", compact_forest(forest))
    _write_json(output_dir / "summary.json", forest.get("summary") or {})
    write_tree(output_dir / "aml_pattern_forest_tree.txt", forest)
    write_report(output_dir / "aml_pattern_forest_review.md", forest)
    return forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--manual-audits", type=Path, default=DEFAULT_MANUAL_AUDITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    forest = run(parse_args())
    print(json.dumps(forest.get("summary") or {}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
