from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_AUDIT = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/motif_audit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_v1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _item_count(item: dict[str, Any]) -> int:
    try:
        return int(item.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _required_ids(items: list[dict[str, Any]], *, min_relative_count: float, max_items: int) -> list[str]:
    if not items:
        return []
    top = max(_item_count(item) for item in items)
    cutoff = max(1, int(round(top * min_relative_count)))
    kept = [
        _item_id(item)
        for item in sorted(items, key=lambda row: (-_item_count(row), _item_id(row)))
        if _item_id(item) and _item_count(item) >= cutoff
    ]
    return kept[:max_items]


def _examples(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for example in row.get("example_occurrences") or []:
        case_id = str(example.get("case_id") or "")
        span = example.get("span") or []
        span_key = tuple(int(x) for x in span[:2]) if len(span) >= 2 else (-1, -1)
        key = (case_id, span_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "case_id": case_id,
                "span": example.get("span"),
                "caption": example.get("caption") or "",
            }
        )
        if len(out) >= limit:
            break
    return out


def _promotion_status(row: dict[str, Any], args: argparse.Namespace) -> str:
    support = int(row.get("support_cases") or 0)
    purity = float(row.get("caption_alias_purity") or 0.0)
    alias = str(row.get("top_caption_alias") or "")
    channel_count = len(row.get("channels") or [])
    geometry_count = len(row.get("top_geometry_clusters") or [])
    if (
        alias
        and support >= int(args.min_named_support)
        and purity >= float(args.min_named_purity)
        and channel_count >= int(args.min_channels)
        and geometry_count >= int(args.min_geometry)
    ):
        return "promote_named_coordination_candidate"
    if support >= int(args.min_structural_support) and channel_count >= int(args.min_channels):
        return "review_structural_coordination_candidate"
    if alias and support >= int(args.min_weak_named_support) and purity >= float(args.min_weak_named_purity):
        return "review_named_low_support_candidate"
    return "diagnostic_coordination_motif"


def _candidate_from_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    required_channels = _required_ids(
        row.get("channels") or [],
        min_relative_count=float(args.min_relative_channel_count),
        max_items=int(args.max_channels),
    )
    required_geometry = _required_ids(
        row.get("top_geometry_clusters") or [],
        min_relative_count=float(args.min_relative_geometry_count),
        max_items=int(args.max_geometry),
    )
    status = _promotion_status(row, args)
    return {
        "candidate_id": "coord_candidate_" + str(row.get("motif_id") or "").strip("<>").lower(),
        "schema_version": "coordination_pattern_promotion_candidate_v1",
        "status": status,
        "source_motif_id": row.get("motif_id"),
        "operator": row.get("operator"),
        "support": {
            "support_cases": int(row.get("support_cases") or 0),
            "occurrences": int(row.get("occurrences") or 0),
        },
        "motion_definition": {
            "parent_signature": (row.get("parents") or [""])[0],
            "required_channels": required_channels,
            "required_geometry_clusters": required_geometry,
            "top_channels": row.get("channels") or [],
            "top_geometry_clusters": row.get("top_geometry_clusters") or [],
            "relation_profile": row.get("relation_profile") or [],
        },
        "naming_diagnostics": {
            "top_caption_alias": row.get("top_caption_alias") or "",
            "caption_alias_purity": float(row.get("caption_alias_purity") or 0.0),
            "top_caption_aliases": row.get("top_caption_aliases") or [],
            "policy": "diagnostic only; motion_definition decides the structural node",
        },
        "review_examples": _examples(row, int(args.examples_per_candidate)),
    }


def build_candidates(audit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        row
        for row in audit.get("motifs") or []
        if str(row.get("operator") or "") == "COORDINATION_MERGE"
    ]
    candidates = [_candidate_from_row(row, args) for row in rows]
    candidates.sort(
        key=lambda item: (
            0 if item["status"] == "promote_named_coordination_candidate" else 1,
            -float(item["naming_diagnostics"]["caption_alias_purity"]),
            -int(item["support"]["support_cases"]),
            str(item["source_motif_id"]),
        )
    )
    status_counts = Counter(str(item["status"]) for item in candidates)
    alias_counts = Counter(
        str(item["naming_diagnostics"]["top_caption_alias"])
        for item in candidates
        if item["naming_diagnostics"]["top_caption_alias"]
    )
    return {
        "schema_version": "coordination_pattern_promotion_candidates_v1",
        "source_motif_audit": str(args.motif_audit),
        "policy": {
            "runtime_tree_policy": "offline review queue only; do not modify AML runtime tree automatically",
            "selection": "motion-only coordination motifs with caption aliases kept as diagnostics",
            "min_named_support": int(args.min_named_support),
            "min_named_purity": float(args.min_named_purity),
            "min_structural_support": int(args.min_structural_support),
        },
        "summary": {
            "coordination_motif_count": len(rows),
            "candidate_count": len(candidates),
            "status_counts": dict(sorted(status_counts.items())),
            "top_alias_counts": dict(alias_counts.most_common(12)),
        },
        "candidates": candidates,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = payload.get("summary") or {}
    lines.append("# Coordination Pattern Promotion Review")
    lines.append("")
    lines.append("This is an offline review queue. It does not change the AML runtime tree.")
    lines.append("")
    lines.append(f"- source: `{payload.get('source_motif_audit')}`")
    lines.append(f"- coordination motifs: `{summary.get('coordination_motif_count')}`")
    lines.append(f"- candidates: `{summary.get('candidate_count')}`")
    lines.append(f"- status counts: `{summary.get('status_counts')}`")
    lines.append("")
    for item in payload.get("candidates") or []:
        naming = item.get("naming_diagnostics") or {}
        motion = item.get("motion_definition") or {}
        support = item.get("support") or {}
        lines.append(f"## {item.get('candidate_id')}")
        lines.append("")
        lines.append(f"- status: `{item.get('status')}`")
        lines.append(f"- source motif: `{item.get('source_motif_id')}`")
        lines.append(f"- support cases: `{support.get('support_cases')}`")
        lines.append(f"- occurrences: `{support.get('occurrences')}`")
        lines.append(f"- top caption alias: `{naming.get('top_caption_alias')}`")
        lines.append(f"- caption alias purity: `{naming.get('caption_alias_purity')}`")
        lines.append(f"- parent signature: `{motion.get('parent_signature')}`")
        lines.append(f"- required channels: `{motion.get('required_channels')}`")
        lines.append(f"- required geometry: `{motion.get('required_geometry_clusters')}`")
        aliases = ", ".join(f"{row['id']}:{row['count']}" for row in naming.get("top_caption_aliases", [])[:6])
        lines.append(f"- alias diagnostics: {aliases}")
        lines.append("")
        lines.append("| case | span | caption |")
        lines.append("| --- | --- | --- |")
        for example in item.get("review_examples") or []:
            caption = str(example.get("caption") or "").replace("|", "\\|")
            lines.append(f"| `{example.get('case_id')}` | `{example.get('span')}` | {caption} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline promotion review queue from multichannel coordination motifs.")
    parser.add_argument("--motif-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-named-support", type=int, default=30)
    parser.add_argument("--min-named-purity", type=float, default=0.70)
    parser.add_argument("--min-structural-support", type=int, default=120)
    parser.add_argument("--min-weak-named-support", type=int, default=20)
    parser.add_argument("--min-weak-named-purity", type=float, default=0.45)
    parser.add_argument("--min-channels", type=int, default=2)
    parser.add_argument("--min-geometry", type=int, default=2)
    parser.add_argument("--min-relative-channel-count", type=float, default=0.50)
    parser.add_argument("--min-relative-geometry-count", type=float, default=0.50)
    parser.add_argument("--max-channels", type=int, default=6)
    parser.add_argument("--max-geometry", type=int, default=10)
    parser.add_argument("--examples-per-candidate", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.motif_audit = str(args.motif_audit)
    payload = build_candidates(_read_json(Path(args.motif_audit)), args)
    _write_json(output_dir / "coordination_pattern_promotion_candidates.json", payload)
    write_report(output_dir / "coordination_pattern_promotion_review.md", payload)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "coordination_pattern_promotion_summary_v1",
            "source_motif_audit": args.motif_audit,
            **(payload.get("summary") or {}),
            "outputs": {
                "candidates": str(output_dir / "coordination_pattern_promotion_candidates.json"),
                "review": str(output_dir / "coordination_pattern_promotion_review.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    print(output_dir / "summary.json")
    print(output_dir / "coordination_pattern_promotion_review.md")


if __name__ == "__main__":
    main()
