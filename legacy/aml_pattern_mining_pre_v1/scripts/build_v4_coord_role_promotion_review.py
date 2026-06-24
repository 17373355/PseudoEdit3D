"""Build a review table for v4 coord-role Motion-BPE families.

This is an offline audit helper. It reads the v4 Motion-BPE family artifact and
motif audit, then writes a compact promotion/component review surface. It does
not modify the AML runtime tree.

Example:
    python scripts/build_v4_coord_role_promotion_review.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BPE_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_3k")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_promotion_review")

CSV_FIELDS = [
    "priority_rank",
    "family_id",
    "status",
    "motion_scope",
    "recommendation",
    "support_cases_sum",
    "motif_count",
    "motion_role_signature",
    "required_channels",
    "required_geometry_clusters",
    "caption_aliases",
    "review_reason",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _top_aliases(family: dict[str, Any]) -> list[dict[str, Any]]:
    return list((family.get("naming_diagnostics") or {}).get("top_caption_aliases") or [])


def _audit_by_id(motif_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("motif_id") or ""): row for row in motif_audit.get("motifs") or []}


def _family_examples(
    family: dict[str, Any],
    audit_rows: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for source in family.get("source_motifs") or []:
        motif_id = str(source.get("motif_id") or "")
        row = audit_rows.get(motif_id)
        if not row:
            continue
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
                    "motif_id": motif_id,
                    "case_id": case_id,
                    "span": example.get("span"),
                    "caption": example.get("caption") or "",
                }
            )
            if len(out) >= limit:
                return out
    return out


def _recommendation(family: dict[str, Any]) -> tuple[str, str]:
    status = str(family.get("status") or "")
    scope = str(family.get("motion_scope") or "")
    support = int(family.get("support_cases_sum") or 0)
    aliases = _top_aliases(family)
    top_alias = str((aliases[0] if aliases else {}).get("id") or "")
    top_alias_count = int((aliases[0] if aliases else {}).get("count") or 0)
    alias_ratio = top_alias_count / max(1, support)

    if status == "composition_candidate":
        if top_alias and alias_ratio >= 0.55 and support >= 24:
            return (
                "promote_review",
                "composition has concentrated language evidence; inspect examples before adding a named pattern node",
            )
        return (
            "composition_review",
            "composition structure is motion-valid but naming is weak or diffuse; keep as candidate until visually checked",
        )
    if scope in {"upper_body_coordination_component", "upper_vertical_coordination_component", "upper_lower_coordination_component"}:
        if top_alias and alias_ratio >= 0.40 and support >= 24:
            return (
                "named_component_review",
                "component has useful language concentration but should not become a full action without parent composition",
            )
        return (
            "component_review",
            "frequent coordination component; useful for tree internals and edit handles, not a full action node",
        )
    if "component" in status or "component" in scope:
        return (
            "component_library",
            "local component should remain a reusable building block unless composed with other channels",
        )
    return (
        "diagnostic_keep",
        "keep as mining evidence; not enough structure for promotion",
    )


def _priority(row: dict[str, Any]) -> tuple[int, int, str]:
    order = {
        "promote_review": 0,
        "composition_review": 1,
        "named_component_review": 2,
        "component_review": 3,
        "component_library": 4,
        "diagnostic_keep": 5,
    }
    return (order.get(str(row.get("recommendation") or ""), 9), -int(row.get("support_cases_sum") or 0), str(row.get("family_id") or ""))


def build_review(
    family_payload: dict[str, Any],
    motif_audit: dict[str, Any],
    *,
    examples_per_family: int,
    source_family_path: Path,
    source_motif_audit_path: Path,
) -> dict[str, Any]:
    audit_rows = _audit_by_id(motif_audit)
    review_rows: list[dict[str, Any]] = []
    for family in family_payload.get("families") or []:
        motion = family.get("motion_definition") or {}
        aliases = _top_aliases(family)
        recommendation, reason = _recommendation(family)
        review_rows.append(
            {
                "family_id": family.get("family_id"),
                "status": family.get("status"),
                "motion_scope": family.get("motion_scope"),
                "recommendation": recommendation,
                "review_reason": reason,
                "support_cases_sum": int(family.get("support_cases_sum") or 0),
                "occurrences_sum": int(family.get("occurrences_sum") or 0),
                "motif_count": int(family.get("motif_count") or 0),
                "motion_role_signature": motion.get("motion_role_signature") or "",
                "required_channels": list(motion.get("required_channels") or []),
                "required_geometry_clusters": list(motion.get("required_geometry_clusters") or []),
                "caption_aliases": aliases,
                "source_motifs": list(family.get("source_motifs") or []),
                "examples": _family_examples(family, audit_rows, examples_per_family),
            }
        )
    review_rows.sort(key=_priority)
    for idx, row in enumerate(review_rows, start=1):
        row["priority_rank"] = idx
    recommendation_counts = Counter(str(row["recommendation"]) for row in review_rows)
    scope_counts = Counter(str(row["motion_scope"]) for row in review_rows)
    status_counts = Counter(str(row["status"]) for row in review_rows)
    return {
        "schema_version": "v4_coord_role_promotion_review_v1",
        "runtime_policy": "offline review only; no runtime AML matching changes",
        "source_family_candidates": str(source_family_path),
        "source_motif_audit": str(source_motif_audit_path),
        "summary": {
            "family_count": len(review_rows),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "motion_scope_counts": dict(sorted(scope_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "review_rows": review_rows,
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
    summary = payload.get("summary") or {}
    lines: list[str] = [
        "# v4 Coord-Role Promotion Review",
        "",
        "This is an offline review artifact. It does not change runtime AML matching.",
        "",
        "## Summary",
        "",
        f"- family_count: `{summary.get('family_count')}`",
        f"- recommendation_counts: `{summary.get('recommendation_counts')}`",
        f"- motion_scope_counts: `{summary.get('motion_scope_counts')}`",
        "",
        "## Labels",
        "",
        "- `promote_review`: possible named composition; needs visual/example review.",
        "- `composition_review`: motion-valid composition; naming or purity is not ready.",
        "- `named_component_review`: named component candidate; not a full action node.",
        "- `component_review`: reusable component or edit handle candidate.",
        "- `component_library`: stable local component; keep below action level.",
        "- `diagnostic_keep`: evidence only.",
        "",
        "## Priority Rows",
        "",
    ]
    for row in rows[:max_rows]:
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("caption_aliases") or [] if item.get("id"))
        geometry = "; ".join(row.get("required_geometry_clusters") or [])
        lines.append(f"### {row.get('priority_rank')}. {row.get('family_id')}")
        lines.append("")
        lines.append(f"- recommendation: `{row.get('recommendation')}`")
        lines.append(f"- status/scope: `{row.get('status')}` / `{row.get('motion_scope')}`")
        lines.append(f"- support: `{row.get('support_cases_sum')}`; motifs: `{row.get('motif_count')}`")
        lines.append(f"- role signature: `{row.get('motion_role_signature')}`")
        lines.append(f"- channels: `{row.get('required_channels')}`")
        lines.append(f"- geometry: {geometry or 'none'}")
        lines.append(f"- caption aliases: {aliases or 'none'}")
        lines.append(f"- reason: {row.get('review_reason')}")
        examples = row.get("examples") or []
        if examples:
            lines.append("")
            lines.append("| case | span | motif | caption |")
            lines.append("| --- | --- | --- | --- |")
            for example in examples[:4]:
                caption = str(example.get("caption") or "").replace("|", "\\|")
                lines.append(f"| `{example.get('case_id')}` | `{example.get('span')}` | `{example.get('motif_id')}` | {caption} |")
        lines.append("")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows in JSON/CSV.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    bpe_dir = Path(args.bpe_dir)
    family_path = Path(args.family_candidates) if args.family_candidates else bpe_dir / "motif_family_candidates.json"
    motif_audit_path = Path(args.motif_audit) if args.motif_audit else bpe_dir / "motif_audit.json"
    payload = build_review(
        _read_json(family_path),
        _read_json(motif_audit_path),
        examples_per_family=int(args.examples_per_family),
        source_family_path=family_path,
        source_motif_audit_path=motif_audit_path,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "promotion_review_table.json", payload)
    write_csv(output_dir / "promotion_review_table.csv", payload.get("review_rows") or [])
    write_markdown(output_dir / "promotion_review_table.md", payload, int(args.max_markdown_rows))
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": "v4_coord_role_promotion_review_summary_v1",
            **(payload.get("summary") or {}),
            "outputs": {
                "review_json": str(output_dir / "promotion_review_table.json"),
                "review_csv": str(output_dir / "promotion_review_table.csv"),
                "review_md": str(output_dir / "promotion_review_table.md"),
                "summary": str(output_dir / "summary.json"),
            },
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 coord-role Motion-BPE promotion/component review.")
    parser.add_argument("--bpe-dir", default=str(DEFAULT_BPE_DIR))
    parser.add_argument("--family-candidates", default="")
    parser.add_argument("--motif-audit", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--examples-per-family", type=int, default=8)
    parser.add_argument("--max-markdown-rows", type=int, default=50)
    args = parser.parse_args()
    run(args)
    output_dir = Path(args.output_dir)
    print(output_dir / "summary.json")
    print(output_dir / "promotion_review_table.md")


if __name__ == "__main__":
    main()
