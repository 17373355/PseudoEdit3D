"""Export the AML Pattern Mining Explorer v1 core artifact bundle.

This consolidates the current v5 explorer into four files:

- evidence_cases.jsonl
- candidate_patterns.jsonl
- pattern_registry.json
- audit_report.md

It does not rerun mining. It reads existing v5 evidence/mining/audit artifacts
and writes a stable, easier-to-review bundle.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("outputs/aml_regression_testset_v2")
DEFAULT_PHASE_AUDIT = ROOT / "aml_pattern_split_axis_phase_closure_v5_stance_width_full_v0/phase_closure_audit.json"
DEFAULT_COVERAGE_AUDIT = ROOT / "aml_pattern_split_axis_full_v5_stance_width_coverage_v0/case_coverage.json"
DEFAULT_COMPOSITION_FOREST = ROOT / "hml3d_composition_pattern_forest_v5_stance_width_full_v0/composition_pattern_forest.json"
DEFAULT_PROGRAM = ROOT / "aml_composable_pattern_program_v1_support_state_reviewed_draft/aml_composable_pattern_program.json"
DEFAULT_OUTPUT_DIR = ROOT / "aml_pattern_mining_explorer_v1"


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _status_from_structure_group(status: str) -> str:
    if status in {"composition_structure_group", "name_aligned_structure_group"}:
        return "review_candidate"
    if status == "transition_structure_group":
        return "split_required"
    return "blocked"


def evidence_case_rows(phase_payload: dict[str, Any], coverage_payload: dict[str, Any]) -> list[dict[str, Any]]:
    coverage_by_case = {str(row.get("case_id")): row for row in coverage_payload.get("rows") or []}
    rows = []
    for row in phase_payload.get("rows") or []:
        case_id = str(row.get("case_id") or "")
        cov = coverage_by_case.get(case_id) or {}
        rows.append(
            {
                "case_id": case_id,
                "num_frames": row.get("num_frames"),
                "caption_alias_ids": row.get("caption_alias_ids") or [],
                "caption": row.get("caption"),
                "evidence_groups": sorted((row.get("group_counts") or {}).keys()),
                "group_counts": row.get("group_counts") or {},
                "negative_group_counts": row.get("negative_group_counts") or {},
                "phase_status": row.get("best_status"),
                "phase_label": row.get("best_label"),
                "case_has_full_rule": bool(cov.get("case_has_full_rule")),
                "window_has_full_rule": bool(cov.get("window_has_full_rule")),
                "missing_full_rule_groups": cov.get("missing_full_rule_case") or [],
            }
        )
    return rows


def axis_candidate_patterns(phase_payload: dict[str, Any], coverage_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = phase_payload.get("summary") or {}
    coverage_summary = summarize_coverage(coverage_payload)
    axis_id = str(summary.get("axis_id") or "unknown_axis")
    groups = ["upper_spread", "vertical_rhythm", "lower_spread"]
    optional = ["bilateral_high_arm_pose", "large_bilateral_arm_arc"]
    target_alias = "jumping_jack" if axis_id == "bilateral_spread_vertical_coordination_v0" else None
    rows = [
        {
            "pattern_id": f"axis:{axis_id}:strict_phase_closed",
            "source": "split_axis_phase_audit",
            "evidence_groups": groups,
            "required_groups": groups,
            "optional_groups": optional,
            "negative_groups": ["floor_or_inverted_support_confound", "low_body_transition_confound", "strong_locomotion_confound", "leg_strike_confound"],
            "phase_status": "phase_closed_all_pairs",
            "support_cases": summary.get("strict_phase_closed_case_count"),
            "support_windows": None,
            "naming_diagnostics": {
                "target_alias": target_alias,
                "precision": summary.get("strict_phase_closed_precision"),
                "recall": summary.get("strict_phase_closed_recall"),
                "target_cases": summary.get("target_case_count"),
                "target_hits": summary.get("strict_phase_closed_target_count"),
            },
            "status": "component",
            "examples": (phase_payload.get("examples") or {}).get("target_phase_closed") or [],
        },
        {
            "pattern_id": f"axis:{axis_id}:phase_connected_or_closed",
            "source": "split_axis_phase_audit",
            "evidence_groups": groups,
            "required_groups": groups,
            "optional_groups": optional,
            "negative_groups": ["floor_or_inverted_support_confound", "low_body_transition_confound", "strong_locomotion_confound", "leg_strike_confound"],
            "phase_status": "phase_connected_or_closed",
            "support_cases": summary.get("phase_connected_or_closed_case_count"),
            "support_windows": None,
            "naming_diagnostics": {
                "target_alias": target_alias,
                "precision": summary.get("phase_connected_or_closed_precision"),
                "recall": summary.get("phase_connected_or_closed_recall"),
                "target_cases": summary.get("target_case_count"),
                "target_hits": summary.get("phase_connected_or_closed_target_count"),
            },
            "status": "component",
            "examples": (phase_payload.get("examples") or {}).get("target_phase_connected") or [],
        },
        {
            "pattern_id": f"axis:{axis_id}:coverage_full_rule",
            "source": "split_axis_coverage_audit",
            "evidence_groups": groups,
            "required_groups": groups,
            "optional_groups": optional,
            "negative_groups": [],
            "phase_status": "not_checked",
            "support_cases": coverage_summary.get("target_case_has_full_rule"),
            "support_windows": coverage_summary.get("target_window_has_full_rule"),
            "naming_diagnostics": {
                "target_alias": target_alias,
                "target_cases": coverage_summary.get("target_case_count"),
                "missing_full_rule_groups": coverage_summary.get("target_missing_full_rule_group_counts"),
            },
            "status": "component",
            "examples": [],
        },
    ]
    return rows


def summarize_coverage(coverage_payload: dict[str, Any]) -> dict[str, Any]:
    rows = coverage_payload.get("rows") or []
    target_rows = [row for row in rows if row.get("target_alias_hit")]
    missing = Counter()
    for row in target_rows:
        missing.update(row.get("missing_full_rule_case") or [])
    return {
        "target_case_count": len(target_rows),
        "target_case_has_full_rule": sum(1 for row in target_rows if row.get("case_has_full_rule")),
        "target_window_has_full_rule": sum(1 for row in target_rows if row.get("window_has_full_rule")),
        "target_missing_full_rule_group_counts": dict(missing.most_common()),
    }


def composition_candidate_patterns(forest: dict[str, Any], *, max_nodes: int) -> list[dict[str, Any]]:
    nodes = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "structure_group"]
    nodes.sort(key=lambda item: (int(item.get("support_cases_max") or 0), int(item.get("variant_count") or 0)), reverse=True)
    rows = []
    for node in nodes[:max_nodes]:
        rows.append(
            {
                "pattern_id": f"composition:{node.get('node_id')}",
                "source": "composition_forest_structure_group",
                "evidence_groups": node.get("family_core_items") or [],
                "required_groups": [],
                "optional_groups": [],
                "negative_groups": [],
                "phase_status": "not_checked",
                "support_cases": node.get("support_cases_max"),
                "support_windows": node.get("occurrences_sum"),
                "naming_diagnostics": {
                    "caption_aliases": node.get("caption_aliases") or [],
                    "caption_name_candidates": node.get("caption_name_candidates") or [],
                },
                "status": _status_from_structure_group(str(node.get("status") or "")),
                "examples": [],
                "metadata": {
                    "motion_structure_label": node.get("motion_structure_label"),
                    "channels": node.get("channels") or [],
                    "zones": node.get("zones") or [],
                    "variant_count": node.get("variant_count"),
                    "structure_score_max": node.get("structure_score_max"),
                },
            }
        )
    return rows


def registry_from_program(program: dict[str, Any], axis_patterns: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for item in program.get("condition_vocabulary") or []:
        status = str(item.get("review_status") or "review_candidate")
        if status not in {"accepted", "component", "split_required", "blocked"}:
            if str(item.get("semantic_level") or "") in {"split_required_candidate", "closure_required_candidate"}:
                status = "split_required"
            elif str(item.get("scope") or "") == "component":
                status = "component"
            else:
                status = "review_candidate"
        entries.append(
            {
                "pattern_id": str(item.get("condition_entry_id") or item.get("condition_id") or item.get("program_node_id")),
                "status": status,
                "name": item.get("motion_structure_label"),
                "scope": item.get("scope"),
                "semantic_level": item.get("semantic_level"),
                "edit_scope": item.get("edit_scope"),
                "condition_weight_default": item.get("condition_weight_default"),
                "evidence_groups": item.get("canonical_roles") or item.get("geometry_roles") or [],
                "naming_diagnostics": {"caption_name_candidates": item.get("caption_name_candidates") or []},
                "source": "reviewed_support_state_program",
            }
        )
    for pattern in axis_patterns:
        entries.append(
            {
                "pattern_id": pattern["pattern_id"],
                "status": pattern["status"],
                "name": pattern["pattern_id"].split(":")[-1],
                "scope": "component_or_closure",
                "semantic_level": "multi_part_coordination",
                "edit_scope": "multi_part",
                "condition_weight_default": 0.0,
                "evidence_groups": pattern.get("evidence_groups") or [],
                "naming_diagnostics": pattern.get("naming_diagnostics") or {},
                "source": pattern.get("source"),
            }
        )
    counts = Counter(str(item.get("status") or "") for item in entries)
    return {
        "schema_version": "aml_pattern_registry_v1",
        "runtime_policy": "motion structure registry; naming diagnostics do not create evidence",
        "status_counts": dict(sorted(counts.items())),
        "entries": entries,
    }


def write_report(path: Path, summary: dict[str, Any], registry: dict[str, Any], phase_summary: dict[str, Any], coverage_summary: dict[str, Any]) -> None:
    lines = ["# AML Pattern Mining Explorer v1 Bundle", ""]
    lines.append("Golden path: evidence extraction -> candidate mining -> audit -> registry.")
    lines.append("")
    lines.append("## Core Files")
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- evidence cases: {summary.get('evidence_case_count')}")
    lines.append(f"- candidate patterns: {summary.get('candidate_pattern_count')}")
    lines.append(f"- registry entries: {len(registry.get('entries') or [])}")
    lines.append(f"- registry status counts: `{registry.get('status_counts')}`")
    lines.append("")
    lines.append("## Bilateral Spread / Vertical Closure")
    lines.append(f"- target cases: {phase_summary.get('target_case_count')}")
    lines.append(
        f"- strict phase-closed: {phase_summary.get('strict_phase_closed_case_count')} cases, "
        f"precision={phase_summary.get('strict_phase_closed_precision')}, "
        f"recall={phase_summary.get('strict_phase_closed_recall')}"
    )
    lines.append(
        f"- connected-or-closed: {phase_summary.get('phase_connected_or_closed_case_count')} cases, "
        f"precision={phase_summary.get('phase_connected_or_closed_precision')}, "
        f"recall={phase_summary.get('phase_connected_or_closed_recall')}"
    )
    lines.append(f"- coverage full-rule target cases: {coverage_summary.get('target_case_has_full_rule')}")
    lines.append(f"- missing full-rule groups: `{coverage_summary.get('target_missing_full_rule_group_counts')}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- BPE is no longer the system name; it is an optional miner.")
    lines.append("- The main current method is evidence-group closure plus phase/split audit.")
    lines.append("- `bilateral_spread_vertical_coordination` is a reusable component/closure, not a directly accepted `jumping_jack` name.")
    lines.append("- Caption/WordNet/TMR diagnostics belong in the audit layer only.")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_bundle(args: argparse.Namespace) -> dict[str, Any]:
    phase_payload = _read_json(Path(args.phase_audit))
    coverage_payload = _read_json(Path(args.coverage_audit))
    forest = _read_json(Path(args.composition_forest), default={"nodes": []})
    program = _read_json(Path(args.program), default={"condition_vocabulary": []})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = evidence_case_rows(phase_payload, coverage_payload)
    axis_patterns = axis_candidate_patterns(phase_payload, coverage_payload)
    composition_patterns = composition_candidate_patterns(forest, max_nodes=int(args.max_composition_candidates))
    candidate_rows = axis_patterns + composition_patterns
    registry = registry_from_program(program, axis_patterns)

    evidence_count = _write_jsonl(output_dir / "evidence_cases.jsonl", evidence_rows)
    candidate_count = _write_jsonl(output_dir / "candidate_patterns.jsonl", candidate_rows)
    _write_json(output_dir / "pattern_registry.json", registry)
    coverage_summary = summarize_coverage(coverage_payload)
    phase_summary = phase_payload.get("summary") or {}
    summary = {
        "schema_version": "aml_pattern_mining_explorer_bundle_v1",
        "inputs": {
            "phase_audit": str(args.phase_audit),
            "coverage_audit": str(args.coverage_audit),
            "composition_forest": str(args.composition_forest),
            "program": str(args.program),
        },
        "outputs": {
            "evidence_cases": str(output_dir / "evidence_cases.jsonl"),
            "candidate_patterns": str(output_dir / "candidate_patterns.jsonl"),
            "pattern_registry": str(output_dir / "pattern_registry.json"),
            "audit_report": str(output_dir / "audit_report.md"),
            "summary": str(output_dir / "summary.json"),
        },
        "evidence_case_count": evidence_count,
        "candidate_pattern_count": candidate_count,
        "registry_entry_count": len(registry.get("entries") or []),
        "registry_status_counts": registry.get("status_counts") or {},
        "phase_summary": phase_summary,
        "coverage_summary": coverage_summary,
    }
    _write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "audit_report.md", summary, registry, phase_summary, coverage_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-audit", type=Path, default=DEFAULT_PHASE_AUDIT)
    parser.add_argument("--coverage-audit", type=Path, default=DEFAULT_COVERAGE_AUDIT)
    parser.add_argument("--composition-forest", type=Path, default=DEFAULT_COMPOSITION_FOREST)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-composition-candidates", type=int, default=75)
    return parser.parse_args()


def main() -> None:
    summary = export_bundle(parse_args())
    print(json.dumps({"ok": True, "output_dir": str(Path(summary["outputs"]["summary"]).parent), "summary": summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
