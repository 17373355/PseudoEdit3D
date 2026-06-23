"""Audit AML program condition manifests before training.

The audit is policy-only: it does not use captions to create conditions and it
does not rewrite the manifest. It flags broad, component-heavy, or suspicious
selected conditions so we can decide what enters training.

Quick run:
    python scripts/audit_aml_program_condition_manifest_v0.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_JSONL = Path(
    "outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/selected_conditions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/audit")

GENERIC_LABEL_PARTS = {
    "arm_coordination",
    "arm_bimanual_coordination",
    "arm_torso_coordination",
    "arm_whole_body_vertical_coordination",
    "torso_whole_body_vertical_coordination",
    "leg_root_rotation_whole_body_vertical_coordination",
    "arm_root_rotation_whole_body_vertical_coordination",
    "root_locomotion_torso_coordination",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _top(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _condition_risk(cond: dict[str, Any], record: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    label = str(cond.get("motion_structure_label") or "")
    level = str(cond.get("semantic_level") or "")
    case_type = str(record.get("case_pattern_type") or "")
    score = float(cond.get("screen_score") or 0.0)
    case_aliases = {str(item) for item in record.get("caption_alias_ids_for_review_only") or []}
    if label in GENERIC_LABEL_PARTS:
        reasons.append("generic_structure_label")
    if level == "component":
        reasons.append("component_only_condition")
    if case_type in {"component_dominant", "diagnostic_or_ambiguous"}:
        reasons.append(f"case_type_{case_type}")
    if score < 0.60:
        reasons.append("low_screen_score")
    if case_aliases and "sit_down" in case_aliases and "low_body" not in label and "torso" not in label:
        reasons.append("sit_caption_but_not_low_body")
    if case_aliases and "jumping_jack" in case_aliases and "bimanual_raise_spread_vertical" not in label and "upper_limb_vertical" not in label:
        reasons.append("jumping_jack_caption_but_not_vertical_bimanual")
    if not reasons:
        return "train_candidate", []
    if "generic_structure_label" in reasons or "component_only_condition" in reasons:
        return "defer_or_auxiliary", reasons
    return "review_before_train", reasons


def audit_manifest(records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    family_rows: dict[str, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_decisions: dict[str, Counter[str]] = defaultdict(Counter)
    level_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    train_condition_count = 0

    for record in records:
        case_decisions: Counter[str] = Counter()
        flagged: list[dict[str, Any]] = []
        for cond in record.get("selected_conditions") or []:
            family = str(cond.get("family_id") or "")
            label = str(cond.get("motion_structure_label") or "")
            decision, reasons = _condition_risk(cond, record)
            decision_counts.update([decision])
            case_decisions.update([decision])
            reason_counts.update(reasons)
            family_counts.update([family])
            family_decisions[family].update([decision])
            level_counts.update([str(cond.get("semantic_level") or "")])
            scope_counts.update([str(cond.get("edit_scope") or "")])
            if decision == "train_candidate":
                train_condition_count += 1
            row = family_rows.setdefault(
                family,
                {
                    "family_id": family,
                    "motion_structure_label": label,
                    "semantic_level_counts": Counter(),
                    "edit_scope_counts": Counter(),
                    "case_type_counts": Counter(),
                    "caption_alias_counts": Counter(),
                    "score_values": [],
                    "condition_count": 0,
                    "decision_counts": Counter(),
                    "risk_reason_counts": Counter(),
                    "examples": [],
                },
            )
            row["condition_count"] += 1
            row["semantic_level_counts"].update([str(cond.get("semantic_level") or "")])
            row["edit_scope_counts"].update([str(cond.get("edit_scope") or "")])
            row["case_type_counts"].update([str(record.get("case_pattern_type") or "")])
            row["score_values"].append(float(cond.get("screen_score") or 0.0))
            row["decision_counts"].update([decision])
            row["risk_reason_counts"].update(reasons)
            for alias in cond.get("caption_name_candidates_for_review_only") or []:
                row["caption_alias_counts"].update([str(alias.get("id") or "")])
            if len(row["examples"]) < 5:
                row["examples"].append(
                    {
                        "case_id": record.get("case_id"),
                        "caption": record.get("reference_prompt"),
                        "case_pattern_type": record.get("case_pattern_type"),
                        "span": (cond.get("slot_values") or {}).get("span"),
                        "score": cond.get("screen_score"),
                        "decision": decision,
                        "risk_reasons": reasons,
                    }
                )
            if decision != "train_candidate":
                flagged.append(
                    {
                        "family_id": family,
                        "motion_structure_label": label,
                        "semantic_level": cond.get("semantic_level"),
                        "edit_scope": cond.get("edit_scope"),
                        "span": (cond.get("slot_values") or {}).get("span"),
                        "score": cond.get("screen_score"),
                        "decision": decision,
                        "risk_reasons": reasons,
                    }
                )
        if flagged:
            case_rows.append(
                {
                    "case_id": record.get("case_id"),
                    "case_pattern_type": record.get("case_pattern_type"),
                    "caption": record.get("reference_prompt"),
                    "selected_count": len(record.get("selected_conditions") or []),
                    "decision_counts": dict(case_decisions),
                    "flagged_conditions": flagged,
                }
            )

    family_audit: list[dict[str, Any]] = []
    for row in family_rows.values():
        scores = row.pop("score_values")
        condition_count = int(row["condition_count"])
        train_count = int(row["decision_counts"].get("train_candidate", 0))
        if train_count == condition_count:
            recommendation = "keep_for_training"
        elif train_count == 0:
            recommendation = "defer_or_auxiliary"
        else:
            recommendation = "split_or_filter"
        family_audit.append(
            {
                "family_id": row["family_id"],
                "motion_structure_label": row["motion_structure_label"],
                "condition_count": condition_count,
                "recommendation": recommendation,
                "decision_counts": dict(row["decision_counts"]),
                "risk_reason_counts": dict(row["risk_reason_counts"]),
                "semantic_level_counts": dict(row["semantic_level_counts"]),
                "edit_scope_counts": dict(row["edit_scope_counts"]),
                "case_type_counts": dict(row["case_type_counts"]),
                "caption_alias_top": _top(row["caption_alias_counts"], top_k),
                "score_min": round(min(scores), 4) if scores else 0.0,
                "score_mean": round(sum(scores) / max(1, len(scores)), 4),
                "score_max": round(max(scores), 4) if scores else 0.0,
                "examples": row["examples"],
            }
        )
    family_audit.sort(
        key=lambda row: (
            str(row["recommendation"]),
            -int(row["condition_count"]),
            str(row["motion_structure_label"]),
        )
    )
    return {
        "schema_version": "aml_program_condition_manifest_audit_v0",
        "num_records": len(records),
        "total_selected_conditions": sum(len(record.get("selected_conditions") or []) for record in records),
        "train_candidate_conditions": train_condition_count,
        "decision_counts": dict(decision_counts),
        "risk_reason_counts": dict(reason_counts),
        "semantic_level_counts": dict(level_counts),
        "edit_scope_counts": dict(scope_counts),
        "family_counts_top": _top(family_counts, top_k),
        "family_audit": family_audit,
        "flagged_cases": case_rows,
    }


def write_report(path: Path, payload: dict[str, Any], max_families: int, max_cases: int) -> None:
    lines = [
        "# AML Program Condition Manifest Audit v0",
        "",
        f"- records: `{payload['num_records']}`",
        f"- selected conditions: `{payload['total_selected_conditions']}`",
        f"- train-candidate conditions: `{payload['train_candidate_conditions']}`",
        f"- decision counts: `{payload['decision_counts']}`",
        f"- risk reasons: `{payload['risk_reason_counts']}`",
        "",
        "## Family Recommendations",
        "",
    ]
    for row in payload.get("family_audit", [])[:max_families]:
        lines.extend(
            [
                f"### {row['motion_structure_label']}",
                f"- family: `{row['family_id']}`",
                f"- recommendation: `{row['recommendation']}`",
                f"- count: `{row['condition_count']}` score_mean: `{row['score_mean']}`",
                f"- decisions: `{row['decision_counts']}`",
                f"- risk: `{row['risk_reason_counts']}`",
                f"- aliases: `{row['caption_alias_top'][:5]}`",
                "",
            ]
        )
        for ex in row.get("examples", [])[:3]:
            lines.append(
                f"  - {ex.get('case_id')} span={ex.get('span')} score={ex.get('score')} "
                f"decision={ex.get('decision')} caption={ex.get('caption')}"
            )
        lines.append("")
    lines.extend(["", "## Flagged Cases", ""])
    for case in payload.get("flagged_cases", [])[:max_cases]:
        lines.append(f"### {case.get('case_id')} | {case.get('case_pattern_type')}")
        lines.append(f"- caption: {case.get('caption')}")
        lines.append(f"- decisions: `{case.get('decision_counts')}`")
        for cond in case.get("flagged_conditions", [])[:5]:
            lines.append(
                f"  - {cond.get('motion_structure_label')} span={cond.get('span')} "
                f"score={cond.get('score')} decision={cond.get('decision')} reasons={cond.get('risk_reasons')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AML program condition manifest quality.")
    parser.add_argument("--manifest-jsonl", default=str(DEFAULT_MANIFEST_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-report-families", type=int, default=60)
    parser.add_argument("--max-report-cases", type=int, default=80)
    args = parser.parse_args()

    records = _load_jsonl(Path(args.manifest_jsonl))
    payload = audit_manifest(records, top_k=int(args.top_k))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "manifest_quality_audit.json", payload)
    write_report(out_dir / "manifest_quality_audit.md", payload, int(args.max_report_families), int(args.max_report_cases))
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(out_dir),
                "summary": {
                    "records": payload["num_records"],
                    "selected_conditions": payload["total_selected_conditions"],
                    "train_candidate_conditions": payload["train_candidate_conditions"],
                    "decision_counts": payload["decision_counts"],
                    "risk_reason_counts": payload["risk_reason_counts"],
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
