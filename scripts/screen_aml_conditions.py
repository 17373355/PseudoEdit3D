from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


QUALITY_FACTORS = {
    "motion_estimate": 1.0,
    "categorical_estimate": 0.96,
    "temporal_span_estimate": 0.94,
    "proxy_motion_estimate": 0.74,
    "approximate_event_count": 0.72,
}

STATUS_ORDER = {
    "stable": 0,
    "candidate": 1,
    "proxy": 2,
    "unknown": 3,
}

DECISION_ORDER = {
    "selected": 0,
    "deferred": 1,
    "dropped": 2,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float], default: float) -> float:
    values = [float(v) for v in values if v is not None]
    return mean(values) if values else default


def _load_jsonl(path: Path, max_cases: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_cases and len(records) >= max_cases:
                break
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _slot_confidence(cond: dict[str, Any]) -> float:
    values = [_safe_float(value, 0.0) for value in (cond.get("slot_confidences") or {}).values()]
    fallback = _safe_float(cond.get("label_confidence"), 0.52)
    return _safe_mean(values, fallback)


def _quality_factor(cond: dict[str, Any]) -> float:
    qualities = list((cond.get("slot_qualities") or {}).values())
    if not qualities:
        return 0.82
    factors = [QUALITY_FACTORS.get(str(quality), 0.82) for quality in qualities]
    return _safe_mean(factors, 0.82)


def _slot_completeness(cond: dict[str, Any]) -> float:
    missing = list(cond.get("missing_required_slots") or [])
    if not missing:
        return 1.0
    known = len(cond.get("slot_values") or {})
    denom = max(known + len(missing), 1)
    return max(0.0, 1.0 - len(missing) / denom)


def condition_score(cond: dict[str, Any]) -> float:
    weight = _safe_float(cond.get("condition_weight"), 0.0)
    if weight <= 0.0:
        return 0.0
    if cond.get("probe_visible") is False:
        return 0.0
    if cond.get("missing_required_slots"):
        return 0.0

    label_confidence = _safe_float(cond.get("label_confidence"), _slot_confidence(cond))
    slot_confidence = _slot_confidence(cond)
    reliability = 0.45 + 0.35 * label_confidence + 0.20 * slot_confidence
    slot_count = len(cond.get("slot_values") or {})
    slot_density = min(1.0, 0.82 + 0.045 * slot_count)
    score = weight * reliability * _quality_factor(cond) * _slot_completeness(cond) * slot_density
    return round(max(0.0, min(1.0, score)), 4)


def decision_reason(cond: dict[str, Any], score: float, threshold: float, decision: str) -> str:
    if _safe_float(cond.get("condition_weight"), 0.0) <= 0.0:
        if cond.get("probe_visible") is False:
            return "zero_weight_probe_hidden"
        if cond.get("missing_required_slots"):
            return "zero_weight_missing_required_slots"
        return "zero_weight"
    if cond.get("missing_required_slots"):
        return "missing_required_slots"
    if decision == "selected":
        if score < threshold:
            return "selected_by_case_minimum"
        return "above_threshold"
    if decision == "deferred":
        return "near_threshold_or_rank_limited"
    return "below_threshold"


def screen_record(
    record: dict[str, Any],
    threshold: float,
    defer_ratio: float,
    max_selected_per_case: int,
    min_selected_per_case: int,
) -> dict[str, Any]:
    conditions = []
    for cond in record.get("conditions") or []:
        copied = dict(cond)
        copied["screen_score"] = condition_score(copied)
        conditions.append(copied)

    ranked = sorted(
        [cond for cond in conditions if cond["screen_score"] > 0.0],
        key=lambda cond: (float(cond["screen_score"]), -int(cond.get("action_index") or 0)),
        reverse=True,
    )
    selected_ids = {
        id(cond)
        for cond in ranked
        if cond["screen_score"] >= threshold
    }
    selected_ids = set(list(selected_ids)[:max_selected_per_case])
    if len(selected_ids) < min_selected_per_case:
        for cond in ranked:
            selected_ids.add(id(cond))
            if len(selected_ids) >= min_selected_per_case:
                break
    selected_ids = set(list(selected_ids)[:max_selected_per_case])

    defer_threshold = threshold * defer_ratio
    for cond in conditions:
        if id(cond) in selected_ids:
            decision = "selected"
        elif cond["screen_score"] >= defer_threshold:
            decision = "deferred"
        else:
            decision = "dropped"
        cond["screen_decision"] = decision
        cond["screen_reason"] = decision_reason(cond, cond["screen_score"], threshold, decision)

    out = dict(record)
    out["schema_version"] = "aml_condition_screening_v1"
    out["source_schema_version"] = record.get("schema_version")
    out["screening_config"] = {
        "threshold": threshold,
        "defer_ratio": defer_ratio,
        "max_selected_per_case": max_selected_per_case,
        "min_selected_per_case": min_selected_per_case,
    }
    out["conditions"] = conditions
    out["selected_condition_indices"] = [
        int(cond.get("action_index") or 0)
        for cond in conditions
        if cond.get("screen_decision") == "selected"
    ]
    out["screen_decision_counts"] = dict(Counter(cond["screen_decision"] for cond in conditions))
    return out


def selected_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "aml_selected_conditions_v1",
        "case_id": record.get("case_id"),
        "num_frames": record.get("num_frames"),
        "reference_prompt": record.get("selected_hml3d_prompt_for_reference_only"),
        "selected_condition_indices": record.get("selected_condition_indices") or [],
        "selected_conditions": [
            {
                "action_index": cond.get("action_index"),
                "family_id": cond.get("family_id"),
                "status": cond.get("status"),
                "condition_weight": cond.get("condition_weight"),
                "screen_score": cond.get("screen_score"),
                "screen_reason": cond.get("screen_reason"),
                "slot_values": cond.get("slot_values") or {},
            }
            for cond in record.get("conditions") or []
            if cond.get("screen_decision") == "selected"
        ],
        "deferred_conditions": [
            {
                "action_index": cond.get("action_index"),
                "family_id": cond.get("family_id"),
                "status": cond.get("status"),
                "condition_weight": cond.get("condition_weight"),
                "screen_score": cond.get("screen_score"),
                "screen_reason": cond.get("screen_reason"),
                "slot_values": cond.get("slot_values") or {},
            }
            for cond in record.get("conditions") or []
            if cond.get("screen_decision") == "deferred"
        ],
    }


def summarize(records: list[dict[str, Any]], source_manifest: str) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    status_by_decision: dict[str, Counter[str]] = defaultdict(Counter)
    family_by_decision: dict[str, Counter[str]] = defaultdict(Counter)
    score_by_decision: dict[str, list[float]] = defaultdict(list)
    total_conditions = 0
    zero_weight = 0
    missing_required = 0
    case_rows = []

    for record in records:
        case_decisions: Counter[str] = Counter()
        case_scores: list[float] = []
        for cond in record.get("conditions") or []:
            total_conditions += 1
            decision = str(cond.get("screen_decision") or "dropped")
            status = str(cond.get("status") or "unknown")
            family = str(cond.get("family_id") or "UNKNOWN")
            score = _safe_float(cond.get("screen_score"), 0.0)
            decision_counts[decision] += 1
            status_counts[status] += 1
            status_by_decision[decision][status] += 1
            family_by_decision[decision][family] += 1
            score_by_decision[decision].append(score)
            case_decisions[decision] += 1
            case_scores.append(score)
            if _safe_float(cond.get("condition_weight"), 0.0) <= 0.0:
                zero_weight += 1
            if cond.get("missing_required_slots"):
                missing_required += 1
        case_rows.append(
            {
                "case_id": record.get("case_id"),
                "num_conditions": len(record.get("conditions") or []),
                "decision_counts": dict(case_decisions),
                "max_score": round(max(case_scores), 4) if case_scores else 0.0,
                "mean_score": round(mean(case_scores), 4) if case_scores else 0.0,
            }
        )

    return {
        "schema_version": "aml_condition_screening_summary_v1",
        "source_manifest": source_manifest,
        "num_cases": len(records),
        "total_conditions": total_conditions,
        "decision_counts": sorted(decision_counts.items(), key=lambda kv: DECISION_ORDER.get(kv[0], 99)),
        "status_counts": sorted(status_counts.items(), key=lambda kv: (STATUS_ORDER.get(kv[0], 99), kv[0])),
        "status_by_decision": {
            key: sorted(value.items(), key=lambda kv: (STATUS_ORDER.get(kv[0], 99), kv[0]))
            for key, value in sorted(status_by_decision.items(), key=lambda kv: DECISION_ORDER.get(kv[0], 99))
        },
        "score_by_decision": {
            key: {
                "count": len(values),
                "mean": round(mean(values), 4) if values else 0.0,
                "min": round(min(values), 4) if values else 0.0,
                "max": round(max(values), 4) if values else 0.0,
            }
            for key, values in sorted(score_by_decision.items(), key=lambda kv: DECISION_ORDER.get(kv[0], 99))
        },
        "top_families_by_decision": {
            key: value.most_common(20)
            for key, value in sorted(family_by_decision.items(), key=lambda kv: DECISION_ORDER.get(kv[0], 99))
        },
        "zero_weight_condition_count": zero_weight,
        "missing_required_condition_count": missing_required,
        "cases": case_rows,
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def write_report(path: Path, summary: dict[str, Any], selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    decision_counts = dict(summary.get("decision_counts") or [])
    status_counts = dict(summary.get("status_counts") or [])
    lines = [
        "# AML Condition Screening v1",
        "",
        "## Inputs",
        "",
        f"- manifest: `{args.manifest}`",
        "",
        "## Rule",
        "",
        f"- score threshold: `{args.threshold}`",
        f"- defer threshold: `{round(args.threshold * args.defer_ratio, 4)}`",
        f"- max selected per case: `{args.max_selected_per_case}`",
        f"- min selected per case: `{args.min_selected_per_case}`",
        "",
        "The score combines condition weight, label confidence, slot confidence, slot quality, and required-slot coverage.",
        "",
        "## Aggregate Counts",
        "",
        f"- cases: `{summary.get('num_cases')}`",
        f"- conditions: `{summary.get('total_conditions')}`",
        f"- selected: `{decision_counts.get('selected', 0)}`",
        f"- deferred: `{decision_counts.get('deferred', 0)}`",
        f"- dropped: `{decision_counts.get('dropped', 0)}`",
        f"- stable: `{status_counts.get('stable', 0)}`",
        f"- candidate: `{status_counts.get('candidate', 0)}`",
        f"- proxy: `{status_counts.get('proxy', 0)}`",
        f"- unknown: `{status_counts.get('unknown', 0)}`",
        f"- zero-weight conditions: `{summary.get('zero_weight_condition_count')}`",
        f"- missing required slots: `{summary.get('missing_required_condition_count')}`",
        "",
        "## Status By Decision",
        "",
    ]
    rows = []
    for decision, values in summary.get("status_by_decision", {}).items():
        row_counts = dict(values)
        rows.append([
            decision,
            row_counts.get("stable", 0),
            row_counts.get("candidate", 0),
            row_counts.get("proxy", 0),
            row_counts.get("unknown", 0),
        ])
    lines.append(_table(["decision", "stable", "candidate", "proxy", "unknown"], rows))
    lines.extend(["", "## Top Selected Families", ""])
    top_selected = (summary.get("top_families_by_decision") or {}).get("selected") or []
    lines.append(_table(["family", "count"], [[family, count] for family, count in top_selected[:20]]))
    lines.extend(["", "## Selected Examples", ""])
    example_rows = []
    for record in selected[: min(len(selected), args.example_cases)]:
        conds = record.get("selected_conditions") or []
        families = ", ".join(str(cond.get("family_id")) for cond in conds[:8])
        scores = ", ".join(str(cond.get("screen_score")) for cond in conds[:8])
        example_rows.append([record.get("case_id"), len(conds), families, scores])
    lines.append(_table(["case_id", "selected_count", "families", "scores"], example_rows))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `screened_conditions.jsonl`: all conditions with `screen_score`, `screen_decision`, and `screen_reason`.",
            "- `selected_conditions.jsonl`: selected/deferred-only compact records for downstream use.",
            "- `summary.json`: aggregate counts and per-case score summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--defer-ratio", type=float, default=0.75)
    parser.add_argument("--max-selected-per-case", type=int, default=8)
    parser.add_argument("--min-selected-per-case", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--example-cases", type=int, default=20)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(manifest, args.max_cases)
    screened = [
        screen_record(
            record=record,
            threshold=args.threshold,
            defer_ratio=args.defer_ratio,
            max_selected_per_case=args.max_selected_per_case,
            min_selected_per_case=args.min_selected_per_case,
        )
        for record in records
    ]
    selected = [selected_record(record) for record in screened]
    summary = summarize(screened, str(manifest))

    _write_jsonl(out_dir / "screened_conditions.jsonl", screened)
    _write_jsonl(out_dir / "selected_conditions.jsonl", selected)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "screening_report.md", summary, selected, args)

    print(f"saved={out_dir}")
    print(f"cases={summary['num_cases']} conditions={summary['total_conditions']}")
    print(f"decisions={summary['decision_counts']}")


if __name__ == "__main__":
    main()
