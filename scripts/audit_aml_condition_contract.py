from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REQUIRED_RECORD_FIELDS = [
    "case_id",
    "num_frames",
    "selected_condition_indices",
    "selected_conditions",
    "deferred_conditions",
]

REQUIRED_CONDITION_FIELDS = [
    "action_index",
    "family_id",
    "status",
    "condition_weight",
    "screen_score",
    "screen_reason",
    "slot_values",
]

STRING_SLOT_NAMES = {"direction", "speed", "angle_bin", "dominant_side"}
SPAN_SLOT_NAMES = {"span"}
NUMERIC_SLOT_NAMES = {
    "distance_m",
    "path_length_m",
    "angle_deg",
    "magnitude",
    "vertical_amplitude_m",
    "mean_vertical_amplitude_m",
    "root_height_gain_m",
    "curvature_rad",
    "circle_score",
    "count",
    "source_event_count",
    "segment_count",
    "locomotion_segment_count",
    "turn_count",
    "left_arm_count",
    "right_arm_count",
    "bimanual_count",
    "raise_spread_count",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if k != "_line_no"}
            f.write(json.dumps(clean, ensure_ascii=True, sort_keys=True) + "\n")


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    return type(value).__name__


def _slot_type_issue(slot_name: str, value: Any) -> str | None:
    if slot_name in SPAN_SLOT_NAMES:
        if not isinstance(value, list) or len(value) != 2:
            return "span_not_len2_list"
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return "span_non_numeric"
        return None
    if slot_name in STRING_SLOT_NAMES:
        if not isinstance(value, str):
            return "expected_string"
        return None
    if slot_name in NUMERIC_SLOT_NAMES:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "expected_numeric"
        return None
    return "unknown_slot_name"


def _condition_rows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for cond in record.get("selected_conditions") or []:
        rows.append(("selected", cond))
    for cond in record.get("deferred_conditions") or []:
        rows.append(("deferred", cond))
    return rows


def audit(records: list[dict[str, Any]], selected_jsonl: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    record_missing_fields: Counter[str] = Counter()
    condition_missing_fields: Counter[str] = Counter()
    slot_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    slot_type_issues: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    selected_count_by_case: list[int] = []
    deferred_count_by_case: list[int] = []
    empty_selected_records: list[dict[str, Any]] = []
    train_ready_records: list[dict[str, Any]] = []
    invalid_condition_examples: list[dict[str, Any]] = []

    for record in records:
        for field in REQUIRED_RECORD_FIELDS:
            if field not in record:
                record_missing_fields[field] += 1
        selected = list(record.get("selected_conditions") or [])
        deferred = list(record.get("deferred_conditions") or [])
        selected_count_by_case.append(len(selected))
        deferred_count_by_case.append(len(deferred))
        if selected:
            train_ready_records.append(record)
        else:
            empty_selected_records.append(record)

        for decision, cond in _condition_rows(record):
            decision_counts[decision] += 1
            status_counts[str(cond.get("status") or "unknown")] += 1
            family_counts[str(cond.get("family_id") or "UNKNOWN")] += 1
            for field in REQUIRED_CONDITION_FIELDS:
                if field not in cond:
                    condition_missing_fields[field] += 1
                    if len(invalid_condition_examples) < 20:
                        invalid_condition_examples.append(
                            {
                                "case_id": record.get("case_id"),
                                "action_index": cond.get("action_index"),
                                "field": field,
                                "issue": "missing_condition_field",
                            }
                        )
            for slot_name, value in (cond.get("slot_values") or {}).items():
                slot_type_counts[slot_name][_type_name(value)] += 1
                issue = _slot_type_issue(slot_name, value)
                if issue:
                    key = f"{slot_name}:{issue}"
                    slot_type_issues[key] += 1
                    if len(invalid_condition_examples) < 20:
                        invalid_condition_examples.append(
                            {
                                "case_id": record.get("case_id"),
                                "action_index": cond.get("action_index"),
                                "slot": slot_name,
                                "issue": issue,
                                "value": value,
                            }
                        )

    total_conditions = sum(decision_counts.values())
    summary = {
        "schema_version": "aml_condition_dataset_contract_audit_v1",
        "source_selected_jsonl": selected_jsonl,
        "num_records": len(records),
        "train_ready_records": len(train_ready_records),
        "empty_selected_records": len(empty_selected_records),
        "total_selected_conditions": decision_counts.get("selected", 0),
        "total_deferred_conditions": decision_counts.get("deferred", 0),
        "total_conditions_in_compact_records": total_conditions,
        "selected_count_by_case": {
            "min": min(selected_count_by_case) if selected_count_by_case else 0,
            "max": max(selected_count_by_case) if selected_count_by_case else 0,
            "mean": round(mean(selected_count_by_case), 4) if selected_count_by_case else 0.0,
        },
        "deferred_count_by_case": {
            "min": min(deferred_count_by_case) if deferred_count_by_case else 0,
            "max": max(deferred_count_by_case) if deferred_count_by_case else 0,
            "mean": round(mean(deferred_count_by_case), 4) if deferred_count_by_case else 0.0,
        },
        "record_missing_fields": dict(record_missing_fields),
        "condition_missing_fields": dict(condition_missing_fields),
        "slot_type_counts": {
            key: dict(value)
            for key, value in sorted(slot_type_counts.items())
        },
        "slot_type_issues": dict(slot_type_issues),
        "status_counts": status_counts.most_common(),
        "family_counts_top30": family_counts.most_common(30),
        "empty_selected_case_ids": [record.get("case_id") for record in empty_selected_records],
        "invalid_condition_examples": invalid_condition_examples,
        "contract_status": "pass" if not record_missing_fields and not condition_missing_fields and not slot_type_issues else "warn",
        "recommended_downstream_policy": {
            "training": "use train_ready_selected_conditions.jsonl",
            "empty_selected_cases": "keep empty_selected_cases.jsonl as no-condition or low-evidence audit cases",
            "deferred_conditions": "do not train as hard positive by default; keep for diagnostics or weak auxiliary conditions",
        },
    }
    return summary, train_ready_records, empty_selected_records


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def write_report(path: Path, summary: dict[str, Any], max_empty_examples: int) -> None:
    lines = [
        "# AML Condition Dataset Contract Audit",
        "",
        "## Source",
        "",
        f"- selected JSONL: `{summary['source_selected_jsonl']}`",
        "",
        "## Contract Status",
        "",
        f"- status: `{summary['contract_status']}`",
        f"- records: `{summary['num_records']}`",
        f"- train-ready records: `{summary['train_ready_records']}`",
        f"- empty-selected records: `{summary['empty_selected_records']}`",
        f"- selected conditions: `{summary['total_selected_conditions']}`",
        f"- deferred conditions: `{summary['total_deferred_conditions']}`",
        "",
        "Train-ready means the case has at least one selected condition.",
        "",
        "## Per-Case Counts",
        "",
        _table(
            ["field", "min", "mean", "max"],
            [
                [
                    "selected",
                    summary["selected_count_by_case"]["min"],
                    summary["selected_count_by_case"]["mean"],
                    summary["selected_count_by_case"]["max"],
                ],
                [
                    "deferred",
                    summary["deferred_count_by_case"]["min"],
                    summary["deferred_count_by_case"]["mean"],
                    summary["deferred_count_by_case"]["max"],
                ],
            ],
        ),
        "",
        "## Field Checks",
        "",
        f"- missing record fields: `{summary['record_missing_fields']}`",
        f"- missing condition fields: `{summary['condition_missing_fields']}`",
        f"- slot type issues: `{summary['slot_type_issues']}`",
        "",
        "## Slot Types",
        "",
    ]
    slot_rows = []
    for slot_name, counts in summary.get("slot_type_counts", {}).items():
        slot_rows.append([slot_name, ", ".join(f"{k}:{v}" for k, v in counts.items())])
    lines.append(_table(["slot", "types"], slot_rows))
    lines.extend(["", "## Top Families", ""])
    lines.append(_table(["family", "count"], summary.get("family_counts_top30", [])[:20]))
    lines.extend(["", "## Empty Selected Cases", ""])
    empty_ids = summary.get("empty_selected_case_ids") or []
    if empty_ids:
        lines.append(", ".join(str(case_id) for case_id in empty_ids[:max_empty_examples]))
        if len(empty_ids) > max_empty_examples:
            lines.append(f"... {len(empty_ids) - max_empty_examples} more")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Recommended Downstream Policy",
            "",
            "- Use `train_ready_selected_conditions.jsonl` for the first dataset smoke.",
            "- Keep `empty_selected_cases.jsonl` as audit material; do not silently train on empty-condition records.",
            "- Keep deferred conditions as diagnostics or weak auxiliary conditions, not hard positives by default.",
            "",
            "## Outputs",
            "",
            "- `dataset_contract.json`: machine-readable audit.",
            "- `dataset_contract.md`: this report.",
            "- `train_ready_selected_conditions.jsonl`: records with at least one selected condition.",
            "- `empty_selected_cases.jsonl`: records with zero selected conditions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-empty-examples", type=int, default=50)
    args = parser.parse_args()

    selected_jsonl = Path(args.selected_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(selected_jsonl)
    summary, train_ready, empty_selected = audit(records, str(selected_jsonl))

    (out_dir / "dataset_contract.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "dataset_contract.md", summary, args.max_empty_examples)
    _write_jsonl(out_dir / "train_ready_selected_conditions.jsonl", train_ready)
    _write_jsonl(out_dir / "empty_selected_cases.jsonl", empty_selected)

    print(f"saved={out_dir}")
    print(
        "records={num_records} train_ready={train_ready_records} empty_selected={empty_selected_records} "
        "selected={total_selected_conditions} deferred={total_deferred_conditions} status={contract_status}".format(**summary)
    )


if __name__ == "__main__":
    main()
