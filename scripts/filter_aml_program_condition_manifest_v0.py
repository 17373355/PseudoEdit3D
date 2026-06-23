"""Filter AML program condition manifests into a conservative train split.

This script is intentionally policy-only. It does not create new conditions and
does not use captions to match motion. It reads the audit policy from
`audit_aml_program_condition_manifest_v0.py`, keeps only train-candidate
selected conditions, and moves the rest to `deferred_conditions`.

Pipeline:
1. Run `export_aml_program_condition_manifest_v0.py` to build a candidate manifest.
2. Run `audit_aml_program_condition_manifest_v0.py` to inspect quality.
3. Run this script to create the first clean encoder-training manifest.

Quick run:
    python scripts/filter_aml_program_condition_manifest_v0.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_aml_program_condition_manifest_v0 import _condition_risk, _write_json


DEFAULT_INPUT_JSONL = Path(
    "outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/selected_conditions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _selected_indices(selected: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    for idx, _cond in enumerate(selected):
        indices.append(idx)
    return indices


def filter_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    moved_reason_counts: Counter[str] = Counter()
    clean_condition_counts: Counter[str] = Counter()
    moved_condition_counts: Counter[str] = Counter()
    clean_record_count = 0

    for record in records:
        clean_selected: list[dict[str, Any]] = []
        moved_deferred: list[dict[str, Any]] = []
        for cond in record.get("selected_conditions") or []:
            decision, reasons = _condition_risk(cond, record)
            decision_counts.update([decision])
            if decision == "train_candidate":
                keep = dict(cond)
                keep["audit_decision"] = decision
                clean_selected.append(keep)
                clean_condition_counts.update([str(cond.get("motion_structure_label") or "")])
            else:
                moved = dict(cond)
                moved["status"] = "deferred_audit_filtered"
                moved["condition_weight"] = 0.0
                moved["audit_decision"] = decision
                moved["audit_risk_reasons"] = reasons
                moved_deferred.append(moved)
                moved_reason_counts.update(reasons)
                moved_condition_counts.update([str(cond.get("motion_structure_label") or "")])

        clean_record = dict(record)
        existing_deferred = list(record.get("deferred_conditions") or [])
        clean_record["selected_conditions"] = clean_selected
        clean_record["deferred_conditions"] = existing_deferred + moved_deferred
        clean_record["selected_condition_indices"] = _selected_indices(clean_selected)
        clean_record["audit_filter"] = {
            "source_selected_count": len(record.get("selected_conditions") or []),
            "kept_train_candidate_count": len(clean_selected),
            "moved_to_deferred_count": len(moved_deferred),
        }
        if clean_selected:
            clean_record_count += 1
        out.append(clean_record)

    summary = {
        "schema_version": "aml_program_condition_manifest_train_clean_v0",
        "records": len(out),
        "train_ready_records": clean_record_count,
        "empty_selected_records": len(out) - clean_record_count,
        "source_selected_conditions": sum(len(record.get("selected_conditions") or []) for record in records),
        "clean_selected_conditions": sum(len(record.get("selected_conditions") or []) for record in out),
        "moved_to_deferred_conditions": sum(
            int((record.get("audit_filter") or {}).get("moved_to_deferred_count") or 0)
            for record in out
        ),
        "audit_decision_counts": dict(decision_counts),
        "moved_reason_counts": dict(moved_reason_counts),
        "clean_condition_labels_top30": clean_condition_counts.most_common(30),
        "moved_condition_labels_top30": moved_condition_counts.most_common(30),
        "policy": {
            "selected_conditions": "only audit train_candidate conditions",
            "deferred_conditions": "original deferred plus audit-filtered selected conditions",
            "caption_usage": "captions are only risk/audit hints, never condition match rules",
        },
    }
    return out, summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AML Program Condition Train-Clean Manifest v0",
        "",
        f"- records: `{summary['records']}`",
        f"- train-ready records: `{summary['train_ready_records']}`",
        f"- empty selected records: `{summary['empty_selected_records']}`",
        f"- source selected conditions: `{summary['source_selected_conditions']}`",
        f"- clean selected conditions: `{summary['clean_selected_conditions']}`",
        f"- moved to deferred: `{summary['moved_to_deferred_conditions']}`",
        f"- audit decisions: `{summary['audit_decision_counts']}`",
        f"- moved reasons: `{summary['moved_reason_counts']}`",
        "",
        "## Clean Labels",
        "",
    ]
    for label, count in summary.get("clean_condition_labels_top30") or []:
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Deferred Labels", ""])
    for label, count in summary.get("moved_condition_labels_top30") or []:
        lines.append(f"- `{label}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter AML program condition manifest for clean training.")
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    records = _load_jsonl(Path(args.input_jsonl))
    filtered, summary = filter_records(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "selected_conditions.jsonl", filtered)
    _write_json(out_dir / "manifest_summary.json", summary)
    write_report(out_dir / "manifest_report.md", summary)
    preview = [record for record in filtered if record.get("selected_conditions")][:20]
    _write_json(out_dir / "selected_conditions_preview.json", preview)
    print(json.dumps({"ok": True, "output_dir": str(out_dir), "summary": summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
