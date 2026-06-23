from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path(
    "outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/group_01/probe/summary.json"
)
DEFAULT_NOTES = Path(
    "outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/"
    "group_01_failure_audit_v0/manual_failure_notes.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/group_01_failure_audit_v0"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _caption_lines(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in row.get("raw_prompt_segments") or []:
        if isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        elif isinstance(item, str):
            out.append(item)
    return out


def _canonical_ids(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for action in row.get("canonical_actions") or []:
        cid = action.get("canonical_id") or action.get("family_id") or action.get("prototype_id")
        if cid:
            out.append(str(cid))
    return out


def build_ledger(summary_rows: list[dict[str, Any]], note_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case = {str(row["case_id"]): row for row in summary_rows}
    records: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    recoverability_counts: Counter[str] = Counter()

    for note in note_rows:
        case_id = str(note["case_id"])
        row = by_case.get(case_id)
        if row is None:
            missing_cases.append(case_id)
            continue
        failure_families = [str(item) for item in note.get("failure_families") or []]
        failure_sources = [str(item) for item in note.get("failure_sources") or []]
        proposed_actions = [str(item) for item in note.get("proposed_actions") or []]
        family_counts.update(failure_families)
        source_counts.update(failure_sources)
        action_counts.update(proposed_actions)
        recoverability_counts.update([str(note.get("recoverability") or "unknown")])
        records.append(
            {
                "case_id": case_id,
                "priority": str(note.get("priority") or "P2"),
                "user_observation": str(note.get("user_observation") or ""),
                "missing_semantics": [str(item) for item in note.get("missing_semantics") or []],
                "failure_families": failure_families,
                "failure_sources": failure_sources,
                "recoverability": str(note.get("recoverability") or "unknown"),
                "proposed_actions": proposed_actions,
                "native_prompt": row.get("native_prompt") or row.get("gt_prompt") or "",
                "aml_auto_prompt": row.get("auto_prompt") or "",
                "hml3d_captions": _caption_lines(row),
                "canonical_ids": _canonical_ids(row),
                "gif_path": (
                    "outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/"
                    f"group_01/gifs/case_{case_id}.gif"
                ),
            }
        )

    return {
        "schema_version": "aml_momask_review_failure_ledger_v0",
        "source_summary": str(DEFAULT_SUMMARY),
        "note_count": len(note_rows),
        "matched_note_count": len(records),
        "missing_case_ids": missing_cases,
        "failure_family_counts": dict(family_counts),
        "failure_source_counts": dict(source_counts),
        "proposed_action_counts": dict(action_counts),
        "recoverability_counts": dict(recoverability_counts),
        "records": records,
    }


def write_report(path: Path, ledger: dict[str, Any]) -> None:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger.get("records") or []:
        families = row.get("failure_families") or ["uncategorized"]
        for family in families:
            by_family[str(family)].append(row)

    lines = [
        "# AML MoMask Review Failure Ledger v0",
        "",
        "This ledger records manual review feedback from the group_01 native-vs-AML MoMask review pack.",
        "It is an audit artifact only: it must not be imported by runtime AML extraction or Motion-BPE learning.",
        "",
        "## Summary",
        "",
        f"- notes: `{ledger['note_count']}`",
        f"- matched cases: `{ledger['matched_note_count']}`",
        f"- missing cases: `{ledger['missing_case_ids']}`",
        f"- failure families: `{ledger['failure_family_counts']}`",
        f"- failure sources: `{ledger['failure_source_counts']}`",
        f"- proposed actions: `{ledger['proposed_action_counts']}`",
        "",
        "## Review Rule",
        "",
        "Use this table to decide what full-HML3D observable/mining pass to run next.",
        "Do not patch individual case ids into the prompt renderer.",
        "",
    ]
    for family, rows in sorted(by_family.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.extend([f"## {family}", "", f"- cases: `{len(rows)}`", ""])
        lines.extend(
            [
                "| case | missing semantics | failure sources | proposed actions | native prompt | AML AutoPrompt | GIF |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            gif = Path(row["gif_path"])
            lines.append(
                "| `{case}` | {missing} | `{sources}` | `{actions}` | {native} | {auto} | [gif]({gif}) |".format(
                    case=row["case_id"],
                    missing=", ".join(row.get("missing_semantics") or []),
                    sources=", ".join(row.get("failure_sources") or []),
                    actions=", ".join(row.get("proposed_actions") or []),
                    native=str(row.get("native_prompt") or "").replace("|", "/"),
                    auto=str(row.get("aml_auto_prompt") or "").replace("|", "/"),
                    gif=gif.as_posix(),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--notes", default=str(DEFAULT_NOTES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary_rows = _load_json(Path(args.summary))
    notes_payload = _load_json(Path(args.notes))
    note_rows = notes_payload.get("notes") or []
    ledger = build_ledger(summary_rows, note_rows)
    ledger["source_summary"] = str(args.summary)
    ledger["source_notes"] = str(args.notes)
    out_dir = Path(args.output_dir)
    _write_json(out_dir / "failure_ledger.json", ledger)
    write_report(out_dir / "failure_ledger.md", ledger)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "matched": ledger["matched_note_count"],
                "families": ledger["failure_family_counts"],
                "sources": ledger["failure_source_counts"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
