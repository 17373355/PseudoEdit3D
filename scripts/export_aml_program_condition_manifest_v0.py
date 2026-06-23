"""Export AML selected_conditions from composable pattern-program search.

This converts motion-to-tree search debug output into the existing AML condition
dataset contract. It does not promote names from captions and does not train on
diagnostic nodes by default.

Pipeline:
1. Read `case_tree_search_results.json` from
   `search_aml_composable_pattern_program_v0.py`.
2. Select one or more program-tree hits per case/window by semantic level,
   score, and review status.
3. Emit `selected_conditions` for trainable/evaluable candidates and
   `deferred_conditions` for diagnostics or weak evidence.
4. Write a compact manifest and a summary report.

Quick run:
    python scripts/export_aml_program_condition_manifest_v0.py

Smoke test:
    python scripts/export_aml_program_condition_manifest_v0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SEARCH_RESULTS = Path(
    "outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0/case_tree_search_results.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0")

SELECTABLE_LEVELS = {
    "whole_body_pattern_candidate": 1.0,
    "multi_part_coordination": 0.85,
    "transition": 0.65,
    "component": 0.50,
    "local_component": 0.45,
}
DEFERRED_LEVELS = {"diagnostic_context"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _slot_values(evidence: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    span = evidence.get("span") or hit.get("span")
    match_detail = hit.get("match_detail") or {}
    values: dict[str, Any] = {}
    if isinstance(span, list) and len(span) == 2:
        values["span"] = [_safe_int(span[0]), _safe_int(span[1])]
    channel_hits = match_detail.get("channel_hits") or []
    values["source_event_count"] = len(evidence.get("member_symbols") or [])
    values["segment_count"] = 1
    if "whole_body_vertical" in channel_hits:
        values["vertical_amplitude_m"] = 0.0
    if "root_locomotion" in channel_hits:
        values["distance_m"] = 0.0
        values["direction"] = "unknown"
    if "root_rotation" in channel_hits:
        values["angle_deg"] = 0.0
        values["angle_bin"] = "unknown"
    if "left_arm" in channel_hits or "right_arm" in channel_hits or "bimanual" in channel_hits:
        values["bimanual_count"] = 1 if "bimanual" in channel_hits else 0
        values["left_arm_count"] = 1 if "left_arm" in channel_hits else 0
        values["right_arm_count"] = 1 if "right_arm" in channel_hits else 0
    return values


def _condition_from_hit(
    *,
    case_id: str,
    window_index: int,
    hit_rank: int,
    hit: dict[str, Any],
    evidence: dict[str, Any],
    decision: str,
    reason: str,
) -> dict[str, Any]:
    condition_entry = hit.get("condition_entry") or {}
    level = str(hit.get("semantic_level") or "unknown")
    score = _safe_float(hit.get("score"))
    level_weight = SELECTABLE_LEVELS.get(level, 0.0)
    status = "candidate" if decision == "selected" else "deferred"
    if level == "whole_body_pattern_candidate" and decision == "selected":
        status = "candidate_full"
    elif level == "multi_part_coordination" and decision == "selected":
        status = "candidate_composed"
    elif level == "transition" and decision == "selected":
        status = "candidate_transition"
    elif level in {"component", "local_component"} and decision == "selected":
        status = "candidate_component"
    return {
        "action_index": int(window_index),
        "hit_rank": int(hit_rank),
        "family_id": str(condition_entry.get("condition_entry_id") or condition_entry.get("condition_id") or hit.get("program_node_id") or "UNKNOWN"),
        "condition_id": str(condition_entry.get("condition_id") or ""),
        "program_node_id": str(hit.get("program_node_id") or ""),
        "motion_structure_label": str(hit.get("motion_structure_label") or ""),
        "status": status,
        "condition_weight": round(score * level_weight, 6) if decision == "selected" else 0.0,
        "screen_score": round(score, 6),
        "screen_reason": reason,
        "case_id": case_id,
        "semantic_level": level,
        "edit_scope": str(hit.get("edit_scope") or ""),
        "composition_policy": str(hit.get("composition_policy") or ""),
        "review_status": str(hit.get("review_status") or ""),
        "scope": str(hit.get("scope") or ""),
        "slot_values": _slot_values(evidence, hit),
        "match_detail": hit.get("match_detail") or {},
        "evidence": {
            "span": evidence.get("span"),
            "channels": evidence.get("channels") or [],
            "zones": evidence.get("zones") or [],
            "event_families": evidence.get("event_families") or [],
            "cluster_ids": evidence.get("cluster_ids") or [],
            "geometry_clusters": evidence.get("geometry_clusters") or [],
            "raw_geometry_clusters": evidence.get("raw_geometry_clusters") or [],
            "observable_refinement_tags": evidence.get("observable_refinement_tags") or [],
        },
        "caption_name_candidates_for_review_only": condition_entry.get("caption_name_candidates") or [],
    }


def _condition_priority(cond: dict[str, Any]) -> tuple[int, float, float, str]:
    level_rank = {
        "whole_body_pattern_candidate": 5,
        "multi_part_coordination": 4,
        "transition": 3,
        "component": 2,
        "local_component": 1,
    }
    return (
        level_rank.get(str(cond.get("semantic_level") or ""), 0),
        _safe_float(cond.get("condition_weight")),
        _safe_float(cond.get("screen_score")),
        str(cond.get("family_id") or ""),
    )


def _dedupe_conditions(conditions: list[dict[str, Any]], limit: int, *, max_per_span: int) -> list[dict[str, Any]]:
    seen_family_span: set[tuple[str, tuple[int, int] | None]] = set()
    span_counts: Counter[tuple[int, int] | None] = Counter()
    level_counts: Counter[str] = Counter()
    level_caps = {
        "whole_body_pattern_candidate": 2,
        "multi_part_coordination": 3,
        "transition": 2,
        "component": 2,
        "local_component": 1,
    }
    out: list[dict[str, Any]] = []
    for cond in sorted(
        conditions,
        key=lambda row: tuple(-value if isinstance(value, (int, float)) else value for value in _condition_priority(row)),
    ):
        span = cond.get("slot_values", {}).get("span")
        span_key = tuple(span) if isinstance(span, list) and len(span) == 2 else None
        family_span_key = (str(cond.get("family_id") or ""), span_key)
        level = str(cond.get("semantic_level") or "")
        if family_span_key in seen_family_span:
            continue
        if span_counts[span_key] >= max_per_span:
            continue
        if level_counts[level] >= level_caps.get(level, 1):
            continue
        seen_family_span.add(family_span_key)
        span_counts[span_key] += 1
        level_counts[level] += 1
        out.append(cond)
        if len(out) >= limit:
            break
    return out


def convert_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for window in case.get("top_windows") or []:
        evidence = window.get("evidence") or {}
        hits = list(window.get("hits") or [])
        for rank, hit in enumerate(hits[: int(args.max_hits_per_window)], start=1):
            level = str(hit.get("semantic_level") or "")
            score = _safe_float(hit.get("score"))
            review_status = str(hit.get("review_status") or "")
            is_diagnostic = level in DEFERRED_LEVELS or review_status == "diagnostic_only"
            if not is_diagnostic and level in SELECTABLE_LEVELS and score >= float(args.min_selected_score):
                selected.append(
                    _condition_from_hit(
                        case_id=str(case.get("case_id") or ""),
                        window_index=_safe_int(window.get("window_index"), 0),
                        hit_rank=rank,
                        hit=hit,
                        evidence=evidence,
                        decision="selected",
                        reason="non_diagnostic_program_hit_above_threshold",
                    )
                )
            elif score >= float(args.min_deferred_score):
                deferred.append(
                    _condition_from_hit(
                        case_id=str(case.get("case_id") or ""),
                        window_index=_safe_int(window.get("window_index"), 0),
                        hit_rank=rank,
                        hit=hit,
                        evidence=evidence,
                        decision="deferred",
                        reason="diagnostic_or_below_selected_threshold",
                    )
                )
    selected = _dedupe_conditions(selected, int(args.max_selected_conditions), max_per_span=int(args.max_selected_per_span))
    selected_indices = list(range(len(selected)))
    return {
        "schema_version": "aml_program_condition_manifest_v0",
        "case_id": str(case.get("case_id") or ""),
        "num_frames": _safe_int(case.get("num_frames"), 0) or None,
        "reference_prompt": (case.get("caption_texts") or [""])[0] if case.get("caption_texts") else "",
        "caption_texts_for_review_only": case.get("caption_texts") or [],
        "caption_alias_ids_for_review_only": case.get("caption_alias_ids") or [],
        "case_pattern_type": (case.get("case_classification") or {}).get("case_pattern_type"),
        "selected_condition_indices": selected_indices,
        "selected_conditions": selected,
        "deferred_conditions": deferred,
    }


def convert_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return [convert_case(case, args) for case in cases]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_status = Counter()
    selected_family = Counter()
    selected_level = Counter()
    selected_scope = Counter()
    deferred_level = Counter()
    case_type = Counter()
    selected_counts = []
    deferred_counts = []
    for record in records:
        case_type.update([str(record.get("case_pattern_type") or "unknown")])
        selected = record.get("selected_conditions") or []
        deferred = record.get("deferred_conditions") or []
        selected_counts.append(len(selected))
        deferred_counts.append(len(deferred))
        for cond in selected:
            selected_status.update([str(cond.get("status") or "")])
            selected_family.update([str(cond.get("family_id") or "")])
            selected_level.update([str(cond.get("semantic_level") or "")])
            selected_scope.update([str(cond.get("edit_scope") or "")])
        for cond in deferred:
            deferred_level.update([str(cond.get("semantic_level") or "")])
    return {
        "schema_version": "aml_program_condition_manifest_summary_v0",
        "num_cases": len(records),
        "train_ready_records": sum(1 for value in selected_counts if value > 0),
        "empty_selected_records": sum(1 for value in selected_counts if value == 0),
        "total_selected_conditions": sum(selected_counts),
        "total_deferred_conditions": sum(deferred_counts),
        "selected_count_min": min(selected_counts) if selected_counts else 0,
        "selected_count_max": max(selected_counts) if selected_counts else 0,
        "selected_count_mean": round(sum(selected_counts) / max(1, len(selected_counts)), 4),
        "case_pattern_type_counts": case_type.most_common(),
        "selected_status_counts": selected_status.most_common(),
        "selected_semantic_level_counts": selected_level.most_common(),
        "selected_edit_scope_counts": selected_scope.most_common(),
        "deferred_semantic_level_counts": deferred_level.most_common(),
        "selected_family_counts_top30": selected_family.most_common(30),
    }


def write_report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]], max_examples: int) -> None:
    lines = [
        "# AML Program Condition Manifest v0",
        "",
        f"- cases: `{summary['num_cases']}`",
        f"- train-ready records: `{summary['train_ready_records']}`",
        f"- empty-selected records: `{summary['empty_selected_records']}`",
        f"- selected conditions: `{summary['total_selected_conditions']}`",
        f"- deferred conditions: `{summary['total_deferred_conditions']}`",
        f"- selected per case: min `{summary['selected_count_min']}`, mean `{summary['selected_count_mean']}`, max `{summary['selected_count_max']}`",
        "",
        "## Selected Semantic Levels",
        "",
    ]
    for key, value in summary.get("selected_semantic_level_counts") or []:
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Selected Edit Scopes", ""])
    for key, value in summary.get("selected_edit_scope_counts") or []:
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Selected Families", ""])
    for key, value in (summary.get("selected_family_counts_top30") or [])[:20]:
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Examples", ""])
    for record in records[:max_examples]:
        lines.append(f"### {record.get('case_id')} | {record.get('case_pattern_type')}")
        if record.get("reference_prompt"):
            lines.append(f"- caption: {record.get('reference_prompt')}")
        selected = record.get("selected_conditions") or []
        deferred = record.get("deferred_conditions") or []
        lines.append(f"- selected={len(selected)} deferred={len(deferred)}")
        for cond in selected[:5]:
            lines.append(
                f"  - score={cond.get('screen_score')} weight={cond.get('condition_weight')} "
                f"level={cond.get('semantic_level')} edit={cond.get('edit_scope')} "
                f"family={cond.get('motion_structure_label')} span={cond.get('slot_values', {}).get('span')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any], max_examples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selected_conditions.jsonl", records)
    _write_json(output_dir / "manifest_summary.json", summary)
    _write_json(output_dir / "selected_conditions_preview.json", records[:max_examples])
    write_report(output_dir / "manifest_report.md", summary, records, max_examples)


def run_self_test() -> None:
    case = {
        "case_id": "case",
        "caption_texts": ["synthetic"],
        "case_classification": {"case_pattern_type": "whole_body_or_full_composition_candidate"},
        "top_windows": [
            {
                "window_index": 1,
                "evidence": {
                    "span": [1, 10],
                    "channels": ["bimanual", "whole_body_vertical"],
                    "zones": ["upper", "vertical"],
                    "event_families": ["BIMANUAL_PERIODIC", "WHOLE_BODY_VERTICAL"],
                    "cluster_ids": ["BI_RAISE_SPREAD", "WB_VERT_UP"],
                    "member_symbols": ["a", "b"],
                },
                "hits": [
                    {
                        "score": 0.9,
                        "program_node_id": "node",
                        "motion_structure_label": "bimanual_vertical",
                        "semantic_level": "whole_body_pattern_candidate",
                        "edit_scope": "whole_body",
                        "composition_policy": "may_bind",
                        "review_status": "review_candidate",
                        "scope": "full_composition_candidate",
                        "match_detail": {"channel_hits": ["bimanual", "whole_body_vertical"]},
                        "condition_entry": {
                            "condition_entry_id": "AMLCPNODE_TEST",
                            "condition_id": "AMLCP_TEST",
                        },
                    }
                ],
            }
        ],
    }
    ns = argparse.Namespace(
        max_hits_per_window=2,
        min_selected_score=0.45,
        min_deferred_score=0.2,
        max_selected_conditions=4,
        max_selected_per_span=2,
    )
    records = convert_cases([case], ns)
    assert len(records[0]["selected_conditions"]) == 1
    assert records[0]["selected_conditions"][0]["family_id"] == "AMLCPNODE_TEST"
    with tempfile.TemporaryDirectory() as tmp:
        summary = summarize(records)
        write_outputs(Path(tmp), records, summary, 5)
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected AML conditions from program-tree search results.")
    parser.add_argument("--search-results", default=str(DEFAULT_SEARCH_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-selected-score", type=float, default=0.45)
    parser.add_argument("--min-deferred-score", type=float, default=0.20)
    parser.add_argument("--max-hits-per-window", type=int, default=4)
    parser.add_argument("--max-selected-conditions", type=int, default=8)
    parser.add_argument("--max-selected-per-span", type=int, default=2)
    parser.add_argument("--max-preview-examples", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    cases = _read_json(Path(args.search_results))
    if not isinstance(cases, list):
        raise ValueError(f"search results must be a list: {args.search_results}")
    records = convert_cases(cases, args)
    summary = summarize(records)
    write_outputs(Path(args.output_dir), records, summary, int(args.max_preview_examples))
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(args.output_dir),
                "summary": summary,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
