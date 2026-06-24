"""Audit motion-evidence split axes on composable AML tree-search results.

This script is deliberately separate from the runtime tree search. It tests
whether a broad pending node can be split by data-defined motion evidence before
we promote it into the AML pattern program.

Typical use:
    python scripts/audit_v1_support_state_split_axes.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_AXIS_SPEC = Path("pseudoedit3d/edit/aml_pattern_split_axes.json")
DEFAULT_SEARCH_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_search_v0"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_split_axis_audit_v0"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id") or ""): dict(row) for row in cases}


def _cluster_set(evidence: dict[str, Any]) -> set[str]:
    clusters = set(str(item) for item in evidence.get("cluster_ids") or [])
    for key in ("geometry_clusters", "raw_geometry_clusters"):
        for item in evidence.get(key) or []:
            clusters.add(str(item).rsplit("/", 1)[-1])
    for symbol in evidence.get("member_symbols") or []:
        head = str(symbol).split("|", 1)[0]
        if "/" in head:
            clusters.add(head.rsplit("/", 1)[-1])
    return clusters


def _group_hits(groups: list[dict[str, Any]], clusters: set[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for group in groups:
        group_clusters = {str(item) for item in group.get("cluster_ids") or []}
        matched = sorted(group_clusters & clusters)
        if matched:
            out = dict(group)
            out["matched_cluster_ids"] = matched
            hits.append(out)
    return hits


def _label_for_axis(axis: dict[str, Any], positive_group_ids: set[str]) -> str:
    for rule in axis.get("label_rules") or []:
        required_all = {str(item) for item in rule.get("require_all_groups") or []}
        if required_all and not required_all <= positive_group_ids:
            continue
        required = {str(item) for item in rule.get("require_any_groups") or []}
        if not required or required & positive_group_ids:
            return str(rule.get("label") or axis.get("default_candidate_label") or axis.get("target_family"))
    return str(axis.get("default_candidate_label") or axis.get("target_family"))


def score_axis_window(axis: dict[str, Any], window: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    evidence = window.get("evidence") or {}
    clusters = _cluster_set(evidence)
    positive_hits = _group_hits(axis.get("positive_groups") or [], clusters)
    negative_hits = _group_hits(axis.get("negative_groups") or [], clusters)
    positive_group_ids = {str(hit.get("group_id") or "") for hit in positive_hits}
    negative_group_ids = {str(hit.get("group_id") or "") for hit in negative_hits}
    required = {str(item) for item in axis.get("required_groups") or []}
    missing_required = sorted(required - positive_group_ids)
    hard_blocks = sorted(
        str(hit.get("group_id") or "")
        for hit in negative_hits
        if bool(hit.get("hard_block"))
    )
    positive_score = sum(float(hit.get("weight") or 0.0) for hit in positive_hits)
    penalty = sum(float(hit.get("penalty") or 0.0) for hit in negative_hits)
    score = max(0.0, positive_score - penalty)
    accepted = not missing_required and not hard_blocks and score >= float(axis.get("min_score") or 0.0)
    aliases = {str(item) for item in case.get("caption_alias_ids") or []}
    audit_aliases = {str(item) for item in axis.get("audit_alias_ids") or []}
    return {
        "case_id": window.get("case_id"),
        "span": window.get("span") or evidence.get("span"),
        "axis_id": axis.get("axis_id"),
        "target_family": axis.get("target_family"),
        "candidate_label": _label_for_axis(axis, positive_group_ids),
        "accepted": accepted,
        "score": round(score, 4),
        "positive_score": round(positive_score, 4),
        "penalty": round(penalty, 4),
        "missing_required_groups": missing_required,
        "hard_block_groups": hard_blocks,
        "positive_group_ids": sorted(positive_group_ids),
        "negative_group_ids": sorted(negative_group_ids),
        "positive_hits": [
            {
                "group_id": hit.get("group_id"),
                "weight": hit.get("weight"),
                "matched_cluster_ids": hit.get("matched_cluster_ids"),
            }
            for hit in positive_hits
        ],
        "negative_hits": [
            {
                "group_id": hit.get("group_id"),
                "penalty": hit.get("penalty"),
                "hard_block": bool(hit.get("hard_block")),
                "matched_cluster_ids": hit.get("matched_cluster_ids"),
            }
            for hit in negative_hits
        ],
        "caption_alias_ids": sorted(aliases),
        "target_alias_hit": bool(aliases & audit_aliases),
        "caption": (case.get("caption_texts") or [""])[0],
        "channels": evidence.get("channels") or [],
        "cluster_ids": sorted(clusters),
        "top_tree_hit": ((window.get("hit_summary") or {}).get("top_hit") or {}),
        "member_symbols": evidence.get("member_symbols") or [],
    }


def audit_axes(spec: dict[str, Any], cases: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    cases_by_id = _case_index(cases)
    all_rows: list[dict[str, Any]] = []
    axis_summaries: list[dict[str, Any]] = []
    for axis in spec.get("axes") or []:
        axis_rows = [
            score_axis_window(axis, window, cases_by_id.get(str(window.get("case_id") or ""), {}))
            for window in windows
        ]
        candidates = [row for row in axis_rows if row["accepted"]]
        alias_rows = [row for row in axis_rows if row["target_alias_hit"]]
        candidate_cases = {str(row["case_id"]) for row in candidates}
        alias_cases = {str(row["case_id"]) for row in alias_rows}
        target_candidate_rows = [row for row in candidates if row["target_alias_hit"]]
        non_target_candidate_rows = [row for row in candidates if not row["target_alias_hit"]]
        missed_alias_rows = [row for row in alias_rows if str(row["case_id"]) not in candidate_cases]
        all_rows.extend(candidates)

        cluster_counter = Counter()
        negative_counter = Counter()
        label_counter = Counter(str(row["candidate_label"]) for row in candidates)
        top_hit_counter = Counter(str((row.get("top_tree_hit") or {}).get("motion_structure_label") or "__none__") for row in candidates)
        for row in candidates:
            cluster_counter.update(row.get("cluster_ids") or [])
            negative_counter.update(row.get("negative_group_ids") or [])
        axis_summaries.append(
            {
                "axis_id": axis.get("axis_id"),
                "target_family": axis.get("target_family"),
                "candidate_window_count": len(candidates),
                "candidate_case_count": len(candidate_cases),
                "target_alias_window_count": len(alias_rows),
                "target_alias_case_count": len(alias_cases),
                "target_alias_case_recall": round(_safe_div(len(candidate_cases & alias_cases), len(alias_cases)), 4),
                "candidate_target_alias_precision": round(_safe_div(len({str(row["case_id"]) for row in target_candidate_rows}), len(candidate_cases)), 4),
                "target_alias_candidate_count": len(target_candidate_rows),
                "non_target_candidate_count": len(non_target_candidate_rows),
                "missed_target_alias_count": len(missed_alias_rows),
                "candidate_label_counts": dict(sorted(label_counter.items())),
                "top_tree_hit_label_counts": dict(top_hit_counter.most_common(10)),
                "top_candidate_clusters": [{"id": key, "count": value} for key, value in cluster_counter.most_common(20)],
                "negative_group_counts": dict(sorted(negative_counter.items())),
                "target_examples": _examples(target_candidate_rows, limit=10),
                "non_target_examples": _examples(non_target_candidate_rows, limit=10),
                "missed_target_examples": _examples(sorted(missed_alias_rows, key=lambda row: -float(row.get("score") or 0.0)), limit=10),
            }
        )

    return {
        "schema_version": "aml_pattern_split_axis_audit_v0",
        "runtime_policy": "motion evidence only; caption aliases are diagnostics for precision/recall estimates",
        "inputs": {
            "axis_schema_version": spec.get("schema_version"),
        },
        "summary": {
            "axis_count": len(spec.get("axes") or []),
            "case_count": len(cases),
            "window_count": len(windows),
            "accepted_window_count": len(all_rows),
            "accepted_case_count": len({str(row["case_id"]) for row in all_rows}),
        },
        "axis_summaries": axis_summaries,
        "accepted_rows": sorted(all_rows, key=lambda row: (str(row.get("axis_id")), str(row.get("case_id")), row.get("span") or [])),
    }


def _examples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id in seen:
            continue
        seen.add(case_id)
        top = row.get("top_tree_hit") or {}
        out.append(
            {
                "case_id": case_id,
                "span": row.get("span"),
                "score": row.get("score"),
                "candidate_label": row.get("candidate_label"),
                "caption_alias_ids": row.get("caption_alias_ids"),
                "caption": row.get("caption"),
                "positive_group_ids": row.get("positive_group_ids"),
                "negative_group_ids": row.get("negative_group_ids"),
                "top_tree_label": top.get("motion_structure_label"),
                "top_tree_level": top.get("semantic_level"),
                "cluster_ids": (row.get("cluster_ids") or [])[:24],
            }
        )
        if len(out) >= limit:
            break
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# AML Pattern Split-Axis Audit v0", ""]
    summary = payload.get("summary") or {}
    lines.append(
        f"axes={summary.get('axis_count')} cases={summary.get('case_count')} "
        f"windows={summary.get('window_count')} accepted_cases={summary.get('accepted_case_count')}"
    )
    lines.append("")
    for axis in payload.get("axis_summaries") or []:
        lines.append(f"## {axis.get('axis_id')}")
        lines.append(f"- target family: `{axis.get('target_family')}`")
        lines.append(
            f"- candidates: windows={axis.get('candidate_window_count')} cases={axis.get('candidate_case_count')}"
        )
        lines.append(
            f"- diagnostic precision={axis.get('candidate_target_alias_precision')} "
            f"recall={axis.get('target_alias_case_recall')} "
            f"missed_alias_windows={axis.get('missed_target_alias_count')}"
        )
        lines.append(f"- candidate labels: `{axis.get('candidate_label_counts')}`")
        lines.append(f"- top tree labels before split-axis: `{axis.get('top_tree_hit_label_counts')}`")
        lines.append(f"- negative groups: `{axis.get('negative_group_counts')}`")
        lines.append("")
        lines.append("### Target-Alias Candidate Examples")
        for ex in axis.get("target_examples") or []:
            lines.append(
                f"- `{ex.get('case_id')}` score={ex.get('score')} label={ex.get('candidate_label')} "
                f"top_tree={ex.get('top_tree_label')}: {ex.get('caption')}"
            )
            lines.append(f"  - groups={ex.get('positive_group_ids')} negatives={ex.get('negative_group_ids')}")
        lines.append("")
        lines.append("### Non-Target Candidate Examples")
        for ex in axis.get("non_target_examples") or []:
            lines.append(
                f"- `{ex.get('case_id')}` score={ex.get('score')} label={ex.get('candidate_label')} "
                f"aliases={ex.get('caption_alias_ids')}: {ex.get('caption')}"
            )
            lines.append(f"  - groups={ex.get('positive_group_ids')} negatives={ex.get('negative_group_ids')}")
        lines.append("")
        lines.append("### Missed Target-Alias Examples")
        for ex in axis.get("missed_target_examples") or []:
            lines.append(
                f"- `{ex.get('case_id')}` score={ex.get('score')} aliases={ex.get('caption_alias_ids')}: {ex.get('caption')}"
            )
            lines.append(f"  - groups={ex.get('positive_group_ids')} negatives={ex.get('negative_group_ids')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = _read_json(Path(args.axis_spec))
    search_dir = Path(args.search_dir)
    cases = _read_json(search_dir / "case_tree_search_results.json")
    windows = _read_json(search_dir / "window_tree_search_results.json")
    return audit_axes(spec, cases, windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-spec", type=Path, default=DEFAULT_AXIS_SPEC)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    spec = {
        "schema_version": "test",
        "axes": [
            {
                "axis_id": "low_transition_test",
                "target_family": "body_level_low_transition",
                "audit_alias_ids": ["sit_down"],
                "required_groups": ["descend_to_low", "low_posture_state"],
                "min_score": 0.5,
                "default_candidate_label": "body_level_descend_to_low_candidate",
                "positive_groups": [
                    {"group_id": "descend_to_low", "weight": 0.3, "cluster_ids": ["WB_LEVEL_DESCEND_TO_LOW"]},
                    {"group_id": "low_posture_state", "weight": 0.3, "cluster_ids": ["WB_LEVEL_LOW_SUSTAINED"]},
                ],
                "negative_groups": [
                    {"group_id": "inverted_support_confound", "penalty": 0.9, "hard_block": True, "cluster_ids": ["WB_SUPPORT_INVERTED"]}
                ],
            }
        ],
    }
    cases = [
        {
            "case_id": "case_a",
            "caption_alias_ids": ["sit_down"],
            "caption_texts": ["a person sits down"],
        },
        {
            "case_id": "case_b",
            "caption_alias_ids": [],
            "caption_texts": ["a person does a hand stand"],
        },
    ]
    windows = [
        {
            "case_id": "case_a",
            "span": [0, 40],
            "evidence": {
                "cluster_ids": ["WB_LEVEL_DESCEND_TO_LOW", "WB_LEVEL_LOW_SUSTAINED"],
                "channels": ["whole_body_state"],
            },
            "hit_summary": {"top_hit": {"motion_structure_label": "sit down"}},
        },
        {
            "case_id": "case_b",
            "span": [0, 40],
            "evidence": {
                "cluster_ids": ["WB_LEVEL_DESCEND_TO_LOW", "WB_LEVEL_LOW_SUSTAINED", "WB_SUPPORT_INVERTED"],
                "channels": ["whole_body_state", "whole_body_support"],
            },
            "hit_summary": {"top_hit": {"motion_structure_label": "cartwheel"}},
        },
    ]
    payload = audit_axes(spec, cases, windows)
    assert payload["summary"]["accepted_case_count"] == 1
    assert payload["accepted_rows"][0]["case_id"] == "case_a"
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        _write_json(output_dir / "split_axis_audit.json", payload)
        write_report(output_dir / "split_axis_audit.md", payload)
    print(json.dumps({"ok": True}, indent=2))


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    payload = run(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "split_axis_audit.json", payload)
    _write_json(output_dir / "summary.json", payload.get("summary") or {})
    write_report(output_dir / "split_axis_audit.md", payload)
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
