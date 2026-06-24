"""Audit whether split-axis groups are phase-aligned, not only co-present.

This script is a generic motion-evidence diagnostic. It does not use captions
to match motion. Caption aliases are copied only to estimate pseudo-GT
precision/recall for review.

Typical full-HML3D v5 run:
    python scripts/audit_split_axis_phase_closure.py \
      --axis-id bilateral_spread_vertical_coordination_v0 \
      --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/case_multichannel_bpe_sequences.jsonl \
      --output-dir outputs/aml_regression_testset_v2/aml_pattern_split_axis_phase_closure_v5_stance_width_full_v0

Smoke test:
    python scripts/audit_split_axis_phase_closure.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AXIS_SPEC = Path("pseudoedit3d/edit/aml_pattern_split_axes.json")
DEFAULT_AXIS_ID = "bilateral_spread_vertical_coordination_v0"
DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/"
    "case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_split_axis_phase_closure_v0")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _parse_case_ids(text: str) -> set[str] | None:
    values = {item.strip() for item in str(text or "").split(",") if item.strip()}
    return values or None


def _axis_by_id(spec: dict[str, Any], axis_id: str) -> dict[str, Any]:
    for axis in spec.get("axes") or []:
        if str(axis.get("axis_id") or "") == axis_id:
            return dict(axis)
    raise KeyError(f"axis not found: {axis_id}")


def _load_case_text(path: Path, case_filter: set[str] | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            if case_filter and case_id not in case_filter:
                continue
            rows[case_id] = {
                "case_id": case_id,
                "num_frames": int(record.get("num_frames") or 0),
                "caption_texts": [str(item) for item in record.get("caption_texts") or []],
                "caption_alias_ids": [str(item) for item in record.get("caption_alias_ids") or []],
            }
    return rows


def _load_channel_units(path: Path, case_filter: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("view") or "").startswith("channel::"):
                continue
            case_id = str(row.get("case_id") or str(row.get("sequence_id") or "").split("::", 1)[0])
            if case_filter and case_id not in case_filter:
                continue
            for token in row.get("tokens") or []:
                span = token.get("span") or []
                if len(span) != 2:
                    continue
                copied = dict(token)
                copied["case_id"] = case_id
                by_case[case_id].append(copied)
    for units in by_case.values():
        units.sort(key=lambda item: (int((item.get("span") or [0, 0])[0]), int((item.get("span") or [0, 0])[1]), str(item.get("symbol") or "")))
    return dict(by_case)


def _cluster_ids(unit: dict[str, Any]) -> set[str]:
    clusters: set[str] = set()
    for key in ("geometry_clusters", "raw_geometry_clusters"):
        for item in unit.get(key) or []:
            clusters.add(str(item).rsplit("/", 1)[-1])
    for key in ("base_symbols", "member_symbols"):
        for symbol in unit.get(key) or []:
            head = str(symbol).split("|", 1)[0]
            if "/" in head:
                clusters.add(head.rsplit("/", 1)[-1])
    symbol_head = str(unit.get("symbol") or "").split("|", 1)[0]
    if "/" in symbol_head:
        clusters.add(symbol_head.rsplit("/", 1)[-1])
    return clusters


def _span(unit: dict[str, Any]) -> list[int]:
    start, end = unit.get("span") or [0, 0]
    return [int(start), int(end)]


def _duration(unit: dict[str, Any]) -> int:
    start, end = _span(unit)
    return max(1, end - start + 1)


def _center(unit: dict[str, Any]) -> float:
    start, end = _span(unit)
    return (start + end) / 2.0


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _overlap(a, b) / max(1, min(_duration(a), _duration(b)))


def _event_row(unit: dict[str, Any], group_id: str, matched: list[str]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "span": _span(unit),
        "channels": [str(item) for item in unit.get("channels") or []],
        "symbol": str(unit.get("symbol") or ""),
        "matched_cluster_ids": matched,
    }


def _match_groups(groups: list[dict[str, Any]], unit: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = _cluster_ids(unit)
    hits: list[dict[str, Any]] = []
    for group in groups:
        group_clusters = {str(item) for item in group.get("cluster_ids") or []}
        matched = sorted(group_clusters & clusters)
        if matched:
            hits.append(_event_row(unit, str(group.get("group_id") or ""), matched))
    return hits


def _closure_rules(axis: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in axis.get("label_rules") or []:
        groups = [str(item) for item in rule.get("require_all_groups") or []]
        if groups:
            rules.append({"label": str(rule.get("label") or axis.get("target_family") or "closure"), "groups": groups})
    if not rules:
        rules.append(
            {
                "label": str(axis.get("default_candidate_label") or axis.get("target_family") or "closure"),
                "groups": [str(item) for item in axis.get("required_groups") or []],
            }
        )
    rules.sort(key=lambda row: (-len(row["groups"]), row["label"]))
    return rules


def _best_pair(
    left_group: str,
    left_events: list[dict[str, Any]],
    right_group: str,
    right_events: list[dict[str, Any]],
    *,
    min_pair_overlap: float,
    max_gap_frames: int,
    max_center_gap_frames: int,
    num_frames: int,
    broad_event_frame_ratio: float,
    broad_event_min_frames: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for left, right in itertools.product(left_events, right_events):
        overlap_ratio = _overlap_ratio(left, right)
        gap = _gap(left, right)
        center_gap = abs(_center(left) - _center(right))
        left_broad = _duration(left) >= broad_event_min_frames and _safe_div(_duration(left), max(1, num_frames)) >= broad_event_frame_ratio
        right_broad = _duration(right) >= broad_event_min_frames and _safe_div(_duration(right), max(1, num_frames)) >= broad_event_frame_ratio
        near_score = max(0.0, 1.0 - (gap / max(1, max_gap_frames))) if gap <= max_gap_frames else 0.0
        center_ok = center_gap <= max_center_gap_frames
        score = overlap_ratio + 0.5 * near_score + (0.25 if center_ok else 0.0)
        if overlap_ratio >= min_pair_overlap and (left_broad or right_broad) and not center_ok:
            relation = "broad_context"
        elif overlap_ratio >= min_pair_overlap and center_ok:
            relation = "overlap"
        elif gap <= max_gap_frames and center_ok:
            relation = "near"
        else:
            relation = "distant"
        row = {
            "groups": [left_group, right_group],
            "relation": relation,
            "phase_aligned": relation in {"overlap", "near"},
            "context_aligned": relation in {"overlap", "near", "broad_context"},
            "overlap_ratio": round(overlap_ratio, 4),
            "gap_frames": int(gap),
            "center_gap_frames": round(center_gap, 2),
            "left_broad_context": left_broad,
            "right_broad_context": right_broad,
            "score": round(score, 4),
            "left": left,
            "right": right,
        }
        if best is None or (float(row["score"]), -float(row["center_gap_frames"])) > (
            float(best["score"]),
            -float(best["center_gap_frames"]),
        ):
            best = row
    return best or {
        "groups": [left_group, right_group],
        "relation": "missing",
        "phase_aligned": False,
        "overlap_ratio": 0.0,
        "gap_frames": None,
        "center_gap_frames": None,
        "score": 0.0,
    }


def _representative_events(
    groups: list[str],
    events_by_group: dict[str, list[dict[str, Any]]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(pair_rows, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        if not row.get("phase_aligned"):
            continue
        for side in ("left", "right"):
            evt = row.get(side) or {}
            group_id = str(evt.get("group_id") or "")
            if group_id and group_id not in selected:
                selected[group_id] = evt
    for group_id in groups:
        if group_id not in selected and events_by_group.get(group_id):
            selected[group_id] = events_by_group[group_id][0]
    return [selected[group_id] for group_id in groups if group_id in selected]


def _phase_status(
    *,
    missing_groups: list[str],
    hard_block_groups: list[str],
    pair_rows: list[dict[str, Any]],
    groups: list[str],
) -> str:
    if missing_groups:
        return "missing_groups"
    if hard_block_groups:
        return "hard_blocked"
    aligned_pairs = [row for row in pair_rows if row.get("phase_aligned")]
    total_pairs = len(pair_rows)
    if total_pairs and len(aligned_pairs) == total_pairs:
        return "phase_closed_all_pairs"
    context_pairs = [row for row in pair_rows if row.get("context_aligned")]
    if total_pairs and len(context_pairs) == total_pairs:
        return "broad_context_closure"
    if not aligned_pairs:
        return "case_level_only"
    graph: dict[str, set[str]] = {group: set() for group in groups}
    for row in aligned_pairs:
        left, right = [str(item) for item in row.get("groups") or []]
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    seen: set[str] = set()
    stack = [groups[0]] if groups else []
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(sorted(graph.get(cur, set()) - seen))
    return "phase_connected_chain" if set(groups) <= seen else "partial_phase"


def _quality_gate(axis: dict[str, Any], events_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    gate = axis.get("phase_quality_gate") or {}
    require_any = [str(item) for item in gate.get("require_any_groups") or []]
    require_all = [str(item) for item in gate.get("require_all_groups") or []]
    present_any = [group for group in require_any if events_by_group.get(group)]
    missing_all = [group for group in require_all if not events_by_group.get(group)]
    pass_any = bool(present_any) if require_any else True
    pass_all = not missing_all
    return {
        "enabled": bool(require_any or require_all),
        "passed": pass_any and pass_all,
        "present_any_groups": present_any,
        "missing_any_groups": [] if pass_any else require_any,
        "missing_all_groups": missing_all,
    }


def _apply_quality_status(status: str, quality: dict[str, Any]) -> str:
    if not quality.get("enabled") or quality.get("passed"):
        return status
    if status == "phase_closed_all_pairs":
        return "phase_closed_low_quality"
    if status == "phase_connected_chain":
        return "phase_connected_low_quality"
    if status == "broad_context_closure":
        return "broad_context_low_quality"
    return status


def score_case_phase(axis: dict[str, Any], case: dict[str, Any], units: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    positive_groups = axis.get("positive_groups") or []
    negative_groups = axis.get("negative_groups") or []
    events_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negative_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        for hit in _match_groups(positive_groups, unit):
            events_by_group[str(hit["group_id"])].append(hit)
        for hit in _match_groups(negative_groups, unit):
            negative_by_group[str(hit["group_id"])].append(hit)

    hard_block_groups = sorted(
        str(group.get("group_id") or "")
        for group in negative_groups
        if bool(group.get("hard_block")) and negative_by_group.get(str(group.get("group_id") or ""))
    )
    aliases = {str(item) for item in case.get("caption_alias_ids") or []}
    audit_aliases = {str(item) for item in axis.get("audit_alias_ids") or []}
    num_frames = int(case.get("num_frames") or 0) or max((int((_span(unit))[1]) for unit in units), default=0)
    group_counts = {group_id: len(rows) for group_id, rows in sorted(events_by_group.items())}
    quality = _quality_gate(axis, events_by_group)
    closure_rows: list[dict[str, Any]] = []
    for rule in _closure_rules(axis):
        groups = [str(item) for item in rule["groups"]]
        missing_groups = [group for group in groups if not events_by_group.get(group)]
        pair_rows: list[dict[str, Any]] = []
        if not missing_groups:
            for left_group, right_group in itertools.combinations(groups, 2):
                pair_rows.append(
                    _best_pair(
                        left_group,
                        events_by_group[left_group],
                        right_group,
                        events_by_group[right_group],
                        min_pair_overlap=float(args.min_pair_overlap),
                        max_gap_frames=int(args.max_gap_frames),
                        max_center_gap_frames=int(args.max_center_gap_frames),
                        num_frames=num_frames,
                        broad_event_frame_ratio=float(args.broad_event_frame_ratio),
                        broad_event_min_frames=int(args.broad_event_min_frames),
                    )
                )
        status = _phase_status(
            missing_groups=missing_groups,
            hard_block_groups=hard_block_groups,
            pair_rows=pair_rows,
            groups=groups,
        )
        status = _apply_quality_status(status, quality)
        reps = _representative_events(groups, events_by_group, pair_rows) if pair_rows else []
        span = [
            min((evt["span"][0] for evt in reps), default=None),
            max((evt["span"][1] for evt in reps), default=None),
        ]
        avg_pair_score = round(sum(float(row.get("score") or 0.0) for row in pair_rows) / max(1, len(pair_rows)), 4)
        closure_rows.append(
            {
                "label": rule["label"],
                "groups": groups,
                "status": status,
                "span": span if span[0] is not None else None,
                "missing_groups": missing_groups,
                "hard_block_groups": hard_block_groups,
                "quality_gate": quality,
                "pair_relations": [
                    {
                        key: row.get(key)
                        for key in [
                            "groups",
                            "relation",
                            "phase_aligned",
                            "context_aligned",
                            "overlap_ratio",
                            "gap_frames",
                            "center_gap_frames",
                            "left_broad_context",
                            "right_broad_context",
                            "score",
                        ]
                    }
                    for row in pair_rows
                ],
                "phase_aligned_pair_count": sum(1 for row in pair_rows if row.get("phase_aligned")),
                "pair_count": len(pair_rows),
                "avg_pair_score": avg_pair_score,
                "representative_events": reps,
            }
        )

    best = max(
        closure_rows,
        key=lambda row: (
            {
                "phase_closed_all_pairs": 6,
                "phase_connected_chain": 5,
                "broad_context_closure": 4,
                "phase_closed_low_quality": 3,
                "phase_connected_low_quality": 3,
                "broad_context_low_quality": 3,
                "partial_phase": 3,
                "case_level_only": 2,
                "hard_blocked": 1,
                "missing_groups": 0,
            }.get(
                str(row.get("status")), 0
            ),
            float(row.get("avg_pair_score") or 0.0),
        ),
        default={},
    )
    return {
        "case_id": str(case.get("case_id") or ""),
        "num_frames": num_frames,
        "caption": (case.get("caption_texts") or [""])[0],
        "caption_alias_ids": sorted(aliases),
        "target_alias_hit": bool(aliases & audit_aliases),
        "group_counts": group_counts,
        "negative_group_counts": {key: len(value) for key, value in sorted(negative_by_group.items())},
        "best_status": best.get("status"),
        "best_label": best.get("label"),
        "best_avg_pair_score": best.get("avg_pair_score"),
        "closure_rows": closure_rows,
    }


def summarize(rows: list[dict[str, Any]], axis: dict[str, Any]) -> dict[str, Any]:
    target_rows = [row for row in rows if row.get("target_alias_hit")]
    non_target_rows = [row for row in rows if not row.get("target_alias_hit")]
    status_counts = Counter(str(row.get("best_status") or "missing") for row in rows)
    target_status_counts = Counter(str(row.get("best_status") or "missing") for row in target_rows)
    non_target_status_counts = Counter(str(row.get("best_status") or "missing") for row in non_target_rows)
    full_positive = {"phase_closed_all_pairs", "phase_connected_chain", "broad_context_closure"}
    strict_positive = {"phase_closed_all_pairs"}
    target_ids = {str(row["case_id"]) for row in target_rows}
    strict_ids = {str(row["case_id"]) for row in rows if row.get("best_status") in strict_positive}
    full_ids = {str(row["case_id"]) for row in rows if row.get("best_status") in full_positive}
    group_counter = Counter()
    missing_counter = Counter()
    pair_relation_counter = Counter()
    for row in target_rows:
        group_counter.update((row.get("group_counts") or {}).keys())
        best = (row.get("closure_rows") or [{}])[0]
        missing_counter.update(best.get("missing_groups") or [])
        for pair in best.get("pair_relations") or []:
            pair_relation_counter.update([str(pair.get("relation") or "")])
    return {
        "axis_id": axis.get("axis_id"),
        "target_family": axis.get("target_family"),
        "case_count": len(rows),
        "target_case_count": len(target_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "target_status_counts": dict(sorted(target_status_counts.items())),
        "non_target_status_counts": dict(sorted(non_target_status_counts.items())),
        "strict_phase_closed_case_count": len(strict_ids),
        "strict_phase_closed_target_count": len(strict_ids & target_ids),
        "strict_phase_closed_precision": round(_safe_div(len(strict_ids & target_ids), len(strict_ids)), 4),
        "strict_phase_closed_recall": round(_safe_div(len(strict_ids & target_ids), len(target_ids)), 4),
        "phase_connected_or_closed_case_count": len(full_ids),
        "phase_connected_or_closed_target_count": len(full_ids & target_ids),
        "phase_connected_or_closed_precision": round(_safe_div(len(full_ids & target_ids), len(full_ids)), 4),
        "phase_connected_or_closed_recall": round(_safe_div(len(full_ids & target_ids), len(target_ids)), 4),
        "target_group_presence_counts": dict(group_counter.most_common()),
        "target_missing_group_counts": dict(missing_counter.most_common()),
        "target_pair_relation_counts": dict(pair_relation_counter.most_common()),
        "quality_gate_enabled": bool((axis.get("phase_quality_gate") or {}).get("require_any_groups") or (axis.get("phase_quality_gate") or {}).get("require_all_groups")),
    }


def _examples(rows: list[dict[str, Any]], status_set: set[str], *, target: bool | None, limit: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if status_set and row.get("best_status") not in status_set:
            continue
        if target is not None and bool(row.get("target_alias_hit")) != target:
            continue
        best = (row.get("closure_rows") or [{}])[0]
        out.append(
            {
                "case_id": row.get("case_id"),
                "status": row.get("best_status"),
                "caption": row.get("caption"),
                "caption_alias_ids": row.get("caption_alias_ids"),
                "group_counts": row.get("group_counts"),
                "missing_groups": best.get("missing_groups"),
                "pair_relations": best.get("pair_relations"),
                "representative_events": best.get("representative_events"),
            }
        )
        if len(out) >= limit:
            break
    return out


def build_payload(axis: dict[str, Any], case_text: dict[str, dict[str, Any]], units_by_case: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    case_ids = sorted(set(case_text) & set(units_by_case))
    rows = [score_case_phase(axis, case_text[case_id], units_by_case[case_id], args) for case_id in case_ids]
    summary = summarize(rows, axis)
    return {
        "schema_version": "aml_split_axis_phase_closure_v0",
        "runtime_policy": "motion-only phase audit; captions are diagnostics only",
        "inputs": {
            "axis_id": axis.get("axis_id"),
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "min_pair_overlap": float(args.min_pair_overlap),
            "max_gap_frames": int(args.max_gap_frames),
        },
        "summary": summary,
        "examples": {
            "target_phase_closed": _examples(rows, {"phase_closed_all_pairs"}, target=True, limit=int(args.example_limit)),
            "target_phase_connected": _examples(rows, {"phase_connected_chain"}, target=True, limit=int(args.example_limit)),
            "target_broad_context": _examples(rows, {"broad_context_closure"}, target=True, limit=int(args.example_limit)),
            "target_case_level_only": _examples(rows, {"case_level_only", "partial_phase"}, target=True, limit=int(args.example_limit)),
            "target_missing": _examples(rows, {"missing_groups"}, target=True, limit=int(args.example_limit)),
            "non_target_phase_closed": _examples(rows, {"phase_closed_all_pairs", "phase_connected_chain", "broad_context_closure"}, target=False, limit=int(args.example_limit)),
        },
        "rows": rows,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    lines = ["# Split-Axis Phase Closure Audit", ""]
    lines.append(f"axis: `{summary.get('axis_id')}`")
    lines.append(f"cases={summary.get('case_count')} target_cases={summary.get('target_case_count')}")
    lines.append("")
    lines.append("## Status Counts")
    lines.append(f"- all: `{summary.get('status_counts')}`")
    lines.append(f"- target: `{summary.get('target_status_counts')}`")
    lines.append(f"- non-target: `{summary.get('non_target_status_counts')}`")
    lines.append("")
    lines.append("## Diagnostic Precision / Recall")
    lines.append(
        f"- strict phase-closed: cases={summary.get('strict_phase_closed_case_count')} "
        f"target={summary.get('strict_phase_closed_target_count')} "
        f"precision={summary.get('strict_phase_closed_precision')} "
        f"recall={summary.get('strict_phase_closed_recall')}"
    )
    lines.append(
        f"- phase connected-or-closed: cases={summary.get('phase_connected_or_closed_case_count')} "
        f"target={summary.get('phase_connected_or_closed_target_count')} "
        f"precision={summary.get('phase_connected_or_closed_precision')} "
        f"recall={summary.get('phase_connected_or_closed_recall')}"
    )
    lines.append("")
    lines.append("## Target Group Diagnostics")
    lines.append(f"- group presence: `{summary.get('target_group_presence_counts')}`")
    lines.append(f"- missing groups: `{summary.get('target_missing_group_counts')}`")
    lines.append(f"- pair relations: `{summary.get('target_pair_relation_counts')}`")
    lines.append("")

    def add_examples(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        for row in rows:
            lines.append(f"### {row.get('case_id')} | {row.get('status')}")
            lines.append(f"- caption: {row.get('caption')}")
            lines.append(f"- aliases: `{row.get('caption_alias_ids')}`")
            lines.append(f"- groups: `{row.get('group_counts')}` missing={row.get('missing_groups')}")
            for pair in row.get("pair_relations") or []:
                lines.append(
                    f"  - pair={pair.get('groups')} relation={pair.get('relation')} "
                    f"overlap={pair.get('overlap_ratio')} gap={pair.get('gap_frames')} "
                    f"center_gap={pair.get('center_gap_frames')}"
                )
            for event in row.get("representative_events") or []:
                lines.append(
                    f"  - event group={event.get('group_id')} span={event.get('span')} "
                    f"clusters={event.get('matched_cluster_ids')} channels={event.get('channels')}"
                )
            lines.append("")

    examples = payload.get("examples") or {}
    add_examples("Target Strict Phase-Closed Examples", examples.get("target_phase_closed") or [])
    add_examples("Target Phase-Connected Examples", examples.get("target_phase_connected") or [])
    add_examples("Target Broad-Context Closure Examples", examples.get("target_broad_context") or [])
    add_examples("Target Case-Level-Only Examples", examples.get("target_case_level_only") or [])
    add_examples("Target Missing-Group Examples", examples.get("target_missing") or [])
    add_examples("Non-Target Phase-Closed/Connected Examples", examples.get("non_target_phase_closed") or [])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = _read_json(Path(args.axis_spec))
    axis = _axis_by_id(spec, str(args.axis_id))
    case_filter = _parse_case_ids(str(args.case_ids or ""))
    case_text = _load_case_text(Path(args.source_corpus), case_filter=case_filter)
    units_by_case = _load_channel_units(Path(args.bpe_sequences), case_filter=case_filter)
    return build_payload(axis, case_text, units_by_case, args)


def run_self_test() -> None:
    axis = {
        "axis_id": "toy_axis",
        "target_family": "toy",
        "audit_alias_ids": ["toy"],
        "label_rules": [{"label": "toy_full", "require_all_groups": ["upper", "lower", "vertical"]}],
        "positive_groups": [
            {"group_id": "upper", "cluster_ids": ["UP"], "weight": 0.3},
            {"group_id": "lower", "cluster_ids": ["LOW"], "weight": 0.3},
            {"group_id": "vertical", "cluster_ids": ["VERT"], "weight": 0.3},
        ],
        "negative_groups": [{"group_id": "block", "cluster_ids": ["BLOCK"], "hard_block": True, "penalty": 1.0}],
    }
    case_text = {
        "a": {"case_id": "a", "num_frames": 50, "caption_texts": ["toy"], "caption_alias_ids": ["toy"]},
        "b": {"case_id": "b", "num_frames": 50, "caption_texts": ["other"], "caption_alias_ids": []},
    }
    units_by_case = {
        "a": [
            {"span": [10, 20], "channels": ["arm"], "geometry_clusters": ["X/UP"], "symbol": "arm/X/UP"},
            {"span": [12, 22], "channels": ["leg"], "geometry_clusters": ["X/LOW"], "symbol": "leg/X/LOW"},
            {"span": [11, 19], "channels": ["vertical"], "geometry_clusters": ["X/VERT"], "symbol": "vertical/X/VERT"},
        ],
        "b": [
            {"span": [1, 5], "channels": ["arm"], "geometry_clusters": ["X/UP"], "symbol": "arm/X/UP"},
            {"span": [30, 35], "channels": ["leg"], "geometry_clusters": ["X/LOW"], "symbol": "leg/X/LOW"},
            {"span": [60, 65], "channels": ["vertical"], "geometry_clusters": ["X/VERT"], "symbol": "vertical/X/VERT"},
        ],
    }
    args = argparse.Namespace(
        source_corpus="toy",
        bpe_sequences="toy",
        min_pair_overlap=0.1,
        max_gap_frames=6,
        max_center_gap_frames=12,
        broad_event_frame_ratio=0.45,
        broad_event_min_frames=48,
        example_limit=4,
    )
    payload = build_payload(axis, case_text, units_by_case, args)
    assert payload["summary"]["target_case_count"] == 1
    assert payload["rows"][0]["best_status"] == "phase_closed_all_pairs"
    assert payload["rows"][1]["best_status"] == "case_level_only"
    with tempfile.TemporaryDirectory() as tmp:
        _write_json(Path(tmp) / "phase_closure_audit.json", payload)
        write_report(Path(tmp) / "phase_closure_audit.md", payload)
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-spec", type=Path, default=DEFAULT_AXIS_SPEC)
    parser.add_argument("--axis-id", default=DEFAULT_AXIS_ID)
    parser.add_argument("--source-corpus", type=Path, default=DEFAULT_SOURCE_CORPUS)
    parser.add_argument("--bpe-sequences", type=Path, default=DEFAULT_BPE_SEQUENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--min-pair-overlap", type=float, default=0.15)
    parser.add_argument("--max-gap-frames", type=int, default=12)
    parser.add_argument("--max-center-gap-frames", type=int, default=18)
    parser.add_argument("--broad-event-frame-ratio", type=float, default=0.45)
    parser.add_argument("--broad-event-min-frames", type=int, default=48)
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    payload = run(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "phase_closure_audit.json", payload)
    _write_json(output_dir / "summary.json", payload.get("summary") or {})
    write_report(output_dir / "phase_closure_audit.md", payload)
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "summary": payload["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
