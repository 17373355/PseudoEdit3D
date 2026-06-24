"""Search the composable AML pattern program with motion-derived evidence.

This is the bridge between mined pattern forests and an AML runtime interface.
It does not use captions as matching rules. Captions are copied only for review.

Pipeline:
1. Load the composable pattern program exported by
   `export_aml_composable_pattern_program_v0.py`.
2. Load full-HML3D multichannel Motion-BPE sequences.
3. Rebuild all-unit coactivation windows for each motion.
4. Convert each window into structural evidence: channels, zones, event
   families, and cluster ids.
5. Search the pattern tree and classify hits by semantic level/edit scope.

Quick run:
    python scripts/search_aml_composable_pattern_program_v0.py --max-cases 250

Inspect selected cases:
    python scripts/search_aml_composable_pattern_program_v0.py --case-ids 003082,003191,007581

Smoke test:
    python scripts/search_aml_composable_pattern_program_v0.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pseudoedit3d.edit.aml_composable_pattern_program import (  # noqa: E402
    load_composable_pattern_program,
    search_program_nodes,
)
from scripts.audit_hml3d_coactivation_recall_v0 import (  # noqa: E402
    build_all_unit_coactivations,
    load_case_text,
    load_channel_sequences,
)
from scripts.build_hml3d_composition_pattern_forest_v0 import _item_zone  # noqa: E402


DEFAULT_PROGRAM_JSON = Path("outputs/aml_regression_testset_v2/aml_composable_pattern_program_v0/aml_composable_pattern_program.json")
DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full/case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0")
DEFAULT_SUPPORT_STATE_PROGRAM_JSON = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_reviewed_draft/"
    "aml_composable_pattern_program.json"
)
DEFAULT_SUPPORT_STATE_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/"
    "hml3d_multichannel_motion_bpe_v4_support_state_full_v0/"
    "case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_SUPPORT_STATE_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_v0"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_case_ids(text: str) -> set[str] | None:
    values = {item.strip() for item in text.split(",") if item.strip()}
    return values or None


def _event_family(cluster: str) -> str:
    return cluster.split("/", 1)[0] if "/" in cluster else cluster


def _cluster_id(cluster: str) -> str:
    return cluster.rsplit("/", 1)[-1]


def coactivation_evidence(unit: dict[str, Any]) -> dict[str, Any]:
    channels = sorted({str(item) for item in unit.get("channels") or []})
    geometry_clusters = sorted({str(item) for item in unit.get("geometry_clusters") or []})
    raw_geometry_clusters = sorted({str(item) for item in unit.get("raw_geometry_clusters") or []})
    all_clusters = sorted(set(geometry_clusters) | set(raw_geometry_clusters))
    return {
        "span": unit.get("span") or [0, 0],
        "channels": channels,
        "zones": sorted({_item_zone(f"{channel}:x") for channel in channels}),
        "event_families": sorted({_event_family(cluster) for cluster in all_clusters}),
        "cluster_ids": sorted({_cluster_id(cluster) for cluster in all_clusters}),
        "geometry_clusters": geometry_clusters,
        "raw_geometry_clusters": raw_geometry_clusters,
        "observable_refinement_tags": sorted({str(item) for item in unit.get("observable_refinement_tags") or []}),
        "member_symbols": [str(item) for item in unit.get("member_symbols") or []],
    }


def _hit_summary(hits: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = Counter(str(hit.get("semantic_level") or "") for hit in hits)
    scope_counts = Counter(str(hit.get("edit_scope") or "") for hit in hits)
    return {
        "hit_count": len(hits),
        "semantic_level_counts": dict(sorted(level_counts.items())),
        "edit_scope_counts": dict(sorted(scope_counts.items())),
        "top_hit": hits[0] if hits else None,
    }


def search_case_windows(
    *,
    case_id: str,
    coactivation_units: list[dict[str, Any]],
    program: dict[str, Any],
    top_k: int,
    min_score: float,
    semantic_priority: bool = False,
    node_kinds: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, unit in enumerate(coactivation_units, start=1):
        evidence = coactivation_evidence(unit)
        hits = search_program_nodes(
            program,
            channels=evidence["channels"],
            zones=evidence["zones"],
            cluster_ids=evidence["cluster_ids"],
            event_families=evidence["event_families"],
            top_k=top_k,
            min_score=min_score,
            semantic_priority=semantic_priority,
            node_kinds=node_kinds,
        )
        rows.append(
            {
                "case_id": case_id,
                "window_index": idx,
                "span": evidence["span"],
                "evidence": evidence,
                "hit_summary": _hit_summary(hits),
                "hits": hits,
            }
        )
    return rows


def classify_case(windows: list[dict[str, Any]]) -> dict[str, Any]:
    best_hits = [window["hits"][0] for window in windows if window.get("hits")]
    if not best_hits:
        return {
            "case_pattern_type": "unmatched_or_local_only",
            "reason": "no program node matched current structural evidence",
        }
    levels = Counter(str(hit.get("semantic_level") or "") for hit in best_hits)
    scopes = Counter(str(hit.get("edit_scope") or "") for hit in best_hits)
    statuses = Counter(str(hit.get("review_status") or "") for hit in best_hits)
    if statuses.get("accepted", 0) >= 1 or levels.get("whole_body_pattern", 0) >= 1:
        pattern_type = "accepted_full_pattern"
    elif levels.get("closure_required_candidate", 0) >= 1:
        pattern_type = "pending_closure_candidate"
    elif levels.get("split_required_candidate", 0) >= 1:
        pattern_type = "pending_split_candidate"
    elif levels.get("whole_body_pattern_candidate", 0) >= 1 or levels.get("multi_part_coordination", 0) >= 1:
        pattern_type = "unreviewed_full_or_composed_candidate"
    elif levels.get("transition", 0) >= 1:
        pattern_type = "transition_candidate"
    elif levels.get("component", 0) >= 1 or levels.get("local_component", 0) >= 1:
        pattern_type = "component_hit"
    else:
        pattern_type = "diagnostic_or_ambiguous"
    priority = {
        "whole_body_pattern": 6,
        "whole_body_pattern_candidate": 5,
        "closure_required_candidate": 5,
        "multi_part_coordination": 4,
        "split_required_candidate": 3,
        "transition": 3,
        "component": 2,
        "local_component": 2,
        "diagnostic_context": 1,
    }
    best_hit = max(
        best_hits,
        key=lambda row: (
            priority.get(str(row.get("semantic_level") or ""), 0),
            float(row.get("score") or 0.0),
        ),
    )
    return {
        "case_pattern_type": pattern_type,
        "best_window_count": len(best_hits),
        "semantic_level_counts": dict(sorted(levels.items())),
        "review_status_counts": dict(sorted(statuses.items())),
        "edit_scope_counts": dict(sorted(scopes.items())),
        "best_hit": best_hit,
    }


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    case_filter = _parse_case_ids(str(args.case_ids or ""))
    case_text = load_case_text(Path(args.source_corpus), max_cases=None)
    effective_max_cases = None if case_filter else args.max_cases
    channel_sequences = load_channel_sequences(Path(args.bpe_sequences), max_cases=effective_max_cases)
    if case_filter:
        channel_sequences = {
            sequence_id: seq
            for sequence_id, seq in channel_sequences.items()
            if sequence_id.split("::", 1)[0] in case_filter
        }
    coactivations = build_all_unit_coactivations(channel_sequences, parallel_overlap_min=float(args.parallel_overlap_min))
    program = load_composable_pattern_program(Path(args.program_json))
    available_kinds = {
        str(node.get("program_node_kind") or "")
        for node in program.get("nodes") or []
        if node.get("match_signature")
    }
    node_kinds = [kind for kind in ["pattern_family", "composition_family", "structure_group"] if kind in available_kinds]
    if not node_kinds:
        node_kinds = None

    case_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, units in coactivations.items():
        case_id = sequence_id.split("::", 1)[0]
        case_windows[case_id].extend(units)
    channel_case_ids = {sequence_id.split("::", 1)[0] for sequence_id in channel_sequences}
    all_case_ids = sorted(case_filter or channel_case_ids)

    case_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for case_id in all_case_ids:
        raw_windows = case_windows.get(case_id, [])
        windows = search_case_windows(
            case_id=case_id,
            coactivation_units=raw_windows,
            program=program,
            top_k=int(args.top_k),
            min_score=float(args.min_score),
            semantic_priority=bool(args.semantic_priority),
            node_kinds=node_kinds,
        )
        window_rows.extend(windows)
        text = case_text.get(case_id, {})
        case_rows.append(
            {
                "case_id": case_id,
                "num_frames": int(text.get("num_frames") or 0) or None,
                "caption_texts": text.get("caption_texts") or [],
                "caption_alias_ids": text.get("caption_alias_ids") or [],
                "window_count": len(windows),
                "case_classification": classify_case(windows),
                "top_windows": sorted(
                    windows,
                    key=lambda row: float(((row.get("hit_summary") or {}).get("top_hit") or {}).get("score") or 0.0),
                    reverse=True,
                )[: int(args.case_top_windows)],
            }
        )

    pattern_counts = Counter(str((row.get("case_classification") or {}).get("case_pattern_type") or "") for row in case_rows)
    hit_level_counts = Counter()
    for window in window_rows:
        top_hit = (window.get("hit_summary") or {}).get("top_hit") or {}
        if top_hit:
            hit_level_counts.update([str(top_hit.get("semantic_level") or "")])

    return {
        "schema_version": "aml_composable_pattern_program_search_v0",
        "runtime_policy": "motion-only tree search debug; captions are review context only",
        "inputs": {
            "program_json": str(args.program_json),
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "parallel_overlap_min": float(args.parallel_overlap_min),
            "min_score": float(args.min_score),
            "semantic_priority": bool(args.semantic_priority),
            "node_kinds": node_kinds or [],
        },
        "summary": {
            "case_count": len(case_rows),
            "channel_case_count": len(channel_case_ids),
            "case_with_coactivation_count": len(case_windows),
            "case_without_coactivation_count": len(case_rows) - len(case_windows),
            "window_count": len(window_rows),
            "case_pattern_type_counts": dict(sorted(pattern_counts.items())),
            "top_hit_semantic_level_counts": dict(sorted(hit_level_counts.items())),
        },
        "cases": case_rows,
        "windows": window_rows,
    }


def write_report(path: Path, payload: dict[str, Any], *, max_cases: int = 40) -> None:
    lines = ["# AML Composable Pattern Program Search v0", ""]
    summary = payload.get("summary") or {}
    inputs = payload.get("inputs") or {}
    lines.append(f"cases={summary.get('case_count')} windows={summary.get('window_count')}")
    lines.append(f"semantic_priority={inputs.get('semantic_priority')} node_kinds={inputs.get('node_kinds')}")
    lines.append("")
    lines.append("## Case Pattern Types")
    for key, value in (summary.get("case_pattern_type_counts") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Top Cases")
    for case in (payload.get("cases") or [])[:max_cases]:
        cls = case.get("case_classification") or {}
        best = cls.get("best_hit") or {}
        captions = case.get("caption_texts") or []
        lines.append(f"### {case.get('case_id')} | {cls.get('case_pattern_type')}")
        if captions:
            lines.append(f"- caption: {captions[0]}")
        if best:
            lines.append(
                f"- best: score={best.get('score')} level={best.get('semantic_level')} "
                f"edit={best.get('edit_scope')} node={best.get('program_node_id')}"
            )
            lines.append(f"- structure: {best.get('motion_structure_label')}")
        for window in case.get("top_windows") or []:
            top = (window.get("hit_summary") or {}).get("top_hit") or {}
            evidence = window.get("evidence") or {}
            if not top:
                continue
            lines.append(
                f"  - span={window.get('span')} score={top.get('score')} "
                f"level={top.get('semantic_level')} edit={top.get('edit_scope')}"
            )
            lines.append(f"    channels={evidence.get('channels')} clusters={evidence.get('cluster_ids')[:8]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "case_tree_search_results.json", payload.get("cases") or [])
    _write_json(output_dir / "window_tree_search_results.json", payload.get("windows") or [])
    _write_json(output_dir / "summary.json", payload.get("summary") or {})
    write_report(output_dir / "search_report.md", payload)


def run_self_test() -> None:
    program = {
        "nodes": [
            {
                "program_node_id": "program_family_test",
                "program_node_kind": "composition_family",
                "motion_structure_label": "bimanual_vertical_coordination",
                "semantic_level": "multi_part_coordination",
                "edit_scope": "multi_part",
                "composition_policy": "may_bind_as_composed_subpattern",
                "review_status": "review_candidate",
                "scope": "full_composition_candidate",
                "match_signature": {
                    "required_channels": ["bimanual", "whole_body_vertical"],
                    "required_zones": ["upper", "vertical"],
                    "required_cluster_ids": ["BI_RAISE_SPREAD", "WB_VERT_UP"],
                    "required_event_families": ["BIMANUAL_PERIODIC", "WHOLE_BODY_VERTICAL"],
                    "min_channel_overlap": 2,
                    "min_cluster_overlap": 0,
                    "min_event_family_overlap": 1,
                },
            }
        ],
        "edges": [],
        "condition_vocabulary": [],
    }
    unit = {
        "span": [1, 12],
        "channels": ["bimanual", "whole_body_vertical"],
        "geometry_clusters": ["BIMANUAL_PERIODIC/BI_RAISE_SPREAD", "WHOLE_BODY_VERTICAL/WB_VERT_UP"],
        "raw_geometry_clusters": [],
        "member_symbols": [],
    }
    rows = search_case_windows(case_id="case", coactivation_units=[unit], program=program, top_k=3, min_score=0.2)
    assert rows[0]["hits"]
    assert classify_case(rows)["case_pattern_type"] == "unreviewed_full_or_composed_candidate"
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(Path(tmp), {"summary": {"case_count": 1, "window_count": 1}, "cases": [], "windows": rows})
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search composable AML pattern program with motion evidence.")
    parser.add_argument("--program-json", default=str(DEFAULT_PROGRAM_JSON))
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-cases", type=int, default=250)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--parallel-overlap-min", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--case-top-windows", type=int, default=4)
    parser.add_argument("--semantic-priority", action="store_true")
    parser.add_argument(
        "--support-state-v1",
        action="store_true",
        help="Use the reviewed support-state v1 program and matching v4 support-state BPE sequence defaults.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.support_state_v1:
        if args.program_json == str(DEFAULT_PROGRAM_JSON):
            args.program_json = str(DEFAULT_SUPPORT_STATE_PROGRAM_JSON)
        if args.bpe_sequences == str(DEFAULT_BPE_SEQUENCES):
            args.bpe_sequences = str(DEFAULT_SUPPORT_STATE_BPE_SEQUENCES)
        if args.output_dir == str(DEFAULT_OUTPUT_DIR):
            args.output_dir = str(DEFAULT_SUPPORT_STATE_OUTPUT_DIR)

    if args.self_test:
        run_self_test()
        return

    payload = run_search(args)
    write_outputs(Path(args.output_dir), payload)
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(args.output_dir),
                "summary": payload.get("summary") or {},
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
