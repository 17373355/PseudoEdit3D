"""Run one Pattern Mining Explorer axis audit bundle.

This is the converged entrypoint for split-axis diagnostics. It keeps coverage,
window scoring, and phase closure under one output directory instead of creating
separate branch artifacts.

Example:
    python scripts/run_pattern_axis_audit.py \
      --axis-id bilateral_spread_vertical_coordination_v0 \
      --search-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_full_v5_stance_width_v0 \
      --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/case_multichannel_bpe_sequences.jsonl \
      --output-dir outputs/aml_regression_testset_v2/aml_pattern_axis_audit_v1_bilateral_spread_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_split_axis_case_coverage as coverage_audit  # noqa: E402
from scripts import audit_split_axis_phase_closure as phase_audit  # noqa: E402
from scripts import audit_v1_support_state_split_axes as split_audit  # noqa: E402


DEFAULT_AXIS_SPEC = Path("pseudoedit3d/edit/aml_pattern_split_axes.json")
DEFAULT_AXIS_ID = "bilateral_spread_vertical_coordination_v0"
DEFAULT_SEARCH_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_full_v5_stance_width_v0"
)
DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/"
    "case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_axis_audit_v1")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_modes(text: str) -> set[str]:
    modes = {item.strip() for item in str(text or "all").split(",") if item.strip()}
    if "all" in modes:
        return {"coverage", "split", "phase"}
    unknown = modes - {"coverage", "split", "phase"}
    if unknown:
        raise ValueError(f"unknown audit modes: {sorted(unknown)}")
    return modes


def _summary_for(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("summary") or payload)


def run_bundle(args: argparse.Namespace) -> dict[str, Any]:
    modes = _parse_modes(str(args.modes))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}

    if "coverage" in modes:
        payload, summary = coverage_audit.run(
            argparse.Namespace(axis_spec=args.axis_spec, axis_id=args.axis_id, search_dir=args.search_dir, output_dir=output_dir)
        )
        _write_json(output_dir / "coverage_audit.json", payload)
        _write_json(output_dir / "coverage_summary.json", summary)
        coverage_audit.write_report(output_dir / "coverage_audit.md", payload, summary)
        outputs["coverage_audit"] = str(output_dir / "coverage_audit.json")
        summaries["coverage"] = summary

    if "split" in modes:
        payload = split_audit.run(argparse.Namespace(axis_spec=args.axis_spec, search_dir=args.search_dir, output_dir=output_dir))
        # Keep only the selected axis in the compact bundle; full rows remain in split_axis_audit.json.
        selected_summaries = [row for row in payload.get("axis_summaries") or [] if str(row.get("axis_id")) == str(args.axis_id)]
        compact = dict(payload)
        compact["axis_summaries"] = selected_summaries
        compact["accepted_rows"] = [row for row in payload.get("accepted_rows") or [] if str(row.get("axis_id")) == str(args.axis_id)]
        _write_json(output_dir / "split_axis_audit.json", compact)
        _write_json(output_dir / "split_summary.json", {"axis_summaries": selected_summaries, "summary": compact.get("summary") or {}})
        split_audit.write_report(output_dir / "split_axis_audit.md", compact)
        outputs["split_axis_audit"] = str(output_dir / "split_axis_audit.json")
        summaries["split"] = {"axis_summaries": selected_summaries, "summary": compact.get("summary") or {}}

    if "phase" in modes:
        payload = phase_audit.run(
            argparse.Namespace(
                axis_spec=args.axis_spec,
                axis_id=args.axis_id,
                source_corpus=args.source_corpus,
                bpe_sequences=args.bpe_sequences,
                output_dir=output_dir,
                case_ids=args.case_ids,
                min_pair_overlap=args.min_pair_overlap,
                max_gap_frames=args.max_gap_frames,
                max_center_gap_frames=args.max_center_gap_frames,
                broad_event_frame_ratio=args.broad_event_frame_ratio,
                broad_event_min_frames=args.broad_event_min_frames,
                example_limit=args.example_limit,
            )
        )
        _write_json(output_dir / "phase_closure_audit.json", payload)
        _write_json(output_dir / "phase_summary.json", payload.get("summary") or {})
        phase_audit.write_report(output_dir / "phase_closure_audit.md", payload)
        outputs["phase_closure_audit"] = str(output_dir / "phase_closure_audit.json")
        summaries["phase"] = payload.get("summary") or {}

    summary = {
        "schema_version": "aml_pattern_axis_audit_bundle_v1",
        "axis_id": str(args.axis_id),
        "modes": sorted(modes),
        "inputs": {
            "axis_spec": str(args.axis_spec),
            "search_dir": str(args.search_dir),
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
        },
        "outputs": outputs,
        "summaries": summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "audit_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Pattern Axis Audit Bundle", ""]
    lines.append(f"axis: `{summary.get('axis_id')}`")
    lines.append(f"modes: `{summary.get('modes')}`")
    lines.append("")
    for mode, data in (summary.get("summaries") or {}).items():
        lines.append(f"## {mode}")
        if mode == "coverage":
            lines.append(
                f"- target cases: {data.get('target_case_count')} | "
                f"full-rule case coverage: {data.get('target_case_has_full_rule')} | "
                f"full-rule window coverage: {data.get('target_window_has_full_rule')}"
            )
            lines.append(f"- missing full-rule groups: `{data.get('target_missing_full_rule_group_counts')}`")
        elif mode == "phase":
            lines.append(
                f"- strict phase closed: {data.get('strict_phase_closed_case_count')} cases, "
                f"precision={data.get('strict_phase_closed_precision')}, "
                f"recall={data.get('strict_phase_closed_recall')}"
            )
            lines.append(
                f"- connected/closed: {data.get('phase_connected_or_closed_case_count')} cases, "
                f"precision={data.get('phase_connected_or_closed_precision')}, "
                f"recall={data.get('phase_connected_or_closed_recall')}"
            )
            lines.append(f"- status counts: `{data.get('status_counts')}`")
        elif mode == "split":
            axis_summaries = data.get("axis_summaries") or []
            for axis in axis_summaries:
                lines.append(
                    f"- candidate cases: {axis.get('candidate_case_count')} | "
                    f"precision={axis.get('candidate_target_alias_precision')} | "
                    f"recall={axis.get('target_alias_case_recall')}"
                )
                lines.append(f"- labels: `{axis.get('candidate_label_counts')}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-spec", type=Path, default=DEFAULT_AXIS_SPEC)
    parser.add_argument("--axis-id", default=DEFAULT_AXIS_ID)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--source-corpus", type=Path, default=DEFAULT_SOURCE_CORPUS)
    parser.add_argument("--bpe-sequences", type=Path, default=DEFAULT_BPE_SEQUENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modes", default="all", help="Comma list: coverage,split,phase or all.")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--min-pair-overlap", type=float, default=0.15)
    parser.add_argument("--max-gap-frames", type=int, default=12)
    parser.add_argument("--max-center-gap-frames", type=int, default=18)
    parser.add_argument("--broad-event-frame-ratio", type=float, default=0.45)
    parser.add_argument("--broad-event-min-frames", type=int, default=48)
    parser.add_argument("--example-limit", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_bundle(args)
    print(
        json.dumps(
            {"ok": True, "output_dir": str(args.output_dir), "summary": {"axis_id": summary.get("axis_id"), "modes": summary.get("modes")}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
