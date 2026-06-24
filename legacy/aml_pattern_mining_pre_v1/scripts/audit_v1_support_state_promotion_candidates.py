"""Audit v1 support-state program hits for promotion decisions.

This script does not change the pattern tree. It aggregates motion-only search
hits by program node and reports whether each pending/component node looks pure
enough to promote, split, downgrade, or keep pending.

Typical use:
    python scripts/audit_v1_support_state_promotion_candidates.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROGRAM_JSON = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_reviewed_draft/"
    "aml_composable_pattern_program.json"
)
DEFAULT_SEARCH_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_search_v0"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_promotion_audit_v0"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _node_index(program: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("program_node_id") or ""): dict(node)
        for node in program.get("nodes") or []
        if node.get("program_node_id")
    }


def _case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id") or ""): dict(row) for row in cases}


def _target_aliases(node: dict[str, Any]) -> set[str]:
    label = str(node.get("motion_structure_label") or "")
    normalized = label.lower().replace("-", " ").replace("/", " ")
    registry = {
        "cartwheel": {"cartwheel"},
        "jumping jack": {"jumping_jack"},
        "sit down": {"sit_down", "sit_down_stand_up"},
        "kneel or fall to knees": {"kneel_or_fall_to_knees"},
        "martial arts": {"martial_arts", "karate_or_martial"},
        "swim like motion": {"swim_like_motion"},
        "cheer or dance": {"cheer_dance", "ballet_dance", "ballroom_dance_path"},
    }
    if normalized in registry:
        return set(registry[normalized])
    if label:
        return {label.replace(" ", "_")}
    return set()


def _score_promotion(row: dict[str, Any]) -> tuple[str, str]:
    status = str(row.get("review_status") or "")
    level = str(row.get("semantic_level") or "")
    case_count = int(row.get("matched_case_count") or 0)
    alias_hit_rate = float(row.get("target_alias_hit_rate") or 0.0)
    no_alias_rate = float(row.get("no_alias_rate") or 0.0)
    mean_score = float(row.get("mean_top_score") or 0.0)
    if status == "accepted":
        return "keep_positive", "already reviewed as a positive full-pattern condition"
    if case_count < 5:
        return "insufficient_support", "too few matched cases for promotion audit"
    if no_alias_rate >= 0.60:
        return "split_or_downgrade", "most hits have no aligned caption alias, so the geometry signature is too broad"
    if alias_hit_rate >= 0.45 and mean_score >= 0.55 and level in {"closure_required_candidate", "split_required_candidate"}:
        return "promote_review", "caption alias alignment and motion score are high enough for visual promotion review"
    if alias_hit_rate >= 0.25:
        return "split_review", "some aligned language exists, but the node is still mixed"
    if status == "component":
        return "keep_component", "node is currently a reusable component and lacks full-pattern purity"
    return "keep_pending", "not pure enough to promote, but may inform future split axes"


def _split_axis_hints(row: dict[str, Any]) -> list[str]:
    label = str(row.get("motion_structure_label") or "")
    clusters = {str(item.get("id") or "") for item in row.get("top_cluster_ids") or []}
    aliases = {str(item.get("id") or "") for item in row.get("caption_aliases") or []}
    hints: list[str] = []
    if label == "sit down":
        if {
            "WB_LEVEL_DESCEND_TO_LOW",
            "WB_LEVEL_RISE_FROM_LOW",
            "WB_LOW_BODY_DOWN_UP_CYCLE",
            "WB_LEVEL_LOW_SUSTAINED",
        } & clusters:
            hints.append(
                "candidate split axis: require ordered body-level transition evidence "
                "(descend-to-low, low hold/sustain, optional rise-from-low)"
            )
        if {"TORSO_HUNCHED_FORWARD", "TORSO_BEND_RECOVER"} & clusters:
            hints.append("candidate split axis: torso hunch/recover should support, not replace, body-level evidence")
        if {"martial_arts", "kneel_or_fall_to_knees", "cheer_dance"} & aliases:
            hints.append("confusers: martial/kneel/cheer aliases show this node still needs negative gates")
    elif label == "jumping jack":
        hints.append(
            "candidate split axis: require bilateral upper raise-spread plus bilateral lower/stance spread "
            "and repeated vertical cycles; upper+vertical alone is only a component"
        )
    elif label == "swim-like motion":
        hints.append("candidate split axis: require prone/floor or horizontal body support before naming as swim-like")
    elif label == "martial arts":
        hints.append("candidate split axis: require strike/guard arm evidence plus directional step/kick; gait+vertical is too broad")
    elif label == "kneel or fall to knees":
        hints.append("candidate split axis: separate kneel/fall-to-knees from crawl/all-fours and sit-down transitions")
    if not hints and str(row.get("recommendation") or "").startswith("split"):
        hints.append("candidate split axis: inspect aligned vs mixed examples before promotion")
    return hints


def build_audit(program: dict[str, Any], cases: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = _node_index(program)
    cases_by_id = _case_index(cases)
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "matched_cases": set(),
            "hit_count": 0,
            "top_hit_count": 0,
            "scores": [],
            "top_scores": [],
            "caption_alias_counter": Counter(),
            "caption_counter": Counter(),
            "cluster_counter": Counter(),
            "channel_counter": Counter(),
            "semantic_level_counter": Counter(),
            "review_status_counter": Counter(),
            "example_rows": [],
        }
    )

    for window in windows:
        case_id = str(window.get("case_id") or "")
        case = cases_by_id.get(case_id, {})
        aliases = [str(item) for item in case.get("caption_alias_ids") or []] or ["__none__"]
        captions = [str(item) for item in case.get("caption_texts") or []]
        evidence = window.get("evidence") or {}
        hits = list(window.get("hits") or [])
        for rank, hit in enumerate(hits, start=1):
            node_id = str(hit.get("program_node_id") or "")
            if node_id not in nodes:
                continue
            row = stats[node_id]
            row["matched_cases"].add(case_id)
            row["hit_count"] += 1
            row["scores"].append(float(hit.get("score") or 0.0))
            row["caption_alias_counter"].update(aliases)
            if captions:
                row["caption_counter"].update([captions[0]])
            row["cluster_counter"].update(str(item) for item in evidence.get("cluster_ids") or [])
            row["channel_counter"].update(str(item) for item in evidence.get("channels") or [])
            row["semantic_level_counter"].update([str(hit.get("semantic_level") or "")])
            row["review_status_counter"].update([str(hit.get("review_status") or "")])
            if rank == 1:
                row["top_hit_count"] += 1
                row["top_scores"].append(float(hit.get("score") or 0.0))
            if len(row["example_rows"]) < 8:
                row["example_rows"].append(
                    {
                        "case_id": case_id,
                        "span": window.get("span"),
                        "rank": rank,
                        "score": round(float(hit.get("score") or 0.0), 4),
                        "caption": captions[0] if captions else "",
                        "caption_alias_ids": aliases,
                        "channels": evidence.get("channels") or [],
                        "cluster_ids": (evidence.get("cluster_ids") or [])[:12],
                    }
                )

    rows: list[dict[str, Any]] = []
    for node_id, row in stats.items():
        node = nodes[node_id]
        target_aliases = _target_aliases(node)
        alias_counter: Counter[str] = row["caption_alias_counter"]
        matched_case_count = len(row["matched_cases"])
        target_alias_hits = sum(count for alias, count in alias_counter.items() if alias in target_aliases)
        no_alias_count = int(alias_counter.get("__none__", 0))
        top_scores = row["top_scores"] or row["scores"]
        out = {
            "program_node_id": node_id,
            "motion_structure_label": node.get("motion_structure_label"),
            "semantic_level": node.get("semantic_level"),
            "review_status": node.get("review_status"),
            "review_decision": node.get("review_decision"),
            "scope": node.get("scope"),
            "composition_policy": node.get("composition_policy"),
            "matched_case_count": matched_case_count,
            "hit_count": int(row["hit_count"]),
            "top_hit_count": int(row["top_hit_count"]),
            "mean_score": round(_safe_div(sum(row["scores"]), len(row["scores"])), 4),
            "mean_top_score": round(_safe_div(sum(top_scores), len(top_scores)), 4),
            "target_aliases": sorted(target_aliases),
            "target_alias_hit_rate": round(_safe_div(target_alias_hits, sum(alias_counter.values())), 4),
            "no_alias_rate": round(_safe_div(no_alias_count, sum(alias_counter.values())), 4),
            "caption_aliases": [{"id": key, "count": value} for key, value in alias_counter.most_common(8)],
            "top_captions": [{"text": key, "count": value} for key, value in row["caption_counter"].most_common(5)],
            "top_channels": [{"id": key, "count": value} for key, value in row["channel_counter"].most_common(10)],
            "top_cluster_ids": [{"id": key, "count": value} for key, value in row["cluster_counter"].most_common(12)],
            "examples": row["example_rows"],
        }
        recommendation, reason = _score_promotion(out)
        out["recommendation"] = recommendation
        out["recommendation_reason"] = reason
        out["split_axis_hints"] = _split_axis_hints(out)
        rows.append(out)

    rows.sort(
        key=lambda item: (
            str(item.get("recommendation") or "") not in {"promote_review", "split_review"},
            -int(item.get("matched_case_count") or 0),
            str(item.get("program_node_id") or ""),
        )
    )
    recommendation_counts = Counter(str(row.get("recommendation") or "") for row in rows)
    return {
        "schema_version": "v1_support_state_promotion_audit_v0",
        "runtime_policy": "caption aliases are diagnostics only; promotion still requires motion evidence and visual review",
        "summary": {
            "program_node_count": len(nodes),
            "audited_node_count": len(rows),
            "case_count": len(cases),
            "window_count": len(windows),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
        },
        "rows": rows,
    }


def write_report(path: Path, payload: dict[str, Any], *, max_rows: int) -> None:
    lines = ["# V1 Support-State Promotion Audit", ""]
    summary = payload.get("summary") or {}
    lines.append(
        f"audited_nodes={summary.get('audited_node_count')} cases={summary.get('case_count')} "
        f"windows={summary.get('window_count')}"
    )
    lines.append("")
    lines.append("## Recommendation Counts")
    for key, value in (summary.get("recommendation_counts") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Node Table")
    lines.append(
        "| node | label | status | cases | top hits | alias hit | no alias | mean top score | recommendation |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload.get("rows", [])[:max_rows]:
        lines.append(
            f"| `{row.get('program_node_id')}` | {row.get('motion_structure_label')} | "
            f"{row.get('review_status')} | {row.get('matched_case_count')} | {row.get('top_hit_count')} | "
            f"{row.get('target_alias_hit_rate')} | {row.get('no_alias_rate')} | {row.get('mean_top_score')} | "
            f"{row.get('recommendation')} |"
        )
    lines.append("")
    lines.append("## Details")
    for row in payload.get("rows", [])[:max_rows]:
        lines.append(f"### {row.get('program_node_id')}")
        lines.append(f"- label: `{row.get('motion_structure_label')}`")
        lines.append(f"- status: `{row.get('review_status')}` / level: `{row.get('semantic_level')}`")
        lines.append(f"- recommendation: `{row.get('recommendation')}`")
        lines.append(f"- reason: {row.get('recommendation_reason')}")
        for hint in row.get("split_axis_hints") or []:
            lines.append(f"- {hint}")
        lines.append(f"- target aliases: `{row.get('target_aliases')}`")
        lines.append(f"- caption aliases: `{row.get('caption_aliases')}`")
        lines.append(f"- top channels: `{row.get('top_channels')}`")
        lines.append(f"- top clusters: `{row.get('top_cluster_ids')}`")
        lines.append("- examples:")
        for ex in row.get("examples") or []:
            lines.append(
                f"  - {ex.get('case_id')} span={ex.get('span')} score={ex.get('score')} "
                f"aliases={ex.get('caption_alias_ids')}: {ex.get('caption')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    program = _read_json(Path(args.program_json))
    search_dir = Path(args.search_dir)
    cases = _read_json(search_dir / "case_tree_search_results.json")
    windows = _read_json(search_dir / "window_tree_search_results.json")
    return build_audit(program, cases, windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program-json", type=Path, default=DEFAULT_PROGRAM_JSON)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-report-rows", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "promotion_audit.json", payload)
    _write_json(output_dir / "summary.json", payload.get("summary") or {})
    write_report(output_dir / "promotion_audit.md", payload, max_rows=int(args.max_report_rows))
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
