"""Diagnose split-axis evidence at case level and coactivation-window level.

This is a debug tool for data-defined split axes. It answers whether a target
case misses a full axis because the required evidence is absent from the case,
or because evidence exists in separate channel windows and does not coactivate.

Typical use:
    python scripts/audit_split_axis_case_coverage.py \
      --axis-id bilateral_spread_vertical_coordination_v0 \
      --search-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v0 \
      --output-dir outputs/aml_regression_testset_v2/aml_pattern_split_axis_jumpaxis_probe_coverage_v0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_v1_support_state_split_axes import DEFAULT_AXIS_SPEC, _cluster_set, _group_hits


DEFAULT_SEARCH_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v0"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_split_axis_jumpaxis_probe_coverage_v0"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _case_key(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or "")


def _groups_for_clusters(axis: dict[str, Any], clusters: set[str]) -> set[str]:
    return {str(hit.get("group_id") or "") for hit in _group_hits(axis.get("positive_groups") or [], clusters)}


def _axis_by_id(spec: dict[str, Any], axis_id: str) -> dict[str, Any]:
    for axis in spec.get("axes") or []:
        if str(axis.get("axis_id") or "") == axis_id:
            return dict(axis)
    raise KeyError(f"axis not found: {axis_id}")


def _label_rules(axis: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for rule in axis.get("label_rules") or []:
        label = str(rule.get("label") or "")
        groups = {str(item) for item in rule.get("require_all_groups") or []}
        groups.update(str(item) for item in rule.get("require_any_groups") or [])
        if label and groups:
            out[label] = groups
    default_label = str(axis.get("default_candidate_label") or axis.get("target_family") or "default")
    out.setdefault(default_label, {str(item) for item in axis.get("required_groups") or []})
    return out


def build_case_coverage(axis: dict[str, Any], cases: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    windows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        windows_by_case[_case_key(window)].append(window)

    audit_aliases = {str(item) for item in axis.get("audit_alias_ids") or []}
    required = {str(item) for item in axis.get("required_groups") or []}
    full_rule_groups = set()
    for groups in _label_rules(axis).values():
        if len(groups) > len(full_rule_groups):
            full_rule_groups = set(groups)

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = _case_key(case)
        aliases = {str(item) for item in case.get("caption_alias_ids") or []}
        case_clusters: set[str] = set()
        window_rows: list[dict[str, Any]] = []
        for idx, window in enumerate(windows_by_case.get(case_id, []), start=1):
            evidence = window.get("evidence") or {}
            clusters = _cluster_set(evidence)
            case_clusters.update(clusters)
            group_ids = _groups_for_clusters(axis, clusters)
            window_rows.append(
                {
                    "window_index": idx,
                    "span": window.get("span") or evidence.get("span"),
                    "groups": sorted(group_ids),
                    "channels": evidence.get("channels") or [],
                    "matched_clusters": sorted(clusters),
                    "top_tree_label": (((window.get("hit_summary") or {}).get("top_hit") or {}).get("motion_structure_label")),
                }
            )
        case_group_ids = _groups_for_clusters(axis, case_clusters)
        best_window_groups = max((set(row["groups"]) for row in window_rows), key=len, default=set())
        rows.append(
            {
                "case_id": case_id,
                "caption_alias_ids": sorted(aliases),
                "target_alias_hit": bool(aliases & audit_aliases),
                "caption": (case.get("caption_texts") or [""])[0],
                "window_count": len(window_rows),
                "case_level_groups": sorted(case_group_ids),
                "best_window_groups": sorted(best_window_groups),
                "case_has_required": required <= case_group_ids,
                "window_has_required": any(required <= set(row["groups"]) for row in window_rows),
                "case_has_full_rule": full_rule_groups <= case_group_ids if full_rule_groups else False,
                "window_has_full_rule": any(full_rule_groups <= set(row["groups"]) for row in window_rows) if full_rule_groups else False,
                "missing_required_case": sorted(required - case_group_ids),
                "missing_full_rule_case": sorted(full_rule_groups - case_group_ids),
                "windows": window_rows[:8],
            }
        )
    return {
        "axis_id": axis.get("axis_id"),
        "target_family": axis.get("target_family"),
        "audit_alias_ids": sorted(audit_aliases),
        "required_groups": sorted(required),
        "full_rule_groups": sorted(full_rule_groups),
        "rows": rows,
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    target_rows = [row for row in rows if row.get("target_alias_hit")]
    counters = {
        "case_level_groups": Counter(),
        "best_window_groups": Counter(),
        "missing_full_rule_case": Counter(),
        "missing_required_case": Counter(),
    }
    for row in target_rows:
        counters["case_level_groups"].update(row.get("case_level_groups") or [])
        counters["best_window_groups"].update(row.get("best_window_groups") or [])
        counters["missing_full_rule_case"].update(row.get("missing_full_rule_case") or [])
        counters["missing_required_case"].update(row.get("missing_required_case") or [])
    summary = {
        "axis_id": payload.get("axis_id"),
        "case_count": len(rows),
        "target_case_count": len(target_rows),
        "target_case_has_required": sum(1 for row in target_rows if row.get("case_has_required")),
        "target_window_has_required": sum(1 for row in target_rows if row.get("window_has_required")),
        "target_case_has_full_rule": sum(1 for row in target_rows if row.get("case_has_full_rule")),
        "target_window_has_full_rule": sum(1 for row in target_rows if row.get("window_has_full_rule")),
        "target_case_level_group_counts": dict(counters["case_level_groups"].most_common()),
        "target_best_window_group_counts": dict(counters["best_window_groups"].most_common()),
        "target_missing_full_rule_group_counts": dict(counters["missing_full_rule_case"].most_common()),
        "target_missing_required_group_counts": dict(counters["missing_required_case"].most_common()),
    }
    return summary


def write_report(path: Path, payload: dict[str, Any], summary: dict[str, Any], *, max_examples: int = 30) -> None:
    lines = ["# Split-Axis Case Coverage Audit", ""]
    lines.append(f"axis: `{summary.get('axis_id')}`")
    lines.append(f"cases={summary.get('case_count')} target_cases={summary.get('target_case_count')}")
    lines.append(
        "target coverage: "
        f"case_required={summary.get('target_case_has_required')} "
        f"window_required={summary.get('target_window_has_required')} "
        f"case_full={summary.get('target_case_has_full_rule')} "
        f"window_full={summary.get('target_window_has_full_rule')}"
    )
    lines.append("")
    lines.append("## Target Group Counts")
    lines.append(f"- case-level groups: `{summary.get('target_case_level_group_counts')}`")
    lines.append(f"- best-window groups: `{summary.get('target_best_window_group_counts')}`")
    lines.append(f"- missing full-rule groups: `{summary.get('target_missing_full_rule_group_counts')}`")
    lines.append("")
    lines.append("## Target Examples Missing Full Rule In Any Window")
    examples = [
        row for row in payload.get("rows") or []
        if row.get("target_alias_hit") and not row.get("window_has_full_rule")
    ]
    for row in examples[:max_examples]:
        lines.append(f"### {row.get('case_id')}: {row.get('caption')}")
        lines.append(f"- case groups: `{row.get('case_level_groups')}`")
        lines.append(f"- best window groups: `{row.get('best_window_groups')}`")
        lines.append(f"- missing full-rule case groups: `{row.get('missing_full_rule_case')}`")
        for window in row.get("windows") or []:
            lines.append(
                f"  - span={window.get('span')} groups={window.get('groups')} "
                f"channels={window.get('channels')} top={window.get('top_tree_label')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _read_json(Path(args.axis_spec))
    axis = _axis_by_id(spec, str(args.axis_id))
    search_dir = Path(args.search_dir)
    cases = _read_json(search_dir / "case_tree_search_results.json")
    windows = _read_json(search_dir / "window_tree_search_results.json")
    payload = build_case_coverage(axis, cases, windows)
    return payload, summarize(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-spec", type=Path, default=DEFAULT_AXIS_SPEC)
    parser.add_argument("--axis-id", default="bilateral_spread_vertical_coordination_v0")
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, summary = run(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "case_coverage.json", payload)
    _write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "case_coverage.md", payload, summary)
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
