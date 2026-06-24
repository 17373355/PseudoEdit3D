"""Draft review decisions for the v1 support-state pattern forest.

This script is a review helper only. It reads the static review pack generated
from the support-state closure forest and fills a first-pass decision draft.
The output is meant to be edited/reviewed before any node is promoted into an
accepted AML vocabulary.

Typical use:
    python scripts/propose_v1_support_state_review_decisions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_PACK = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_review_pack"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/"
    "aml_pattern_forest_v1_support_state_full_v0_review_decisions_draft"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _role_ids(row: dict[str, Any]) -> list[str]:
    motion = row.get("motion_summary") or {}
    roles = motion.get("canonical_role_items") or []
    out: list[str] = []
    for item in roles:
        if isinstance(item, dict):
            role = str(item.get("id") or "")
        else:
            role = str(item)
        if role:
            out.append(role)
    return out


def _alias_ids(row: dict[str, Any], limit: int = 5) -> list[str]:
    return [str(item.get("id") or "") for item in (row.get("language_aliases") or [])[:limit] if item.get("id")]


def _example_case_ids(row: dict[str, Any]) -> list[str]:
    return [str(item.get("case_id") or "") for item in (row.get("examples") or []) if item.get("case_id")]


def _has_role(row: dict[str, Any], needle: str) -> bool:
    return any(needle in role for role in _role_ids(row))


def _support_summary(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence") or {}
    return {
        "source_candidate_count": int(evidence.get("source_candidate_count") or 0),
        "support_cases_max": int(evidence.get("support_cases_max") or 0),
        "support_cases_sum": int(evidence.get("support_cases_sum") or 0),
        "example_count": int(row.get("example_count") or len(row.get("examples") or [])),
    }


def propose_decision(row: dict[str, Any]) -> tuple[str, str]:
    """Return an editable first-pass decision and rationale.

    The rules intentionally use review status, structural scope, and motion
    roles. The action name is used only for human-readable rationale, not as
    motion evidence.
    """

    status = str(row.get("status") or "")
    scope = str(row.get("scope") or "")
    roles = set(_role_ids(row))

    if status == "review_candidate":
        if scope == "full_pattern_candidate" and (
            "whole_body_support:inverted_support" in roles
            or _has_role(row, "inversion_or_acrobatics")
        ):
            return (
                "promote",
                "examples share an explicit inverted-support/acrobatic full-body structure; this is the cleanest current full-pattern promotion candidate",
            )
        if scope == "floor_or_prone_pattern_candidate":
            return (
                "split",
                "floor/prone support is useful, but the family still mixes prone swimming, lying/get-up, sit/kneel-like transitions, and mime-like upper-body motion",
            )
        if scope == "full_upper_lower_body_candidate":
            return (
                "downgrade_to_component",
                "upper/lower coordination is real but the evidence is broad or low-support; keep it as a reusable coordination component until a purer full pattern appears",
            )
        return (
            "downgrade_to_component",
            "review candidate is structurally plausible but not specific enough for a named full-pattern node",
        )

    if status == "split_required":
        if "component" in scope:
            return (
                "downgrade_to_component",
                "scope conflict indicates this is a reusable component, not a complete action pattern",
            )
        if scope in {"transition_pattern_candidate", "floor_or_prone_pattern_candidate", "full_upper_lower_body_candidate"}:
            return (
                "split_axis_confirmed",
                "candidate has useful evidence but needs an additional structural split before promotion",
            )
        return (
            "split_axis_confirmed",
            "candidate is explicitly marked as needing split before promotion",
        )

    if status == "composition_needs_closure":
        if scope == "full_upper_lower_body_candidate":
            return (
                "needs_closure",
                "near-complete full-body structure is present, but a missing or unstable role prevents direct promotion",
            )
        if scope in {"transition_pattern_candidate", "floor_or_prone_pattern_candidate"}:
            return (
                "needs_closure",
                "transition/floor evidence is present but remains too broad; closure or support-state splitting is needed",
            )
        return (
            "needs_closure",
            "composition candidate needs one more stable structural role before promotion",
        )

    return (
        "pending",
        "no automatic decision rule matched; keep for manual review",
    )


def build_decision_payload(template: dict[str, Any], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {str(row.get("family_id") or ""): row for row in manifest}
    decisions: list[dict[str, Any]] = []

    for item in template.get("decisions") or []:
        family_id = str(item.get("family_id") or "")
        row = by_family.get(family_id, {})
        decision, rationale = propose_decision(row or item)
        allowed = set(item.get("allowed_decisions") or [])
        if allowed and decision not in allowed:
            decision = "pending"
            rationale = f"draft rule suggested an unsupported decision; allowed decisions are {sorted(allowed)}"

        output = dict(item)
        output["decision"] = decision
        output["notes"] = rationale
        output["motion_roles"] = _role_ids(row)[:12]
        output["language_aliases"] = _alias_ids(row)
        output["support_summary"] = _support_summary(row)
        output["example_case_ids"] = _example_case_ids(row) or item.get("example_case_ids") or []
        decisions.append(output)

    summary: dict[str, int] = {}
    for item in decisions:
        key = str(item.get("decision") or "pending")
        summary[key] = summary.get(key, 0) + 1

    return {
        "schema_version": "v1_support_state_review_decision_draft_v0",
        "source_template": str(DEFAULT_REVIEW_PACK / "review_decision_template.json"),
        "source_manifest": str(DEFAULT_REVIEW_PACK / "review_pack_manifest.json"),
        "instructions": (
            "Editable first-pass decisions. Captions and names are naming hints only; "
            "promotion still requires user review of the PNG sheets."
        ),
        "summary": summary,
        "decisions": decisions,
    }


def _format_examples(item: dict[str, Any], limit: int = 8) -> str:
    cases = [str(case_id) for case_id in item.get("example_case_ids") or [] if case_id]
    if not cases:
        return "-"
    shown = ", ".join(cases[:limit])
    if len(cases) > limit:
        shown += f", ... (+{len(cases) - limit})"
    return shown


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = [
        "# v1 Support-State Pattern Review Decisions Draft",
        "",
        "This is an editable decision draft. It does not promote nodes by itself.",
        "",
        "Decision counts:",
        "",
    ]
    for decision, count in sorted((payload.get("summary") or {}).items()):
        lines.append(f"- `{decision}`: {count}")
    lines.extend(
        [
            "",
            "| family | status | scope | name hint | draft decision | examples | image | note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for item in payload.get("decisions") or []:
        image = str(item.get("image_path") or "")
        lines.append(
            "| "
            f"`{item.get('family_id')}` | "
            f"`{item.get('status')}` | "
            f"`{item.get('scope')}` | "
            f"{item.get('name_candidate')} | "
            f"`{item.get('decision')}` | "
            f"{_format_examples(item)} | "
            f"`{image}` | "
            f"{item.get('notes')} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Gate",
            "",
            "- `promote`: can become the first accepted v1 node after visual confirmation.",
            "- `split_axis_confirmed` / `needs_closure`: keep as mining TODOs; do not expose as final action labels.",
            "- `downgrade_to_component`: keep as component-library evidence only.",
            "- `split`: rerun split/closure logic before promotion.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack-dir", type=Path, default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = args.review_pack_dir / "review_decision_template.json"
    manifest_path = args.review_pack_dir / "review_pack_manifest.json"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    template = _read_json(template_path)
    manifest = _read_json(manifest_path)
    payload = build_decision_payload(template, manifest)
    payload["source_template"] = str(template_path)
    payload["source_manifest"] = str(manifest_path)
    payload["output_dir"] = str(output_dir)

    _write_json(output_dir / "review_decisions_draft.json", payload)
    write_markdown(payload, output_dir / "review_decisions_draft.md")
    _write_json(output_dir / "summary.json", {"decision_counts": payload["summary"], "output_dir": str(output_dir)})

    print(json.dumps({"ok": True, "decision_counts": payload["summary"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
