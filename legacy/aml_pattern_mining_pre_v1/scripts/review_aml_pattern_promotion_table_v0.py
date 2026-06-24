"""Self-review the AML pattern promotion table.

This is an offline audit helper. It reads the promotion review table and writes
a structured decision for each dense family. Decisions are based on motion
structure, alias concentration, and text examples; they are not runtime rules.

Example:
    python scripts/review_aml_pattern_promotion_table_v0.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0/"
    "promotion_review_table.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _has_any(values: list[str], parts: tuple[str, ...]) -> bool:
    return any(any(part in value for part in parts) for value in values)


def _alias_text(row: dict[str, Any]) -> str:
    aliases = row.get("caption_aliases") or []
    return ", ".join(f"{item.get('id')}:{item.get('count')}" for item in aliases[:6]) or "none"


def _top_alias(row: dict[str, Any]) -> tuple[str, int]:
    aliases = row.get("caption_aliases") or []
    if not aliases:
        return "", 0
    return str(aliases[0].get("id") or ""), int(aliases[0].get("count") or 0)


def _examples_text(row: dict[str, Any]) -> str:
    return " ".join(str(item.get("caption") or "").lower() for item in row.get("examples") or [])


def _geometry_bucket(geometry: list[str], channels: list[str]) -> str:
    if _has_any(geometry, ("LEG_FORWARD_GAIT_SWING", "LOCO_GAIT_CONTEXT", "WB_VERT_GAIT_BOUNCE", "LOCO_ARM_SWING")):
        return "gait_context"
    if _has_any(geometry, ("LEG_FORWARD_HOLD_POSE", "LEG_FORWARD_KICK_IMPULSE", "LEG_FORWARD_HOP_OR_KICK_IMPULSE", "LEG_FORWARD_UNRESOLVED")):
        return "leg_forward_refined"
    if _has_any(geometry, ("LL_KICK_FORWARD", "RL_KICK_FORWARD")) and "whole_body_vertical" in channels:
        return "legacy_leg_vertical"
    if _has_any(geometry, ("LL_KICK_FORWARD", "RL_KICK_FORWARD")):
        return "legacy_leg"
    if _has_any(geometry, ("LOW_BODY_DESCENT", "LOW_BODY_RISE", "LOW_BODY_DOWN_UP", "SQUAT_HOLD", "LOW_BODY_HOLD")):
        return "low_body"
    if _has_any(geometry, ("WB_VERT_ARM_RAISE", "BILATERAL_VERTICAL_ARM_CYCLE", "RAISE_SPREAD_VERTICAL", "HANDS_CLOSE_VERTICAL")):
        return "arm_vertical_coordination"
    if _has_any(geometry, ("WB_VERT_", "VERT_GENERIC", "VERTICAL", "JUMP_UP_IMPULSE", "SALIENT_DESCENT")):
        return "vertical"
    if _has_any(geometry, ("BILATERAL_ARM_PERIODIC", "ISOLATED_ARM_PERIODIC", "BIMANUAL_CONTEXT_ARM_PERIODIC", "LA_REPEAT", "RA_REPEAT", "LA_NEAR_FAR", "RA_NEAR_FAR")):
        return "arm_periodic"
    if _has_any(geometry, ("BILATERAL_HIGH_POSE", "HIGH_POSE", "LA_HAND_HIGH", "RA_HAND_HIGH")):
        return "arm_posture"
    if _has_any(geometry, ("BI_RAISE", "BI_SPREAD", "BI_HANDS_CLOSE", "BIMANUAL_PERIODIC")):
        return "bimanual_component"
    if _has_any(geometry, ("TORSO_",)):
        return "torso"
    if _has_any(geometry, ("WB_ROT", "LOCO_TURN")):
        return "turn"
    if _has_any(geometry, ("LOCO_ROOT_DRIFT",)):
        return "root_drift"
    if _has_any(geometry, ("LOCO_PATH_FRAGMENT", "LOCO_TRANSLATION", "LOCO_", "ROOT_CIRCULAR_PATH")):
        return "root_locomotion"
    return "generic"


def _review_row(row: dict[str, Any]) -> dict[str, Any]:
    family_id = str(row.get("family_id") or "")
    rec = str(row.get("recommendation") or "")
    channels = list(row.get("channels") or [])
    geometry = list(row.get("geometry_clusters") or [])
    geometry_bucket = _geometry_bucket(geometry, channels)
    top_alias, top_alias_count = _top_alias(row)
    examples = _examples_text(row)

    final_decision = "keep_diagnostic"
    proposed_scope = "diagnostic"
    proposed_name = "unnamed_motion_evidence"
    confidence = "medium"
    reason = "insufficient structural or naming purity for promotion"
    v2_fix = "keep as evidence; revisit after Motion-BPE v2 purity audit"
    needs_visual_review = False

    if rec == "already_reviewed":
        final_decision = "keep_accepted_reference"
        proposed_scope = "accepted_pattern_reference"
        proposed_name = "jumping_jack_full_coordination_reference"
        confidence = "high"
        reason = "already linked to reviewed v0 forest with concentrated jumping-jack examples"
        v2_fix = "none"
    elif rec == "composition_review":
        if geometry_bucket == "gait_context":
            final_decision = "downgrade_to_component"
            proposed_scope = "gait_context_component"
            proposed_name = "gait_context_coordination_component"
            confidence = "high"
            reason = "v2 refinement identifies this as gait/root/leg/arm context rather than a clean full action"
            v2_fix = "already split into gait-context observables; use as context evidence only"
        elif geometry_bucket == "leg_forward_refined":
            final_decision = "downgrade_to_component"
            proposed_scope = "leg_forward_component"
            proposed_name = "leg_forward_pose_or_impulse_component"
            confidence = "medium"
            reason = "leg-forward evidence is now separated from gait, but examples still need composition before naming a full action"
            v2_fix = "compose with support, low-body, and root context before promoting kick/lunge-like patterns"
        elif geometry_bucket == "legacy_leg_vertical":
            final_decision = "downgrade_to_component"
            proposed_scope = "gait_or_hop_component"
            proposed_name = "leg_forward_swing_with_vertical_bounce_component"
            confidence = "high"
            reason = "examples are mostly walking, stopping, turning, or path fragments; leg-forward clusters conflate gait swing with kick-like motion"
            v2_fix = "split leg-forward observable into gait swing, kick impulse, hold pose, and lunge step using contact/root-locomotion/duration"
        elif geometry_bucket == "legacy_leg":
            final_decision = "downgrade_to_component"
            proposed_scope = "gait_leg_component"
            proposed_name = "bilateral_leg_forward_swing_component"
            confidence = "high"
            reason = "bilateral leg-forward coactivation mostly describes gait stepping rather than a named action"
            v2_fix = "separate gait-phase leg swing from kick/lunge candidates before building action nodes"
        elif geometry_bucket == "arm_posture":
            final_decision = "downgrade_to_component"
            proposed_scope = "upper_body_pose_component"
            proposed_name = "both_hands_high_pose_component"
            confidence = "medium"
            reason = "both-hand-high coordination is useful, but examples span gestures and jumping-jack-like motions"
            v2_fix = "compose with vertical/lower-body packets before promoting as jumping-jack-like full pattern"
        elif geometry_bucket in {"arm_periodic", "arm_vertical_coordination", "bimanual_component"}:
            final_decision = "downgrade_to_component"
            if geometry_bucket == "arm_vertical_coordination" or ("whole_body_vertical" in channels and not {"left_arm", "right_arm"}.issubset(set(channels))):
                proposed_scope = "arm_vertical_coordination_component"
                proposed_name = "single_arm_periodic_with_vertical_motion_component"
            else:
                proposed_scope = "upper_body_periodic_component"
                proposed_name = "bilateral_arm_periodic_component"
            confidence = "high"
            reason = "arm periodic motion appears in many actions and aliases are diffuse"
            v2_fix = "split by arm path direction, symmetry, object-mime hints, and coupling to lower-body/root motion"
        elif geometry_bucket == "low_body":
            final_decision = "downgrade_to_component"
            proposed_scope = "low_body_transition_component"
            proposed_name = "low_body_transition_or_hold_component"
            confidence = "medium"
            reason = "low-body evidence is useful for sit/stand/kneel/squat-like patterns but still lacks full support/contact semantics"
            v2_fix = "compose low-body transition with torso/root/support evidence before promotion"
        elif geometry_bucket == "turn":
            final_decision = "downgrade_to_component"
            proposed_scope = "root_rotation_component"
            proposed_name = "root_turn_context_component"
            confidence = "high"
            reason = "turn geometry describes path orientation or rotation context, not a complete action"
            v2_fix = "already split by angle, tempo, and path/isolated context"
        elif geometry_bucket in {"root_drift", "root_locomotion"}:
            final_decision = "downgrade_to_component"
            proposed_scope = "root_locomotion_component"
            proposed_name = "root_path_or_drift_component"
            confidence = "high" if geometry_bucket == "root_drift" else "medium"
            reason = "root motion is path or weak drift context and should not become a full action node by itself"
            v2_fix = "already split root drift, path fragments, gait context, and translation context"
        elif geometry_bucket == "torso":
            final_decision = "downgrade_to_component"
            proposed_scope = "torso_context_component"
            proposed_name = "torso_context_component"
            confidence = "medium"
            reason = "torso posture/periodicity is reusable context across many actions"
            v2_fix = "already split torso by low-body, locomotion, vertical, sustained, and periodic context"
        else:
            final_decision = "needs_visual_review"
            proposed_scope = "uncertain_composition"
            proposed_name = "uncertain_multichannel_coordination"
            confidence = "low"
            reason = "text examples are not enough to decide whether this is a reusable component or a full pattern"
            v2_fix = "inspect motion examples before promotion"
            needs_visual_review = True
    elif rec == "component_review":
        final_decision = "keep_component"
        proposed_scope = "component_library"
        confidence = "high"
        if geometry_bucket == "gait_context":
            proposed_name = "gait_context_component"
            reason = "v2 separates gait leg swing, gait bounce, locomotion-coupled arms, and gait root context; keep as context/component"
            v2_fix = "already refined; downstream should prevent these from naming kick/jump/sport patterns alone"
        elif geometry_bucket == "vertical":
            proposed_name = "whole_body_vertical_motion_component"
            reason = "vertical motion is common across gait, jump, sit/stand, and dance; it is a component only"
            v2_fix = "split vertical events by amplitude, support phase, and coupling to locomotion/arms"
        elif geometry_bucket in {"leg_forward_refined", "legacy_leg"}:
            proposed_name = "leg_forward_motion_component"
            reason = "leg-forward signal is a reusable component and needs context before becoming kick/lunge/sit-related semantics"
            v2_fix = "already split gait swing, hold pose, and impulse-like forward leg evidence"
        elif geometry_bucket in {"arm_periodic", "arm_posture", "arm_vertical_coordination", "bimanual_component"}:
            proposed_name = "arm_motion_component"
            reason = "arm motion is frequent and semantically ambiguous without object/style/context evidence"
            v2_fix = "v2 splits symmetry, vertical coupling, locomotion coupling, bimanual context, and high-pose context; still component only"
        elif geometry_bucket == "torso":
            proposed_name = "torso_motion_component"
            reason = "torso posture/periodic motion appears in sit, bow, duck, swim, and many mimes"
            v2_fix = "already split torso posture by low-body, locomotion, vertical, and periodic context"
        elif geometry_bucket == "turn":
            proposed_name = "root_rotation_component"
            reason = "root turns are path components, not full action labels"
            v2_fix = "already split turn angle, tempo, and path role"
        elif geometry_bucket in {"root_locomotion", "root_drift"}:
            proposed_name = "root_locomotion_component"
            reason = "root locomotion is context for many actions"
            v2_fix = "already split path fragments, weak drift, gait context, and translation context"
        elif geometry_bucket == "low_body":
            proposed_name = "low_body_posture_component"
            reason = "low-body posture supports sit/kneel/squat/cartwheel proxies but is not a full action alone"
            v2_fix = "v2 separates descent, rise, down-up cycle, sustained hold, gait context, and leg-extension context"
        else:
            proposed_name = "generic_motion_component"
            reason = "frequent local sequence, but no stable full-action semantics"
            v2_fix = "keep as component until a cleaner composition emerges"
    elif rec == "name_only_review":
        final_decision = "keep_name_alignment_only"
        proposed_scope = "naming_sidecar"
        confidence = "high"
        if top_alias in {"jumping_jack"}:
            proposed_name = "jumping_jack_component_name_evidence"
            reason = "caption aliases are concentrated, but structure is only an arm/hand component"
            v2_fix = "compose with vertical and lower-body coordination before pattern promotion"
        elif top_alias in {"sit_down", "kneel_or_fall_to_knees", "sit_down_stand_up"}:
            proposed_name = "low_body_state_name_evidence"
            reason = "caption aliases point to low-body transitions, but structure lacks transition/support detail"
            v2_fix = "add sit/stand/kneel transition observables before naming a full pattern"
        elif top_alias in {"cheer_dance"}:
            proposed_name = "bimanual_gesture_name_evidence"
            reason = "caption aliases point to style/activity, but structure is only bimanual motion"
            v2_fix = "keep style/activity labels in naming sidecar; do not promote without stable motion composition"
        else:
            proposed_name = f"{top_alias or 'caption'}_name_evidence"
            reason = "caption name concentration is useful for naming only"
            v2_fix = "align to motion nodes after structural promotion"
    elif rec == "diagnostic_keep":
        final_decision = "keep_diagnostic"
        proposed_scope = "diagnostic_only"
        proposed_name = "low_support_or_noisy_coordination"
        confidence = "medium"
        reason = "low support and diffuse aliases; not enough for tree growth"
        v2_fix = "use only as evidence for future split/merge decisions"

    if any(word in examples for word in ("walk", "turn", "forward", "backward", "circle", "path", "stops")) and rec == "composition_review":
        if final_decision == "needs_visual_review":
            final_decision = "downgrade_to_component"
            proposed_scope = "path_or_gait_component"
            proposed_name = "path_coupled_coordination_component"
            confidence = "medium"
            reason = "caption examples mostly describe path or gait behavior"
            v2_fix = "separate path/gait context from action-specific coordination"

    return {
        "priority_rank": row.get("priority_rank"),
        "family_id": family_id,
        "source_recommendation": rec,
        "final_decision": final_decision,
        "proposed_scope": proposed_scope,
        "proposed_name": proposed_name,
        "confidence": confidence,
        "needs_visual_review": needs_visual_review,
        "support_cases_sum": row.get("support_cases_sum"),
        "motif_count": row.get("motif_count"),
        "channels": channels,
        "geometry_clusters": geometry,
        "caption_aliases": row.get("caption_aliases") or [],
        "alias_summary": _alias_text(row),
        "reason": reason,
        "v2_fix": v2_fix,
        "examples": row.get("examples") or [],
    }


def build_self_review(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [_review_row(row) for row in payload.get("review_rows") or []]
    decision_counts = Counter(row["final_decision"] for row in rows)
    scope_counts = Counter(row["proposed_scope"] for row in rows)
    visual_count = sum(1 for row in rows if row.get("needs_visual_review"))
    return {
        "schema_version": "aml_pattern_promotion_self_review_v0",
        "source": str(DEFAULT_INPUT),
        "summary": {
            "reviewed_family_count": len(rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
            "needs_visual_review_count": visual_count,
        },
        "review_rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("review_rows") or []
    lines = ["# AML Pattern Promotion Self-Review v0", ""]
    lines.append("This is a text/structure review of the promotion table.")
    lines.append("Only rows marked `needs_visual_review` require human motion inspection.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary = payload.get("summary") or {}
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.get('priority_rank')}. {row.get('family_id')}")
        lines.append("")
        lines.append(f"- decision: `{row.get('final_decision')}`")
        lines.append(f"- scope: `{row.get('proposed_scope')}`")
        lines.append(f"- proposed_name: `{row.get('proposed_name')}`")
        lines.append(f"- confidence: `{row.get('confidence')}`")
        lines.append(f"- support: `{row.get('support_cases_sum')}`; motifs: `{row.get('motif_count')}`")
        lines.append(f"- channels: `{row.get('channels')}`")
        lines.append(f"- geometry: `{row.get('geometry_clusters')}`")
        lines.append(f"- aliases: {row.get('alias_summary')}")
        lines.append(f"- reason: {row.get('reason')}")
        lines.append(f"- v2_fix: {row.get('v2_fix')}")
        examples = row.get("examples") or []
        if examples:
            lines.append("- example captions:")
            for example in examples[:3]:
                lines.append(
                    f"  - `{example.get('case_id')}` span={example.get('span')} "
                    f"{example.get('caption')}"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(Path(args.input))
    review = build_self_review(payload)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "promotion_self_review.json", review)
    _write_json(output_dir / "promotion_self_review_summary.json", review.get("summary") or {})
    write_markdown(output_dir / "promotion_self_review.md", review)
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    review = run(parse_args())
    print(json.dumps(review.get("summary") or {}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
