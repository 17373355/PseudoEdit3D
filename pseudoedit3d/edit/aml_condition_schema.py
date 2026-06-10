from __future__ import annotations

from typing import Any


REQUIRED_APPROX_SLOTS: dict[str, tuple[str, ...]] = {
    "TRANSLATING_GAIT": ("span", "direction", "distance_m", "speed"),
    "TRANSLATING_GAIT_SEGMENT": ("span", "direction", "distance_m", "speed"),
    "IN_PLACE_GAIT": ("span", "direction", "count", "vertical_amplitude_m"),
    "IN_PLACE_GAIT_PROXY": ("span", "direction", "count", "vertical_amplitude_m", "source_event_count"),
    "BALLISTIC_TRANSLATION": ("span", "direction", "distance_m", "vertical_amplitude_m"),
    "BALLISTIC_TRANSLATION_SEGMENT": ("span", "direction", "distance_m", "vertical_amplitude_m"),
    "WEAK_BALLISTIC_CANDIDATE": ("span", "direction", "distance_m", "vertical_amplitude_m"),
    "VERTICAL_JUMP": ("span", "direction", "vertical_amplitude_m"),
    "VERTICAL_JUMP_SEGMENT": ("span", "direction", "vertical_amplitude_m"),
    "JUMPING_JACK": ("span", "direction", "count", "vertical_amplitude_m"),
    "ROTATION_DOMINANT": ("span", "direction", "angle_deg", "angle_bin"),
    "TURN_SEGMENT": ("span", "direction", "angle_deg", "angle_bin"),
    "RECOVERY_STEP_SEGMENT": ("span", "direction", "distance_m", "speed"),
    "TERMINAL_STILL": ("span", "direction"),
    "BIMANUAL_HANDS_CLOSE": ("span", "direction", "count"),
    "BIMANUAL_ARM_MIME_CANDIDATE": ("span", "direction", "source_event_count", "bimanual_count"),
    "UNILATERAL_ARM_MIME_CANDIDATE": ("span", "direction", "source_event_count", "dominant_side"),
    "TORSO_HUNCHED_FORWARD": ("span", "direction", "source_event_count", "magnitude"),
    "LEFT_HAND_RAISED_HIGH": ("span", "direction", "source_event_count", "magnitude"),
    "RIGHT_HAND_RAISED_HIGH": ("span", "direction", "source_event_count", "magnitude"),
    "SQUAT_HOLD": ("span", "direction", "source_event_count", "magnitude"),
    "SQUAT_REPETITION": ("span", "direction", "count", "magnitude|vertical_amplitude_m"),
    "SQUAT_ARM_LIFT": ("span", "direction", "count", "magnitude|vertical_amplitude_m"),
    "LEFT_LEG_KICK_FORWARD": ("span", "direction", "source_event_count", "magnitude"),
    "RIGHT_LEG_KICK_FORWARD": ("span", "direction", "source_event_count", "magnitude"),
    "LEG_FORWARD_POSE_CANDIDATE": ("span", "direction", "dominant_side", "source_event_count"),
    "DANCE_LEG_POSE_CANDIDATE": ("span", "direction", "dominant_side", "source_event_count"),
    "CIRCULAR_WALK_PATH": ("span", "direction", "path_length_m", "curvature_rad", "circle_score"),
    "CLIMB_UP_OVER_PROXY": ("span", "direction", "root_height_gain_m", "source_event_count"),
    "CARTWHEEL_CANDIDATE": ("span", "direction", "source_event_count"),
    "INVERTED_ACROBATICS_CANDIDATE": ("span", "direction", "source_event_count"),
    "ACROBATIC_SEQUENCE_CANDIDATE": ("span", "direction", "segment_count", "source_event_count"),
    "CELEBRATORY_DANCE_GESTURE": (
        "span",
        "direction",
        "turn_count",
        "locomotion_segment_count",
        "raise_spread_count",
        "bimanual_count",
    ),
    "STATIC_OR_SUBTLE_STATE_PROXY": ("span", "source_event_count"),
}


STATUS_CONDITION_WEIGHTS = {
    "stable": 1.0,
    "candidate": 0.7,
    "proxy": 0.5,
    "unknown": 0.0,
}


def required_approx_slots(family_id: str) -> tuple[str, ...]:
    return REQUIRED_APPROX_SLOTS.get(family_id, ("span",))


def slot_requirement_satisfied(requirement: str, approx_slots: dict[str, Any]) -> bool:
    return any(slot_key in approx_slots for slot_key in requirement.split("|"))


def missing_required_slots(family_id: str, approx_slots: dict[str, Any]) -> list[str]:
    return [
        requirement
        for requirement in required_approx_slots(family_id)
        if not slot_requirement_satisfied(requirement, approx_slots)
    ]


def slot_values(approx_slots: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, slot in approx_slots.items():
        if isinstance(slot, dict) and "value" in slot:
            out[key] = slot["value"]
    return out


def slot_confidences(approx_slots: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, slot in approx_slots.items():
        if not isinstance(slot, dict):
            continue
        try:
            out[key] = round(float(slot.get("confidence")), 4)
        except (TypeError, ValueError):
            continue
    return out


def slot_qualities(approx_slots: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, slot in approx_slots.items():
        if isinstance(slot, dict) and slot.get("quality") is not None:
            out[key] = str(slot["quality"])
    return out


def action_condition_weight(semantic_family: dict[str, Any], missing_slots: list[str]) -> float:
    status = str(semantic_family.get("status") or "unknown")
    if missing_slots or status == "unknown" or semantic_family.get("probe_visible") is False:
        return 0.0
    return float(STATUS_CONDITION_WEIGHTS.get(status, 0.0))
