from __future__ import annotations

from typing import Any

from .coarse_event_utils import (
    _count_peaks,
    _coverage,
    _duration,
    _event_by_index,
    _event_sort_key,
    _gap,
    _indexed_events,
    _is_after,
    _magnitude,
    _mean_magnitude,
    _overlap_ratio,
    _span,
    _speed_from_event,
    _total_frames,
)
from .coarse_pattern_counts import (
    bilateral_gesture_evidence,
    bilateral_rhythmic_evidence,
    low_body_repetition_evidence,
)


def _locomotion_axis(events: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    active = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
    ]
    turn = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
    ]
    best = max(active, key=lambda evt: (_magnitude(evt), _duration(evt)), default=None)
    distance = _magnitude(best) if best else 0.0
    state = "translate" if distance >= 0.35 else ("micro_translate" if distance > 0.0 else "none")
    segments = [
        {
            "event_index": int(evt["event_index"]),
            "span": list(_span(evt)),
            "direction": str(evt.get("direction", "unknown")),
            "speed": _speed_from_event(evt),
            "distance_m": round(_magnitude(evt), 4),
            "duration": _duration(evt),
        }
        for evt in active
    ]
    return {
        "state": state,
        "direction": str(best.get("direction", "none")) if best else "none",
        "speed": _speed_from_event(best),
        "distance_m": round(distance, 4),
        "coverage": round(_coverage(active, total_frames), 4),
        "best_event_index": int(best["event_index"]) if best else None,
        "segments": segments,
        "turn_event_indices": [int(evt["event_index"]) for evt in turn],
    }


def _best_locomotion_overlap(evt: dict[str, Any], locomotion_events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best_evt = None
    best_score = 0.0
    for loco in locomotion_events:
        score = max(_overlap_ratio(evt, loco), 0.35 if _gap(evt, loco) <= 4 else 0.0)
        if score > best_score:
            best_evt = loco
            best_score = score
    return best_evt, best_score


def _vertical_axis(events: list[dict[str, Any]], locomotion_events: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    vertical = [evt for evt in events if evt.get("super_family") == "WHOLE_BODY_VERTICAL"]
    repeated = [
        evt
        for evt in vertical
        if str(evt.get("cluster_id", "")) in {"WB_VERT_REP", "WB_VERT_REP_ALT", "WB_VERT_CYCLE"}
        or evt.get("role") == "repeated_phase"
    ]
    primitives = [evt for evt in vertical if evt not in repeated]
    max_amp = max((_magnitude(evt) for evt in vertical), default=0.0)
    up_count = sum(1 for evt in primitives if str(evt.get("cluster_id", "")) == "WB_VERT_UP")
    down_count = sum(1 for evt in primitives if str(evt.get("cluster_id", "")) == "WB_VERT_DOWN")
    phase_count = max((int(evt.get("count") or 1) for evt in repeated), default=0)
    primitive_up_count = _count_peaks(
        [
            evt
            for evt in primitives
            if str(evt.get("cluster_id", "")) == "WB_VERT_UP" and _magnitude(evt) >= 0.10
        ],
        min_gap=6,
    )
    primitive_cycle_count = max(primitive_up_count, min(up_count, down_count) if down_count else primitive_up_count)
    repeat_count = max(phase_count, primitive_cycle_count)
    best_overlap_evt = None
    best_overlap = 0.0
    for evt in vertical:
        loco, score = _best_locomotion_overlap(evt, locomotion_events)
        if score > best_overlap:
            best_overlap_evt = loco
            best_overlap = score
    has_repetition = repeat_count >= 2 or (up_count + down_count) >= 4
    if not vertical:
        kind = "none"
    elif has_repetition and best_overlap >= 0.35 and max_amp < 0.16:
        kind = "low_gait_bounce"
    elif has_repetition and max_amp < 0.12:
        kind = "low_gait_bounce"
    elif has_repetition and max_amp >= 0.12:
        kind = "repeated_jump"
    elif max_amp >= 0.16:
        kind = "jump"
    elif max_amp >= 0.07 and up_count and down_count:
        kind = "crouch_release_or_hop"
    else:
        kind = "minor_height_change"
    return {
        "kind": kind,
        "max_amplitude_m": round(max_amp, 4),
        "repeat_count": int(repeat_count),
        "phase_repeat_count": int(phase_count),
        "primitive_cycle_count": int(primitive_cycle_count),
        "mean_amplitude_m": round(_mean_magnitude(vertical), 4),
        "up_events": int(up_count),
        "down_events": int(down_count),
        "coverage": round(_coverage(vertical, total_frames), 4),
        "locomotion_overlap": round(best_overlap, 4),
        "locomotion_coupled_event_index": int(best_overlap_evt["event_index"]) if best_overlap_evt else None,
        "locomotion_coupled_direction": str(best_overlap_evt.get("direction", "none")) if best_overlap_evt else "none",
        "event_indices": [int(evt["event_index"]) for evt in vertical],
    }


def _rotation_axis(events: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    rotation = [evt for evt in events if evt.get("super_family") == "WHOLE_BODY_ROTATION"]
    best = max(rotation, key=lambda evt: (_magnitude(evt), _duration(evt)), default=None)
    cluster = str(best.get("cluster_id", "")) if best else ""
    if not best:
        angle_bin = "none"
    elif "MULTI" in cluster:
        angle_bin = "multi"
    elif "FULL" in cluster:
        angle_bin = "full"
    elif "THREE_QTR" in cluster:
        angle_bin = "three_quarter"
    elif "HALF" in cluster:
        angle_bin = "half"
    elif "QTR" in cluster:
        angle_bin = "quarter"
    else:
        angle_bin = "small"
    return {
        "state": "turn" if best else "none",
        "direction": str(best.get("direction", "none")) if best else "none",
        "angle_deg": round(_magnitude(best), 2) if best else 0.0,
        "angle_bin": angle_bin,
        "coverage": round(_coverage(rotation, total_frames), 4),
        "best_event_index": int(best["event_index"]) if best else None,
        "event_indices": [int(evt["event_index"]) for evt in rotation],
    }


def _limb_axis(events: list[dict[str, Any]]) -> dict[str, Any]:
    left = [evt for evt in events if evt.get("super_family") == "LEFT_ARM_PERIODIC"]
    right = [evt for evt in events if evt.get("super_family") == "RIGHT_ARM_PERIODIC"]
    bimanual = [evt for evt in events if evt.get("super_family") == "BIMANUAL_PERIODIC"]
    left_loco = [evt for evt in left if "LOCO" in str(evt.get("cluster_id", ""))]
    right_loco = [evt for evt in right if "LOCO" in str(evt.get("cluster_id", ""))]
    bi_clusters = [str(evt.get("cluster_id", "")) for evt in bimanual]
    raise_spread_count = sum(1 for c in bi_clusters if c == "BI_RAISE_SPREAD")
    spread_count = sum(1 for c in bi_clusters if c in {"BI_SPREAD", "BI_OUT"})
    close_count = sum(1 for c in bi_clusters if "HANDS_CLOSE" in c)
    raise_count = sum(1 for c in bi_clusters if c in {"BI_UP", "BI_RAISE"})
    if raise_spread_count >= 2:
        bimanual_pattern = "raise_spread_repeated"
    elif raise_spread_count:
        bimanual_pattern = "raise_spread"
    elif close_count:
        bimanual_pattern = "hands_close"
    elif raise_count:
        bimanual_pattern = "raise"
    elif spread_count:
        bimanual_pattern = "spread"
    else:
        bimanual_pattern = "none"
    if bimanual:
        dominance = "both_arms"
    elif len(left) > len(right):
        dominance = "left_arm"
    elif len(right) > len(left):
        dominance = "right_arm"
    elif left or right:
        dominance = "both_single_arms"
    else:
        dominance = "none"
    return {
        "dominance": dominance,
        "bimanual_pattern": bimanual_pattern,
        "bimanual_count": len(bimanual),
        "raise_spread_count": int(raise_spread_count),
        "spread_count": int(spread_count),
        "hands_close_count": int(close_count),
        "raise_count": int(raise_count),
        "left_arm_count": len(left),
        "right_arm_count": len(right),
        "left_locomotion_coupled_count": len(left_loco),
        "right_locomotion_coupled_count": len(right_loco),
        "bimanual_event_indices": [int(evt["event_index"]) for evt in bimanual],
        "arm_locomotion_event_indices": [int(evt["event_index"]) for evt in left_loco + right_loco],
    }


def _support_gait_axis(locomotion: dict[str, Any], vertical: dict[str, Any], limbs: dict[str, Any]) -> dict[str, Any]:
    arm_loco = int(limbs["left_locomotion_coupled_count"]) + int(limbs["right_locomotion_coupled_count"])
    if locomotion["state"] == "translate" and arm_loco:
        state = "alternating_proxy"
        confidence = 0.62
    elif locomotion["state"] == "translate" and vertical["kind"] in {"low_gait_bounce", "minor_height_change"}:
        state = "root_stride_proxy"
        confidence = 0.5
    elif arm_loco and vertical["kind"] in {"low_gait_bounce", "crouch_release_or_hop"}:
        state = "in_place_gait_proxy"
        confidence = 0.52
    else:
        state = "unknown"
        confidence = 0.0
    return {
        "state": state,
        "source": "event_proxy",
        "confidence": confidence,
        "evidence_event_indices": list(limbs["arm_locomotion_event_indices"]),
    }


def _in_place_gait_name(vertical: dict[str, Any], limbs: dict[str, Any]) -> tuple[str, float, str]:
    """Separate walk/jog/run in place using event-derived intensity only."""
    repeat_count = int(vertical.get("repeat_count") or 0)
    phase_count = int(vertical.get("phase_repeat_count") or 0)
    arm_loco = int(limbs.get("left_locomotion_coupled_count") or 0) + int(limbs.get("right_locomotion_coupled_count") or 0)
    vertical_amp = float(vertical.get("max_amplitude_m") or 0.0)
    mean_amp = float(vertical.get("mean_amplitude_m") or 0.0)

    if vertical_amp >= 0.14 or (vertical_amp >= 0.11 and phase_count >= 5 and arm_loco >= 3):
        return "run_in_place", 0.56, "high in-place gait intensity"
    if phase_count >= 4 or arm_loco >= 3 or (repeat_count >= 6 and mean_amp >= 0.075):
        return "jog_in_place", 0.53, "moderate in-place gait intensity"
    return "walk_in_place", 0.50, "low in-place gait intensity"


def _state_axis(events: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_STATE"
        and evt.get("cluster_id") == "WB_TERMINAL_STILL"
    ]
    best = max(terminal, key=lambda evt: (_duration(evt), float(evt.get("confidence", 0.0))), default=None)
    return {
        "terminal": "still" if best else "unknown",
        "best_event_index": int(best["event_index"]) if best else None,
        "span": list(_span(best)) if best else None,
        "event_indices": [int(evt["event_index"]) for evt in terminal],
    }


def _temporal_axis(events: list[dict[str, Any]]) -> dict[str, Any]:
    group_map = {
        "WHOLE_BODY_LOCOMOTION": "locomotion",
        "WHOLE_BODY_VERTICAL": "vertical",
        "WHOLE_BODY_ROTATION": "rotation",
        "WHOLE_BODY_PATH": "path",
        "WHOLE_BODY_CLIMB": "climb",
        "WHOLE_BODY_ACROBATICS": "acrobatics",
        "BIMANUAL_PERIODIC": "bimanual",
        "LEFT_ARM_PERIODIC": "left_arm",
        "RIGHT_ARM_PERIODIC": "right_arm",
        "LEFT_ARM_POSTURE": "left_arm",
        "RIGHT_ARM_POSTURE": "right_arm",
        "LEFT_LEG_ACTION": "left_leg",
        "RIGHT_LEG_ACTION": "right_leg",
        "TORSO_PERIODIC": "torso",
        "TORSO_POSTURE": "torso",
        "WHOLE_BODY_STATE": "state",
    }
    sketch: list[str] = []
    last = None
    for evt in sorted(events, key=_event_sort_key):
        group = group_map.get(str(evt.get("super_family", "")), "other")
        if group == last:
            continue
        sketch.append(group)
        last = group
        if len(sketch) >= 20:
            break
    return {
        "ordered_groups": sketch,
        "event_count": len(events),
    }


def _motion_patterns_axis(events: list[dict[str, Any]], axes: dict[str, Any]) -> dict[str, Any]:
    """Compose reusable motion-pattern evidence from Layer-3 axes.

    These are not final semantic action names. They are intermediate evidence
    summaries consumed by prototype assignment and later candidate expansion.
    """
    locomotion = axes.get("locomotion") or {}
    vertical = axes.get("vertical") or {}
    limbs = axes.get("limb_coordination") or {}
    total_frames = int(axes.get("total_frames") or 0)

    loco_direction = str(locomotion.get("direction", "none"))
    loco_speed = str(locomotion.get("speed", "unknown"))
    vertical_amp = float(vertical.get("max_amplitude_m") or 0.0)
    repeat_count = int(vertical.get("repeat_count") or 0)
    raise_spread_count = int(limbs.get("raise_spread_count") or 0)
    rotation_count = len((axes.get("rotation") or {}).get("event_indices") or [])
    locomotion_segment_count = len(locomotion.get("segments") or [])

    coupled_loco = _event_by_index(events, vertical.get("locomotion_coupled_event_index"))
    coupled_distance = _magnitude(coupled_loco) if coupled_loco else 0.0
    coupled_direction = str(coupled_loco.get("direction", loco_direction)) if coupled_loco else loco_direction
    coupled_speed = _speed_from_event(coupled_loco) if coupled_loco else loco_speed
    vertical_up_events = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
    ]
    strongest_vertical = max(vertical_up_events, key=_magnitude, default=None)
    recovery_vertical_overlap = _overlap_ratio(coupled_loco, strongest_vertical) if coupled_loco and strongest_vertical else 0.0
    recovery_after_vertical = bool(coupled_loco and strongest_vertical and _is_after(coupled_loco, strongest_vertical))

    return {
        "coupled_locomotion": {
            "event_index": int(coupled_loco["event_index"]) if coupled_loco else None,
            "direction": coupled_direction,
            "speed": coupled_speed,
            "distance_m": round(coupled_distance, 4),
        },
        "post_vertical_recovery_step": {
            "overlap_with_vertical": round(recovery_vertical_overlap, 4),
            "is_after_vertical": recovery_after_vertical,
            "speed": coupled_speed,
            "distance_m": round(coupled_distance, 4),
            "vertical_amplitude_m": round(vertical_amp, 4),
            "source_event_indices": [
                int(evt["event_index"]) for evt in (coupled_loco, strongest_vertical) if evt is not None
            ],
        },
        "low_body_repetition": low_body_repetition_evidence(events, total_frames),
        "bilateral_rhythmic_coordination": bilateral_rhythmic_evidence(
            vertical,
            limbs,
            vertical_amplitude_m=vertical_amp,
        ),
        "bilateral_rhythmic_gesture": bilateral_gesture_evidence(
            limbs,
            rotation_count=rotation_count,
            locomotion_segment_count=locomotion_segment_count,
        ),
    }


def build_event_coarse_signature(
    program_or_events: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    total_frames: int | None = None,
) -> dict[str, Any]:
    """Build a motion identity summary using only Layer-3 AML events."""
    events = _indexed_events(program_or_events)
    n_frames = _total_frames(events, total_frames)
    locomotion_events = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
    ]
    locomotion = _locomotion_axis(events, n_frames)
    vertical = _vertical_axis(events, locomotion_events, n_frames)
    rotation = _rotation_axis(events, n_frames)
    limbs = _limb_axis(events)
    support_gait = _support_gait_axis(locomotion, vertical, limbs)
    state = _state_axis(events)
    temporal = _temporal_axis(events)
    motion_patterns = _motion_patterns_axis(
        events,
        {
            "total_frames": n_frames,
            "locomotion": locomotion,
            "vertical": vertical,
            "rotation": rotation,
            "limb_coordination": limbs,
        },
    )
    return {
        "version": "event_coarse_signature_v1",
        "source": "layer3_events_only",
        "total_frames": n_frames,
        "locomotion": locomotion,
        "vertical": vertical,
        "rotation": rotation,
        "support_gait": support_gait,
        "limb_coordination": limbs,
        "state": state,
        "temporal": temporal,
        "motion_patterns": motion_patterns,
    }
