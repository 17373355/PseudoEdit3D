from __future__ import annotations

from typing import Any


def _event_sort_key(evt: dict[str, Any]) -> tuple[int, int, str, str, int]:
    return (
        int(evt.get("start_frame", -1)),
        int(evt.get("end_frame", -1)),
        str(evt.get("super_family", "")),
        str(evt.get("cluster_id", "")),
        int(evt.get("event_index", -1)),
    )


def _span(evt: dict[str, Any]) -> tuple[int, int]:
    return int(evt.get("start_frame", -1)), int(evt.get("end_frame", -1))


def _duration(evt: dict[str, Any]) -> int:
    s, e = _span(evt)
    return max(0, e - s + 1)


def _magnitude(evt: dict[str, Any]) -> float:
    value = evt.get("magnitude")
    if value is None:
        value = evt.get("signed_delta")
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _mean_magnitude(events: list[dict[str, Any]]) -> float:
    values = [_magnitude(evt) for evt in events if _magnitude(evt) > 0.0]
    return sum(values) / max(1, len(values))


def _overlap_frames(a: dict[str, Any], b: dict[str, Any]) -> int:
    s1, e1 = _span(a)
    s2, e2 = _span(b)
    return max(0, min(e1, e2) - max(s1, s2) + 1)


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = _overlap_frames(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1, min(_duration(a), _duration(b)))


def _gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    s1, e1 = _span(a)
    s2, e2 = _span(b)
    if e1 < s2:
        return s2 - e1
    if e2 < s1:
        return s1 - e2
    return 0


def _is_after(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a.get("start_frame", -1)) > int(b.get("end_frame", -1))


def _indexed_events(program_or_events: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if isinstance(program_or_events, dict):
        raw_events = list(program_or_events.get("events") or [])
    else:
        raw_events = list(program_or_events or [])
    out: list[dict[str, Any]] = []
    for idx, evt in enumerate(raw_events):
        copied = dict(evt)
        copied["event_index"] = int(copied.get("event_index", idx))
        out.append(copied)
    return sorted(out, key=_event_sort_key)


def _total_frames(events: list[dict[str, Any]], total_frames: int | None) -> int:
    if total_frames:
        return int(total_frames)
    return max((int(evt.get("end_frame", 0)) for evt in events), default=0)


def _coverage(events: list[dict[str, Any]], total_frames: int) -> float:
    if not events or total_frames <= 0:
        return 0.0
    spans = sorted((_span(evt) for evt in events if _duration(evt) > 0))
    if not spans:
        return 0.0
    merged: list[list[int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    frames = sum(e - s + 1 for s, e in merged)
    return frames / max(1, total_frames)


def _count_peaks(events: list[dict[str, Any]], *, min_gap: int = 6) -> int:
    starts = sorted(int(evt.get("start_frame", -1)) for evt in events if int(evt.get("start_frame", -1)) >= 0)
    if not starts:
        return 0
    picked = [starts[0]]
    for start in starts[1:]:
        if start - picked[-1] >= min_gap:
            picked.append(start)
    return len(picked)


def _speed_from_event(evt: dict[str, Any] | None) -> str:
    if not evt:
        return "unknown"
    cluster = str(evt.get("cluster_id", ""))
    if "FAST" in cluster:
        return "fast"
    if "SLOW" in cluster:
        return "slow"
    if "MEDIUM" in cluster:
        return "medium"
    meta = evt.get("metadata") or {}
    try:
        mean_speed = float(meta.get("mean_speed"))
    except (TypeError, ValueError):
        return "unknown"
    if mean_speed >= 0.04:
        return "fast"
    if mean_speed <= 0.022:
        return "slow"
    return "medium"


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
    }


def assign_seeded_prototype(signature: dict[str, Any], events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    locomotion = signature.get("locomotion") or {}
    vertical = signature.get("vertical") or {}
    rotation = signature.get("rotation") or {}
    support_gait = signature.get("support_gait") or {}
    limbs = signature.get("limb_coordination") or {}

    loco_state = str(locomotion.get("state", "none"))
    loco_direction = str(locomotion.get("direction", "none"))
    loco_speed = str(locomotion.get("speed", "unknown"))
    loco_distance = float(locomotion.get("distance_m") or 0.0)
    vertical_kind = str(vertical.get("kind", "none"))
    vertical_amp = float(vertical.get("max_amplitude_m") or 0.0)
    vertical_loco_overlap = float(vertical.get("locomotion_overlap") or 0.0)
    bimanual_pattern = str(limbs.get("bimanual_pattern", "none"))
    support_state = str(support_gait.get("state", "unknown"))
    repeat_count = int(vertical.get("repeat_count") or 0)
    raise_spread_count = int(limbs.get("raise_spread_count") or 0)
    rotation_count = len(rotation.get("event_indices") or [])
    locomotion_segment_count = len(locomotion.get("segments") or [])
    coupled_loco = _event_by_index(events or [], vertical.get("locomotion_coupled_event_index"))
    coupled_distance = _magnitude(coupled_loco) if coupled_loco else 0.0
    coupled_direction = str(coupled_loco.get("direction", loco_direction)) if coupled_loco else loco_direction
    coupled_speed = _speed_from_event(coupled_loco) if coupled_loco else loco_speed
    vertical_events = [
        evt
        for evt in (events or [])
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
    ]
    strongest_vertical = max(vertical_events, key=_magnitude, default=None)
    post_jump_recovery = False
    if coupled_loco and strongest_vertical:
        post_jump_recovery = (
            _overlap_ratio(coupled_loco, strongest_vertical) < 0.20
            and _is_after(coupled_loco, strongest_vertical)
            and coupled_speed == "slow"
            and coupled_distance < 0.65
            and vertical_amp >= 0.24
        )

    squat_events = [
        evt for evt in (events or [])
        if evt.get("super_family") == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
    ]
    squat_count = len(squat_events)
    squat_coverage = _coverage(squat_events, int(signature.get("total_frames") or 0))
    bimanual_near_squat = [
        evt for evt in (events or [])
        if evt.get("super_family") == "BIMANUAL_PERIODIC"
        and any(_overlap_ratio(evt, squat) >= 0.15 or _gap(evt, squat) <= 8 for squat in squat_events)
    ]
    if (
        squat_count >= 2
        and vertical_kind in {"jump", "repeated_jump", "crouch_release_or_hop"}
        and squat_coverage >= 0.25
        and loco_distance < 0.60
    ):
        return {
            "prototype_id": "SQUAT_ARM_LIFT" if bimanual_near_squat else "SQUAT_REPETITION",
            "name_hint": "squat_arm_lift" if bimanual_near_squat else "squat_repetition",
            "confidence": 0.70 + min(0.10, 0.02 * squat_count),
            "primary_direction": "low",
            "count": squat_count,
            "rationale": "repeated low-body posture dominates vertical motion, so treat vertical cycles as squats rather than jumps",
        }

    if (
        vertical_kind == "repeated_jump"
        and bimanual_pattern in {"raise_spread", "raise_spread_repeated"}
        and repeat_count >= 4
        and raise_spread_count >= 4
        and vertical_loco_overlap < 0.35
        and loco_distance < 0.45
    ):
        return {
            "prototype_id": "JUMPING_JACK",
            "name_hint": "jumping_jack",
            "confidence": 0.76 if bimanual_pattern == "raise_spread_repeated" else 0.62,
            "primary_direction": "in_place",
            "rationale": "repeated vertical jump events co-occur with bimanual raise-spread events",
        }

    if (
        bimanual_pattern == "raise_spread_repeated"
        and raise_spread_count >= 3
        and vertical_kind in {"low_gait_bounce", "minor_height_change", "none"}
        and vertical_amp < 0.12
        and (rotation_count >= 2 or locomotion_segment_count >= 3)
        and loco_distance < 0.80
    ):
        return {
            "prototype_id": "CELEBRATORY_DANCE_GESTURE",
            "name_hint": "cheer_dance_like",
            "confidence": 0.58 + min(0.12, 0.02 * raise_spread_count),
            "primary_direction": "in_place",
            "turn_count": rotation_count,
            "locomotion_segment_count": locomotion_segment_count,
            "raise_spread_count": raise_spread_count,
            "bimanual_count": int(limbs.get("bimanual_count") or 0),
            "global_alias_evidence": "dance_or_rhythm+overhead_clap_or_cheer",
            "rationale": "repeated bimanual raise-spread with small turns and low gait bounce suggests a celebratory dance-like gesture",
        }

    if (
        vertical_kind in {"jump", "repeated_jump"}
        and loco_state == "translate"
        and coupled_direction in {"forward", "backward", "left", "right", "mixed"}
        and (vertical_loco_overlap >= 0.30 or coupled_distance >= 0.55)
        and vertical_amp >= 0.16
        and not post_jump_recovery
    ):
        return {
            "prototype_id": "BALLISTIC_TRANSLATION",
            "name_hint": f"jump_{coupled_direction}",
            "confidence": 0.68 + min(0.2, vertical_amp),
            "primary_direction": coupled_direction,
            "primary_locomotion_event_index": int(coupled_loco["event_index"]) if coupled_loco else None,
            "primary_locomotion_distance_m": round(coupled_distance, 4),
            "primary_locomotion_speed": coupled_speed,
            "rationale": "salient vertical impulse overlaps or neighbors a translating root event",
        }

    if vertical_kind in {"jump", "repeated_jump"} and loco_distance < 0.45 and vertical_amp >= 0.12:
        return {
            "prototype_id": "VERTICAL_JUMP",
            "name_hint": "jump_in_place" if repeat_count <= 1 else "repeated_jump_in_place",
            "confidence": 0.62 + min(0.2, vertical_amp),
            "primary_direction": "in_place",
            "rationale": "salient vertical impulse without reliable root translation",
        }

    if loco_state == "translate":
        name_hint = "run" if loco_speed == "fast" else "walk"
        return {
            "prototype_id": "TRANSLATING_GAIT",
            "name_hint": name_hint,
            "confidence": 0.62 if support_state != "unknown" else 0.50,
            "primary_direction": loco_direction,
            "rationale": "dominant root translation summarized as a gait-level family",
        }

    if support_state == "in_place_gait_proxy" or (
        vertical_kind == "low_gait_bounce" and int(limbs.get("left_locomotion_coupled_count") or 0) + int(limbs.get("right_locomotion_coupled_count") or 0)
    ):
        del repeat_count
        in_place_name, confidence, intensity_rationale = _in_place_gait_name(vertical, limbs)
        return {
            "prototype_id": "IN_PLACE_GAIT",
            "name_hint": in_place_name,
            "confidence": confidence,
            "primary_direction": "in_place",
            "rationale": f"arm swing and low vertical cycles indicate a gait-like in-place motion; {intensity_rationale}",
        }

    if rotation.get("state") == "turn":
        return {
            "prototype_id": "ROTATION_DOMINANT",
            "name_hint": "turn",
            "confidence": 0.58,
            "primary_direction": str(rotation.get("direction", "unknown")),
            "rationale": "rotation is the strongest whole-body event",
        }

    if bimanual_pattern != "none":
        return {
            "prototype_id": "BIMANUAL_ACTION",
            "name_hint": bimanual_pattern,
            "confidence": 0.45,
            "primary_direction": "none",
            "rationale": "bimanual events are the most interpretable family",
        }

    return {
        "prototype_id": "EVENT_SEQUENCE",
        "name_hint": "generic_motion",
        "confidence": 0.30,
        "primary_direction": "none",
        "rationale": "no stable coarse family matched the event signature",
    }


def _indices_by_family(events: list[dict[str, Any]], family: str) -> set[int]:
    return {int(evt["event_index"]) for evt in events if evt.get("super_family") == family}


def _event_by_index(events: list[dict[str, Any]], idx: int | None) -> dict[str, Any] | None:
    if idx is None:
        return None
    for evt in events:
        if int(evt.get("event_index", -1)) == int(idx):
            return evt
    return None


def _cover_primary_events(events: list[dict[str, Any]], signature: dict[str, Any], prototype: dict[str, Any]) -> set[int]:
    pid = str(prototype.get("prototype_id", ""))
    covered: set[int] = set()
    locomotion = signature.get("locomotion") or {}
    vertical = signature.get("vertical") or {}
    limbs = signature.get("limb_coordination") or {}
    support = signature.get("support_gait") or {}

    if pid == "JUMPING_JACK":
        covered.update(_indices_by_family(events, "WHOLE_BODY_VERTICAL"))
        covered.update(_indices_by_family(events, "BIMANUAL_PERIODIC"))
        covered.update(_indices_by_family(events, "LEFT_ARM_PERIODIC"))
        covered.update(_indices_by_family(events, "RIGHT_ARM_PERIODIC"))
        covered.update(_indices_by_family(events, "TORSO_PERIODIC"))
    elif pid == "BALLISTIC_TRANSLATION":
        primary_loco = _event_by_index(events, prototype.get("primary_locomotion_event_index"))
        if primary_loco is None:
            primary_loco = _event_by_index(events, locomotion.get("best_event_index"))
        if primary_loco:
            covered.add(int(primary_loco["event_index"]))
            for evt in events:
                if evt.get("super_family") in {"WHOLE_BODY_VERTICAL", "BIMANUAL_PERIODIC", "TORSO_PERIODIC"}:
                    if _overlap_ratio(evt, primary_loco) >= 0.25 or _gap(evt, primary_loco) <= 8:
                        covered.add(int(evt["event_index"]))
                if evt.get("super_family") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC"}:
                    if _overlap_ratio(evt, primary_loco) >= 0.20 or _gap(evt, primary_loco) <= 8:
                        covered.add(int(evt["event_index"]))
    elif pid == "VERTICAL_JUMP":
        covered.update(_indices_by_family(events, "WHOLE_BODY_VERTICAL"))
        for evt in events:
            if evt.get("super_family") in {"BIMANUAL_PERIODIC", "TORSO_PERIODIC"}:
                covered.add(int(evt["event_index"]))
            if (
                evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
                and str(evt.get("cluster_id", "")).startswith("LOCO_ACTIVE")
                and _magnitude(evt) < 0.35
            ):
                for vert in events:
                    if vert.get("super_family") != "WHOLE_BODY_VERTICAL":
                        continue
                    if _overlap_ratio(evt, vert) >= 0.20 or _gap(evt, vert) <= 4:
                        covered.add(int(evt["event_index"]))
                        break
    elif pid == "TRANSLATING_GAIT":
        loco_ids = [int(seg["event_index"]) for seg in locomotion.get("segments") or []]
        covered.update(loco_ids)
        covered.update(support.get("evidence_event_indices") or [])
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_VERTICAL" and str(vertical.get("kind")) in {"low_gait_bounce", "minor_height_change"}:
                for idx in loco_ids:
                    loco = _event_by_index(events, idx)
                    if loco and (_overlap_ratio(evt, loco) >= 0.25 or _gap(evt, loco) <= 4):
                        covered.add(int(evt["event_index"]))
            if evt.get("super_family") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC"} and "LOCO" in str(evt.get("cluster_id", "")):
                covered.add(int(evt["event_index"]))
    elif pid == "IN_PLACE_GAIT":
        covered.update(_indices_by_family(events, "WHOLE_BODY_VERTICAL"))
        covered.update(_indices_by_family(events, "LEFT_ARM_PERIODIC"))
        covered.update(_indices_by_family(events, "RIGHT_ARM_PERIODIC"))
        covered.update(_indices_by_family(events, "BIMANUAL_PERIODIC"))
        covered.update(_indices_by_family(events, "TORSO_PERIODIC"))
    elif pid == "CELEBRATORY_DANCE_GESTURE":
        covered.update(_indices_by_family(events, "BIMANUAL_PERIODIC"))
        covered.update(_indices_by_family(events, "WHOLE_BODY_ROTATION"))
        covered.update(_indices_by_family(events, "WHOLE_BODY_VERTICAL"))
        covered.update(_indices_by_family(events, "TORSO_PERIODIC"))
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION" and _magnitude(evt) < 0.45:
                covered.add(int(evt["event_index"]))
    elif pid == "ROTATION_DOMINANT":
        covered.update(_indices_by_family(events, "WHOLE_BODY_ROTATION"))
        covered.update(locomotion.get("turn_event_indices") or [])
    elif pid == "BIMANUAL_ACTION":
        covered.update(limbs.get("bimanual_event_indices") or [])
    elif pid in {"SQUAT_REPETITION", "SQUAT_ARM_LIFT"}:
        covered.update(
            int(evt["event_index"])
            for evt in events
            if evt.get("super_family") == "WHOLE_BODY_POSTURE"
            and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        )
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_VERTICAL":
                covered.add(int(evt["event_index"]))
        if pid == "SQUAT_ARM_LIFT":
            covered.update(
                int(evt["event_index"])
                for evt in events
                if evt.get("super_family") == "BIMANUAL_PERIODIC"
            )
    return {int(x) for x in covered}


def _event_ref(evt: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_index": int(evt.get("event_index", -1)),
        "family": str(evt.get("super_family", "")),
        "cluster": str(evt.get("cluster_id", "")),
        "direction": str(evt.get("direction", "")),
        "span": list(_span(evt)),
        "magnitude": evt.get("magnitude"),
        "count": evt.get("count"),
    }


def _probe_alias(action: dict[str, Any]) -> str:
    pid = str(action.get("prototype_id", ""))
    name = str(action.get("name_hint", ""))
    if pid in {"TRANSLATING_GAIT", "TRANSLATING_GAIT_SEGMENT"}:
        if name == "run":
            return "run"
        return "walk"
    if pid == "IN_PLACE_GAIT":
        if name == "run_in_place":
            return "run in place"
        if name == "jog_in_place":
            return "jog in place"
        return "walk in place"
    if pid == "CELEBRATORY_DANCE_GESTURE":
        return "cheer-like dance gesture"
    if pid == "JUMPING_JACK":
        return "jumping jacks"
    if pid in {"BALLISTIC_TRANSLATION", "BALLISTIC_TRANSLATION_SEGMENT"}:
        direction = str(action.get("primary_direction", "in_place"))
        return f"jump {direction}" if direction != "in_place" else "jump"
    if pid in {"VERTICAL_JUMP", "VERTICAL_JUMP_SEGMENT"}:
        return "jump upward"
    if pid in {"ROTATION_DOMINANT", "TURN_SEGMENT"}:
        return "turn"
    if pid == "TERMINAL_STILL":
        return "stand still"
    if pid == "BIMANUAL_HANDS_CLOSE":
        return "bring hands together"
    if pid == "BIMANUAL_ACTION":
        return str(action.get("name_hint", "bimanual action")).replace("_", " ")
    if pid == "SQUAT_REPETITION":
        return "repeated squat"
    if pid == "SQUAT_ARM_LIFT":
        return "squat with arm lift"
    if pid == "ACROBATIC_SEQUENCE_CANDIDATE":
        return "acrobatic inverted sequence"
    if pid == "LEG_FORWARD_POSE_CANDIDATE":
        return "leg forward pose"
    if pid == "DANCE_LEG_POSE_CANDIDATE":
        return "dance-like raised leg pose"
    return str(action.get("name_hint", "motion")).replace("_", " ")


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    slots: dict[str, Any] = {
        "span": action.get("span"),
        "direction": action.get("primary_direction"),
        "confidence": action.get("confidence"),
        "covered_event_indices": list(action.get("covered_event_indices") or []),
    }
    for key in (
        "count",
        "distance_m",
        "speed",
        "angle_deg",
        "angle_bin",
        "vertical_amplitude_m",
        "mean_vertical_amplitude_m",
        "global_alias_evidence",
        "source_event_clusters",
        "turn_count",
        "locomotion_segment_count",
        "raise_spread_count",
        "bimanual_count",
        "source_event_cluster",
        "source_event_family",
        "semantic_proxy",
        "path_length_m",
        "circle_score",
        "curvature_rad",
        "root_height_gain_m",
        "segment_count",
        "source_event_count",
        "source_event_spans",
        "dominant_side",
        "hidden_by_semantic_family",
    ):
        if action.get(key) is not None:
            slots[key] = action.get(key)
    return {
        "canonical_id": str(action.get("prototype_id", "EVENT_SEQUENCE")),
        "family": str(action.get("prototype_id", "EVENT_SEQUENCE")),
        "probe_alias": _probe_alias(action),
        "surface_name_hint": action.get("name_hint"),
        "slots": slots,
    }


def _attach_action_metadata(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in actions:
        item = dict(action)
        item["probe_alias"] = _probe_alias(item)
        item["canonical"] = _canonical_action(item)
        out.append(item)
    return out


def _vertical_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evt for evt in events if evt.get("super_family") == "WHOLE_BODY_VERTICAL"]


def _bimanual_raise_spread_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evt
        for evt in events
        if evt.get("super_family") == "BIMANUAL_PERIODIC"
        and str(evt.get("cluster_id", "")) == "BI_RAISE_SPREAD"
    ]


def _jumping_jack_repeat_count(events: list[dict[str, Any]], vertical_axis: dict[str, Any]) -> int:
    vertical_ups = [
        evt
        for evt in _vertical_events(events)
        if str(evt.get("cluster_id", "")) == "WB_VERT_UP" and _magnitude(evt) >= 0.10
    ]
    arm_raise_spread = _bimanual_raise_spread_events(events)
    vertical_count = _count_peaks(vertical_ups, min_gap=6)
    arm_count = _count_peaks(arm_raise_spread, min_gap=6)
    phase_count = int(vertical_axis.get("phase_repeat_count") or 0)
    down_count = int(vertical_axis.get("down_events") or 0)
    if down_count >= 4:
        return down_count + 1
    paired_count = min(vertical_count, arm_count) if vertical_count and arm_count else max(vertical_count, arm_count)
    if paired_count >= 4:
        return paired_count
    return max(paired_count, phase_count, int(vertical_axis.get("repeat_count") or 0))


def _hands_close_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close_clusters = {"BI_HANDS_CLOSE", "BI_HANDS_CLOSE_RAISE"}
    close_events = sorted(
        [
            evt
            for evt in events
            if int(evt["event_index"]) not in covered
            and evt.get("super_family") == "BIMANUAL_PERIODIC"
            and str(evt.get("cluster_id", "")) in close_clusters
        ],
        key=_event_sort_key,
    )
    groups: list[list[dict[str, Any]]] = []
    for evt in close_events:
        if not groups or _gap(groups[-1][-1], evt) > 20:
            groups.append([evt])
        else:
            groups[-1].append(evt)
    for group in groups:
        clusters = sorted({str(evt.get("cluster_id", "")) for evt in group})
        indices = [int(evt["event_index"]) for evt in group]
        spans = [_span(evt) for evt in group]
        has_raise = any(cluster == "BI_HANDS_CLOSE_RAISE" for cluster in clusters)
        confidence = min(0.68, max(float(evt.get("confidence", 0.0)) for evt in group) + 0.04)
        out.append(
            {
                "prototype_id": "BIMANUAL_HANDS_CLOSE",
                "name_hint": "hands_close_raise" if has_raise else "hands_close",
                "primary_direction": "inward",
                "confidence": confidence,
                "span": [min(s for s, _ in spans), max(e for _, e in spans)],
                "count": len(group),
                "source_event_clusters": clusters,
                "global_alias_evidence": "clap_or_hands_together",
                "covered_event_indices": indices,
            }
        )
        covered.update(indices)
    return out


def _candidate_jump_segments(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    locos = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
        and _magnitude(evt) >= 0.45
    ]
    verticals = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
        and _magnitude(evt) >= 0.09
    ]
    out: list[dict[str, Any]] = []
    for vert in verticals:
        best_loco = None
        best_score = 0.0
        for loco in locos:
            score = max(_overlap_ratio(vert, loco), 0.35 if _gap(vert, loco) <= 5 else 0.0)
            if score > best_score:
                best_loco = loco
                best_score = score
        if best_loco is None or best_score < 0.30:
            continue
        lidx = int(best_loco["event_index"])
        vidx = int(vert["event_index"])
        if lidx in covered and vidx in covered:
            continue
        start = min(int(best_loco["start_frame"]), int(vert["start_frame"]))
        end = max(int(best_loco["end_frame"]), int(vert["end_frame"]))
        local_cover = {lidx, vidx}
        for evt in events:
            if evt.get("super_family") in {"BIMANUAL_PERIODIC", "TORSO_PERIODIC"}:
                if _overlap_ratio(evt, vert) >= 0.20 or _overlap_ratio(evt, best_loco) >= 0.20 or _gap(evt, vert) <= 4:
                    local_cover.add(int(evt["event_index"]))
        strong = _magnitude(vert) >= 0.14
        out.append(
            {
                "prototype_id": "BALLISTIC_TRANSLATION_SEGMENT" if strong else "WEAK_BALLISTIC_CANDIDATE",
                "name_hint": f"jump_{best_loco.get('direction', 'unknown')}",
                "primary_direction": str(best_loco.get("direction", "unknown")),
                "confidence": (0.62 if strong else 0.42) + min(0.18, _magnitude(vert)),
                "span": [start, end],
                "speed": _speed_from_event(best_loco),
                "distance_m": round(_magnitude(best_loco), 4),
                "vertical_amplitude_m": round(_magnitude(vert), 4),
                "probe_visible": bool(strong),
                "covered_event_indices": sorted(local_cover),
            }
        )
    return out


def _recovery_step_segments(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    verticals = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
        and _magnitude(evt) >= 0.20
    ]
    locos = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
        and _magnitude(evt) < 0.70
        and _speed_from_event(evt) == "slow"
    ]
    out: list[dict[str, Any]] = []
    for loco in locos:
        if int(loco["event_index"]) in covered:
            continue
        prior_vertical = [
            vert
            for vert in verticals
            if int(loco.get("start_frame", -1)) > int(vert.get("end_frame", -1))
            and int(loco.get("start_frame", -1)) - int(vert.get("end_frame", -1)) <= 14
        ]
        if not prior_vertical:
            continue
        best_vert = max(prior_vertical, key=_magnitude)
        direction = str(loco.get("direction", "unknown"))
        out.append(
            {
                "prototype_id": "RECOVERY_STEP_SEGMENT",
                "name_hint": f"recovery_step_{direction}",
                "primary_direction": direction,
                "confidence": 0.58,
                "span": list(_span(loco)),
                "distance_m": round(_magnitude(loco), 4),
                "speed": _speed_from_event(loco),
                "source_jump_event_index": int(best_vert["event_index"]),
                "covered_event_indices": [
                    int(evt["event_index"])
                    for evt in events
                    if int(evt["event_index"]) == int(loco["event_index"])
                    or (
                        evt.get("super_family") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC", "BIMANUAL_PERIODIC"}
                        and (_overlap_ratio(evt, loco) >= 0.20 or _gap(evt, loco) <= 5)
                    )
                ],
            }
        )
    return out


def _semantic_candidate_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    semantic_map = {
        ("TORSO_POSTURE", "TORSO_HUNCHED_FORWARD"): ("TORSO_HUNCHED_FORWARD", "hunched_forward", "forward", 0.61),
        ("LEFT_ARM_POSTURE", "LA_HAND_HIGH"): ("LEFT_HAND_RAISED_HIGH", "left_hand_high", "up", 0.60),
        ("RIGHT_ARM_POSTURE", "RA_HAND_HIGH"): ("RIGHT_HAND_RAISED_HIGH", "right_hand_high", "up", 0.60),
        ("WHOLE_BODY_POSTURE", "WB_SQUAT_HOLD"): ("SQUAT_HOLD", "squat_hold", "low", 0.68),
        ("LEFT_LEG_ACTION", "LL_KICK_FORWARD"): ("LEFT_LEG_KICK_FORWARD", "left_kick_forward", "forward_up", 0.66),
        ("RIGHT_LEG_ACTION", "RL_KICK_FORWARD"): ("RIGHT_LEG_KICK_FORWARD", "right_kick_forward", "forward_up", 0.66),
        ("LEFT_LEG_ACTION", "LL_LEG_FORWARD_POSE"): ("LEG_FORWARD_POSE_CANDIDATE", "left_leg_forward_pose", "forward", 0.64),
        ("RIGHT_LEG_ACTION", "RL_LEG_FORWARD_POSE"): ("LEG_FORWARD_POSE_CANDIDATE", "right_leg_forward_pose", "forward", 0.64),
        ("WHOLE_BODY_PATH", "ROOT_CIRCULAR_PATH"): ("CIRCULAR_WALK_PATH", "walk_circle_path", "circular", 0.66),
        ("WHOLE_BODY_CLIMB", "CLIMB_UP_OVER_PROXY"): ("CLIMB_UP_OVER_PROXY", "climb_up_over_proxy", "up_over", 0.64),
        ("WHOLE_BODY_ACROBATICS", "WB_CARTWHEEL_CANDIDATE"): ("CARTWHEEL_CANDIDATE", "cartwheel_candidate", "inverted", 0.63),
        ("WHOLE_BODY_ACROBATICS", "WB_INVERTED_ROTATION_CANDIDATE"): ("INVERTED_ACROBATICS_CANDIDATE", "inverted_acrobatics_candidate", "inverted", 0.59),
    }
    semantic_events = [
        evt
        for evt in events
        if (str(evt.get("super_family", "")), str(evt.get("cluster_id", ""))) in semantic_map
    ]
    locomotion_events = [
        evt for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
    ]

    def make_action(
        proto_id: str,
        name_hint: str,
        direction: str,
        group: list[dict[str, Any]],
        confidence: float,
    ) -> dict[str, Any]:
        spans = [_span(evt) for evt in group]
        indices = [int(evt["event_index"]) for evt in group]
        metadata = group[0].get("metadata") or {}
        action = {
            "prototype_id": proto_id,
            "name_hint": name_hint,
            "primary_direction": direction,
            "confidence": min(0.82, max([confidence] + [float(evt.get("confidence", 0.0)) for evt in group])),
            "span": [min(s for s, _ in spans), max(e for _, e in spans)],
            "semantic_proxy": True,
            "source_event_family": str(group[0].get("super_family", "")),
            "source_event_cluster": str(group[0].get("cluster_id", "")),
            "source_event_count": len(group),
            "source_event_spans": [list(span) for span in spans],
            "covered_event_indices": indices,
        }
        magnitude = max((_magnitude(evt) for evt in group), default=0.0)
        if magnitude > 0.0:
            action["magnitude"] = round(magnitude, 4)
        if str(group[0].get("cluster_id", "")) == "ROOT_CIRCULAR_PATH":
            action["path_length_m"] = round(float(metadata.get("path_length", magnitude) or 0.0), 4)
            action["circle_score"] = round(float(metadata.get("circle_score", 0.0) or 0.0), 4)
            action["curvature_rad"] = round(float(metadata.get("curvature", group[0].get("signed_delta") or 0.0) or 0.0), 4)
        if str(group[0].get("cluster_id", "")) == "CLIMB_UP_OVER_PROXY":
            action["root_height_gain_m"] = round(float(metadata.get("root_height_gain", magnitude) or 0.0), 4)
        return action

    acrobatics = [
        evt for evt in semantic_events
        if str(evt.get("super_family", "")) == "WHOLE_BODY_ACROBATICS"
    ]
    acrobatic_cover: set[int] = set()
    if acrobatics:
        spans = [_span(evt) for evt in acrobatics]
        proto_id = "ACROBATIC_SEQUENCE_CANDIDATE" if len(acrobatics) >= 2 else semantic_map[
            (str(acrobatics[0].get("super_family", "")), str(acrobatics[0].get("cluster_id", "")))
        ][0]
        action = make_action(
            proto_id,
            "acrobatic_inverted_sequence" if proto_id == "ACROBATIC_SEQUENCE_CANDIDATE" else "cartwheel_candidate",
            "inverted",
            acrobatics,
            0.70 if proto_id == "ACROBATIC_SEQUENCE_CANDIDATE" else 0.65,
        )
        action["segment_count"] = len(acrobatics)
        out.append(action)
        acrobatic_cover.update(int(evt["event_index"]) for evt in acrobatics)
        acro_span_evt = {
            "start_frame": min(s for s, _ in spans),
            "end_frame": max(e for _, e in spans),
        }
        for evt in semantic_events:
            if evt.get("super_family") in {"LEFT_LEG_ACTION", "RIGHT_LEG_ACTION", "WHOLE_BODY_POSTURE", "TORSO_POSTURE"}:
                if _overlap_ratio(evt, acro_span_evt) >= 0.15 or _gap(evt, acro_span_evt) <= 6:
                    acrobatic_cover.add(int(evt["event_index"]))
        for evt in events:
            if evt.get("super_family") in {"WHOLE_BODY_VERTICAL", "WHOLE_BODY_LOCOMOTION"}:
                if _overlap_ratio(evt, acro_span_evt) >= 0.15:
                    acrobatic_cover.add(int(evt["event_index"]))
        covered.update(acrobatic_cover)

    climb_events = [
        evt for evt in semantic_events
        if str(evt.get("super_family", "")) == "WHOLE_BODY_CLIMB"
        and str(evt.get("cluster_id", "")) == "CLIMB_UP_OVER_PROXY"
    ]
    climb_cover: set[int] = set()
    if climb_events:
        climb_span_evt = {
            "start_frame": min(int(evt["start_frame"]) for evt in climb_events),
            "end_frame": max(int(evt["end_frame"]) for evt in climb_events),
        }
        for evt in semantic_events:
            if evt.get("super_family") in {"LEFT_LEG_ACTION", "RIGHT_LEG_ACTION", "WHOLE_BODY_POSTURE", "TORSO_POSTURE"}:
                if _overlap_ratio(evt, climb_span_evt) >= 0.15:
                    climb_cover.add(int(evt["event_index"]))
        covered.update(climb_cover)

    squats = [
        evt for evt in semantic_events
        if str(evt.get("super_family", "")) == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        and int(evt["event_index"]) not in acrobatic_cover
    ]
    primary_squat_ids = covered.intersection({int(evt["event_index"]) for evt in squats})
    if len(squats) >= 2 and not primary_squat_ids:
        bimanual_nearby = [
            evt for evt in events
            if evt.get("super_family") == "BIMANUAL_PERIODIC"
            and any(_overlap_ratio(evt, squat) >= 0.15 or _gap(evt, squat) <= 8 for squat in squats)
        ]
        proto_id = "SQUAT_ARM_LIFT" if bimanual_nearby else "SQUAT_REPETITION"
        action = make_action(
            proto_id,
            "squat_arm_lift" if bimanual_nearby else "squat_repetition",
            "low",
            squats,
            0.72,
        )
        action["count"] = len(squats)
        if bimanual_nearby:
            action["covered_event_indices"] = sorted(set(action["covered_event_indices"]) | {int(evt["event_index"]) for evt in bimanual_nearby})
            action["source_event_clusters"] = sorted({str(evt.get("cluster_id", "")) for evt in bimanual_nearby})
        out.append(action)
        covered.update(int(x) for x in action["covered_event_indices"])
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_VERTICAL":
                if any(_overlap_ratio(evt, squat) >= 0.15 or _gap(evt, squat) <= 6 for squat in squats):
                    covered.add(int(evt["event_index"]))

    hand_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    leg_pose_groups: dict[str, list[dict[str, Any]]] = {}
    torso_events: list[dict[str, Any]] = []
    for evt in semantic_events:
        idx = int(evt["event_index"])
        if idx in acrobatic_cover:
            continue
        key = (str(evt.get("super_family", "")), str(evt.get("cluster_id", "")))
        if key not in semantic_map:
            continue
        proto_id, name_hint, direction, confidence = semantic_map[key]
        if proto_id in {"LEFT_LEG_KICK_FORWARD", "RIGHT_LEG_KICK_FORWARD", "LEG_FORWARD_POSE_CANDIDATE"} and idx in climb_cover:
            continue
        if proto_id == "TORSO_HUNCHED_FORWARD":
            torso_events.append(evt)
            continue
        if proto_id in {"LEFT_HAND_RAISED_HIGH", "RIGHT_HAND_RAISED_HIGH"}:
            hand_groups.setdefault((proto_id, name_hint), []).append(evt)
            continue
        if proto_id == "LEG_FORWARD_POSE_CANDIDATE":
            leg_pose_groups.setdefault(name_hint, []).append(evt)
            continue
        if proto_id in {"LEFT_LEG_KICK_FORWARD", "RIGHT_LEG_KICK_FORWARD"}:
            gait_like = any(
                _overlap_ratio(evt, loco) >= 0.40
                and _magnitude(loco) >= 1.0
                and _speed_from_event(loco) != "slow"
                for loco in locomotion_events
            )
            if gait_like:
                covered.add(idx)
                continue
        if proto_id == "SQUAT_HOLD" and len(squats) >= 2:
            continue
        action = make_action(proto_id, name_hint, direction, [evt], confidence)
        out.append(action)
        if proto_id in {
            "CIRCULAR_WALK_PATH",
            "CLIMB_UP_OVER_PROXY",
            "CARTWHEEL_CANDIDATE",
            "INVERTED_ACROBATICS_CANDIDATE",
            "SQUAT_HOLD",
        }:
            covered.add(idx)
    for (proto_id, name_hint), group in hand_groups.items():
        action = make_action(proto_id, name_hint, "up", group, 0.60)
        action["count"] = len(group)
        out.append(action)
    if torso_events:
        action = make_action("TORSO_HUNCHED_FORWARD", "hunched_forward", "forward", torso_events, 0.61)
        action["count"] = len(torso_events)
        out.append(action)
    for name_hint, group in leg_pose_groups.items():
        has_turn = any(evt.get("super_family") == "WHOLE_BODY_ROTATION" and _magnitude(evt) >= 120.0 for evt in events)
        has_high_hand = bool(hand_groups)
        proto_id = "DANCE_LEG_POSE_CANDIDATE" if has_turn and has_high_hand else "LEG_FORWARD_POSE_CANDIDATE"
        action = make_action(proto_id, name_hint, "forward", group, 0.68 if proto_id == "DANCE_LEG_POSE_CANDIDATE" else 0.64)
        action["count"] = len(group)
        action["dominant_side"] = "left" if name_hint.startswith("left") else "right"
        out.append(action)
    return out


def _secondary_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for evt in events:
        idx = int(evt["event_index"])
        if idx in covered:
            continue
        family = evt.get("super_family")
        cluster = str(evt.get("cluster_id", ""))
        if family == "WHOLE_BODY_LOCOMOTION" and cluster.startswith("LOCO_") and not cluster.startswith("LOCO_TURN_") and _magnitude(evt) >= 0.35:
            actions.append(
                {
                    "prototype_id": "TRANSLATING_GAIT_SEGMENT",
                    "name_hint": "run" if _speed_from_event(evt) == "fast" else "walk",
                    "primary_direction": str(evt.get("direction", "unknown")),
                    "confidence": 0.46,
                    "span": list(_span(evt)),
                    "speed": _speed_from_event(evt),
                    "distance_m": round(_magnitude(evt), 4),
                    "covered_event_indices": [idx],
                }
            )
            covered.add(idx)
        elif family == "WHOLE_BODY_ROTATION" and _magnitude(evt) >= 45.0:
            local_cover = {idx}
            for other in events:
                if other.get("super_family") == "WHOLE_BODY_LOCOMOTION" and str(other.get("cluster_id", "")).startswith("LOCO_TURN_"):
                    if _overlap_ratio(evt, other) >= 0.25 or _gap(evt, other) <= 4:
                        local_cover.add(int(other["event_index"]))
            actions.append(
                {
                    "prototype_id": "TURN_SEGMENT",
                    "name_hint": "turn",
                    "primary_direction": str(evt.get("direction", "unknown")),
                    "confidence": 0.50,
                    "span": list(_span(evt)),
                    "angle_deg": round(_magnitude(evt), 2),
                    "angle_bin": str(evt.get("cluster_id", "")),
                    "covered_event_indices": sorted(local_cover),
                }
            )
            covered.update(local_cover)
    return actions


def _apply_semantic_dominance(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep low-level trajectory evidence in canonical slots, but hide it from probe prompts
    when a stronger composed semantic family already explains the same span."""
    dominant = [
        action for action in actions
        if action.get("prototype_id") in {
            "CLIMB_UP_OVER_PROXY",
            "ACROBATIC_SEQUENCE_CANDIDATE",
            "CARTWHEEL_CANDIDATE",
            "DANCE_LEG_POSE_CANDIDATE",
        }
    ]
    if not dominant:
        return actions
    out: list[dict[str, Any]] = []
    for action in actions:
        item = dict(action)
        pid = str(item.get("prototype_id", ""))
        if pid in {
            "BALLISTIC_TRANSLATION",
            "BALLISTIC_TRANSLATION_SEGMENT",
            "VERTICAL_JUMP",
            "VERTICAL_JUMP_SEGMENT",
            "CIRCULAR_WALK_PATH",
            "TRANSLATING_GAIT_SEGMENT",
        }:
            for dom in dominant:
                dom_pid = str(dom.get("prototype_id", ""))
                if _overlap_ratio(item, dom) < 0.20:
                    continue
                if dom_pid == "CLIMB_UP_OVER_PROXY" and pid in {"BALLISTIC_TRANSLATION", "BALLISTIC_TRANSLATION_SEGMENT", "CIRCULAR_WALK_PATH", "TRANSLATING_GAIT_SEGMENT"}:
                    item["probe_visible"] = False
                    item["hidden_by_semantic_family"] = dom_pid
                    break
                if dom_pid in {"ACROBATIC_SEQUENCE_CANDIDATE", "CARTWHEEL_CANDIDATE"} and pid in {
                    "BALLISTIC_TRANSLATION",
                    "BALLISTIC_TRANSLATION_SEGMENT",
                    "VERTICAL_JUMP_SEGMENT",
                    "TRANSLATING_GAIT_SEGMENT",
                }:
                    item["probe_visible"] = False
                    item["hidden_by_semantic_family"] = dom_pid
                    break
                if dom_pid == "DANCE_LEG_POSE_CANDIDATE" and pid in {
                    "BALLISTIC_TRANSLATION",
                    "BALLISTIC_TRANSLATION_SEGMENT",
                    "LEFT_LEG_KICK_FORWARD",
                    "RIGHT_LEG_KICK_FORWARD",
                    "TRANSLATING_GAIT_SEGMENT",
                }:
                    item["probe_visible"] = False
                    item["hidden_by_semantic_family"] = dom_pid
                    break
        out.append(item)
    return out


def build_coarse_action_program(
    program_or_events: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    total_frames: int | None = None,
    max_residual_events: int = 4,
) -> dict[str, Any]:
    events = _indexed_events(program_or_events)
    signature = build_event_coarse_signature(events, total_frames=total_frames)
    prototype = assign_seeded_prototype(signature, events)
    if prototype.get("prototype_id") == "JUMPING_JACK":
        repeat_count = _jumping_jack_repeat_count(events, signature.get("vertical", {}))
        prototype = {
            **prototype,
            "count": repeat_count,
            "vertical_amplitude_m": signature.get("vertical", {}).get("max_amplitude_m"),
            "mean_vertical_amplitude_m": signature.get("vertical", {}).get("mean_amplitude_m"),
        }
    covered = _cover_primary_events(events, signature, prototype)
    primary_span = [1, signature.get("total_frames", 0)]
    primary_idx = signature.get("locomotion", {}).get("best_event_index")
    if prototype.get("primary_locomotion_event_index") is not None:
        primary_idx = prototype.get("primary_locomotion_event_index")
    if prototype.get("prototype_id") == "ROTATION_DOMINANT":
        primary_idx = signature.get("rotation", {}).get("best_event_index")
    primary_evt = _event_by_index(events, primary_idx)
    if primary_evt:
        primary_span = list(_span(primary_evt))
    if prototype.get("prototype_id") in {"JUMPING_JACK", "VERTICAL_JUMP", "IN_PLACE_GAIT", "CELEBRATORY_DANCE_GESTURE"} and covered:
        covered_events = [_event_by_index(events, idx) for idx in covered]
        spans = [_span(evt) for evt in covered_events if evt]
        if spans:
            primary_span = [min(s for s, _ in spans), max(e for _, e in spans)]
    primary_action = {
        **prototype,
        "span": primary_span,
        "covered_event_indices": sorted(covered),
    }
    if prototype.get("prototype_id") == "TRANSLATING_GAIT":
        primary_action["speed"] = signature.get("locomotion", {}).get("speed")
        primary_action["distance_m"] = signature.get("locomotion", {}).get("distance_m")
    if prototype.get("prototype_id") == "BALLISTIC_TRANSLATION":
        primary_action["distance_m"] = prototype.get("primary_locomotion_distance_m")
        primary_action["speed"] = prototype.get("primary_locomotion_speed")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if prototype.get("prototype_id") in {"JUMPING_JACK", "VERTICAL_JUMP", "IN_PLACE_GAIT"}:
        primary_action["count"] = prototype.get("count") or signature.get("vertical", {}).get("repeat_count")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if prototype.get("prototype_id") in {"SQUAT_REPETITION", "SQUAT_ARM_LIFT"}:
        primary_action["count"] = prototype.get("count")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if prototype.get("prototype_id") == "CELEBRATORY_DANCE_GESTURE":
        primary_action["turn_count"] = prototype.get("turn_count")
        primary_action["locomotion_segment_count"] = prototype.get("locomotion_segment_count")
        primary_action["raise_spread_count"] = prototype.get("raise_spread_count")
        primary_action["bimanual_count"] = prototype.get("bimanual_count")
        primary_action["global_alias_evidence"] = prototype.get("global_alias_evidence")
    actions = [primary_action]
    actions.extend(_semantic_candidate_actions(events, covered))
    for action in _recovery_step_segments(events, covered):
        covered.update(int(x) for x in action.get("covered_event_indices") or [])
        actions.append(action)
    actions.extend(_hands_close_actions(events, covered))
    actions.extend(_secondary_actions(events, covered))
    for action in _candidate_jump_segments(events, covered):
        if action.get("prototype_id") != "WEAK_BALLISTIC_CANDIDATE":
            covered.update(int(x) for x in action.get("covered_event_indices") or [])
        actions.append(action)
    state = signature.get("state") or {}
    if state.get("terminal") == "still":
        state_idx = state.get("best_event_index")
        state_evt = _event_by_index(events, state_idx)
        if state_evt:
            covered.add(int(state_evt["event_index"]))
            actions.append(
                {
                    "prototype_id": "TERMINAL_STILL",
                    "name_hint": "stand_still",
                    "primary_direction": "still",
                    "confidence": float(state_evt.get("confidence", 0.0)),
                    "span": list(_span(state_evt)),
                    "covered_event_indices": [int(state_evt["event_index"])],
                }
            )
    actions = _apply_semantic_dominance(actions)
    actions = sorted(actions, key=lambda item: (int(item.get("span", [0, 0])[0]), int(item.get("span", [0, 0])[1])))
    actions = _attach_action_metadata(actions)
    residual = [evt for evt in events if int(evt["event_index"]) not in covered]
    return {
        "version": "coarse_action_program_v2",
        "signature": signature,
        "prototype": prototype,
        "coarse_actions": actions,
        "canonical_actions": [dict(action["canonical"]) for action in actions],
        "covered_event_indices": sorted(covered),
        "covered_events": [_event_ref(evt) for evt in events if int(evt["event_index"]) in covered],
        "residual_events": [_event_ref(evt) for evt in residual[:max_residual_events]],
        "residual_event_count": len(residual),
    }
