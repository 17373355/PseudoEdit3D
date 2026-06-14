from __future__ import annotations

from typing import Any

from .coarse_event_utils import (
    _count_peaks,
    _coverage,
    _gap,
    _magnitude,
    _overlap_ratio,
)


def bimanual_periodic_events(events: list[dict[str, Any]], clusters: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        evt
        for evt in events
        if evt.get("super_family") == "BIMANUAL_PERIODIC"
        and (clusters is None or str(evt.get("cluster_id", "")) in clusters)
    ]


def low_body_repetition_evidence(events: list[dict[str, Any]], total_frames: int) -> dict[str, Any]:
    low_body_events = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
    ]
    bimanual_near_low_body = [
        evt
        for evt in events
        if evt.get("super_family") == "BIMANUAL_PERIODIC"
        and any(_overlap_ratio(evt, low) >= 0.15 or _gap(evt, low) <= 8 for low in low_body_events)
    ]
    return {
        "event_count": len(low_body_events),
        "coverage": round(_coverage(low_body_events, total_frames), 4),
        "has_bimanual_arm_lift": bool(bimanual_near_low_body),
        "source_event_indices": [int(evt["event_index"]) for evt in low_body_events],
        "bimanual_event_indices": [int(evt["event_index"]) for evt in bimanual_near_low_body],
    }


def bilateral_rhythmic_cycle_count(events: list[dict[str, Any]], vertical_axis: dict[str, Any]) -> int:
    vertical_ups = [
        evt
        for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
        and _magnitude(evt) >= 0.10
    ]
    bimanual_cycles = bimanual_periodic_events(events, {"BI_RAISE_SPREAD"})
    vertical_count = _count_peaks(vertical_ups, min_gap=6)
    arm_count = _count_peaks(bimanual_cycles, min_gap=6)
    phase_count = int(vertical_axis.get("phase_repeat_count") or 0)
    down_count = int(vertical_axis.get("down_events") or 0)
    if down_count >= 4:
        return down_count + 1
    paired_count = min(vertical_count, arm_count) if vertical_count and arm_count else max(vertical_count, arm_count)
    if paired_count >= 4:
        return paired_count
    return max(paired_count, phase_count, int(vertical_axis.get("repeat_count") or 0))


def bimanual_cycle_peak_count(events: list[dict[str, Any]]) -> int:
    return _count_peaks(events, min_gap=6)


def bilateral_rhythmic_evidence(
    vertical_axis: dict[str, Any],
    limb_axis: dict[str, Any],
    *,
    vertical_amplitude_m: float,
) -> dict[str, Any]:
    raise_spread_count = int(limb_axis.get("raise_spread_count") or 0)
    return {
        "count": int(vertical_axis.get("repeat_count") or 0),
        "raise_spread_count": raise_spread_count,
        "bimanual_count": int(limb_axis.get("bimanual_count") or 0),
        "vertical_amplitude_m": round(vertical_amplitude_m, 4),
        "mean_vertical_amplitude_m": vertical_axis.get("mean_amplitude_m"),
        "source_event_indices": list(vertical_axis.get("event_indices") or []) + list(limb_axis.get("bimanual_event_indices") or []),
    }


def bilateral_gesture_evidence(
    limb_axis: dict[str, Any],
    *,
    rotation_count: int,
    locomotion_segment_count: int,
) -> dict[str, Any]:
    return {
        "turn_count": int(rotation_count),
        "locomotion_segment_count": int(locomotion_segment_count),
        "raise_spread_count": int(limb_axis.get("raise_spread_count") or 0),
        "bimanual_count": int(limb_axis.get("bimanual_count") or 0),
        "source_event_indices": list(limb_axis.get("bimanual_event_indices") or []),
    }
