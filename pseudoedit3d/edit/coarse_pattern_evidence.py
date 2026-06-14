from __future__ import annotations

from typing import Any

from .aml_pattern_tree import (
    action_pattern_metadata_for_family,
    action_pattern_metadata_for_node,
    event_proxy_action_fields,
    event_proxy_for_event,
    prototype_from_node_outputs,
    select_composed_pattern_match,
    select_sparse_pattern_match,
)
from .aml_proto_registry import active_proto_id, proto_in_group, registry_map
from .coarse_event_utils import (
    _duration,
    _event_sort_key,
    _gap,
    _magnitude,
    _mean_magnitude,
    _overlap_frames,
    _overlap_ratio,
    _span,
    _speed_from_event,
)
from .coarse_pattern_counts import (
    bimanual_cycle_peak_count,
    bilateral_rhythmic_cycle_count,
    bimanual_periodic_events,
)

_active_proto_id = active_proto_id
_proto_in_group = proto_in_group
_registry_map = registry_map


def _emitter_spec(name: str) -> dict[str, Any]:
    spec = _registry_map("semantic_action_emitters", name)
    if not spec:
        raise KeyError(f"missing semantic action emitter spec: {name}")
    return spec


def _emitter_proto_id(name: str, default: str = "") -> str:
    proto_id = _emitter_spec(name).get("prototype_id")
    if not proto_id and default:
        return default
    if not proto_id:
        raise KeyError(f"missing prototype_id in semantic action emitter spec: {name}")
    return str(proto_id)


def _vertical_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evt for evt in events if evt.get("super_family") == "WHOLE_BODY_VERTICAL"]


_bimanual_periodic_events = bimanual_periodic_events
_cyclic_coordination_count = bilateral_rhythmic_cycle_count


def _action_from_sparse_match(ctx: dict[str, Any], group: list[dict[str, Any]]) -> dict[str, Any] | None:
    match = select_sparse_pattern_match(ctx)
    if match is None:
        return None
    prototype = prototype_from_node_outputs(match.node, ctx)
    spans = [_span(evt) for evt in group]
    action = {
        "prototype_id": str(prototype.get("prototype_id", match.family_id or "")),
        "name_hint": str(prototype.get("name_hint", match.family_id or "")),
        "primary_direction": str(prototype.get("primary_direction", "unknown")),
        "confidence": min(0.82, max([float(prototype.get("confidence", 0.0) or 0.0)] + [float(evt.get("confidence", 0.0)) for evt in group])),
        "span": [min(s for s, _ in spans), max(e for _, e in spans)],
        "covered_event_indices": [int(evt["event_index"]) for evt in group],
        "source_event_indices": [int(evt["event_index"]) for evt in group],
        "source_event_count": len(group),
        "source_event_spans": [list(span) for span in spans],
        "semantic_proxy": True,
    }
    for key, value in prototype.items():
        if key not in {"prototype_id", "name_hint", "primary_direction", "confidence"} and value is not None:
            action[key] = value
    return action


def _bimanual_contact_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close_clusters = {"BI_HANDS_CLOSE", "BI_HANDS_CLOSE_RAISE"}
    close_events = sorted(
        [
            evt
            for evt in events
            if int(evt["event_index"]) not in covered
            and evt in _bimanual_periodic_events(events, close_clusters)
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
        action = _action_from_sparse_match(
            {
                "pattern_kind": "bimanual_periodic_contact",
                "event_count": len(group),
                "source_event_clusters": clusters,
                "has_raise": any(cluster == "BI_HANDS_CLOSE_RAISE" for cluster in clusters),
                "confidence": min(0.68, max(float(evt.get("confidence", 0.0)) for evt in group) + 0.04),
            },
            group,
        )
        if action is None:
            continue
        action["covered_event_indices"] = indices
        out.append(action)
        covered.update(indices)
    return out


def _cyclic_bilateral_coordination_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    _ = covered
    bimanual_cycle_events = _bimanual_periodic_events(events, {"BI_RAISE_SPREAD"})
    if len(bimanual_cycle_events) < 3:
        return []
    vertical_events = [
        evt for evt in _vertical_events(events)
        if (
            str(evt.get("cluster_id", "")) in {"WB_VERT_UP", "WB_VERT_DOWN", "WB_VERT_REP", "WB_VERT_REP_ALT", "WB_VERT_CYCLE"}
            or evt.get("role") == "repeated_phase"
        )
        and (_magnitude(evt) >= 0.015 or evt.get("role") == "repeated_phase")
    ]
    if not vertical_events:
        return []

    def has_strong_translation(span_evt: dict[str, Any]) -> bool:
        return any(
            evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
            and str(evt.get("cluster_id", "")).startswith("LOCO_")
            and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
            and _magnitude(evt) >= 1.2
            and _overlap_ratio(evt, span_evt) >= 0.25
            for evt in events
        )

    def make_candidate(arm_group: list[dict[str, Any]], vertical_pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        arm_group = sorted(arm_group, key=_event_sort_key)
        vertical_group = [
            evt for evt in sorted(vertical_pool, key=_event_sort_key)
            if _magnitude(evt) >= 0.04 or evt.get("role") == "repeated_phase"
        ]
        paired_arms = [
            arm for arm in arm_group
            if any(_overlap_ratio(arm, vert) >= 0.05 or _gap(arm, vert) <= 28 for vert in vertical_group)
        ]
        if len(arm_group) < 3 or len(vertical_group) < 2 or len(paired_arms) < 2:
            return None
        spans = [_span(evt) for evt in arm_group + vertical_group]
        span_evt = {"start_frame": min(s for s, _ in spans), "end_frame": max(e for _, e in spans)}
        if has_strong_translation(span_evt):
            return None
        local_cover = {
            int(evt["event_index"])
            for evt in events
            if evt.get("super_family") in {
                "BIMANUAL_PERIODIC",
                "LEFT_ARM_POSTURE",
                "RIGHT_ARM_POSTURE",
                "LEFT_ARM_PERIODIC",
                "RIGHT_ARM_PERIODIC",
                "WHOLE_BODY_VERTICAL",
                "WHOLE_BODY_POSTURE",
            }
            and (_overlap_ratio(evt, span_evt) >= 0.05 or _gap(evt, span_evt) <= 6)
        }
        repeat_count = bimanual_cycle_peak_count(paired_arms)
        vertical_axis = {
            "phase_repeat_count": max((int(evt.get("count") or 0) for evt in vertical_group), default=0),
            "down_events": sum(1 for evt in vertical_group if str(evt.get("cluster_id", "")) == "WB_VERT_DOWN"),
            "repeat_count": max(1, len(vertical_group)),
        }
        count = max(repeat_count, _cyclic_coordination_count(arm_group + vertical_group, vertical_axis))
        action = _action_from_sparse_match(
            {
                "pattern_kind": "bilateral_limb_vertical_coordination",
                "bimanual_event_count": len(arm_group),
                "vertical_event_count": len(vertical_group),
                "paired_bimanual_count": len(paired_arms),
                "near_strong_translation": has_strong_translation(span_evt),
                "cycle_count": max(2, count),
                "vertical_amplitude_m": round(max((_magnitude(evt) for evt in vertical_group), default=0.0), 4),
                "mean_vertical_amplitude_m": round(_mean_magnitude(vertical_group), 4),
                "confidence": 0.66 + min(0.14, 0.02 * len(paired_arms)),
            },
            arm_group + vertical_group,
        )
        if action is not None:
            action["covered_event_indices"] = sorted(local_cover)
            action["source_event_indices"] = sorted(int(evt["event_index"]) for evt in arm_group + vertical_group)
        return action

    global_candidate = make_candidate(bimanual_cycle_events, vertical_events)
    if global_candidate:
        return [global_candidate]

    evidence_spans = [_span(evt) for evt in bimanual_cycle_events + vertical_events]
    evidence_span = {
        "start_frame": min(s for s, _ in evidence_spans),
        "end_frame": max(e for _, e in evidence_spans),
    }
    blockers = sorted(
        [
            evt for evt in events
            if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
            and str(evt.get("cluster_id", "")).startswith("LOCO_")
            and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
            and _magnitude(evt) >= 1.2
            and _overlap_ratio(evt, evidence_span) >= 0.10
        ],
        key=_event_sort_key,
    )
    if not blockers:
        return []

    windows: list[dict[str, int]] = []
    cursor = int(evidence_span["start_frame"])
    for blocker in blockers:
        start, end = _span(blocker)
        if cursor < start:
            windows.append({"start_frame": cursor, "end_frame": start - 1})
        cursor = max(cursor, end + 1)
    if cursor <= int(evidence_span["end_frame"]):
        windows.append({"start_frame": cursor, "end_frame": int(evidence_span["end_frame"])})

    actions: list[dict[str, Any]] = []
    for window in windows:
        arm_group = [evt for evt in bimanual_cycle_events if _overlap_frames(evt, window) > 0]
        vertical_group = [evt for evt in vertical_events if _overlap_frames(evt, window) > 0]
        candidate = make_candidate(arm_group, vertical_group)
        if candidate:
            actions.append(candidate)
    return actions


def _vertical_impulse_translation_pair_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
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
        action = _action_from_sparse_match(
            {
                "pattern_kind": "vertical_impulse_translation_pair",
                "strength": "strong" if strong else "weak",
                "direction": str(best_loco.get("direction", "unknown")),
                "confidence": (0.62 if strong else 0.42) + min(0.18, _magnitude(vert)),
                "speed": _speed_from_event(best_loco),
                "distance_m": round(_magnitude(best_loco), 4),
                "vertical_amplitude_m": round(_magnitude(vert), 4),
            },
            [best_loco, vert],
        )
        if action is None:
            continue
        action["span"] = [start, end]
        action["covered_event_indices"] = sorted(local_cover)
        action["source_event_indices"] = [lidx, vidx]
        out.append(action)
    return out


def _post_vertical_translation_recovery_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
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
        local_cover = [
            int(evt["event_index"])
            for evt in events
            if int(evt["event_index"]) == int(loco["event_index"])
            or (
                evt.get("super_family") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC", "BIMANUAL_PERIODIC"}
                and (_overlap_ratio(evt, loco) >= 0.20 or _gap(evt, loco) <= 5)
            )
        ]
        action = _action_from_sparse_match(
            {
                "pattern_kind": "post_vertical_translation_recovery",
                "direction": direction,
                "confidence": 0.58,
                "distance_m": round(_magnitude(loco), 4),
                "speed": _speed_from_event(loco),
                "source_jump_event_index": int(best_vert["event_index"]),
            },
            [loco, best_vert],
        )
        if action is None:
            continue
        action["span"] = list(_span(loco))
        action["covered_event_indices"] = local_cover
        action["source_event_indices"] = [int(loco["event_index"]), int(best_vert["event_index"])]
        out.append(action)
    return out


def _make_semantic_proxy_action(
    proto_id: str,
    name_hint: str,
    direction: str,
    group: list[dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    spans = [_span(evt) for evt in group]
    indices = [int(evt["event_index"]) for evt in group]
    metadata = group[0].get("metadata") or {}
    pattern_metadata = {}
    event_proxy_node = event_proxy_for_event(group[0])
    if event_proxy_node and str(event_proxy_node.get("family_id", "")) == proto_id:
        pattern_metadata = action_pattern_metadata_for_node(event_proxy_node)
    if not pattern_metadata:
        pattern_metadata = action_pattern_metadata_for_family(
            _active_proto_id(proto_id),
            preferred_node_types=("composed_candidate", "event_proxy", "primary"),
        )
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
        "source_event_indices": indices,
        **pattern_metadata,
    }
    magnitude = max((_magnitude(evt) for evt in group), default=0.0)
    if magnitude > 0.0:
        action["magnitude"] = round(magnitude, 4)
        action["unit"] = group[0].get("unit")
    if str(group[0].get("cluster_id", "")) == "ROOT_CIRCULAR_PATH":
        action["path_length_m"] = round(float(metadata.get("path_length", magnitude) or 0.0), 4)
        action["circle_score"] = round(float(metadata.get("circle_score", 0.0) or 0.0), 4)
        action["curvature_rad"] = round(float(metadata.get("curvature", group[0].get("signed_delta") or 0.0) or 0.0), 4)
    if str(group[0].get("cluster_id", "")) == "CLIMB_UP_OVER_PROXY":
        action["root_height_gain_m"] = round(float(metadata.get("root_height_gain", magnitude) or 0.0), 4)
    return action


def _make_composed_semantic_action(ctx: dict[str, Any], group: list[dict[str, Any]]) -> dict[str, Any] | None:
    match = select_composed_pattern_match(ctx)
    if match is None:
        return None
    prototype = prototype_from_node_outputs(match.node, ctx)
    action = _make_semantic_proxy_action(
        str(prototype.get("prototype_id", match.family_id or "")),
        str(prototype.get("name_hint", match.family_id or "")),
        str(prototype.get("primary_direction", "unknown")),
        group,
        float(prototype.get("confidence", 0.0) or 0.0),
    )
    for key, value in prototype.items():
        if key not in {"prototype_id", "name_hint", "primary_direction", "confidence"} and value is not None:
            action[key] = value
    return action


def _near_strong_translation(span_evt: dict[str, Any], locomotion_events: list[dict[str, Any]]) -> bool:
    for loco in locomotion_events:
        if _magnitude(loco) < 0.90:
            continue
        if _overlap_ratio(span_evt, loco) >= 0.20 or _gap(span_evt, loco) <= 8:
            return True
    return False


def _leg_forward_near(leg_forward_events: list[dict[str, Any]], span_evt: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        evt for evt in leg_forward_events
        if _overlap_ratio(evt, span_evt) >= 0.08 or _gap(evt, span_evt) <= 16
    ]


def _collect_inversion_semantic_actions(
    events: list[dict[str, Any]],
    semantic_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    inversion_events = [
        evt for evt in semantic_events
        if str(evt.get("super_family", "")) == "WHOLE_BODY_ACROBATICS"
    ]
    inversion_cover: set[int] = set()
    if not inversion_events:
        return [], inversion_cover
    action = _make_composed_semantic_action(
        {
            "pattern_kind": "acrobatic_sequence",
            "segment_count": len(inversion_events),
        },
        inversion_events,
    )
    if action is None:
        return [], inversion_cover
    inversion_cover.update(
        int(evt["event_index"])
        for evt in events
        if evt.get("super_family") in {
            "WHOLE_BODY_ACROBATICS",
            "WHOLE_BODY_VERTICAL",
            "WHOLE_BODY_ROTATION",
            "WHOLE_BODY_POSTURE",
            "TORSO_POSTURE",
        }
        and any(_overlap_ratio(evt, inv) >= 0.10 or _gap(evt, inv) <= 12 for inv in inversion_events)
    )
    action["covered_event_indices"] = sorted(inversion_cover)
    action["source_event_indices"] = [int(evt["event_index"]) for evt in inversion_events]
    return [action], inversion_cover


def _collect_ground_transition_cover(semantic_events: list[dict[str, Any]]) -> set[int]:
    ground_transition_cover: set[int] = set()
    for evt in semantic_events:
        semantic_node = event_proxy_for_event(evt)
        if semantic_node is None:
            continue
        proto_id, _, _, _ = event_proxy_action_fields(semantic_node)
        if _proto_in_group(proto_id, "dominance", "dominant_groups", "SIT_STAND_TRANSITION"):
            ground_transition_cover.add(int(evt["event_index"]))
    return ground_transition_cover


def _semantic_proxy_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evt
        for evt in events
        if event_proxy_for_event(evt) is not None
    ]


def _translation_locomotion_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evt for evt in events
        if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION"
        and str(evt.get("cluster_id", "")).startswith("LOCO_")
        and not str(evt.get("cluster_id", "")).startswith("LOCO_TURN_")
    ]


def _low_body_posture_events(
    semantic_events: list[dict[str, Any]],
    inversion_cover: set[int],
) -> list[dict[str, Any]]:
    return [
        evt for evt in semantic_events
        if str(evt.get("super_family", "")) == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        and int(evt["event_index"]) not in inversion_cover
    ]


def _leg_forward_candidate_events(
    semantic_events: list[dict[str, Any]],
    inversion_cover: set[int],
    ground_transition_cover: set[int],
) -> list[dict[str, Any]]:
    return [
        evt for evt in semantic_events
        if int(evt["event_index"]) not in inversion_cover
        and int(evt["event_index"]) not in ground_transition_cover
        and str(evt.get("super_family", "")) in {"LEFT_LEG_ACTION", "RIGHT_LEG_ACTION"}
        and str(evt.get("cluster_id", "")) in {
            "LL_KICK_FORWARD",
            "RL_KICK_FORWARD",
            "LL_LEG_FORWARD_POSE",
            "RL_LEG_FORWARD_POSE",
        }
    ]


def _collect_low_body_vertical_transition_actions(
    events: list[dict[str, Any]],
    semantic_events: list[dict[str, Any]],
    locomotion_events: list[dict[str, Any]],
    leg_forward_events: list[dict[str, Any]],
    inversion_cover: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    vertical_down = [
        evt for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_DOWN"
        and _magnitude(evt) >= 0.18
    ]
    vertical_up = [
        evt for evt in events
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL"
        and str(evt.get("cluster_id", "")) == "WB_VERT_UP"
        and _magnitude(evt) >= 0.18
    ]
    long_low_events = [
        evt for evt in semantic_events
        if int(evt["event_index"]) not in inversion_cover
        and str(evt.get("super_family", "")) == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        and _duration(evt) >= 24
        and _magnitude(evt) >= 0.24
        and not _near_strong_translation(evt, locomotion_events)
    ]
    low_body_vertical_transition_actions: list[dict[str, Any]] = []
    low_body_vertical_transition_cover: set[int] = set()
    for low_evt in long_low_events:
        related_down = [
            evt for evt in vertical_down
            if _overlap_ratio(evt, low_evt) >= 0.10 or _gap(evt, low_evt) <= 12
        ]
        related_up = [
            evt for evt in vertical_up
            if _overlap_ratio(evt, low_evt) >= 0.10 or _gap(evt, low_evt) <= 16
        ]
        if not related_down and not related_up:
            continue
        nearby_legs = _leg_forward_near(leg_forward_events, low_evt)
        side_count = len({
            "left" if str(evt.get("super_family", "")) == "LEFT_LEG_ACTION" else "right"
            for evt in nearby_legs
        })
        strong_unilateral_leg = (
            side_count == 1
            and any(_magnitude(evt) >= 0.45 or _duration(evt) >= 16 for evt in nearby_legs)
        )
        low_start, _ = _span(low_evt)
        if strong_unilateral_leg and not related_down and low_start > 5:
            continue
        phase_events = [low_evt] + related_down + related_up
        phase_order = "none"
        if related_down and related_up:
            phase_order = "down_then_up" if min(int(evt["start_frame"]) for evt in related_down) <= min(int(evt["start_frame"]) for evt in related_up) else "up_then_down"
        composed_ctx = {
            "pattern_kind": "low_body_vertical_transition",
            "has_low_body_posture": True,
            "has_vertical_down": bool(related_down),
            "has_vertical_up": bool(related_up),
            "strong_unilateral_leg_without_down": bool(strong_unilateral_leg and not related_down),
            "near_strong_translation": _near_strong_translation(low_evt, locomotion_events),
            "phase_order": phase_order,
        }
        action = _make_composed_semantic_action(composed_ctx, phase_events)
        if action is None:
            continue
        low_body_vertical_transition_actions.append(action)
        local_cover = {int(evt["event_index"]) for evt in phase_events}
        low_body_vertical_transition_cover.update(local_cover)
    return low_body_vertical_transition_actions, low_body_vertical_transition_cover


def _collect_low_body_leg_pair_actions(
    events: list[dict[str, Any]],
    semantic_events: list[dict[str, Any]],
    locomotion_events: list[dict[str, Any]],
    leg_forward_events: list[dict[str, Any]],
    inversion_cover: set[int],
    ground_transition_cover: set[int],
    low_body_vertical_transition_cover: set[int],
) -> tuple[list[dict[str, Any]], set[int], dict[int, str]]:
    out: list[dict[str, Any]] = []
    low_body_leg_pair_cover: set[int] = set()
    low_body_leg_pair_cover_owner: dict[int, str] = {}
    low_body_events = [
        evt for evt in semantic_events
        if int(evt["event_index"]) not in inversion_cover
        and int(evt["event_index"]) not in low_body_vertical_transition_cover
        and str(evt.get("super_family", "")) == "WHOLE_BODY_POSTURE"
        and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        and _duration(evt) >= 18
        and _magnitude(evt) >= 0.22
        and not _near_strong_translation(evt, locomotion_events)
    ]
    leg_forward_pair_events = [
        evt for evt in leg_forward_events
        if int(evt["event_index"]) not in inversion_cover
        and int(evt["event_index"]) not in low_body_vertical_transition_cover
        and int(evt["event_index"]) not in ground_transition_cover
    ]
    used_low_ids: set[int] = set()
    for leg_evt in sorted(leg_forward_pair_events, key=_event_sort_key):
        leg_idx = int(leg_evt["event_index"])
        side = "left" if str(leg_evt.get("super_family", "")) == "LEFT_LEG_ACTION" else "right"
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for low_evt in low_body_events:
            low_idx = int(low_evt["event_index"])
            if low_idx in used_low_ids:
                continue
            overlap = _overlap_ratio(leg_evt, low_evt)
            gap = _gap(leg_evt, low_evt)
            leg_start, leg_end = _span(leg_evt)
            low_start, low_end = _span(low_evt)
            enters_low_after_leg = low_start >= leg_start - 4 and low_start <= leg_end + 20
            already_low_near_leg = low_start < leg_start - 4 and low_end <= leg_end + 24
            leg_low_temporally_local = leg_start <= low_end and (enters_low_after_leg or already_low_near_leg)
            if leg_low_temporally_local and (overlap >= 0.10 or gap <= 20):
                candidates.append((overlap, -gap, low_evt))
        if not candidates:
            continue
        _, _, best_low = max(candidates, key=lambda item: (item[0], item[1], _duration(item[2])))
        if _duration(best_low) >= 70 and _duration(leg_evt) >= 30:
            continue
        low_idx = int(best_low["event_index"])
        group = [leg_evt, best_low]
        spans = [_span(evt) for evt in group]
        span_evt = {"start_frame": min(s for s, _ in spans), "end_frame": max(e for _, e in spans)}
        if _near_strong_translation(span_evt, locomotion_events):
            continue
        for evt in semantic_events:
            idx = int(evt["event_index"])
            if idx in {leg_idx, low_idx} or idx in inversion_cover or idx in ground_transition_cover:
                continue
            if evt.get("super_family") in {"TORSO_POSTURE", "WHOLE_BODY_POSTURE"}:
                if _overlap_ratio(evt, span_evt) >= 0.10 or _gap(evt, span_evt) <= 8:
                    group.append(evt)
        composed_ctx = {
            "pattern_kind": "low_body_leg_forward_pair",
            "has_leg_forward": True,
            "has_low_body_posture": True,
            "near_strong_translation": _near_strong_translation(span_evt, locomotion_events),
            "side": side,
            "leg_duration": _duration(leg_evt),
            "low_body_duration": _duration(best_low),
        }
        action = _make_composed_semantic_action(composed_ctx, group)
        if action is None:
            continue
        action["source_event_clusters"] = sorted({str(evt.get("cluster_id", "")) for evt in group})
        out.append(action)
        action_cover = {int(evt["event_index"]) for evt in group}
        cover_owner = str(action.get("prototype_id") or _registry_map("fallback_entrypoints").get("unknown_source_family") or "EVENT_SEQUENCE")
        low_body_leg_pair_cover.update(action_cover)
        for idx in action_cover:
            low_body_leg_pair_cover_owner[idx] = cover_owner
        used_low_ids.add(low_idx)
    return out, low_body_leg_pair_cover, low_body_leg_pair_cover_owner


def _collect_low_body_repetition_actions(
    events: list[dict[str, Any]],
    low_body_postures: list[dict[str, Any]],
    covered: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    primary_low_body_ids = covered.intersection({int(evt["event_index"]) for evt in low_body_postures})
    if len(low_body_postures) < 2 or primary_low_body_ids:
        return [], set()
    bimanual_nearby = [
        evt for evt in events
        if evt.get("super_family") == "BIMANUAL_PERIODIC"
        and any(_overlap_ratio(evt, posture) >= 0.15 or _gap(evt, posture) <= 8 for posture in low_body_postures)
    ]
    emit_key = "low_body_repetition_with_arm_lift" if bimanual_nearby else "low_body_repetition"
    emit_spec = _emitter_spec(emit_key)
    action = _make_semantic_proxy_action(
        _emitter_proto_id(emit_key),
        str(emit_spec.get("name_hint") or ""),
        str(emit_spec.get("primary_direction") or "low"),
        low_body_postures,
        float(emit_spec.get("confidence") or 0.72),
    )
    action["count"] = len(low_body_postures)
    action["lexical_alias_candidates"] = list(emit_spec.get("lexical_alias_candidates") or [])
    if bimanual_nearby:
        bimanual_indices = {int(evt["event_index"]) for evt in bimanual_nearby}
        action["covered_event_indices"] = sorted(set(action["covered_event_indices"]) | bimanual_indices)
        action["source_event_indices"] = sorted(set(action.get("source_event_indices") or []) | bimanual_indices)
        action["source_event_count"] = len(action["source_event_indices"])
        action["bimanual_count"] = len(bimanual_nearby)
        action["source_event_clusters"] = sorted({str(evt.get("cluster_id", "")) for evt in bimanual_nearby})
    cover = {int(x) for x in action["covered_event_indices"]}
    for evt in events:
        if evt.get("super_family") == "WHOLE_BODY_VERTICAL":
            if any(_overlap_ratio(evt, posture) >= 0.15 or _gap(evt, posture) <= 6 for posture in low_body_postures):
                cover.add(int(evt["event_index"]))
    return [action], cover


def _collect_residual_semantic_proxy_actions(
    events: list[dict[str, Any]],
    semantic_events: list[dict[str, Any]],
    locomotion_events: list[dict[str, Any]],
    low_body_postures: list[dict[str, Any]],
    inversion_cover: set[int],
    ground_transition_cover: set[int],
    low_body_vertical_transition_cover: set[int],
    low_body_leg_pair_cover: set[int],
    low_body_leg_pair_cover_owner: dict[int, str],
    covered: set[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hand_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    leg_pose_groups: dict[str, list[dict[str, Any]]] = {}
    aggregate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for evt in semantic_events:
        idx = int(evt["event_index"])
        if idx in inversion_cover:
            continue
        semantic_node = event_proxy_for_event(evt)
        if semantic_node is None:
            continue
        proto_id, name_hint, direction, confidence = event_proxy_action_fields(semantic_node)
        if idx in covered and _proto_in_group(proto_id, "semantic_action_groups", "skip_when_already_covered"):
            continue
        if idx in low_body_vertical_transition_cover and _proto_in_group(proto_id, "semantic_cover_suppression", "low_body_vertical_transition_cover"):
            continue
        if idx in low_body_leg_pair_cover and _proto_in_group(proto_id, "semantic_cover_suppression", "low_body_leg_pair_cover"):
            continue
        if _proto_in_group(proto_id, "semantic_cover_suppression", "ground_transition_cover") and idx in ground_transition_cover:
            continue
        if _proto_in_group(proto_id, "semantic_action_groups", "aggregate_by_group"):
            aggregate_groups.setdefault((proto_id, name_hint, direction), []).append(evt)
            continue
        if _proto_in_group(proto_id, "semantic_action_groups", "defer_to_state_axis"):
            continue
        if _proto_in_group(proto_id, "semantic_action_groups", "hand_high"):
            hand_groups.setdefault((proto_id, name_hint), []).append(evt)
            continue
        if _proto_in_group(proto_id, "semantic_action_groups", "leg_forward_pose"):
            leg_pose_groups.setdefault(name_hint, []).append(evt)
            continue
        if _proto_in_group(proto_id, "semantic_action_groups", "leg_kick"):
            gait_like = any(
                _overlap_ratio(evt, loco) >= 0.40
                and _magnitude(loco) >= 1.0
                and _speed_from_event(loco) != "slow"
                for loco in locomotion_events
            )
            if gait_like:
                action = _make_semantic_proxy_action(proto_id, name_hint, direction, [evt], confidence)
                action["probe_visible"] = False
                action["hidden_by_semantic_family"] = "TRANSLATING_GAIT"
                action["hidden_by_pattern_cover"] = "gait_like_leg_swing"
                out.append(action)
                covered.add(idx)
                continue
        if _proto_in_group(proto_id, "semantic_action_groups", "suppress_when_repeated_low_body") and len(low_body_postures) >= 2:
            continue
        action = _make_semantic_proxy_action(proto_id, name_hint, direction, [evt], confidence)
        out.append(action)
        if _proto_in_group(proto_id, "semantic_action_groups", "emit_and_cover"):
            covered.add(idx)
    for (proto_id, name_hint), group in hand_groups.items():
        action = _make_semantic_proxy_action(proto_id, name_hint, "up", group, 0.60)
        action["count"] = len(group)
        out.append(action)
    for (proto_id, name_hint, direction), group in aggregate_groups.items():
        aggregate_node = event_proxy_for_event(group[0])
        _, _, _, confidence = event_proxy_action_fields(aggregate_node) if aggregate_node else ("", "", direction, 0.0)
        action = _make_semantic_proxy_action(proto_id, name_hint, direction, group, confidence)
        action["count"] = len(group)
        out.append(action)
    for name_hint, group in leg_pose_groups.items():
        has_turn = any(evt.get("super_family") == "WHOLE_BODY_ROTATION" and _magnitude(evt) >= 120.0 for evt in events)
        has_high_hand = bool(hand_groups)
        side = "left" if name_hint.startswith("left") else "right"
        action = _make_composed_semantic_action(
            {
                "pattern_kind": "leg_forward_pose_context",
                "has_leg_forward_pose": True,
                "has_turn": has_turn,
                "has_high_hand": has_high_hand,
                "name_hint": name_hint,
                "side": side,
            },
            group,
        )
        if action is None:
            leg_node = event_proxy_for_event(group[0])
            proto_id, proxy_name_hint, direction, confidence = event_proxy_action_fields(leg_node) if leg_node else ("", name_hint, "forward", 0.0)
            action = _make_semantic_proxy_action(proto_id, proxy_name_hint or name_hint, direction, group, confidence)
        action["count"] = len(group)
        action["dominant_side"] = side
        pair_cover_owners = sorted({
            low_body_leg_pair_cover_owner[int(evt["event_index"])]
            for evt in group
            if int(evt["event_index"]) in low_body_leg_pair_cover_owner
        })
        if pair_cover_owners:
            action["probe_visible"] = False
            action["hidden_by_semantic_family"] = pair_cover_owners[0]
            action["hidden_by_pattern_cover"] = "low_body_leg_pair_cover"
        out.append(action)
    return out


def _semantic_candidate_actions(events: list[dict[str, Any]], covered: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    semantic_events = _semantic_proxy_events(events)
    locomotion_events = _translation_locomotion_events(events)

    inversion_actions, inversion_cover = _collect_inversion_semantic_actions(events, semantic_events)
    out.extend(inversion_actions)
    covered.update(inversion_cover)

    ground_transition_cover = _collect_ground_transition_cover(semantic_events)
    covered.update(ground_transition_cover)

    low_body_postures = _low_body_posture_events(semantic_events, inversion_cover)
    leg_forward_events = _leg_forward_candidate_events(semantic_events, inversion_cover, ground_transition_cover)

    low_transition_actions, low_body_vertical_transition_cover = _collect_low_body_vertical_transition_actions(
        events,
        semantic_events,
        locomotion_events,
        leg_forward_events,
        inversion_cover,
    )
    out.extend(low_transition_actions)
    covered.update(low_body_vertical_transition_cover)

    low_pair_actions, low_body_leg_pair_cover, low_body_leg_pair_cover_owner = _collect_low_body_leg_pair_actions(
        events,
        semantic_events,
        locomotion_events,
        leg_forward_events,
        inversion_cover,
        ground_transition_cover,
        low_body_vertical_transition_cover,
    )
    out.extend(low_pair_actions)
    covered.update(low_body_leg_pair_cover)

    low_repetition_actions, low_repetition_cover = _collect_low_body_repetition_actions(events, low_body_postures, covered)
    out.extend(low_repetition_actions)
    covered.update(low_repetition_cover)

    out.extend(
        _collect_residual_semantic_proxy_actions(
            events,
            semantic_events,
            locomotion_events,
            low_body_postures,
            inversion_cover,
            ground_transition_cover,
            low_body_vertical_transition_cover,
            low_body_leg_pair_cover,
            low_body_leg_pair_cover_owner,
            covered,
        )
    )
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
            action = _action_from_sparse_match(
                {
                    "pattern_kind": "residual_translating_gait_segment",
                    "direction": str(evt.get("direction", "unknown")),
                    "confidence": 0.46,
                    "speed": _speed_from_event(evt),
                    "distance_m": round(_magnitude(evt), 4),
                },
                [evt],
            )
            if action is None:
                continue
            action["span"] = list(_span(evt))
            action["covered_event_indices"] = [idx]
            actions.append(action)
            covered.add(idx)
        elif family == "WHOLE_BODY_LOCOMOTION" and cluster.startswith("LOCO_TURN_") and _magnitude(evt) >= 30.0:
            has_rotation_driver = any(
                other.get("super_family") == "WHOLE_BODY_ROTATION"
                and _magnitude(other) >= 45.0
                and (_overlap_ratio(evt, other) >= 0.25 or _gap(evt, other) <= 4)
                for other in events
            )
            if has_rotation_driver:
                continue
            action = _action_from_sparse_match(
                {
                    "pattern_kind": "residual_turn_segment",
                    "direction": str(evt.get("direction", "unknown")),
                    "confidence": 0.46,
                    "angle_deg": round(_magnitude(evt), 2),
                    "angle_bin": cluster,
                },
                [evt],
            )
            if action is None:
                continue
            action["span"] = list(_span(evt))
            action["covered_event_indices"] = [idx]
            actions.append(action)
            covered.add(idx)
        elif family == "WHOLE_BODY_ROTATION" and _magnitude(evt) >= 45.0:
            local_cover = {idx}
            for other in events:
                if other.get("super_family") == "WHOLE_BODY_LOCOMOTION" and str(other.get("cluster_id", "")).startswith("LOCO_TURN_"):
                    if _overlap_ratio(evt, other) >= 0.25 or _gap(evt, other) <= 4:
                        local_cover.add(int(other["event_index"]))
            action = _action_from_sparse_match(
                {
                    "pattern_kind": "residual_turn_segment",
                    "direction": str(evt.get("direction", "unknown")),
                    "confidence": 0.50,
                    "angle_deg": round(_magnitude(evt), 2),
                    "angle_bin": str(evt.get("cluster_id", "")),
                },
                [evt],
            )
            if action is None:
                continue
            action["span"] = list(_span(evt))
            action["covered_event_indices"] = sorted(local_cover)
            actions.append(action)
            covered.update(local_cover)
    return actions
