from __future__ import annotations

from typing import Any

from .aml_pattern_tree import (
    PatternMatch,
    action_pattern_metadata_for_node,
    event_proxy_action_fields,
    event_proxy_for_event,
    match_pattern_tree,
    pattern_node_for_family,
    prototype_from_node_outputs,
    select_primary_pattern_match,
)
from .aml_proto_registry import active_proto_id, proto_in_group, registry_map, registry_set
from .coarse_action_materializer import _attach_action_metadata
from .coarse_axes import build_event_coarse_signature, _in_place_gait_name
from .coarse_pattern_evidence import (
    _bimanual_contact_actions,
    _cyclic_bilateral_coordination_actions,
    _cyclic_coordination_count,
    _post_vertical_translation_recovery_actions,
    _semantic_candidate_actions,
    _secondary_actions,
    _vertical_impulse_translation_pair_actions,
)
from .coarse_event_utils import (
    _event_by_index,
    _event_cluster_counts,
    _event_family_counts,
    _event_indices_for_families,
    _event_ref,
    _gap,
    _indexed_events,
    _indices_by_family,
    _magnitude,
    _overlap_ratio,
    _span,
)


_active_proto_id = active_proto_id
_proto_in_group = proto_in_group
_registry_map = registry_map
_registry_set = registry_set




def _no_salient_event_prototype() -> dict[str, Any]:
    fallback_family = str(_registry_map("fallback_entrypoints").get("no_salient_event_family") or "")
    node = pattern_node_for_family(fallback_family, preferred_node_types=("primary", "composed_candidate"))
    family_id = str(node.get("family_id", fallback_family)) if node else fallback_family
    return {
        "prototype_id": family_id,
        "name_hint": str((node.get("defaults") or {}).get("name_hint") or "no_salient_layer3_event") if node else "no_salient_layer3_event",
        "confidence": 0.20,
        "primary_direction": str((node.get("defaults") or {}).get("primary_direction") or "unknown") if node else "unknown",
        "source_event_count": 0,
        "source_event_indices": [],
        "probe_visible": False,
        **(action_pattern_metadata_for_node(node) if node else {}),
        "rationale": "no salient Layer-3 event was extracted, so preserve a conservative no-evidence proxy instead of an unknown action family",
    }


def _prototype_from_pattern_match(match: PatternMatch, ctx: dict[str, Any]) -> dict[str, Any]:
    node = match.node
    derived = node.get("derived_output")
    if derived == "in_place_gait":
        name, confidence, rationale = _in_place_gait_name(ctx["vertical"], ctx["limbs"])
        return {
            "prototype_id": str(node.get("family_id", "")),
            "name_hint": name,
            "confidence": confidence,
            "primary_direction": "in_place",
            "pattern_node_id": match.node_id,
            "pattern_path": match.path,
            "pattern_taxonomy_parent_id": node.get("taxonomy_parent_id"),
            "rationale": f"arm swing and low vertical cycles indicate a gait-like in-place motion; {rationale}",
        }
    if derived == "in_place_gait_proxy":
        events = ctx["events"]
        source_indices = sorted(
            set(_event_indices_for_families(events, {"WHOLE_BODY_VERTICAL"}))
            | set(_event_indices_for_families(events, {"BIMANUAL_PERIODIC"}))
            | set(_event_indices_for_families(events, {"TORSO_PERIODIC"}))
        )
        return {
            "prototype_id": str(node.get("family_id", "")),
            "name_hint": "in_place_bounce_gait_proxy",
            "confidence": 0.46,
            "primary_direction": "in_place",
            "count": ctx["repeat_count"],
            "vertical_amplitude_m": ctx["vertical_amp"],
            "mean_vertical_amplitude_m": ctx["vertical"].get("mean_amplitude_m"),
            "source_event_count": len(source_indices),
            "source_event_indices": source_indices,
            "pattern_node_id": match.node_id,
            "pattern_path": match.path,
            "pattern_taxonomy_parent_id": node.get("taxonomy_parent_id"),
            "rationale": "low repeated vertical cycles with bimanual/torso motion and little translation are safer as an in-place gait proxy than an unknown bimanual family",
        }
    return prototype_from_node_outputs(node, ctx)


def _pattern_match_summary(match: PatternMatch) -> dict[str, Any]:
    return {
        "node_id": match.node_id,
        "family_id": match.family_id,
        "path": match.path,
        "depth": match.depth,
        "taxonomy_parent_id": match.node.get("taxonomy_parent_id"),
        "lexical_aliases": list(match.node.get("lexical_aliases") or []),
    }


def _select_seeded_pattern_prototype(ctx: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    matches = match_pattern_tree(ctx, node_type="primary")
    match_summaries = [_pattern_match_summary(match) for match in matches]
    primary_match = select_primary_pattern_match(ctx)
    if primary_match is None:
        return None, match_summaries
    prototype = _prototype_from_pattern_match(primary_match, ctx)
    prototype["pattern_tree_matches"] = match_summaries
    return prototype, match_summaries


def _prototype_context(signature: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    locomotion = signature.get("locomotion") or {}
    vertical = signature.get("vertical") or {}
    rotation = signature.get("rotation") or {}
    support_gait = signature.get("support_gait") or {}
    limbs = signature.get("limb_coordination") or {}
    patterns = signature.get("motion_patterns") or {}
    arm_loco_count = int(limbs.get("left_locomotion_coupled_count") or 0) + int(limbs.get("right_locomotion_coupled_count") or 0)
    upper_body = _upper_body_evidence(events, limbs)
    return {
        "events": events,
        "signature": signature,
        "locomotion": locomotion,
        "vertical": vertical,
        "rotation": rotation,
        "support_gait": support_gait,
        "limbs": limbs,
        "patterns": patterns,
        "loco_state": str(locomotion.get("state", "none")),
        "loco_direction": str(locomotion.get("direction", "none")),
        "loco_speed": str(locomotion.get("speed", "unknown")),
        "loco_distance": float(locomotion.get("distance_m") or 0.0),
        "vertical_kind": str(vertical.get("kind", "none")),
        "vertical_amp": float(vertical.get("max_amplitude_m") or 0.0),
        "vertical_loco_overlap": float(vertical.get("locomotion_overlap") or 0.0),
        "bimanual_pattern": str(limbs.get("bimanual_pattern", "none")),
        "support_state": str(support_gait.get("state", "unknown")),
        "repeat_count": int(vertical.get("repeat_count") or 0),
        "raise_spread_count": int(limbs.get("raise_spread_count") or 0),
        "rotation_count": len(rotation.get("event_indices") or []),
        "locomotion_segment_count": len(locomotion.get("segments") or []),
        "arm_loco_count": arm_loco_count,
        "event_count": len(events),
        "upper_body": upper_body,
        "bimanual_count": upper_body["bimanual_count"],
        "left_arm_count": upper_body["left_arm_count"],
        "right_arm_count": upper_body["right_arm_count"],
    }


def _upper_body_evidence(events: list[dict[str, Any]], limbs: dict[str, Any]) -> dict[str, Any]:
    arm_families = {
        "LEFT_ARM_PERIODIC",
        "RIGHT_ARM_PERIODIC",
        "LEFT_ARM_POSTURE",
        "RIGHT_ARM_POSTURE",
        "BIMANUAL_PERIODIC",
        "TORSO_POSTURE",
        "TORSO_PERIODIC",
        "WHOLE_BODY_POSTURE",
    }
    arm_indices = _event_indices_for_families(events, arm_families)
    bimanual_count = int(limbs.get("bimanual_count") or 0)
    left_arm_count = int(limbs.get("left_arm_count") or 0)
    right_arm_count = int(limbs.get("right_arm_count") or 0)
    left_posture_count = sum(1 for evt in events if evt.get("super_family") == "LEFT_ARM_POSTURE")
    right_posture_count = sum(1 for evt in events if evt.get("super_family") == "RIGHT_ARM_POSTURE")
    torso_posture_count = sum(1 for evt in events if evt.get("super_family") == "TORSO_POSTURE")
    state_count = sum(1 for evt in events if evt.get("super_family") == "WHOLE_BODY_STATE")
    total_arm_signal = left_arm_count + right_arm_count + left_posture_count + right_posture_count
    subtle_indices = _event_indices_for_families(events, {"TORSO_POSTURE", "WHOLE_BODY_STATE"})
    dominant_side = "left" if (left_arm_count + left_posture_count) >= (right_arm_count + right_posture_count) else "right"
    return {
        "arm_event_indices": arm_indices,
        "arm_event_count": len(arm_indices),
        "bimanual_count": bimanual_count,
        "left_arm_count": left_arm_count,
        "right_arm_count": right_arm_count,
        "left_posture_count": left_posture_count,
        "right_posture_count": right_posture_count,
        "total_arm_signal": total_arm_signal,
        "dominant_side": dominant_side,
        "torso_posture_count": torso_posture_count,
        "state_count": state_count,
        "subtle_state_event_indices": subtle_indices,
        "subtle_state_event_count": len(subtle_indices),
    }


def assign_seeded_prototype(signature: dict[str, Any], events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    events = list(events or [])
    if not events:
        return _no_salient_event_prototype()
    ctx = _prototype_context(signature, events)
    pattern_prototype, pattern_matches = _select_seeded_pattern_prototype(ctx)
    if pattern_prototype is not None:
        return pattern_prototype
    raise RuntimeError("primary pattern tree did not produce a fallback match")


def _cover_primary_events(events: list[dict[str, Any]], signature: dict[str, Any], prototype: dict[str, Any]) -> set[int]:
    pid = _active_proto_id(str(prototype.get("prototype_id", "")))
    covered: set[int] = set()
    locomotion = signature.get("locomotion") or {}
    vertical = signature.get("vertical") or {}
    limbs = signature.get("limb_coordination") or {}
    support = signature.get("support_gait") or {}
    cover_specs = _registry_map("primary_action_metadata", "primary_event_cover")
    spec = cover_specs.get(pid) if isinstance(cover_specs, dict) else None
    if not isinstance(spec, dict):
        return covered
    mode = str(spec.get("mode", ""))

    if mode == "families":
        for family in spec.get("families") or []:
            covered.update(_indices_by_family(events, str(family)))
    elif mode == "ballistic_translation":
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
    elif mode == "vertical_jump":
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
    elif mode == "translating_gait":
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
    elif mode == "prototype_source_indices":
        covered.update(int(x) for x in prototype.get("source_event_indices") or [])
    elif mode == "families_with_weak_locomotion":
        for family in spec.get("families") or []:
            covered.update(_indices_by_family(events, str(family)))
        weak_max = float(spec.get("weak_locomotion_max_m", 0.0) or 0.0)
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_LOCOMOTION" and _magnitude(evt) < weak_max:
                covered.add(int(evt["event_index"]))
    elif mode == "rotation":
        covered.update(_indices_by_family(events, "WHOLE_BODY_ROTATION"))
        covered.update(locomotion.get("turn_event_indices") or [])
    elif mode == "limb_indices":
        covered.update(limbs.get(str(spec.get("limb_key", ""))) or [])
    elif mode == "low_body_repetition":
        covered.update(
            int(evt["event_index"])
            for evt in events
            if evt.get("super_family") == "WHOLE_BODY_POSTURE"
            and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        )
        for evt in events:
            if evt.get("super_family") == "WHOLE_BODY_VERTICAL":
                covered.add(int(evt["event_index"]))
        if spec.get("include_bimanual"):
            covered.update(
                int(evt["event_index"])
                for evt in events
                if evt.get("super_family") == "BIMANUAL_PERIODIC"
            )
    return {int(x) for x in covered}


def _primary_source_event_indices(events: list[dict[str, Any]], signature: dict[str, Any], prototype: dict[str, Any]) -> set[int]:
    pid = _active_proto_id(str(prototype.get("prototype_id", "")))
    if prototype.get("source_event_indices") is not None:
        return {int(x) for x in prototype.get("source_event_indices") or []}

    locomotion = signature.get("locomotion") or {}
    vertical = signature.get("vertical") or {}
    rotation = signature.get("rotation") or {}
    limbs = signature.get("limb_coordination") or {}
    cover_specs = _registry_map("primary_action_metadata", "primary_event_cover")
    spec = cover_specs.get(pid) if isinstance(cover_specs, dict) else None
    mode = str((spec or {}).get("mode", ""))

    if mode == "translating_gait":
        return {int(seg["event_index"]) for seg in locomotion.get("segments") or []}
    if mode == "vertical_jump":
        return set(int(idx) for idx in vertical.get("event_indices") or [])
    if mode == "rotation":
        return set(int(idx) for idx in rotation.get("event_indices") or [])
    if mode == "ballistic_translation":
        source = set()
        primary_loco = _event_by_index(events, prototype.get("primary_locomotion_event_index"))
        if primary_loco is None:
            primary_loco = _event_by_index(events, locomotion.get("best_event_index"))
        if primary_loco:
            source.add(int(primary_loco["event_index"]))
        source.update(int(idx) for idx in vertical.get("event_indices") or [])
        return source
    if mode == "families":
        families = {str(family) for family in (spec or {}).get("families") or []}
        if pid == "IN_PLACE_GAIT":
            families = families.intersection({"WHOLE_BODY_VERTICAL", "LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC", "BIMANUAL_PERIODIC"})
        return set(_event_indices_for_families(events, families)) if families else set()
    if mode == "families_with_weak_locomotion":
        families = {str(family) for family in (spec or {}).get("families") or []}
        families.discard("WHOLE_BODY_LOCOMOTION")
        return set(_event_indices_for_families(events, families)) if families else set()
    if mode == "limb_indices":
        return set(int(idx) for idx in limbs.get(str((spec or {}).get("limb_key", ""))) or [])
    if mode == "low_body_repetition":
        source = {
            int(evt["event_index"])
            for evt in events
            if evt.get("super_family") == "WHOLE_BODY_POSTURE"
            and str(evt.get("cluster_id", "")) == "WB_SQUAT_HOLD"
        }
        if (spec or {}).get("include_bimanual"):
            source.update(_event_indices_for_families(events, {"BIMANUAL_PERIODIC"}))
        return source
    return set()


def _apply_semantic_dominance(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep low-level trajectory evidence in canonical slots, but hide it from probe prompts
    when a stronger composed semantic family already explains the same span."""
    dominant_prototypes = _registry_set("dominance", "dominant_prototypes")
    hideable_targets = _registry_set("dominance", "hideable_targets")
    dominant_groups = {
        str(name): {str(item) for item in value}
        for name, value in _registry_map("dominance", "dominant_groups").items()
        if isinstance(value, list)
    }
    hide_by_dominant = {
        str(name): {str(item) for item in value}
        for name, value in _registry_map("dominance", "hide_by_dominant").items()
        if isinstance(value, list)
    }

    def hide_targets_for(dominant_pid: str) -> set[str]:
        out = set(hide_by_dominant.get(dominant_pid, set()))
        for group_name, members in dominant_groups.items():
            if dominant_pid in members:
                out.update(hide_by_dominant.get(group_name, set()))
        return out

    dominant = [
        action for action in actions
        if str(action.get("prototype_id", "")) in dominant_prototypes
    ]
    if not dominant:
        return actions
    out: list[dict[str, Any]] = []
    for action in actions:
        item = dict(action)
        pid = str(item.get("prototype_id", ""))
        if pid in hideable_targets:
            for dom in dominant:
                dom_pid = str(dom.get("prototype_id", ""))
                if _overlap_ratio(item, dom) < 0.20:
                    continue
                if pid in hide_targets_for(dom_pid):
                    item["probe_visible"] = False
                    item["hidden_by_semantic_family"] = dom_pid
                    break
        out.append(item)
    return out


def _drop_redundant_fallback_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_redundant = _registry_set("fallback_actions", "redundant")
    explanatory_indices: set[int] = set()
    for action in actions:
        if str(action.get("prototype_id", "")) in fallback_redundant:
            continue
        explanatory_indices.update(int(x) for x in action.get("covered_event_indices") or [])

    out: list[dict[str, Any]] = []
    for action in actions:
        pid = str(action.get("prototype_id", ""))
        if pid not in fallback_redundant:
            out.append(action)
            continue
        source_indices = set(int(x) for x in action.get("source_event_indices") or action.get("covered_event_indices") or [])
        if not source_indices and explanatory_indices:
            continue
        if source_indices and source_indices.issubset(explanatory_indices):
            continue
        out.append(action)
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
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "bilateral_rhythmic_count_prototypes"):
        repeat_count = _cyclic_coordination_count(events, signature.get("vertical", {}))
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
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "primary_index_from_rotation"):
        primary_idx = signature.get("rotation", {}).get("best_event_index")
    primary_evt = _event_by_index(events, primary_idx)
    if primary_evt:
        primary_span = list(_span(primary_evt))
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "covered_span_prototypes") and covered:
        covered_events = [_event_by_index(events, idx) for idx in covered]
        spans = [_span(evt) for evt in covered_events if evt]
        if spans:
            primary_span = [min(s for s, _ in spans), max(e for _, e in spans)]
    primary_source_indices = _primary_source_event_indices(events, signature, prototype)
    primary_action = {
        **prototype,
        "span": primary_span,
        "covered_event_indices": sorted(covered),
    }
    if primary_source_indices:
        primary_action["source_event_indices"] = sorted(primary_source_indices)
    covered_event_list = [evt for evt in events if int(evt["event_index"]) in covered]
    if covered_event_list:
        primary_action["covered_event_family_counts"] = _event_family_counts(covered_event_list)
        primary_action["covered_event_cluster_counts"] = _event_cluster_counts(covered_event_list)
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "source_event_detail_prototypes"):
        source_indices = primary_source_indices
        source_events = [evt for evt in events if int(evt["event_index"]) in source_indices]
        primary_action["source_event_count"] = len(source_events)
        primary_action["source_event_indices"] = [int(evt["event_index"]) for evt in source_events]
        primary_action["source_event_family_counts"] = _event_family_counts(source_events)
        primary_action["source_event_cluster_counts"] = _event_cluster_counts(source_events)
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "locomotion_metric_prototypes"):
        primary_action["speed"] = signature.get("locomotion", {}).get("speed")
        primary_action["distance_m"] = signature.get("locomotion", {}).get("distance_m")
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "ballistic_metric_prototypes"):
        primary_action["distance_m"] = prototype.get("primary_locomotion_distance_m")
        primary_action["speed"] = prototype.get("primary_locomotion_speed")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "rotation_metric_prototypes"):
        primary_action["angle_deg"] = signature.get("rotation", {}).get("angle_deg")
        primary_action["angle_bin"] = signature.get("rotation", {}).get("angle_bin")
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "vertical_count_prototypes"):
        primary_action["count"] = prototype.get("count") or signature.get("vertical", {}).get("repeat_count")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "squat_vertical_prototypes"):
        primary_action["count"] = prototype.get("count")
        primary_action["vertical_amplitude_m"] = signature.get("vertical", {}).get("max_amplitude_m")
        primary_action["mean_vertical_amplitude_m"] = signature.get("vertical", {}).get("mean_amplitude_m")
    if _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "bilateral_rhythmic_gesture_prototypes"):
        primary_action["turn_count"] = prototype.get("turn_count")
        primary_action["locomotion_segment_count"] = prototype.get("locomotion_segment_count")
        primary_action["raise_spread_count"] = prototype.get("raise_spread_count")
        primary_action["bimanual_count"] = prototype.get("bimanual_count")
        primary_action["lexical_alias_candidates"] = prototype.get("lexical_alias_candidates")
    actions = [primary_action]
    if not _proto_in_group(str(prototype.get("prototype_id", "")), "primary_action_metadata", "skip_bilateral_rhythmic_secondary"):
        for action in _cyclic_bilateral_coordination_actions(events, covered):
            covered.update(int(x) for x in action.get("covered_event_indices") or [])
            actions.append(action)
    semantic_actions = _semantic_candidate_actions(events, covered)
    for action in semantic_actions:
        covered.update(int(x) for x in action.get("covered_event_indices") or [])
    actions.extend(semantic_actions)
    for action in _post_vertical_translation_recovery_actions(events, covered):
        covered.update(int(x) for x in action.get("covered_event_indices") or [])
        actions.append(action)
    actions.extend(_bimanual_contact_actions(events, covered))
    actions.extend(_secondary_actions(events, covered))
    for action in _vertical_impulse_translation_pair_actions(events, covered):
        if _active_proto_id(str(action.get("prototype_id", ""))) != "WEAK_BALLISTIC_CANDIDATE":
            covered.update(int(x) for x in action.get("covered_event_indices") or [])
        actions.append(action)
    state = signature.get("state") or {}
    if state.get("terminal") == "still":
        state_idx = state.get("best_event_index")
        state_evt = _event_by_index(events, state_idx)
        if state_evt:
            terminal_node = event_proxy_for_event(state_evt)
            proto_id, name_hint, direction, confidence = (
                event_proxy_action_fields(terminal_node)
                if terminal_node
                else ("", "stand_still", "still", float(state_evt.get("confidence", 0.0)))
            )
            covered.add(int(state_evt["event_index"]))
            actions.append(
                {
                    "prototype_id": proto_id,
                    "name_hint": name_hint,
                    "primary_direction": direction,
                    "confidence": max(float(state_evt.get("confidence", 0.0)), confidence),
                    "span": list(_span(state_evt)),
                    "covered_event_indices": [int(state_evt["event_index"])],
                }
            )
    actions = _apply_semantic_dominance(actions)
    actions = _drop_redundant_fallback_actions(actions)
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
