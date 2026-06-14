from __future__ import annotations

from typing import Any

from .aml_family_taxonomy import family_taxonomy_metadata
from .aml_pattern_tree import action_pattern_metadata_for_family
from .aml_proto_registry import active_proto_id, proto_in_group, registry_map


_active_proto_id = active_proto_id
_proto_in_group = proto_in_group
_registry_map = registry_map


def _surface_transform(value: str, transform: str | None) -> str:
    if transform == "underscore_to_space":
        return value.replace("_", " ")
    return value


def _surface_field_value(action: dict[str, Any], spec: dict[str, Any]) -> str:
    path = str(spec.get("path", ""))
    value = action.get(path) if path else None
    if value is None:
        value = spec.get("default")
    if value is None:
        return ""
    text = str(value)
    mapping = spec.get("map")
    if isinstance(mapping, dict):
        text = str(mapping.get(text, text))
    return _surface_transform(text, spec.get("transform"))


def _resolve_surface_alias_spec(action: dict[str, Any], spec: dict[str, Any]) -> str:
    if not spec:
        return ""
    if "template" in spec:
        fields = {
            str(key): _surface_field_value(action, dict(value))
            for key, value in (spec.get("fields") or {}).items()
            if isinstance(value, dict)
        }
        if fields and all(value for value in fields.values()):
            return str(spec["template"]).format(**fields)
        if spec.get("fallback") is not None:
            return str(spec["fallback"])
    if "field" in spec:
        value = action.get(str(spec["field"]))
        if value is None:
            value = spec.get("default") or spec.get("fallback")
        if value is not None:
            return _surface_transform(str(value), spec.get("transform"))
    if spec.get("default") is not None:
        return str(spec["default"])
    if spec.get("fallback") is not None:
        return str(spec["fallback"])
    return ""


def _probe_alias(action: dict[str, Any]) -> str:
    pid = _active_proto_id(str(action.get("prototype_id", "")))
    alias_config = _registry_map("probe_aliases")
    by_prototype = alias_config.get("by_prototype") or {}
    spec = by_prototype.get(pid) if isinstance(by_prototype, dict) else None
    if not isinstance(spec, dict):
        spec = alias_config.get("default") if isinstance(alias_config.get("default"), dict) else {}
    alias = _resolve_surface_alias_spec(action, dict(spec))
    return alias or str(action.get("name_hint", "motion")).replace("_", " ")


def _semantic_family_descriptor(action: dict[str, Any]) -> dict[str, Any]:
    source_pid = str(action.get("prototype_id") or _registry_map("fallback_entrypoints").get("unknown_source_family") or "UNKNOWN")
    pid = _active_proto_id(source_pid)
    confidence = float(action.get("confidence") or 0.0)
    unknown_semantic_families = _registry_map("semantic_family_status", "unknown")
    if pid in unknown_semantic_families:
        status = "unknown"
        family_id = str(unknown_semantic_families[pid])
        label = "unknown semantic family"
    elif _proto_in_group(pid, "semantic_family_status", "proxy") or pid.endswith("_PROXY"):
        status = "proxy"
        family_id = pid
        label = str(action.get("name_hint") or pid).replace("_", " ")
    elif _proto_in_group(pid, "semantic_family_status", "candidate") or pid.endswith("_CANDIDATE") or action.get("semantic_proxy"):
        status = "candidate"
        family_id = pid
        label = str(action.get("name_hint") or pid).replace("_", " ")
    else:
        status = "stable"
        family_id = pid
        label = str(action.get("name_hint") or pid).replace("_", " ")

    if action.get("semantic_proxy"):
        source = "semantic_joint_proxy"
    elif pid in unknown_semantic_families:
        source = "fallback_event_signature"
    else:
        source = "layer3_event_signature"
    taxonomy = family_taxonomy_metadata(family_id)
    return {
        "family_id": family_id,
        "source_family": source_pid,
        "active_family": pid,
        "status": status,
        "label": label,
        "label_confidence": round(confidence, 4),
        "motion_only": True,
        "source": source,
        "probe_visible": action.get("probe_visible", True) is not False,
        "taxonomy_parent_id": taxonomy.get("taxonomy_parent_id"),
        "taxonomy_parent_label": taxonomy.get("taxonomy_parent_label"),
        "taxonomy_recoverability": taxonomy.get("taxonomy_recoverability"),
        "taxonomy_evidence_axes": taxonomy.get("taxonomy_evidence_axes") or [],
        "taxonomy_secondary_parent_ids": taxonomy.get("taxonomy_secondary_parent_ids") or [],
        "ambiguity_boundary": taxonomy.get("ambiguity_boundary"),
        "pattern_node_id": action.get("pattern_node_id"),
        "pattern_path": action.get("pattern_path") or [],
        "pattern_taxonomy_parent_id": action.get("pattern_taxonomy_parent_id"),
    }


def _numeric_range(value: float, unit: str, status: str) -> list[float]:
    if unit == "frame":
        margin = 2.0 if status == "stable" else 4.0
    elif unit == "count":
        margin = 0.0 if status == "stable" else 1.0
    elif unit == "deg":
        margin = max(5.0, abs(value) * (0.06 if status == "stable" else 0.12))
    elif unit == "rad":
        margin = max(0.05, abs(value) * (0.06 if status == "stable" else 0.12))
    elif unit == "score":
        margin = 0.05 if status == "stable" else 0.10
    else:
        margin = max(0.03, abs(value) * (0.08 if status == "stable" else 0.15))
    lo = max(0.0, value - margin) if unit in {"m", "count", "frame", "score"} else value - margin
    hi = value + margin
    return [round(lo, 4), round(hi, 4)]


def _slot_confidence(action: dict[str, Any], semantic_family: dict[str, Any]) -> float:
    confidence = float(action.get("confidence") or semantic_family.get("label_confidence") or 0.0)
    status = str(semantic_family.get("status", "stable"))
    if status == "candidate":
        confidence *= 0.90
    elif status == "proxy":
        confidence *= 0.82
    elif status == "unknown":
        confidence *= 0.60
    return round(max(0.0, min(1.0, confidence)), 4)


def _approx_slot(
    action: dict[str, Any],
    semantic_family: dict[str, Any],
    key: str,
    *,
    unit: str | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    value = action.get(key)
    if value is None:
        return None
    status = str(semantic_family.get("status", "stable"))
    slot_source = source or ("semantic_joint_proxy" if action.get("semantic_proxy") else "layer3_event_signature")
    confidence = _slot_confidence(action, semantic_family)
    if isinstance(value, bool):
        return {
            "value": bool(value),
            "confidence": confidence,
            "source": slot_source,
            "quality": "boolean_observation",
        }
    if isinstance(value, int) and unit == "count":
        numeric = float(value)
        return {
            "value": int(value),
            "range": [int(x) for x in _numeric_range(numeric, "count", status)],
            "unit": "count",
            "confidence": confidence,
            "source": slot_source,
            "quality": "approximate_event_count" if status != "stable" else "event_count",
        }
    if isinstance(value, (int, float)) and unit is not None:
        numeric = float(value)
        return {
            "value": round(numeric, 4),
            "range": _numeric_range(numeric, unit, status),
            "unit": unit,
            "confidence": confidence,
            "source": slot_source,
            "quality": "motion_estimate" if status == "stable" else f"{status}_motion_estimate",
        }
    return {
        "value": value,
        "confidence": confidence,
        "source": slot_source,
        "quality": "categorical_estimate",
    }


def _approx_slots(action: dict[str, Any], semantic_family: dict[str, Any]) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    span = action.get("span")
    if isinstance(span, list) and len(span) == 2:
        status = str(semantic_family.get("status", "stable"))
        margin = 2 if status == "stable" else 4
        slots["span"] = {
            "value": [int(span[0]), int(span[1])],
            "unit": "frame",
            "frame_margin": margin,
            "confidence": _slot_confidence(action, semantic_family),
            "source": "union_of_covered_event_spans",
            "quality": "temporal_span_estimate",
        }
    numeric_units = {
        "count": "count",
        "segment_count": "count",
        "source_event_count": "count",
        "turn_count": "count",
        "locomotion_segment_count": "count",
        "raise_spread_count": "count",
        "bimanual_count": "count",
        "left_arm_count": "count",
        "right_arm_count": "count",
        "distance_m": "m",
        "path_length_m": "m",
        "root_height_gain_m": "m",
        "vertical_amplitude_m": "m",
        "mean_vertical_amplitude_m": "m",
        "angle_deg": "deg",
        "curvature_rad": "rad",
        "circle_score": "score",
    }
    for key, unit in numeric_units.items():
        slot = _approx_slot(action, semantic_family, key, unit=unit)
        if slot is not None:
            slots[key] = slot
    raw_unit = action.get("unit")
    if action.get("magnitude") is not None and raw_unit in {"m", "deg", "rad", "ratio", "score"}:
        slot_unit = "score" if raw_unit == "ratio" else str(raw_unit)
        slot = _approx_slot(action, semantic_family, "magnitude", unit=slot_unit)
        if slot is not None:
            slots["magnitude"] = slot
    for key in ("primary_direction", "speed", "angle_bin", "dominant_side"):
        value = action.get(key)
        if value is not None and str(value) not in {"", "none", "unknown"}:
            slot = _approx_slot(action, semantic_family, key)
            if slot is not None:
                alias = "direction" if key == "primary_direction" else key
                slots[alias] = slot
    return slots


def _canonical_action(action: dict[str, Any]) -> dict[str, Any]:
    semantic_family = _semantic_family_descriptor(action)
    approx_slots = _approx_slots(action, semantic_family)
    canonical_id = _active_proto_id(str(action.get("prototype_id") or _registry_map("fallback_entrypoints").get("unknown_source_family") or "UNKNOWN"))
    slots: dict[str, Any] = {
        "span": action.get("span"),
        "direction": action.get("primary_direction"),
        "confidence": action.get("confidence"),
        "covered_event_indices": list(action.get("covered_event_indices") or []),
        "semantic_family_id": semantic_family["family_id"],
        "semantic_family_status": semantic_family["status"],
        "approx_slots": approx_slots,
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
        "lexical_alias_candidates",
        "source_event_clusters",
        "turn_count",
        "locomotion_segment_count",
        "raise_spread_count",
        "bimanual_count",
        "left_arm_count",
        "right_arm_count",
        "source_event_cluster",
        "source_event_family",
        "source_event_indices",
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
        "hidden_by_pattern_cover",
        "pattern_evidence",
        "phase_order",
        "magnitude",
        "unit",
        "source_event_family_counts",
        "source_event_cluster_counts",
        "covered_event_family_counts",
        "covered_event_cluster_counts",
        "pattern_node_id",
        "pattern_path",
        "pattern_taxonomy_parent_id",
    ):
        if action.get(key) is not None:
            slots[key] = action.get(key)
    return {
        "canonical_id": canonical_id,
        "family": canonical_id,
        "source_family": str(action.get("prototype_id") or _registry_map("fallback_entrypoints").get("unknown_source_family") or "UNKNOWN"),
        "probe_alias": _probe_alias(action),
        "surface_name_hint": action.get("name_hint"),
        "semantic_family": semantic_family,
        "approx_slots": approx_slots,
        "slots": slots,
    }


def _attach_action_metadata(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in actions:
        item = dict(action)
        item["active_prototype_id"] = _active_proto_id(str(item.get("prototype_id", "")))
        if not item.get("pattern_node_id"):
            item.update(
                action_pattern_metadata_for_family(
                    item["active_prototype_id"],
                    preferred_node_types=("primary", "composed_candidate", "event_proxy"),
                )
            )
        item["probe_alias"] = _probe_alias(item)
        item["semantic_family"] = _semantic_family_descriptor(item)
        item["approx_slots"] = _approx_slots(item, item["semantic_family"])
        item["canonical"] = _canonical_action(item)
        out.append(item)
    return out
