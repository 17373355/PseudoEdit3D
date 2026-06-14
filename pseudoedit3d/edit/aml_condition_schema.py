from __future__ import annotations

from typing import Any

from .aml_proto_registry import active_proto_id, registry_map


STATUS_CONDITION_WEIGHTS = {
    "stable": 1.0,
    "candidate": 0.7,
    "proxy": 0.5,
    "unknown": 0.0,
}


def required_approx_slots(family_id: str) -> tuple[str, ...]:
    schema = registry_map("condition_schema")
    required = schema.get("required_approx_slots") or {}
    default = schema.get("default_required_approx_slots") or ["span"]
    slots = required.get(active_proto_id(family_id)) if isinstance(required, dict) else None
    if slots is None:
        slots = default
    return tuple(str(item) for item in slots)


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
