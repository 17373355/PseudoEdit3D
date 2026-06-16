from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .aml_pattern_tree import action_pattern_metadata_for_node, event_proxy_action_fields, event_proxy_for_event
from .aml_proto_registry import active_proto_id


_SIDECAR_PATH = Path(__file__).with_name("aml_semantic_alias_sidecar.json")


@lru_cache(maxsize=1)
def semantic_alias_sidecar() -> dict[str, Any]:
    return json.loads(_SIDECAR_PATH.read_text(encoding="utf-8"))


def _caption_text(captions: list[str] | str | None) -> str:
    if captions is None:
        return ""
    if isinstance(captions, str):
        return captions.lower()
    return " ".join(str(item) for item in captions).lower()


def _rule_matches_caption(rule: dict[str, Any], text: str) -> bool:
    if not text:
        return False
    for pattern in rule.get("negative_caption_patterns") or []:
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            return False
    for pattern in rule.get("caption_patterns") or []:
        if re.search(str(pattern), text, flags=re.IGNORECASE):
            return True
    return False


def matched_caption_alias_rules(captions: list[str] | str | None) -> list[dict[str, Any]]:
    text = _caption_text(captions)
    rules = [
        dict(rule)
        for rule in semantic_alias_sidecar().get("rules") or []
        if isinstance(rule, dict) and _rule_matches_caption(rule, text)
    ]
    rules.sort(key=lambda rule: (-int(rule.get("priority") or 0), str(rule.get("alias_id") or "")))
    return rules


def _action_count(action: dict[str, Any]) -> int:
    for key in ("count", "raise_spread_count", "bimanual_count", "source_event_count", "segment_count"):
        value = action.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _span(evt: dict[str, Any]) -> tuple[int, int]:
    return int(evt.get("start_frame", 0)), int(evt.get("end_frame", evt.get("start_frame", 0)))


def _event_magnitude(evt: dict[str, Any]) -> float:
    try:
        return float(evt.get("magnitude") or abs(float(evt.get("signed_delta") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _rule_compatible_with_action(rule: dict[str, Any], action: dict[str, Any]) -> bool:
    compatible = {active_proto_id(str(item)) for item in rule.get("compatible_families") or []}
    if not compatible:
        return False
    action_family = active_proto_id(str(action.get("prototype_id") or action.get("active_prototype_id") or ""))
    if action_family not in compatible:
        return False
    min_count = rule.get("min_action_count")
    if min_count is not None and _action_count(action) < int(min_count):
        return False
    return True


def _rule_compatible_with_event(rule: dict[str, Any], evt: dict[str, Any]) -> bool:
    node = event_proxy_for_event(evt)
    if node is None:
        return False
    proto_id, _, _, _ = event_proxy_action_fields(node)
    if active_proto_id(proto_id) not in {active_proto_id(str(item)) for item in rule.get("compatible_families") or []}:
        return False
    min_magnitude = rule.get("min_event_magnitude")
    if min_magnitude is not None and _event_magnitude(evt) < float(min_magnitude):
        return False
    return True


def _action_start(action: dict[str, Any]) -> int:
    span = action.get("span")
    if isinstance(span, list) and span:
        try:
            return int(span[0])
        except (TypeError, ValueError):
            return 0
    return 0


def _candidate_sort_key(rule: dict[str, Any], action_index: int, action: dict[str, Any]) -> tuple[int, int, float, int, int]:
    family_order = [active_proto_id(str(item)) for item in rule.get("compatible_families") or []]
    action_family = active_proto_id(str(action.get("prototype_id") or action.get("active_prototype_id") or ""))
    try:
        family_rank = family_order.index(action_family)
    except ValueError:
        family_rank = len(family_order)
    return (
        family_rank,
        1 if action.get("probe_visible") is False else 0,
        -float(action.get("confidence") or 0.0),
        -_action_count(action),
        _action_start(action) * 10000 + action_index,
    )


def _alias_payload(rule: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    action_conf = float(action.get("confidence") or 0.0)
    rule_conf = float(rule.get("confidence") or 0.0)
    return {
        "alias_id": str(rule.get("alias_id") or ""),
        "label": str(rule.get("label") or rule.get("alias_id") or ""),
        "clause": str(rule.get("clause") or rule.get("label") or ""),
        "confidence": round(min(0.92, max(action_conf, 0.1) * max(rule_conf, 0.1)), 4),
        "source": "caption_compatible_with_geometry",
        "priority": int(rule.get("priority") or 0),
        "matched_family": active_proto_id(str(action.get("prototype_id") or action.get("active_prototype_id") or "")),
    }


def _action_from_event_for_rule(rule: dict[str, Any], evt: dict[str, Any]) -> dict[str, Any] | None:
    node = event_proxy_for_event(evt)
    if node is None:
        return None
    proto_id, name_hint, direction, base_confidence = event_proxy_action_fields(node)
    start, end = _span(evt)
    magnitude = _event_magnitude(evt)
    action = {
        "prototype_id": proto_id,
        "name_hint": name_hint,
        "primary_direction": direction,
        "confidence": min(0.82, max(float(base_confidence or 0.0), float(evt.get("confidence") or 0.0))),
        "span": [start, end],
        "semantic_proxy": True,
        "source_event_family": str(evt.get("super_family", "")),
        "source_event_cluster": str(evt.get("cluster_id", "")),
        "source_event_count": 1,
        "source_event_spans": [[start, end]],
        "covered_event_indices": [int(evt["event_index"])],
        "source_event_indices": [int(evt["event_index"])],
        "recovered_for_semantic_alias": str(rule.get("alias_id") or ""),
        **action_pattern_metadata_for_node(node),
    }
    if magnitude > 0.0:
        action["magnitude"] = round(magnitude, 4)
        action["unit"] = evt.get("unit")
    return action


def attach_caption_semantic_aliases(
    actions: list[dict[str, Any]],
    captions: list[str] | str | None,
) -> list[dict[str, Any]]:
    """Attach caption-assisted names to existing geometry actions.

    The alias is only attached when the caption rule and an already-detected
    geometry family agree. This intentionally does not create new actions.
    """

    rules = matched_caption_alias_rules(captions)
    if not rules:
        return [dict(action) for action in actions]

    out: list[dict[str, Any]] = [dict(action) for action in actions]
    used_aliases: set[str] = set()
    used_groups: set[str] = set()
    for rule in rules:
        alias_id = str(rule.get("alias_id") or "")
        alias_group = str(rule.get("alias_group") or alias_id)
        if not alias_id or alias_id in used_aliases or alias_group in used_groups:
            continue
        candidates = [
            (idx, action)
            for idx, action in enumerate(out)
            if not isinstance(action.get("semantic_alias"), dict)
            and action.get("probe_visible") is not False
            and _rule_compatible_with_action(rule, action)
        ]
        if not candidates:
            continue
        idx, action = min(candidates, key=lambda item: _candidate_sort_key(rule, item[0], item[1]))
        item = dict(action)
        item["semantic_alias"] = _alias_payload(rule, item)
        aliases = list(item.get("lexical_alias_candidates") or [])
        if alias_id not in aliases:
            aliases.append(alias_id)
        item["lexical_alias_candidates"] = aliases
        out[idx] = item
        used_aliases.add(alias_id)
        used_groups.add(alias_group)
    return out


def recover_caption_alias_actions(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    captions: list[str] | str | None,
) -> list[dict[str, Any]]:
    rules = matched_caption_alias_rules(captions)
    if not rules:
        return []
    existing_aliases = {
        str((action.get("semantic_alias") or {}).get("alias_id") or "")
        for action in actions
        if isinstance(action.get("semantic_alias"), dict)
    }
    recovered: list[dict[str, Any]] = []
    used_event_indices: set[int] = set()
    for rule in rules:
        alias_id = str(rule.get("alias_id") or "")
        if not alias_id or alias_id in existing_aliases:
            continue
        candidates = [
            evt for evt in events
            if int(evt.get("event_index", -1)) not in used_event_indices
            and _rule_compatible_with_event(rule, evt)
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda evt: (_event_magnitude(evt), _span(evt)[1] - _span(evt)[0]))
        action = _action_from_event_for_rule(rule, best)
        if action is None:
            continue
        aliased = attach_caption_semantic_aliases([action], captions)
        if not aliased or not isinstance(aliased[0].get("semantic_alias"), dict):
            continue
        recovered.append(aliased[0])
        used_event_indices.add(int(best["event_index"]))
        existing_aliases.add(alias_id)
    return recovered


def caption_alias_audit(
    actions: list[dict[str, Any]],
    captions: list[str] | str | None,
) -> dict[str, Any]:
    rules = matched_caption_alias_rules(captions)
    return {
        "matched_alias_ids": [str(rule.get("alias_id") or "") for rule in rules],
        "attached_aliases": [
            dict(action["semantic_alias"])
            for action in actions
            if isinstance(action.get("semantic_alias"), dict)
        ],
    }
