"""Mine composition-closure candidates from v4 coord-role coactivations.

This audit sits above the v4 Motion-BPE coord-role output. It does not add
runtime AML rules. It groups frequent coactivated role sets into reviewable
composition candidates, so complete multi-part patterns can be inspected
separately from reusable components.

Typical use:
    python scripts/build_v4_coord_role_composition_closure.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BPE_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_3k")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_closure_review")

CSV_FIELDS = [
    "rank",
    "candidate_id",
    "recommendation",
    "composition_scope",
    "support_cases",
    "occurrences",
    "score",
    "canonical_role_items",
    "channels",
    "zones",
    "caption_aliases",
    "discriminative_role_coverage",
    "suppressed_discriminative_roles",
    "promotion_blockers",
    "reason",
]

DISCRIMINATIVE_ROLE_NAMES = {
    "inversion_or_acrobatics",
    "arm_large_arc_or_orbit",
    "arm_vertical_cycle",
    "bimanual_spread_cycle",
    "hand_head_proximity",
    "arm_reach_retract",
    "leg_lateral_cycle",
    "leg_forward_action",
    "body_level_cycle",
    "body_level_down",
    "body_level_up",
    "body_level_low",
    "body_low_posture",
    "vertical_low_body_transition",
    "root_rotation",
    "floor_low_horizontal_support",
    "hand_floor_low_support",
    "inverted_support",
}

ALIAS_COMPATIBLE_SCOPES = {
    "jumping_jack": {"full_upper_lower_body_candidate"},
    "cartwheel": {"inversion_acrobatic_candidate"},
    "swim_like_motion": {"floor_prone_or_mime_candidate"},
    "fly_like_motion": {"floor_prone_or_mime_candidate"},
    "sit_down": {"body_level_transition_component", "full_upper_lower_body_candidate"},
    "sit_down_stand_up": {"body_level_transition_component", "full_upper_lower_body_candidate"},
    "martial_arts": {"full_upper_lower_body_candidate"},
    "ballet_dance": {"full_upper_lower_body_candidate"},
    "cheer_dance": {"full_upper_lower_body_candidate"},
    "basketball_dribble": {"full_upper_lower_body_candidate", "upper_lower_coordination_component"},
    "tennis_like": {"upper_lower_coordination_component", "upper_vertical_coordination_component", "full_upper_lower_body_candidate"},
    "jump_rope": {"full_upper_lower_body_candidate"},
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _case_caption_index(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if not cache_path or not cache_path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            index[case_id] = {
                "caption_texts": list(row.get("caption_texts") or []),
                "caption_alias_ids": list(row.get("caption_alias_ids") or []),
            }
    return index


def _record_cache_from_summary(summary_path: Path) -> Path | None:
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    cache = summary.get("record_cache") or {}
    cache_path = cache.get("record_cache")
    return Path(cache_path) if cache_path else None


def _parse_coord_sig(symbol: str) -> list[str]:
    if not symbol.startswith("COORD_SIG[") or not symbol.endswith("]"):
        return []
    body = symbol[len("COORD_SIG[") : -1]
    return [part for part in body.split("+") if ":" in part]


def _canonical_role(role: str, channel: str) -> str | None:
    role = role.strip()
    if role == "low_body_posture":
        if channel == "whole_body_vertical":
            return "vertical_low_body_transition"
        if channel == "whole_body_state":
            return "body_low_posture"
        if channel == "torso":
            return "torso_low_body_context"
        if channel in {"left_arm", "right_arm", "bimanual"}:
            return None
        return "low_body_posture"
    if role.startswith("leg_lateral"):
        return "leg_lateral_cycle"
    if role.startswith("leg_forward"):
        return "leg_forward_action"
    if "arm_large_arc" in role or "arm_orbit" in role:
        return "arm_large_arc_or_orbit"
    if "bilateral_arm_vertical_cycle" in role:
        return "bimanual_spread_cycle" if channel == "bimanual" else "arm_vertical_cycle"
    if "vertical_coupled_arm" in role:
        return "arm_vertical_cycle"
    if "hand_near_head" in role or "hand_approach_head" in role or "hand_leave_head" in role:
        return "hand_head_proximity"
    if "arm_reach" in role:
        return "arm_reach_retract"
    if role in {"vertical_change", "vertical_impulse"} or role.startswith("vertical_"):
        return "vertical_rhythm"
    if role.startswith("body_level_cycle") or role.endswith("_cycle"):
        return "body_level_cycle"
    if role.startswith("body_level_down"):
        return "body_level_down"
    if role.startswith("body_level_up"):
        return "body_level_up"
    if role.startswith("body_level_low") or role == "low_body_level":
        return "body_level_low"
    if role.startswith("floor_low_horizontal_support"):
        return "floor_low_horizontal_support"
    if role.startswith("hand_floor_low_support"):
        return "hand_floor_low_support"
    if role.startswith("inverted_support"):
        return "inverted_support"
    if role.startswith("body_support_state"):
        return "body_support_state"
    if role.startswith("inversion_or_acrobatics"):
        return "inversion_or_acrobatics"
    if "turn" in role or "rotation" in role:
        return "root_rotation"
    if "loco" in role or "gait" in role:
        return "root_locomotion"
    return role


def _role_item(raw_role: str) -> str | None:
    if ":" not in raw_role:
        return None
    channel, role = raw_role.split(":", 1)
    channel = channel.strip()
    role = role.strip()
    if not channel or not role:
        return None
    canonical = _canonical_role(role, channel)
    if not canonical:
        return None
    return f"{channel}:{canonical}"


def _channel_from_item(item: str) -> str:
    return item.split(":", 1)[0]


def _role_from_item(item: str) -> str:
    return item.split(":", 1)[1] if ":" in item else item


def _zone_from_item(item: str) -> str:
    channel = _channel_from_item(item)
    role = _role_from_item(item)
    if role in {"vertical_rhythm", "vertical_low_body_transition"}:
        return "vertical"
    if role in {"inversion_or_acrobatics", "inverted_support"}:
        return "inversion"
    if role in {"floor_low_horizontal_support", "hand_floor_low_support", "body_support_state"}:
        return "support"
    return _zone(channel)


def _zone(channel: str) -> str:
    if channel in {"left_arm", "right_arm", "bimanual"}:
        return "upper"
    if channel in {"left_leg", "right_leg"}:
        return "lower"
    if channel == "whole_body_vertical":
        return "vertical"
    if channel in {"whole_body_state", "torso"}:
        return "posture"
    if channel in {"root_locomotion", "root_rotation"}:
        return "root"
    if channel == "whole_body_support":
        return "support"
    if channel == "acrobatics_or_inversion":
        return "inversion"
    return channel


def _role_priority(item: str) -> tuple[int, str]:
    role = _role_from_item(item)
    channel = _channel_from_item(item)
    priority = {
        "arm_large_arc_or_orbit": 0,
        "arm_vertical_cycle": 1,
        "bimanual_spread_cycle": 1,
        "leg_lateral_cycle": 2,
        "vertical_rhythm": 3,
        "body_level_cycle": 4,
        "body_level_down": 4,
        "body_level_up": 4,
        "low_body_posture": 5,
        "body_low_posture": 5,
        "torso_low_body_context": 5,
        "vertical_low_body_transition": 5,
        "leg_forward_action": 6,
        "hand_head_proximity": 7,
        "arm_reach_retract": 8,
        "floor_low_horizontal_support": -2,
        "hand_floor_low_support": -2,
        "inverted_support": -2,
        "inversion_or_acrobatics": -1,
    }.get(role, 20)
    return (priority, channel)


def _items_from_token(token: dict[str, Any]) -> tuple[str, ...]:
    raw_roles = [str(role) for role in token.get("member_roles") or []]
    if not raw_roles:
        raw_roles = _parse_coord_sig(str(token.get("symbol") or ""))
    items = sorted({item for role in raw_roles if (item := _role_item(role))}, key=_role_priority)
    return tuple(items)


def _is_discriminative_item(item: str) -> bool:
    return _role_from_item(item) in DISCRIMINATIVE_ROLE_NAMES


def collect_occurrences(
    sequence_path: Path,
    caption_index: dict[str, dict[str, Any]],
    *,
    max_transaction_items: int,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    with sequence_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("view") != "coactivation":
                continue
            case_id = str(row.get("case_id") or "")
            caption = caption_index.get(case_id) or {}
            for token in row.get("tokens") or []:
                items = _items_from_token(token)
                if len(items) < 2:
                    continue
                if len(items) > max_transaction_items:
                    items = tuple(sorted(items, key=_role_priority)[:max_transaction_items])
                occurrences.append(
                    {
                        "case_id": case_id,
                        "span": token.get("span") or [],
                        "symbol": token.get("symbol") or "",
                        "items": items,
                        "channels": sorted({_channel_from_item(item) for item in items}),
                        "zones": sorted({_zone(_channel_from_item(item)) for item in items}),
                        "geometry_clusters": sorted(str(x) for x in token.get("geometry_clusters") or []),
                        "raw_geometry_clusters": sorted(str(x) for x in token.get("raw_geometry_clusters") or []),
                        "exact_roles": list(token.get("member_roles") or _parse_coord_sig(str(token.get("symbol") or ""))),
                        "caption": (caption.get("caption_texts") or [""])[0],
                        "caption_alias_ids": list(caption.get("caption_alias_ids") or []),
                    }
                )
    return occurrences


def _itemset_score(itemset: tuple[str, ...], support_cases: int) -> tuple[float, str, str]:
    channels = {_channel_from_item(item) for item in itemset}
    zones = {_zone_from_item(item) for item in itemset}
    roles = {_role_from_item(item) for item in itemset}
    has_upper = "upper" in zones
    has_lower = "lower" in zones
    has_vertical = "vertical" in zones
    has_posture = "posture" in zones
    has_root = "root" in zones
    has_inversion = "inversion" in zones
    has_support = "support" in zones
    has_floor_support = bool({"floor_low_horizontal_support", "hand_floor_low_support"} & roles)
    has_inverted_support = "inverted_support" in roles
    has_both_arms = {"left_arm", "right_arm"}.issubset(channels) or "bimanual" in channels
    has_both_legs = {"left_leg", "right_leg"}.issubset(channels)

    score = math.log1p(support_cases)
    score += 0.75 * len(zones) + 0.15 * len(channels)
    if has_inversion:
        score += 2.5
    if has_support:
        score += 1.6
    if has_floor_support and has_upper:
        score += 1.0
    if has_inverted_support:
        score += 2.0
    if has_upper and has_lower:
        score += 1.8
    if has_upper and has_vertical:
        score += 1.2
    if has_lower and has_vertical:
        score += 1.0
    if has_posture and (has_lower or has_vertical):
        score += 0.8
    if has_both_arms:
        score += 0.6
    if has_both_legs:
        score += 0.8
    if {"arm_large_arc_or_orbit", "arm_vertical_cycle", "bimanual_spread_cycle"} & roles and has_lower:
        score += 0.8
    if "hand_head_proximity" in roles and (has_posture or has_vertical or has_lower):
        score += 0.6
    if "arm_reach_retract" in roles and (has_posture or has_vertical or has_lower):
        score += 0.4
    if len(zones) == 1 and not (has_both_arms or has_both_legs):
        score -= 1.6

    if has_floor_support and (has_upper or has_posture):
        scope = "floor_prone_or_mime_candidate"
        reason = "floor-support evidence is composed with upper-body or posture evidence"
    elif has_inverted_support and (has_upper or has_lower or has_posture or has_inversion):
        scope = "inversion_acrobatic_candidate"
        reason = "inverted support evidence is composed with limb, posture, or acrobatic evidence"
    elif has_inversion and (has_upper or has_lower or has_posture):
        scope = "inversion_acrobatic_candidate"
        reason = "inversion/acrobatic evidence is composed with limb or posture evidence"
    elif has_upper and has_lower and (has_vertical or has_posture):
        scope = "full_upper_lower_body_candidate"
        reason = "upper/lower evidence is composed with vertical or posture evidence"
    elif has_upper and has_vertical:
        scope = "upper_vertical_coordination_component"
        reason = "upper-body evidence is coordinated with vertical body motion"
    elif has_upper and has_lower:
        scope = "upper_lower_coordination_component"
        reason = "upper and lower limbs coactivate without enough whole-body evidence"
    elif has_lower and (has_vertical or has_posture):
        scope = "lower_body_transition_component"
        reason = "lower-limb evidence is coordinated with vertical/posture evidence"
    elif has_posture and has_vertical:
        scope = "body_level_transition_component"
        reason = "body level and vertical/posture evidence coactivate"
    elif has_both_arms:
        scope = "bilateral_upper_component"
        reason = "bilateral upper-body component"
    elif has_both_legs:
        scope = "bilateral_lower_component"
        reason = "bilateral lower-body component"
    else:
        scope = "local_or_ambiguous_component"
        reason = "not enough cross-zone evidence for full composition"
    return round(score, 4), scope, reason


def mine_closure_candidates(
    occurrences: list[dict[str, Any]],
    *,
    min_support_cases: int,
    min_specific_support_cases: int,
    max_itemset_size: int,
    max_candidates: int,
    examples_per_candidate: int,
    min_scope_candidates: int,
) -> dict[str, Any]:
    occurrence_counts: Counter[tuple[str, ...]] = Counter()
    case_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    alias_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    geometry_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    exact_item_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for occ in occurrences:
        items = tuple(occ["items"])
        exact_items = set(items)
        for exact_role in occ.get("exact_roles") or []:
            item = _role_item(str(exact_role))
            if item:
                exact_items.add(item)
        max_size = min(max_itemset_size, len(items))
        for size in range(2, max_size + 1):
            for itemset in itertools.combinations(items, size):
                occurrence_counts[itemset] += 1
                case_sets[itemset].add(str(occ["case_id"]))
                for alias in occ.get("caption_alias_ids") or []:
                    alias_counts[itemset][str(alias)] += 1
                for geo in occ.get("geometry_clusters") or []:
                    geometry_counts[itemset][str(geo)] += 1
                for item in exact_items:
                    if _is_discriminative_item(item):
                        exact_item_counts[itemset][item] += 1

    rows: list[dict[str, Any]] = []
    for itemset, occurrence_count in occurrence_counts.items():
        support_cases = len(case_sets[itemset])
        specificity = _specificity_bucket(itemset, support_cases, min_support_cases, min_specific_support_cases)
        if specificity == "below_threshold":
            continue
        score, scope, reason = _itemset_score(itemset, support_cases)
        zones = sorted({_zone_from_item(item) for item in itemset})
        if scope == "local_or_ambiguous_component" and support_cases < min_support_cases * 2:
            continue
        channels = sorted({_channel_from_item(item) for item in itemset})
        discriminative_coverage = _discriminative_coverage(
            itemset,
            exact_item_counts[itemset],
            occurrence_count=int(occurrence_count),
        )
        rows.append(
            {
                "itemset": itemset,
                "support_cases": support_cases,
                "occurrences": int(occurrence_count),
                "score": score,
                "specificity_bucket": specificity,
                "composition_scope": scope,
                "reason": reason,
                "channels": channels,
                "zones": zones,
                "caption_aliases": [{"id": key, "count": val} for key, val in alias_counts[itemset].most_common(8)],
                "geometry_clusters": [{"id": key, "count": val} for key, val in geometry_counts[itemset].most_common(16)],
                "discriminative_role_coverage": discriminative_coverage,
                "suppressed_discriminative_roles": [
                    item["id"] for item in discriminative_coverage if bool(item.get("suppressed"))
                ],
            }
        )

    rows.sort(key=lambda row: _candidate_sort_key(row))
    rows = _mark_subset_relations(rows)
    rows = _apply_recommendations(rows)
    rows.sort(key=lambda row: _candidate_sort_key(row))
    selected = _select_candidates(rows, max_candidates=max_candidates, min_scope_candidates=min_scope_candidates)
    selected_itemsets = {tuple(row["itemset"]) for row in selected}
    example_map = _examples_for_itemsets(occurrences, selected_itemsets, examples_per_candidate)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        itemset = tuple(row["itemset"])
        out = dict(row)
        out["rank"] = idx
        out["candidate_id"] = f"coord_role_closure_{idx:04d}"
        out["canonical_role_items"] = list(itemset)
        out.pop("itemset", None)
        out["examples"] = example_map.get(itemset, [])
        out_rows.append(out)

    recommendation_counts = Counter(str(row["recommendation"]) for row in out_rows)
    scope_counts = Counter(str(row["composition_scope"]) for row in out_rows)
    return {
        "schema_version": "v4_coord_role_composition_closure_v1",
        "runtime_policy": "offline review only; no runtime AML matching changes",
        "summary": {
            "occurrence_count": len(occurrences),
            "candidate_count": len(out_rows),
            "raw_candidate_count": len(rows),
            "min_support_cases": min_support_cases,
            "min_specific_support_cases": min_specific_support_cases,
            "max_itemset_size": max_itemset_size,
            "min_scope_candidates": min_scope_candidates,
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "composition_scope_counts": dict(sorted(scope_counts.items())),
            "specificity_bucket_counts": dict(sorted(Counter(str(row.get("specificity_bucket")) for row in out_rows).items())),
        },
        "candidates": out_rows,
    }


def _discriminative_coverage(
    itemset: tuple[str, ...],
    exact_counts: Counter[str],
    *,
    occurrence_count: int,
) -> list[dict[str, Any]]:
    itemset_items = set(itemset)
    rows: list[dict[str, Any]] = []
    for item, count in exact_counts.most_common():
        if count <= 0:
            continue
        ratio = count / max(1, occurrence_count)
        if item in itemset_items or ratio >= 0.20:
            rows.append(
                {
                    "id": item,
                    "count": int(count),
                    "ratio": round(ratio, 4),
                    "in_itemset": item in itemset_items,
                    "suppressed": item not in itemset_items and ratio >= 0.35,
                }
            )
    rows.sort(key=lambda row: (not bool(row["suppressed"]), not bool(row["in_itemset"]), -float(row["ratio"]), str(row["id"])))
    return rows


def _specificity_bucket(
    itemset: tuple[str, ...],
    support_cases: int,
    min_support_cases: int,
    min_specific_support_cases: int,
) -> str:
    if support_cases >= min_support_cases:
        return "high_support"
    if support_cases < min_specific_support_cases:
        return "below_threshold"
    channels = {_channel_from_item(item) for item in itemset}
    zones = {_zone_from_item(item) for item in itemset}
    roles = {_role_from_item(item) for item in itemset}
    has_inversion = "inversion_or_acrobatics" in roles
    has_support = bool({"floor_low_horizontal_support", "hand_floor_low_support", "inverted_support"} & roles)
    has_floor_support = bool({"floor_low_horizontal_support", "hand_floor_low_support"} & roles)
    has_inverted_support = "inverted_support" in roles
    has_large_arc = "arm_large_arc_or_orbit" in roles
    has_lateral = "leg_lateral_cycle" in roles
    has_vertical = bool({"vertical_rhythm", "vertical_low_body_transition"} & roles)
    has_body_level = bool({"body_level_cycle", "body_level_down", "body_level_up", "body_level_low", "body_low_posture"} & roles)
    has_hand_head = "hand_head_proximity" in roles
    has_bimanual = "bimanual" in channels or {"left_arm", "right_arm"}.issubset(channels)
    cross_zone = len(zones) >= 2
    if (has_support or has_inversion) and cross_zone:
        return "high_specificity"
    if has_floor_support and (has_hand_head or has_large_arc or has_body_level):
        return "high_specificity"
    if has_inverted_support and (has_large_arc or has_lateral or has_vertical or has_body_level):
        return "high_specificity"
    if has_large_arc and has_lateral and (has_vertical or has_body_level or has_bimanual):
        return "high_specificity"
    if has_hand_head and has_lateral and (has_body_level or has_vertical):
        return "high_specificity"
    return "below_threshold"


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    recommendation_order = {
        "promote_review": 0,
        "composition_review": 1,
        "named_component_review": 2,
        "component_review": 3,
        "diagnostic_keep": 4,
    }
    scope_order = {
        "inversion_acrobatic_candidate": 0,
        "floor_prone_or_mime_candidate": 0,
        "full_upper_lower_body_candidate": 0,
        "upper_lower_coordination_component": 1,
        "upper_vertical_coordination_component": 2,
        "lower_body_transition_component": 3,
        "body_level_transition_component": 4,
        "bilateral_upper_component": 5,
        "bilateral_lower_component": 6,
        "local_or_ambiguous_component": 7,
    }
    bucket_order = {"high_support": 0, "high_specificity": 1}
    suppressed_discriminative_count = len(row.get("suppressed_discriminative_roles") or [])
    itemset = tuple(row.get("itemset") or row.get("canonical_role_items") or [])
    discriminative_count = sum(1 for item in itemset if _is_discriminative_item(str(item)))
    return (
        recommendation_order.get(str(row.get("recommendation")), 9),
        scope_order.get(str(row.get("composition_scope")), 9),
        bucket_order.get(str(row.get("specificity_bucket")), 9),
        0 if row.get("is_near_closed", True) else 1,
        suppressed_discriminative_count,
        -discriminative_count,
        -float(row.get("score") or 0.0),
        -int(row.get("support_cases") or 0),
    )


def _select_candidates(rows: list[dict[str, Any]], *, max_candidates: int, min_scope_candidates: int) -> list[dict[str, Any]]:
    high_support_budget = max(1, int(max_candidates * 0.75))
    high_specific_budget = max_candidates - high_support_budget
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    scope_counts = Counter(str(row.get("composition_scope") or "") for row in rows)
    scope_names = [
        scope
        for scope, count in sorted(scope_counts.items(), key=lambda item: (item[0] != "inversion_acrobatic_candidate", item[0]))
        if scope and count >= max(1, min_scope_candidates)
    ]
    if min_scope_candidates > 0 and scope_names:
        scope_budget_total = min(max_candidates // 3, min_scope_candidates * len(scope_names))
        per_scope = max(1, min(min_scope_candidates, scope_budget_total // max(1, len(scope_names))))
        for scope in scope_names:
            taken = 0
            for row in rows:
                itemset = tuple(row["itemset"])
                if itemset in seen or str(row.get("composition_scope") or "") != scope:
                    continue
                selected.append(row)
                seen.add(itemset)
                taken += 1
                if taken >= per_scope or len(selected) >= scope_budget_total or len(selected) >= max_candidates:
                    break
            if len(selected) >= max_candidates:
                break

    for bucket, budget in (("high_support", high_support_budget), ("high_specificity", high_specific_budget)):
        taken = 0
        remaining = max_candidates - len(selected)
        if remaining <= 0:
            break
        budget = min(budget, remaining)
        for row in rows:
            itemset = tuple(row["itemset"])
            if itemset in seen or row.get("specificity_bucket") != bucket:
                continue
            selected.append(row)
            seen.add(itemset)
            taken += 1
            if taken >= budget:
                break
    if len(selected) < max_candidates:
        for row in rows:
            itemset = tuple(row["itemset"])
            if itemset in seen:
                continue
            selected.append(row)
            seen.add(itemset)
            if len(selected) >= max_candidates:
                break
    selected.sort(key=_candidate_sort_key)
    return selected


def _apply_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        aliases = list(row.get("caption_aliases") or [])
        support = int(row.get("support_cases") or 0)
        top = aliases[0] if aliases else {}
        second = aliases[1] if len(aliases) > 1 else {}
        top_count = int(top.get("count") or 0)
        second_count = int(second.get("count") or 0)
        alias_total = sum(int(item.get("count") or 0) for item in aliases)
        support_ratio = top_count / max(1, support)
        dominance = top_count / max(1, alias_total)
        scope = str(row.get("composition_scope") or "")
        top_alias = str(top.get("id") or "")
        near_closed = bool(row.get("is_near_closed", True))
        has_suppressed_discriminative = bool(row.get("suppressed_discriminative_roles"))
        incompatible_alias_scope = not _alias_scope_compatible(top_alias, scope)
        concentrated_name = (
            top_count >= 10
            and support >= 12
            and support_ratio >= 0.45
            and dominance >= 0.55
            and second_count <= max(2, int(top_count * 0.60))
        )
        blockers: list[str] = []
        if not near_closed:
            blockers.append("not_near_closed")
        if has_suppressed_discriminative:
            blockers.append("suppressed_discriminative_roles")
        if incompatible_alias_scope:
            blockers.append("top_alias_motion_scope_conflict")
        if not concentrated_name:
            blockers.append("low_or_diffuse_caption_alias_purity")
        if scope in {"full_upper_lower_body_candidate", "inversion_acrobatic_candidate", "floor_prone_or_mime_candidate"}:
            row["recommendation"] = "promote_review" if not blockers else "composition_review"
        elif scope.endswith("_component"):
            row["recommendation"] = "named_component_review" if concentrated_name else "component_review"
        else:
            row["recommendation"] = "diagnostic_keep"
        row["promotion_blockers"] = blockers
        row["name_purity"] = {
            "top_alias": top.get("id"),
            "top_count": top_count,
            "second_count": second_count,
            "support_ratio": round(support_ratio, 4),
            "dominance": round(dominance, 4),
            "alias_scope_compatible": not incompatible_alias_scope,
        }
    return rows


def _alias_scope_compatible(alias_id: str, scope: str) -> bool:
    if not alias_id:
        return True
    compatible_scopes = ALIAS_COMPATIBLE_SCOPES.get(alias_id)
    if not compatible_scopes:
        return True
    return scope in compatible_scopes


def _mark_subset_relations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    itemsets = [(set(row["itemset"]), row) for row in rows]
    for row in rows:
        row["subset_of"] = []
        row["is_near_closed"] = True
    for small_set, small_row in itemsets:
        small_support = int(small_row.get("support_cases") or 0)
        for big_set, big_row in itemsets:
            if small_set == big_set or not small_set < big_set:
                continue
            big_support = int(big_row.get("support_cases") or 0)
            if big_support >= int(0.75 * small_support):
                small_row["subset_of"].append(
                    {
                        "canonical_role_items": list(big_row["itemset"]),
                        "support_cases": big_support,
                    }
                )
                small_row["is_near_closed"] = False
                break
    return rows


def _examples_for_itemsets(
    occurrences: list[dict[str, Any]],
    itemsets: set[tuple[str, ...]],
    limit: int,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    examples: dict[tuple[str, ...], list[dict[str, Any]]] = {itemset: [] for itemset in itemsets}
    seen: dict[tuple[str, ...], set[tuple[str, tuple[int, int]]]] = {itemset: set() for itemset in itemsets}
    itemset_sets = {itemset: set(itemset) for itemset in itemsets}
    for occ in occurrences:
        occ_items = set(occ["items"])
        for itemset, required in itemset_sets.items():
            if len(examples[itemset]) >= limit or not required.issubset(occ_items):
                continue
            span = occ.get("span") or []
            span_key = tuple(int(x) for x in span[:2]) if len(span) >= 2 else (-1, -1)
            key = (str(occ.get("case_id")), span_key)
            if key in seen[itemset]:
                continue
            seen[itemset].add(key)
            examples[itemset].append(
                {
                    "case_id": occ.get("case_id"),
                    "span": occ.get("span"),
                    "symbol": occ.get("symbol"),
                    "caption": occ.get("caption"),
                    "caption_alias_ids": occ.get("caption_alias_ids"),
                    "exact_roles": occ.get("exact_roles"),
                    "geometry_clusters": occ.get("geometry_clusters"),
                }
            )
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(
                f"{item.get('id')}:{item.get('count')}"
                + (f"@{item.get('ratio')}" if item.get("ratio") is not None else "")
                + ("*" if item.get("suppressed") else "")
                for item in value
            )
        return "; ".join(str(item) for item in value)
    return str(value or "")


def write_markdown(path: Path, payload: dict[str, Any], *, max_rows: int) -> None:
    summary = payload.get("summary") or {}
    rows = payload.get("candidates") or []
    lines: list[str] = [
        "# v4 Coord-Role Composition Closure Review",
        "",
        "This is an offline review artifact. It groups frequent coactivated role",
        "itemsets into candidate compositions. Caption aliases are diagnostics only.",
        "",
        "## Summary",
        "",
        f"- occurrence_count: `{summary.get('occurrence_count')}`",
        f"- raw_candidate_count: `{summary.get('raw_candidate_count')}`",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- min_support_cases: `{summary.get('min_support_cases')}`",
        f"- min_specific_support_cases: `{summary.get('min_specific_support_cases')}`",
        f"- max_itemset_size: `{summary.get('max_itemset_size')}`",
        f"- recommendation_counts: `{summary.get('recommendation_counts')}`",
        f"- composition_scope_counts: `{summary.get('composition_scope_counts')}`",
        f"- specificity_bucket_counts: `{summary.get('specificity_bucket_counts')}`",
        "",
        "## How To Read",
        "",
        "- `canonical_role_items` is the motion-derived structure key.",
        "- `caption_aliases` helps naming review but did not create the candidate.",
        "- `specificity_bucket=high_specificity` preserves lower-frequency discriminative structures.",
        "- `discriminative_role_coverage` lists exact motion roles frequently present in examples; `*` means the role is suppressed by the current itemset and should be reviewed.",
        "- `promote_review` means the structure is full-body and language evidence is concentrated.",
        "- `promotion_blockers` explains why a full-body candidate was kept below promotion.",
        "- `component_review` means the row is useful below action level or as an edit handle.",
        "",
        "## Candidates",
        "",
    ]
    for row in rows[:max_rows]:
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("caption_aliases") or [] if item.get("id"))
        role_coverage = ", ".join(
            f"{item['id']}:{item['count']}@{item['ratio']}{'*' if item.get('suppressed') else ''}"
            for item in row.get("discriminative_role_coverage") or []
        )
        blockers = ", ".join(str(item) for item in row.get("promotion_blockers") or [])
        lines.extend(
            [
                f"### {row.get('rank')}. {row.get('candidate_id')}",
                "",
                f"- recommendation: `{row.get('recommendation')}`",
                f"- scope: `{row.get('composition_scope')}`",
                f"- support: `{row.get('support_cases')}` cases / `{row.get('occurrences')}` occurrences",
                f"- score: `{row.get('score')}`; near_closed: `{row.get('is_near_closed')}`; specificity: `{row.get('specificity_bucket')}`",
                f"- channels: `{row.get('channels')}`",
                f"- zones: `{row.get('zones')}`",
                f"- canonical_role_items: `{row.get('canonical_role_items')}`",
                f"- discriminative role coverage: {role_coverage or 'none'}",
                f"- promotion blockers: {blockers or 'none'}",
                f"- caption aliases: {aliases or 'none'}",
                f"- reason: {row.get('reason')}",
                "",
                "| case | span | symbol | caption |",
                "| --- | --- | --- | --- |",
            ]
        )
        for example in row.get("examples") or []:
            caption = str(example.get("caption") or "").replace("|", "\\|")
            lines.append(
                f"| `{example.get('case_id')}` | `{example.get('span')}` | `{example.get('symbol')}` | {caption} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_alias_index(path: Path, payload: dict[str, Any], *, max_rows_per_alias: int) -> None:
    alias_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("candidates") or []:
        for alias in row.get("caption_aliases") or []:
            alias_id = str(alias.get("id") or "")
            if not alias_id:
                continue
            alias_rows[alias_id].append(
                {
                    "count": int(alias.get("count") or 0),
                    "candidate_id": row.get("candidate_id"),
                    "rank": row.get("rank"),
                    "recommendation": row.get("recommendation"),
                    "specificity_bucket": row.get("specificity_bucket"),
                    "support_cases": row.get("support_cases"),
                    "canonical_role_items": row.get("canonical_role_items"),
                    "suppressed_discriminative_roles": row.get("suppressed_discriminative_roles") or [],
                    "promotion_blockers": row.get("promotion_blockers") or [],
                    "example": (row.get("examples") or [{}])[0],
                }
            )

    lines = [
        "# v4 Coord-Role Closure Alias Index",
        "",
        "Caption aliases are diagnostics only. Use this file to find which motion",
        "structures currently align with a language label before visual review.",
        "",
    ]
    for alias_id in sorted(alias_rows, key=lambda key: (-sum(r["count"] for r in alias_rows[key]), key)):
        rows = sorted(alias_rows[alias_id], key=lambda r: (-r["count"], int(r.get("rank") or 999999)))
        lines.extend([f"## {alias_id}", ""])
        lines.append("| alias_count | rank | candidate | recommendation | specificity | support | roles | example |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows[:max_rows_per_alias]:
            example = row.get("example") or {}
            caption = str(example.get("caption") or "").replace("|", "\\|")
            example_text = f"`{example.get('case_id')}` {caption}" if example else ""
            roles = "<br>".join(str(item) for item in row.get("canonical_role_items") or [])
            suppressed = [str(item) for item in row.get("suppressed_discriminative_roles") or []]
            if suppressed:
                roles += "<br>suppressed: " + "<br>".join(suppressed)
            blockers = [str(item) for item in row.get("promotion_blockers") or []]
            if blockers:
                roles += "<br>blockers: " + "<br>".join(blockers)
            lines.append(
                f"| `{row['count']}` | `{row.get('rank')}` | `{row.get('candidate_id')}` | "
                f"`{row.get('recommendation')}` | `{row.get('specificity_bucket')}` | "
                f"`{row.get('support_cases')}` | {roles} | {example_text} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-dir", type=Path, default=DEFAULT_BPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--record-cache", type=Path, default=None)
    parser.add_argument("--min-support-cases", type=int, default=8)
    parser.add_argument("--min-specific-support-cases", type=int, default=4)
    parser.add_argument("--max-itemset-size", type=int, default=6)
    parser.add_argument("--max-transaction-items", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--min-scope-candidates", type=int, default=8, help="Reserve a small candidate budget for each structural scope before global ranking.")
    parser.add_argument("--examples-per-candidate", type=int, default=5)
    parser.add_argument("--max-report-rows", type=int, default=180)
    parser.add_argument("--max-alias-index-rows", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_path = args.sequence_file or args.bpe_dir / "case_multichannel_bpe_sequences.jsonl"
    summary_path = args.bpe_dir / "summary.json"
    record_cache = args.record_cache or _record_cache_from_summary(summary_path)
    caption_index = _case_caption_index(record_cache)
    occurrences = collect_occurrences(
        sequence_path,
        caption_index,
        max_transaction_items=args.max_transaction_items,
    )
    payload = mine_closure_candidates(
        occurrences,
        min_support_cases=args.min_support_cases,
        min_specific_support_cases=args.min_specific_support_cases,
        max_itemset_size=args.max_itemset_size,
        max_candidates=args.max_candidates,
        examples_per_candidate=args.examples_per_candidate,
        min_scope_candidates=args.min_scope_candidates,
    )
    payload["sources"] = {
        "bpe_dir": str(args.bpe_dir),
        "sequence_file": str(sequence_path),
        "record_cache": str(record_cache) if record_cache else None,
    }

    _write_json(output_dir / "composition_closure_candidates.json", payload)
    write_csv(output_dir / "composition_closure_candidates.csv", payload.get("candidates") or [])
    write_markdown(output_dir / "composition_closure_review.md", payload, max_rows=args.max_report_rows)
    write_alias_index(output_dir / "caption_alias_index.md", payload, max_rows_per_alias=args.max_alias_index_rows)
    summary = dict(payload.get("summary") or {})
    summary["outputs"] = {
        "composition_closure_candidates": str(output_dir / "composition_closure_candidates.json"),
        "composition_closure_candidates_csv": str(output_dir / "composition_closure_candidates.csv"),
        "composition_closure_review": str(output_dir / "composition_closure_review.md"),
        "caption_alias_index": str(output_dir / "caption_alias_index.md"),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
