"""Build a composition-level motion pattern forest from all-channel units.

This is an offline mining experiment. It does not use text targets to create
motion structure and does not create runtime AML rules.

Pipeline:
1. Load channel BPE sequences from the multichannel Motion-BPE run.
2. Rebuild all-unit coactivation transactions.
3. Mine frequent cross-channel itemsets and keep closed candidates.
4. Attach caption-alias counters for naming diagnostics only.
5. Export a reviewable composition candidate forest.

Quick run:
    python scripts/build_hml3d_composition_pattern_forest_v0.py

Smoke test:
    python scripts/build_hml3d_composition_pattern_forest_v0.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_hml3d_coactivation_recall_v0 import (  # noqa: E402
    build_all_unit_coactivations,
    load_case_text,
    load_channel_sequences,
)


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_BPE_SEQUENCES = Path(
    "outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full/case_multichannel_bpe_sequences.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v0")


ROOTS = [
    ("composition_full_candidates", "full-body composition candidates"),
    ("transition_candidates", "transition composition candidates"),
    ("component_coordination_candidates", "component coordination candidates"),
    ("context_diagnostic_candidates", "context diagnostic candidates"),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_id(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "unnamed"


def _safe_div(num: int | float, denom: int | float) -> float:
    return 0.0 if not denom else float(num) / float(denom)


def _parse_coord_sig(symbol: str) -> frozenset[str]:
    if not symbol.startswith("COORD_SIG[") or not symbol.endswith("]"):
        return frozenset()
    inner = symbol[len("COORD_SIG[") : -1]
    items: set[str] = set()
    for part in inner.split("+"):
        if ":" not in part:
            continue
        channel, cluster_blob = part.split(":", 1)
        channel = channel.strip()
        for cluster in cluster_blob.split("&"):
            cluster = cluster.strip()
            if channel and cluster:
                items.add(f"{channel}:{cluster}")
    return frozenset(items)


def _item_channel(item: str) -> str:
    return item.split(":", 1)[0] if ":" in item else "other"


def _item_geometry(item: str) -> str:
    return item.split(":", 1)[1] if ":" in item else item


def _normalize_laterality(text: str) -> str:
    replacements = [
        ("LEFT_ARM_", "ARM_"),
        ("RIGHT_ARM_", "ARM_"),
        ("LEFT_LEG_", "LEG_"),
        ("RIGHT_LEG_", "LEG_"),
        ("LA_", "ARM_"),
        ("RA_", "ARM_"),
        ("LL_", "LEG_"),
        ("RL_", "LEG_"),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _coarse_geometry_role(geometry: str, all_items: frozenset[str] | None = None) -> str:
    """Fold equivalent geometry labels into structural roles for family grouping.

    The original cluster id is still stored on each variant. This role is only
    used to decide whether two closed itemsets belong to the same family.
    """
    geom = _normalize_laterality(geometry)
    flags = _geometry_flags(all_items or frozenset())
    if "BI_RAISE_SPREAD" in geom:
        return "BIMANUAL_PERIODIC/BI_RAISE_SPREAD"
    if "BI_HANDS_CLOSE" in geom:
        return "BIMANUAL_PERIODIC/BI_HANDS_CLOSE"
    if "BI_RAISE_LOW_BODY" in geom or "BI_SPREAD_LOW_BODY" in geom:
        return "BIMANUAL_PERIODIC/BI_LOW_BODY_CONTEXT"
    if "BILATERAL_HIGH_POSE" in geom:
        return "ARM_POSTURE/ARM_BILATERAL_HIGH_POSE"
    if "BILATERAL_VERTICAL_ARM_CYCLE" in geom:
        return "ARM_PERIODIC/ARM_BILATERAL_VERTICAL_CYCLE"
    if "VERTICAL_COUPLED_ARM_PERIODIC_UP_DOWN" in geom:
        return "ARM_PERIODIC/ARM_VERTICAL_COUPLED_UP_DOWN"
    if "VERT_ARM_RAISE" in geom:
        return "WHOLE_BODY_VERTICAL/WB_VERT_UPPER_LIMB_COORDINATION"
    if "WB_VERT_LOW_BODY" in geom:
        return "WHOLE_BODY_VERTICAL/WB_VERT_LOW_BODY_PHASE"
    if "WB_LOW_BODY" in geom or "WB_SQUAT" in geom:
        return "WHOLE_BODY_POSTURE/WB_LOW_BODY_PHASE"
    if "TORSO_HUNCHED_FORWARD" in geom and "LOW_BODY" in geom:
        return "TORSO_POSTURE/TORSO_FORWARD_LOW_BODY_CONTEXT"
    if "TORSO_BEND_RECOVER" in geom and "LOW_BODY" in geom:
        return "TORSO_PERIODIC/TORSO_BEND_LOW_BODY_CONTEXT"
    if "TORSO_OSC_FB" in geom and "LOW_BODY" in geom:
        return "TORSO_PERIODIC/TORSO_OSC_LOW_BODY_CONTEXT"
    if "TORSO_BEND_RECOVER" in geom and "VERTICAL" in geom:
        return "TORSO_PERIODIC/TORSO_BEND_VERTICAL_CONTEXT"
    if "TORSO_OSC_FB" in geom and "VERTICAL" in geom:
        return "TORSO_PERIODIC/TORSO_OSC_VERTICAL_CONTEXT"
    if "LEG_FORWARD_HOLD_POSE" in geom:
        return "LEG_ACTION/LEG_FORWARD_HOLD_POSE"
    if "LEG_FORWARD_KICK_IMPULSE" in geom or "LEG_FORWARD_HOP_OR_KICK_IMPULSE" in geom:
        return "LEG_ACTION/LEG_FORWARD_IMPULSE"
    if "LOCO_GAIT_CONTEXT" in geom:
        return "ROOT_LOCOMOTION/LOCO_GAIT_CONTEXT"
    if "ROOT_DRIFT" in geom:
        return "ROOT_LOCOMOTION/ROOT_DRIFT"
    if "PATH_FRAGMENT" in geom:
        return "ROOT_LOCOMOTION/PATH_FRAGMENT"
    if "WB_VERT_UP_REFINED_GENERIC" in geom or "WB_VERT_DOWN_REFINED_GENERIC" in geom:
        if flags["arm_raise"] or flags["bimanual_vertical"]:
            return "WHOLE_BODY_VERTICAL/WB_VERT_UPPER_LIMB_CONTEXT"
        if flags["low_body"]:
            return "WHOLE_BODY_VERTICAL/WB_VERT_LOW_BODY_CONTEXT"
        return "WHOLE_BODY_VERTICAL/WB_VERT_GENERIC_PHASE"
    return geom


def _normalized_item(item: str, all_items: frozenset[str] | None = None) -> str:
    channel = _item_channel(item)
    geometry = _coarse_geometry_role(_item_geometry(item), all_items)
    if channel in {"left_arm", "right_arm"}:
        channel = "arm"
    elif channel in {"left_leg", "right_leg"}:
        channel = "leg"
    return f"{channel}:{geometry}"


def _item_zone(item: str) -> str:
    channel = _item_channel(item)
    if channel in {"left_arm", "right_arm", "bimanual"}:
        return "upper"
    if channel in {"left_leg", "right_leg", "whole_body_state"}:
        return "lower"
    if channel == "whole_body_vertical":
        return "vertical"
    if channel in {"root_locomotion", "root_rotation"}:
        return "root"
    if channel == "torso":
        return "torso"
    return "other"


def _geometry_flags(items: frozenset[str]) -> dict[str, bool]:
    geometry = {_item_geometry(item) for item in items}
    zones = {_item_zone(item) for item in items}
    has_upper = "upper" in zones
    has_lower = "lower" in zones
    has_vertical = "vertical" in zones
    has_root = "root" in zones
    has_torso = "torso" in zones
    low_body = any("LOW_BODY" in item or "SQUAT" in item for item in geometry)
    gait_like = any("GAIT" in item or "LOCO_ARM_SWING" in item for item in geometry)
    root_context = any("ROOT_DRIFT" in item or "PATH_FRAGMENT" in item or "LOCO_GAIT_CONTEXT" in item for item in geometry)
    generic = any("REFINED_GENERIC" in item or "COMPOSITE" in item for item in geometry)
    bimanual_vertical = any("RAISE_SPREAD_VERTICAL" in item or "HANDS_CLOSE_VERTICAL" in item for item in geometry)
    arm_raise = any("HIGH_POSE" in item or "ARM_RAISE" in item or "BILATERAL_VERTICAL_ARM" in item for item in geometry)
    leg_pose = any("LEG_FORWARD_HOLD_POSE" in item or "LEG_FORWARD_KICK_IMPULSE" in item or "LEG_FORWARD_HOP_OR_KICK_IMPULSE" in item for item in geometry)
    return {
        "has_upper": has_upper,
        "has_lower": has_lower,
        "has_vertical": has_vertical,
        "has_root": has_root,
        "has_torso": has_torso,
        "low_body": low_body,
        "gait_like": gait_like,
        "root_context": root_context,
        "generic": generic,
        "bimanual_vertical": bimanual_vertical,
        "arm_raise": arm_raise,
        "leg_pose": leg_pose,
    }


def _is_context_item(item: str, all_items: frozenset[str]) -> bool:
    geom = _item_geometry(item)
    flags = _geometry_flags(all_items)
    if "ROOT_DRIFT" in geom or "PATH_FRAGMENT" in geom or "LOCO_GAIT_CONTEXT" in geom:
        return True
    if "REFINED_GENERIC" in geom and (flags["arm_raise"] or flags["low_body"] or flags["bimanual_vertical"]):
        return True
    if _item_channel(item) == "torso" and "TORSO_OSC_FB_VERTICAL_CONTEXT" in geom and flags["bimanual_vertical"]:
        return True
    if _item_channel(item) == "torso" and "LOCO_CONTEXT" in geom and not flags["low_body"]:
        return True
    return False


def _family_core_items(items: frozenset[str]) -> frozenset[str]:
    core = {
        _normalized_item(item, items)
        for item in items
        if not _is_context_item(item, items)
    }
    if len(core) >= 2:
        return frozenset(core)
    return frozenset(_normalized_item(item, items) for item in items)


def _salience(item: str) -> tuple[int, str]:
    geom = _item_geometry(item)
    score = 0
    if any(key in geom for key in ["ARM_RAISE", "HIGH_POSE", "RAISE_SPREAD", "HANDS_CLOSE"]):
        score += 5
    if any(key in geom for key in ["LOW_BODY", "SQUAT", "HUNCHED_FORWARD", "LEG_FORWARD_HOLD", "KICK_IMPULSE"]):
        score += 4
    if any(key in geom for key in ["JUMP", "VERT_ARM_RAISE", "VERT_LOW_BODY"]):
        score += 4
    if "VERT" in geom:
        score += 2
    if "GAIT_CONTEXT" in geom or "ROOT_DRIFT" in geom or "REFINED_GENERIC" in geom:
        score -= 3
    return (-score, item)


def _limited_items(items: frozenset[str], limit: int) -> tuple[str, ...]:
    return tuple(sorted(items, key=_salience)[:limit])


def _combo_iter(items: tuple[str, ...], min_size: int, max_size: int) -> Any:
    upper = min(max_size, len(items))
    for size in range(max(2, min_size), upper + 1):
        yield from itertools.combinations(items, size)


def build_transactions(
    coactivation_sequences: dict[str, list[dict[str, Any]]],
    *,
    max_transaction_items: int,
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    for sequence_id, seq in coactivation_sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            items = _parse_coord_sig(str(unit.get("symbol") or ""))
            if len(items) < 2:
                continue
            limited = frozenset(_limited_items(items, max_transaction_items))
            if len(limited) < 2:
                continue
            transactions.append(
                {
                    "case_id": case_id,
                    "span": unit.get("span"),
                    "items": limited,
                    "source_symbol": unit.get("symbol"),
                }
            )
    return transactions


def count_candidate_itemsets(
    transactions: list[dict[str, Any]],
    *,
    min_itemset_size: int,
    max_itemset_size: int,
    min_occurrences: int,
) -> Counter[frozenset[str]]:
    counts: Counter[frozenset[str]] = Counter()
    for tx in transactions:
        items = tuple(sorted(tx["items"]))
        for combo in _combo_iter(items, min_itemset_size, max_itemset_size):
            counts[frozenset(combo)] += 1
    return Counter({itemset: count for itemset, count in counts.items() if count >= min_occurrences})


def collect_itemset_cases(
    transactions: list[dict[str, Any]],
    candidates: set[frozenset[str]],
    *,
    example_limit: int,
) -> dict[frozenset[str], dict[str, Any]]:
    rows: dict[frozenset[str], dict[str, Any]] = {
        itemset: {
            "occurrences": 0,
            "case_ids": set(),
            "examples": [],
            "source_symbols": Counter(),
        }
        for itemset in candidates
    }
    candidates_by_item: dict[str, list[frozenset[str]]] = defaultdict(list)
    for itemset in candidates:
        for item in itemset:
            candidates_by_item[item].append(itemset)

    for tx in transactions:
        items = set(tx["items"])
        if not items:
            continue
        seed_item = min(items, key=lambda item: len(candidates_by_item.get(item, [])))
        for itemset in candidates_by_item.get(seed_item, []):
            if not itemset.issubset(items):
                continue
            row = rows[itemset]
            row["occurrences"] += 1
            row["case_ids"].add(str(tx["case_id"]))
            row["source_symbols"][str(tx.get("source_symbol") or "")] += 1
            if len(row["examples"]) < example_limit:
                row["examples"].append(
                    {
                        "case_id": tx["case_id"],
                        "span": tx.get("span"),
                        "source_symbol": tx.get("source_symbol"),
                    }
                )
    return rows


def filter_closed_itemsets(
    rows: dict[frozenset[str], dict[str, Any]],
    *,
    support_tolerance: float,
) -> set[frozenset[str]]:
    itemsets = list(rows)
    item_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, itemset in enumerate(itemsets):
        for item in itemset:
            item_to_indices[item].append(idx)

    closed: set[frozenset[str]] = set()
    for idx, itemset in enumerate(itemsets):
        case_ids = rows[itemset]["case_ids"]
        support = len(case_ids)
        if support == 0:
            continue
        probe_item = min(itemset, key=lambda item: len(item_to_indices[item]))
        is_closed = True
        for other_idx in item_to_indices[probe_item]:
            if other_idx == idx:
                continue
            other = itemsets[other_idx]
            if len(other) <= len(itemset) or not itemset.issubset(other):
                continue
            other_support = len(rows[other]["case_ids"])
            if other_support >= int(math.ceil(support * (1.0 - support_tolerance))):
                is_closed = False
                break
        if is_closed:
            closed.add(itemset)
    return closed


def _structure_score(items: frozenset[str], support: int) -> float:
    zones = {_item_zone(item) for item in items}
    channels = {_item_channel(item) for item in items}
    flags = _geometry_flags(items)
    score = math.log1p(support) + 0.35 * len(channels) + 0.50 * len(zones)
    if flags["has_upper"] and flags["has_vertical"]:
        score += 1.50
    if flags["has_lower"] and flags["has_vertical"]:
        score += 1.00
    if flags["has_torso"] and (flags["low_body"] or flags["has_vertical"]):
        score += 0.80
    if flags["low_body"]:
        score += 0.80
    if flags["bimanual_vertical"]:
        score += 1.40
    if flags["arm_raise"]:
        score += 0.70
    if flags["leg_pose"]:
        score += 0.60
    if flags["gait_like"]:
        score -= 1.50
    if flags["root_context"]:
        score -= 1.20
    if flags["generic"] and not (flags["low_body"] or flags["arm_raise"] or flags["bimanual_vertical"]):
        score -= 0.80
    if len(zones) <= 1:
        score -= 1.25
    return round(score, 6)


def _scope_and_status(items: frozenset[str], support: int, score: float, aliases: list[dict[str, Any]]) -> tuple[str, str, str]:
    zones = {_item_zone(item) for item in items}
    flags = _geometry_flags(items)
    top_alias_count = int(aliases[0].get("count") or 0) if aliases else 0
    alias_purity = _safe_div(top_alias_count, support)
    cross_zone = len(zones - {"other"}) >= 2
    upper_only = zones <= {"upper"}
    lower_only = zones <= {"lower"}
    if cross_zone and score >= 6.0 and (flags["has_vertical"] or flags["low_body"] or flags["has_torso"]):
        status = "name_aligned_composition_candidate" if alias_purity >= 0.25 and top_alias_count >= 8 else "composition_candidate"
        return status, "full_or_transition_composition_candidate", "composition_full_candidates"
    if cross_zone and (flags["low_body"] or flags["has_torso"]):
        return "transition_candidate", "transition_or_posture_composition_candidate", "transition_candidates"
    if upper_only or lower_only:
        return "component_candidate", "component_coordination_candidate", "component_coordination_candidates"
    return "diagnostic_candidate", "context_or_ambiguous_candidate", "context_diagnostic_candidates"


def _itemset_aliases(case_ids: set[str], case_text: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for case_id in case_ids:
        aliases = [str(item) for item in (case_text.get(case_id) or {}).get("caption_alias_ids") or []]
        counter.update(aliases or ["__NO_ALIAS__"])
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit) if key != "__NO_ALIAS__"]


def _examples_with_text(examples: list[dict[str, Any]], case_text: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for example in examples:
        case_id = str(example.get("case_id") or "")
        record = case_text.get(case_id, {})
        out.append(
            {
                "case_id": case_id,
                "span": example.get("span"),
                "caption_texts": record.get("caption_texts") or [],
                "caption_alias_ids": record.get("caption_alias_ids") or [],
                "source_symbol": example.get("source_symbol"),
            }
        )
    return out


def _node_display_name(itemset: frozenset[str], aliases: list[dict[str, Any]]) -> str:
    if aliases and int(aliases[0].get("count") or 0) >= 8:
        return str(aliases[0].get("id") or "")
    geometry = sorted(_item_geometry(item) for item in itemset)
    return " + ".join(geometry[:4])


def _merge_aliases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in row.get("caption_aliases") or []:
            alias = str(item.get("id") or "")
            if alias:
                counter[alias] += int(item.get("count") or 0)
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def _structure_label_from_core(core_items: list[str]) -> str:
    text = " ".join(core_items)
    has_upper_vertical = "WB_VERT_UPPER_LIMB" in text
    has_raise_spread = "BI_RAISE_SPREAD" in text
    has_hands_close = "BI_HANDS_CLOSE" in text
    has_low_body = "WB_LOW_BODY_PHASE" in text or "WB_VERT_LOW_BODY_PHASE" in text
    has_torso_low = "TORSO_FORWARD_LOW_BODY" in text or "TORSO_BEND_LOW_BODY" in text or "TORSO_OSC_LOW_BODY" in text
    has_leg_impulse = "LEG_FORWARD_IMPULSE" in text
    has_leg_hold = "LEG_FORWARD_HOLD_POSE" in text
    has_acrobatic = "WB_INVERTED_ROTATION" in text
    if has_acrobatic:
        return "inverted_body_coordination"
    if has_raise_spread and has_upper_vertical:
        return "bimanual_raise_spread_vertical_coordination"
    if has_hands_close and has_upper_vertical:
        return "hands_close_vertical_coordination"
    if has_low_body and has_torso_low:
        return "low_body_torso_transition"
    if has_low_body:
        return "low_body_vertical_transition"
    if has_leg_impulse:
        return "leg_forward_impulse_coordination"
    if has_leg_hold:
        return "leg_forward_hold_coordination"
    if has_upper_vertical:
        return "upper_limb_vertical_coordination"
    if has_raise_spread:
        return "bimanual_raise_spread_coordination"
    if has_hands_close:
        return "hands_close_coordination"
    zones = sorted({item.split(":", 1)[0] for item in core_items})
    if zones:
        return "_".join(zones[:4]) + "_coordination"
    return "composition_coordination"


def _family_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def _family_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    status = str(row.get("status") or "")
    status_rank = 0 if status.startswith("name_aligned") else 1 if status == "composition_candidate" else 2
    return (status_rank, -float(row.get("structure_score") or 0.0), -int(row.get("support_cases") or 0), str(row.get("node_id") or ""))


def _family_node_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    status = str(row.get("status") or "")
    status_rank = 0 if status.startswith("name_aligned") else 1 if status == "composition_family" else 2
    return (
        status_rank,
        -float(row.get("structure_score_max") or 0.0),
        -int(row.get("support_cases_max") or 0),
        str(row.get("node_id") or ""),
    )


def cluster_candidate_rows(rows: list[dict[str, Any]], *, similarity_threshold: float, top_k: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=_family_sort_key)
    families: list[dict[str, Any]] = []
    for row in sorted_rows:
        core = frozenset(row.get("family_core_items") or [])
        best_family: dict[str, Any] | None = None
        best_score = 0.0
        for family in families:
            family_core = frozenset(family.get("family_core_items") or [])
            score = _family_similarity(core, family_core)
            if score > best_score:
                best_score = score
                best_family = family
        if best_family is not None and best_score >= similarity_threshold:
            best_family["variants"].append(row)
            best_family["family_core_items"] = sorted(set(best_family.get("family_core_items") or []) | set(core))
            best_family["support_cases_max"] = max(int(best_family.get("support_cases_max") or 0), int(row.get("support_cases") or 0))
            best_family["support_cases_sum"] = int(best_family.get("support_cases_sum") or 0) + int(row.get("support_cases") or 0)
            best_family["occurrences_sum"] = int(best_family.get("occurrences_sum") or 0) + int(row.get("occurrences") or 0)
            best_family["structure_score_max"] = max(float(best_family.get("structure_score_max") or 0.0), float(row.get("structure_score") or 0.0))
            best_family["zones"] = sorted(set(best_family.get("zones") or []) | set(row.get("zones") or []))
            best_family["channels"] = sorted(set(best_family.get("channels") or []) | set(row.get("channels") or []))
            best_family["geometry_clusters"] = sorted(set(best_family.get("geometry_clusters") or []) | set(row.get("geometry_clusters") or []))
        else:
            families.append(
                {
                    "variants": [row],
                    "family_core_items": sorted(core),
                    "support_cases_max": int(row.get("support_cases") or 0),
                    "support_cases_sum": int(row.get("support_cases") or 0),
                    "occurrences_sum": int(row.get("occurrences") or 0),
                    "structure_score_max": float(row.get("structure_score") or 0.0),
                    "zones": list(row.get("zones") or []),
                    "channels": list(row.get("channels") or []),
                    "geometry_clusters": list(row.get("geometry_clusters") or []),
                }
            )

    out: list[dict[str, Any]] = []
    for idx, family in enumerate(families, start=1):
        variants = sorted(family["variants"], key=_family_sort_key)
        aliases = _merge_aliases(variants, top_k)
        representative = variants[0]
        statuses = Counter(str(row.get("status") or "") for row in variants)
        if any(status.startswith("name_aligned") for status in statuses):
            status = "name_aligned_composition_family"
        elif statuses.get("composition_candidate"):
            status = "composition_family"
        elif statuses.get("transition_candidate"):
            status = "transition_family"
        elif statuses.get("component_candidate"):
            status = "component_family"
        else:
            status = "diagnostic_family"
        structure_label = _structure_label_from_core(sorted(family.get("family_core_items") or []))
        family_node = {
            "node_id": f"composition_family_{idx:04d}_{_safe_id(structure_label)}",
            "node_kind": "composition_family",
            "status": status,
            "scope": representative.get("scope"),
            "display_name": structure_label,
            "motion_structure_label": structure_label,
            "support_cases_max": int(family.get("support_cases_max") or 0),
            "support_cases_sum": int(family.get("support_cases_sum") or 0),
            "occurrences_sum": int(family.get("occurrences_sum") or 0),
            "structure_score_max": round(float(family.get("structure_score_max") or 0.0), 6),
            "variant_count": len(variants),
            "channels": sorted(family.get("channels") or []),
            "zones": sorted(family.get("zones") or []),
            "geometry_clusters": sorted(family.get("geometry_clusters") or []),
            "family_core_items": sorted(family.get("family_core_items") or []),
            "caption_aliases": aliases,
            "caption_name_candidates": aliases,
            "caption_alias_purity_max": max(float(row.get("caption_alias_purity") or 0.0) for row in variants),
            "parent_root": representative.get("parent_root"),
            "variants": variants,
        }
        out.append(family_node)
    out.sort(key=_family_node_sort_key)
    return out


def build_structure_groups(families: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        root = str(family.get("parent_root") or "context_diagnostic_candidates")
        label = str(family.get("motion_structure_label") or "composition_coordination")
        grouped[(root, label)].append(family)

    groups: list[dict[str, Any]] = []
    for idx, ((root, label), rows) in enumerate(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])), start=1):
        rows = sorted(rows, key=_family_node_sort_key)
        aliases = _merge_aliases(rows, top_k)
        status_counts = Counter(str(row.get("status") or "") for row in rows)
        if any(status.startswith("name_aligned") for status in status_counts):
            status = "name_aligned_structure_group"
        elif status_counts.get("composition_family"):
            status = "composition_structure_group"
        elif status_counts.get("transition_family"):
            status = "transition_structure_group"
        elif status_counts.get("component_family"):
            status = "component_structure_group"
        else:
            status = "diagnostic_structure_group"
        core_items = sorted({item for row in rows for item in (row.get("family_core_items") or [])})
        groups.append(
            {
                "node_id": f"structure_group_{idx:04d}_{_safe_id(label)}",
                "node_kind": "structure_group",
                "status": status,
                "scope": rows[0].get("scope"),
                "display_name": label,
                "motion_structure_label": label,
                "parent_root": root,
                "family_count": len(rows),
                "variant_count": sum(int(row.get("variant_count") or 0) for row in rows),
                "support_cases_max": max(int(row.get("support_cases_max") or 0) for row in rows),
                "support_cases_sum": sum(int(row.get("support_cases_sum") or 0) for row in rows),
                "occurrences_sum": sum(int(row.get("occurrences_sum") or 0) for row in rows),
                "structure_score_max": round(max(float(row.get("structure_score_max") or 0.0) for row in rows), 6),
                "channels": sorted({channel for row in rows for channel in (row.get("channels") or [])}),
                "zones": sorted({zone for row in rows for zone in (row.get("zones") or [])}),
                "family_core_items": core_items,
                "caption_aliases": aliases,
                "caption_name_candidates": aliases,
                "family_node_ids": [str(row.get("node_id") or "") for row in rows],
            }
        )
    groups.sort(key=lambda row: (str(row.get("parent_root") or ""), _family_node_sort_key(row)))
    root_ranks: Counter[str] = Counter()
    for group in groups:
        root = str(group.get("parent_root") or "")
        root_ranks[root] += 1
        group["priority_rank"] = int(root_ranks[root])
    return groups


def build_forest(args: argparse.Namespace) -> dict[str, Any]:
    case_text = load_case_text(Path(args.source_corpus), max_cases=args.max_cases)
    channel_sequences = load_channel_sequences(Path(args.bpe_sequences), max_cases=args.max_cases)
    coactivation_sequences = build_all_unit_coactivations(channel_sequences, parallel_overlap_min=float(args.parallel_overlap_min))
    transactions = build_transactions(coactivation_sequences, max_transaction_items=int(args.max_transaction_items))
    counts = count_candidate_itemsets(
        transactions,
        min_itemset_size=int(args.min_itemset_size),
        max_itemset_size=int(args.max_itemset_size),
        min_occurrences=int(args.min_occurrences),
    )
    collected = collect_itemset_cases(transactions, set(counts), example_limit=int(args.example_limit))
    frequent = {
        itemset: row
        for itemset, row in collected.items()
        if len(row["case_ids"]) >= int(args.min_support_cases)
    }
    closed = filter_closed_itemsets(frequent, support_tolerance=float(args.closed_support_tolerance))

    root_nodes = [
        {
            "node_id": root_id,
            "node_kind": "composition_root",
            "status": "root",
            "scope": "candidate_group",
            "display_name": display_name,
        }
        for root_id, display_name in ROOTS
    ]
    nodes: list[dict[str, Any]] = list(root_nodes)
    edges: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for rank, itemset in enumerate(sorted(closed, key=lambda key: (-len(frequent[key]["case_ids"]), -len(key), tuple(sorted(key)))), start=1):
        row = frequent[itemset]
        support = len(row["case_ids"])
        if support < int(args.min_support_cases):
            continue
        score = _structure_score(itemset, support)
        if score < float(args.min_structure_score):
            continue
        aliases = _itemset_aliases(row["case_ids"], case_text, int(args.top_k))
        status, scope, root_id = _scope_and_status(itemset, support, score, aliases)
        channels = sorted({_item_channel(item) for item in itemset})
        zones = sorted({_item_zone(item) for item in itemset})
        geometry = sorted({_item_geometry(item) for item in itemset})
        node_id = f"composition_candidate_{rank:05d}_{_safe_id('_'.join(zones))}"
        payload = {
            "node_id": node_id,
            "node_kind": "composition_variant",
            "status": status,
            "scope": scope,
            "display_name": _node_display_name(itemset, aliases),
            "support_cases": support,
            "occurrences": int(row["occurrences"]),
            "structure_score": score,
            "channels": channels,
            "zones": zones,
            "geometry_clusters": geometry,
            "items": sorted(itemset),
            "family_core_items": sorted(_family_core_items(itemset)),
            "caption_aliases": aliases,
            "caption_alias_purity": round(_safe_div(int(aliases[0].get("count") or 0) if aliases else 0, support), 6),
            "top_source_symbols": [
                {"symbol": symbol, "count": int(value)}
                for symbol, value in row["source_symbols"].most_common(min(5, int(args.top_k)))
            ],
            "examples": _examples_with_text(row["examples"], case_text),
            "parent_root": root_id,
        }
        candidate_rows.append(payload)

    families = cluster_candidate_rows(
        candidate_rows,
        similarity_threshold=float(args.family_similarity_threshold),
        top_k=int(args.top_k),
    )
    if int(args.max_nodes) > 0:
        families = families[: int(args.max_nodes)]
    structure_groups = build_structure_groups(families, top_k=int(args.top_k))
    variant_nodes: list[dict[str, Any]] = []
    group_by_key = {
        (str(group.get("parent_root") or ""), str(group.get("motion_structure_label") or "")): group
        for group in structure_groups
    }
    for group in structure_groups:
        nodes.append(group)
        edges.append({"parent_node_id": group["parent_root"], "child_node_id": group["node_id"], "edge_type": "structure_group_member"})
    for idx, family in enumerate(families, start=1):
        family["priority_rank"] = idx
        variants = family.pop("variants")
        group = group_by_key[(str(family.get("parent_root") or ""), str(family.get("motion_structure_label") or ""))]
        family["parent_structure_group_id"] = group["node_id"]
        nodes.append(family)
        edges.append({"parent_node_id": group["node_id"], "child_node_id": family["node_id"], "edge_type": "composition_family_member"})
        for variant_idx, variant in enumerate(variants[: int(args.max_variants_per_family)], start=1):
            variant = dict(variant)
            variant["node_id"] = f"{family['node_id']}__variant_{variant_idx:03d}"
            variant["priority_rank"] = variant_idx
            variant["parent_family_id"] = family["node_id"]
            variant_nodes.append(variant)
            nodes.append(variant)
            edges.append({"parent_node_id": family["node_id"], "child_node_id": variant["node_id"], "edge_type": "composition_variant_member"})

    status_counts = Counter(str(node.get("status") or "") for node in families)
    scope_counts = Counter(str(node.get("scope") or "") for node in families)
    root_counts = Counter(str(node.get("parent_root") or "") for node in families)
    structure_group_status_counts = Counter(str(node.get("status") or "") for node in structure_groups)
    structure_group_root_counts = Counter(str(node.get("parent_root") or "") for node in structure_groups)
    return {
        "schema_version": "hml3d_composition_pattern_forest_v0",
        "runtime_policy": "offline composition candidate forest; text aliases are naming diagnostics only",
        "inputs": {
            "source_corpus": str(args.source_corpus),
            "bpe_sequences": str(args.bpe_sequences),
            "parallel_overlap_min": float(args.parallel_overlap_min),
        },
        "thresholds": {
            "min_occurrences": int(args.min_occurrences),
            "min_support_cases": int(args.min_support_cases),
            "min_itemset_size": int(args.min_itemset_size),
            "max_itemset_size": int(args.max_itemset_size),
            "max_transaction_items": int(args.max_transaction_items),
            "closed_support_tolerance": float(args.closed_support_tolerance),
            "min_structure_score": float(args.min_structure_score),
        },
        "summary": {
            "case_text_count": len(case_text),
            "channel_sequence_count": len(channel_sequences),
            "coactivation_sequence_count": len(coactivation_sequences),
            "transaction_count": len(transactions),
            "frequent_itemset_count": len(counts),
            "closed_itemset_count": len(closed),
            "raw_candidate_itemset_count": len(candidate_rows),
            "candidate_node_count": len(families),
            "structure_group_count": len(structure_groups),
            "variant_node_count": len(variant_nodes),
            "status_counts": dict(sorted(status_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
            "root_counts": dict(sorted(root_counts.items())),
            "structure_group_status_counts": dict(sorted(structure_group_status_counts.items())),
            "structure_group_root_counts": dict(sorted(structure_group_root_counts.items())),
        },
        "nodes": nodes,
        "edges": edges,
    }


def write_tree(path: Path, forest: dict[str, Any], *, max_children: int) -> None:
    node_by_id = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if child in node_by_id:
            children[parent].append(node_by_id[child])
    lines = ["# HML3D Composition Pattern Forest v0", ""]
    summary = forest.get("summary") or {}
    lines.append(
        f"structure_groups={summary.get('structure_group_count')} families={summary.get('candidate_node_count')} "
        f"variants={summary.get('variant_node_count')} "
        f"transactions={summary.get('transaction_count')} closed_itemsets={summary.get('closed_itemset_count')}"
    )
    lines.append("")
    for root in [node for node in forest.get("nodes") or [] if node.get("node_kind") == "composition_root"]:
        root_id = str(root.get("node_id") or "")
        groups = sorted(children.get(root_id, []), key=lambda row: int(row.get("priority_rank") or 999999))
        lines.append(f"- {root.get('display_name')} [{len(groups)} structure groups]")
        for group in groups[:max_children]:
            aliases = ", ".join(f"{item['id']}:{item['count']}" for item in group.get("caption_aliases", [])[:4])
            core = ", ".join(group.get("family_core_items", [])[:5])
            lines.append(
                f"  - {group.get('node_id')} | {group.get('status')} | families={group.get('family_count')} "
                f"variants={group.get('variant_count')} support_max={group.get('support_cases_max')} "
                f"score_max={group.get('structure_score_max')} | aliases={aliases or 'none'}"
            )
            lines.append(f"    core: {core}")
            for family in sorted(children.get(str(group.get("node_id")), []), key=lambda item: int(item.get("priority_rank") or 999999))[:5]:
                f_aliases = ", ".join(f"{item['id']}:{item['count']}" for item in family.get("caption_aliases", [])[:2])
                lines.append(
                    f"    - family {family.get('priority_rank')}: {family.get('node_id')} | variants={family.get('variant_count')} "
                    f"support_max={family.get('support_cases_max')} score_max={family.get('structure_score_max')} aliases={f_aliases or 'none'}"
                )
                for variant in sorted(children.get(str(family.get("node_id")), []), key=lambda item: int(item.get("priority_rank") or 999999))[:3]:
                    v_aliases = ", ".join(f"{item['id']}:{item['count']}" for item in variant.get("caption_aliases", [])[:2])
                    v_geometry = ", ".join(variant.get("geometry_clusters", [])[:4])
                    lines.append(
                        f"      - variant {variant.get('priority_rank')}: support={variant.get('support_cases')} "
                        f"score={variant.get('structure_score')} aliases={v_aliases or 'none'}"
                    )
                    lines.append(f"        geometry: {v_geometry}")
        if len(groups) > max_children:
            lines.append(f"    ... {len(groups) - max_children} more structure groups")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, forest: dict[str, Any], *, detail_limit: int) -> None:
    candidates = [node for node in forest.get("nodes") or [] if node.get("node_kind") == "composition_family"]
    candidates.sort(key=lambda row: int(row.get("priority_rank") or 999999))
    node_by_id = {str(node.get("node_id") or ""): node for node in forest.get("nodes") or []}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in forest.get("edges") or []:
        parent = str(edge.get("parent_node_id") or "")
        child = str(edge.get("child_node_id") or "")
        if child in node_by_id:
            children[parent].append(node_by_id[child])
    summary = forest.get("summary") or {}
    lines = [
        "# HML3D Composition Pattern Forest v0",
        "",
        "Offline mining report. Caption aliases are naming diagnostics only.",
        "",
        "## Summary",
        "",
        f"- transactions: `{summary.get('transaction_count')}`",
        f"- frequent itemsets: `{summary.get('frequent_itemset_count')}`",
        f"- closed itemsets: `{summary.get('closed_itemset_count')}`",
        f"- candidate families: `{summary.get('candidate_node_count')}`",
        f"- raw itemset candidates: `{summary.get('raw_candidate_itemset_count')}`",
        f"- exported variants: `{summary.get('variant_node_count')}`",
        f"- status counts: `{summary.get('status_counts')}`",
        f"- root counts: `{summary.get('root_counts')}`",
        "",
        "## Top Candidates",
        "",
        "| rank | family | status | variants | support max | score max | aliases | geometry |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in candidates[:detail_limit]:
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("caption_aliases", [])[:4])
        geometry = ", ".join(row.get("geometry_clusters", [])[:5])
        lines.append(
            f"| {row.get('priority_rank')} | `{row.get('node_id')}` | {row.get('status')} | "
            f"{row.get('variant_count')} | {row.get('support_cases_max')} | {row.get('structure_score_max')} | {aliases or 'none'} | {geometry} |"
        )
    for row in candidates[:detail_limit]:
        variants = sorted(children.get(str(row.get("node_id")), []), key=lambda item: int(item.get("priority_rank") or 999999))
        lines.extend(
            [
                "",
                f"## {row.get('priority_rank')}. {row.get('node_id')}",
                "",
                f"- status: `{row.get('status')}`",
                f"- scope: `{row.get('scope')}`",
                f"- variant count: `{row.get('variant_count')}`",
                f"- support max: `{row.get('support_cases_max')}`",
                f"- support sum: `{row.get('support_cases_sum')}`",
                f"- occurrences sum: `{row.get('occurrences_sum')}`",
                f"- score max: `{row.get('structure_score_max')}`",
                f"- zones: `{row.get('zones')}`",
                f"- channels: `{row.get('channels')}`",
                f"- caption aliases: `{row.get('caption_aliases')}`",
                f"- geometry: `{row.get('geometry_clusters')}`",
                "- top variants:",
            ]
        )
        for variant in variants[:6]:
            aliases = ", ".join(f"{item['id']}:{item['count']}" for item in variant.get("caption_aliases", [])[:3])
            geometry = ", ".join(variant.get("geometry_clusters", [])[:5])
            lines.append(
                f"  - variant `{variant.get('priority_rank')}` support={variant.get('support_cases')} "
                f"score={variant.get('structure_score')} aliases={aliases or 'none'}"
            )
            lines.append(f"    geometry: {geometry}")
            for example in (variant.get("examples") or [])[:3]:
                captions = " / ".join(str(text).replace("\n", " ") for text in (example.get("caption_texts") or [])[:2])
                lines.append(f"    - `{example.get('case_id')}` span={example.get('span')}: {captions}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, forest: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "composition_pattern_forest.json", forest)
    compact = {
        key: forest[key]
        for key in ["schema_version", "runtime_policy", "inputs", "thresholds", "summary"]
    }
    compact["nodes"] = [
        {
            key: node.get(key)
            for key in [
                "node_id",
                "node_kind",
                "status",
                "scope",
                "display_name",
                "support_cases",
                "support_cases_max",
                "support_cases_sum",
                "structure_score",
                "structure_score_max",
                "variant_count",
                "channels",
                "zones",
                "geometry_clusters",
                "family_core_items",
                "caption_aliases",
                "caption_alias_purity",
                "caption_alias_purity_max",
                "priority_rank",
                "parent_family_id",
            ]
            if key in node
        }
        for node in forest.get("nodes") or []
    ]
    compact["edges"] = forest.get("edges") or []
    _write_json(output_dir / "composition_pattern_forest_compact.json", compact)
    _write_json(output_dir / "summary.json", forest["summary"])
    write_tree(output_dir / "composition_pattern_forest_tree.txt", forest, max_children=int(args.tree_children))
    write_report(output_dir / "composition_pattern_forest_report.md", forest, detail_limit=int(args.report_detail_limit))


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        seq = root / "seq.jsonl"
        source.write_text(
            "\n".join(
                [
                    json.dumps({"case_id": "a", "caption_texts": ["does jumping jacks"], "caption_alias_ids": ["jumping_jack"]}),
                    json.dumps({"case_id": "b", "caption_texts": ["does jumping jacks"], "caption_alias_ids": ["jumping_jack"]}),
                    json.dumps({"case_id": "c", "caption_texts": ["walks"], "caption_alias_ids": ["walk"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        left = {
            "symbol": "left_arm/LEFT_ARM_POSTURE/LA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT",
            "span": [0, 10],
            "channels": ["left_arm"],
            "geometry_clusters": ["LEFT_ARM_POSTURE/LA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT"],
        }
        right = {
            "symbol": "right_arm/RIGHT_ARM_POSTURE/RA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT",
            "span": [0, 10],
            "channels": ["right_arm"],
            "geometry_clusters": ["RIGHT_ARM_POSTURE/RA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT"],
        }
        vert = {
            "symbol": "whole_body_vertical/WHOLE_BODY_VERTICAL/WB_VERT_ARM_RAISE_COUPLED",
            "span": [0, 10],
            "channels": ["whole_body_vertical"],
            "geometry_clusters": ["WHOLE_BODY_VERTICAL/WB_VERT_ARM_RAISE_COUPLED"],
        }
        rows = []
        for case in ["a", "b"]:
            rows.extend(
                [
                    {"sequence_id": f"{case}::channel::left_arm", "case_id": case, "view": "channel::left_arm", "tokens": [left]},
                    {"sequence_id": f"{case}::channel::right_arm", "case_id": case, "view": "channel::right_arm", "tokens": [right]},
                    {"sequence_id": f"{case}::channel::whole_body_vertical", "case_id": case, "view": "channel::whole_body_vertical", "tokens": [vert]},
                ]
            )
        seq.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        args = argparse.Namespace(
            source_corpus=str(source),
            bpe_sequences=str(seq),
            output_dir=str(root / "out"),
            max_cases=None,
            parallel_overlap_min=0.3,
            min_occurrences=2,
            min_support_cases=2,
            min_itemset_size=2,
            max_itemset_size=4,
            max_transaction_items=12,
            closed_support_tolerance=0.0,
            min_structure_score=0.0,
            max_nodes=20,
            family_similarity_threshold=0.75,
            max_variants_per_family=8,
            example_limit=2,
            top_k=6,
            tree_children=10,
            report_detail_limit=5,
        )
        forest = build_forest(args)
        assert forest["summary"]["candidate_node_count"] >= 1
        assert any("WB_VERT_ARM_RAISE_COUPLED" in item for node in forest["nodes"] for item in node.get("geometry_clusters", []))
    print(json.dumps({"ok": True}, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build composition-level pattern forest from all-channel coactivations.")
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--parallel-overlap-min", type=float, default=0.30)
    parser.add_argument("--min-occurrences", type=int, default=20)
    parser.add_argument("--min-support-cases", type=int, default=20)
    parser.add_argument("--min-itemset-size", type=int, default=2)
    parser.add_argument("--max-itemset-size", type=int, default=5)
    parser.add_argument("--max-transaction-items", type=int, default=12)
    parser.add_argument("--closed-support-tolerance", type=float, default=0.0)
    parser.add_argument("--min-structure-score", type=float, default=4.0)
    parser.add_argument("--max-nodes", type=int, default=0, help="Maximum exported families; 0 keeps all candidates.")
    parser.add_argument("--family-similarity-threshold", type=float, default=0.75)
    parser.add_argument("--max-variants-per-family", type=int, default=12)
    parser.add_argument("--example-limit", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tree-children", type=int, default=60)
    parser.add_argument("--report-detail-limit", type=int, default=40)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    forest = build_forest(args)
    write_outputs(Path(args.output_dir), forest, args)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "summary": forest["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
