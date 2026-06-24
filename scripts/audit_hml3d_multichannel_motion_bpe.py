"""Audit HumanML3D multi-channel Motion-BPE.

What this script does:
1. Read the Layer3 event corpus.
2. Convert each case into channel events, overlap relations, and packets.
3. Learn per-channel temporal motifs first.
4. Promote frequently overlapping cross-channel motifs into coordination motifs.
5. Optionally refine coarse Layer3 geometry labels for the Motion-BPE view.
6. Write motif/family/forest artifacts for manual inspection.

Text policy:
- Captions are saved only as examples/diagnostics.
- Text keywords do not create motion tokens and do not affect BPE merges.

Quick check:
    python scripts/audit_hml3d_multichannel_motion_bpe.py --self-test

Small tuning run:
    python scripts/audit_hml3d_multichannel_motion_bpe.py \
      --max-records 200 \
      --num-merges 32 --min-pair-count 4 --min-pair-support 3 \
      --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug \
      --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_cache

Full run:
    python scripts/audit_hml3d_multichannel_motion_bpe.py \
      --num-merges 96 --min-pair-count 120 --min-pair-support 60 \
      --write-heavy-corpora \
      --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1 \
      --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1_cache

Tune these first:
- --num-merges: maximum learned motif count.
- --channel-merge-ratio: fraction of merge budget used for per-channel motifs.
- --min-pair-count: minimum total pair frequency.
- --min-pair-support: minimum distinct case support.
- --parallel-overlap-min: stricter/looser parallel packet grouping.
- --lead-lag-gap-max: stricter/looser short temporal relation grouping.

Cache note:
- Changing BPE thresholds reuses --cache-dir.
- Changing source/max-records/overlap/gap/relation/refinement flags rebuilds the cache.
- Check summary.json -> record_cache.status: hit / miss_built / rebuilt.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1")
DEFAULT_HML3D_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")
CACHE_SCHEMA_VERSION = "hml3d_multichannel_record_cache_v5_support_sidecars"

CHANNEL_ORDER = [
    "root_locomotion",
    "root_rotation",
    "whole_body_vertical",
    "whole_body_state",
    "torso",
    "left_arm",
    "right_arm",
    "bimanual",
    "left_leg",
    "right_leg",
    "acrobatics_or_inversion",
    "whole_body_support",
    "other",
]

CHANNEL_RANK = {channel: idx for idx, channel in enumerate(CHANNEL_ORDER)}


def _read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= max_records:
                break
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "metadata": cache_dir / "cache_metadata.json",
        "records": cache_dir / "multi_channel_record_cache.jsonl",
        "summary": cache_dir / "record_cache_summary.json",
    }


def _cache_config(args: argparse.Namespace) -> dict[str, Any]:
    arm_sidecar_enabled = _use_arm_trajectory_sidecar(args)
    body_sidecar_enabled = _use_body_level_sidecar(args)
    arm_reach_sidecar_enabled = _use_arm_reach_sidecar(args)
    hand_proximity_sidecar_enabled = _use_hand_proximity_sidecar(args)
    leg_lateral_sidecar_enabled = _use_leg_lateral_sidecar(args)
    body_support_sidecar_enabled = _use_body_support_sidecar(args)
    raw_joint_sidecar_enabled = (
        arm_sidecar_enabled
        or body_sidecar_enabled
        or arm_reach_sidecar_enabled
        or hand_proximity_sidecar_enabled
        or leg_lateral_sidecar_enabled
        or body_support_sidecar_enabled
    )
    return {
        "source_signature": _source_signature(Path(args.source_corpus)),
        "max_records": args.max_records,
        "parallel_overlap_min": float(args.parallel_overlap_min),
        "lead_lag_gap_max": int(args.lead_lag_gap_max),
        "observable_refinement": str(getattr(args, "observable_refinement", "v1")),
        "arm_trajectory_sidecar": arm_sidecar_enabled,
        "arm_reach_sidecar": arm_reach_sidecar_enabled,
        "hand_proximity_sidecar": hand_proximity_sidecar_enabled,
        "leg_lateral_sidecar": leg_lateral_sidecar_enabled,
        "body_level_sidecar": body_sidecar_enabled,
        "body_support_sidecar": body_support_sidecar_enabled,
        "hml3d_joints3d": str(Path(getattr(args, "hml3d_root", DEFAULT_HML3D_ROOT)) / "joints3d.pth") if raw_joint_sidecar_enabled else "",
        "arm_span_gap": int(getattr(args, "arm_span_gap", 6)),
        "arm_span_pad": int(getattr(args, "arm_span_pad", 3)),
        "arm_min_radius": float(getattr(args, "arm_min_radius", 0.18)),
        "arm_circle_min_abs_deg": float(getattr(args, "arm_circle_min_abs_deg", 540.0)),
        "arm_circle_max_radius_cv": float(getattr(args, "arm_circle_max_radius_cv", 0.38)),
        "arm_large_arc_min_abs_deg": float(getattr(args, "arm_large_arc_min_abs_deg", 180.0)),
        "arm_large_arc_min_path": float(getattr(args, "arm_large_arc_min_path", 0.70)),
        "arm_large_arc_min_range": float(getattr(args, "arm_large_arc_min_range", 0.40)),
        "arm_large_arc_max_radius_cv": float(getattr(args, "arm_large_arc_max_radius_cv", 0.55)),
        "arm_min_planarity": float(getattr(args, "arm_min_planarity", 0.80)),
        "arm_reach_span_gap": int(getattr(args, "arm_reach_span_gap", 4)),
        "arm_reach_span_pad": int(getattr(args, "arm_reach_span_pad", 3)),
        "arm_reach_min_delta": float(getattr(args, "arm_reach_min_delta", 0.16)),
        "arm_reach_min_path": float(getattr(args, "arm_reach_min_path", 0.22)),
        "arm_reach_min_peak": float(getattr(args, "arm_reach_min_peak", 0.24)),
        "arm_reach_retract_ratio": float(getattr(args, "arm_reach_retract_ratio", 0.55)),
        "arm_reach_min_forward_component": float(getattr(args, "arm_reach_min_forward_component", 0.10)),
        "hand_head_proximity_threshold": float(getattr(args, "hand_head_proximity_threshold", 0.30)),
        "hand_head_min_run": int(getattr(args, "hand_head_min_run", 6)),
        "hand_head_hold_min_duration": int(getattr(args, "hand_head_hold_min_duration", 10)),
        "hand_head_transition_window": int(getattr(args, "hand_head_transition_window", 14)),
        "hand_head_min_delta": float(getattr(args, "hand_head_min_delta", 0.14)),
        "hand_head_merge_gap": int(getattr(args, "hand_head_merge_gap", 4)),
        "leg_lateral_threshold": float(getattr(args, "leg_lateral_threshold", 0.16)),
        "leg_lateral_min_run": int(getattr(args, "leg_lateral_min_run", 5)),
        "leg_lateral_hold_min_duration": int(getattr(args, "leg_lateral_hold_min_duration", 10)),
        "leg_lateral_transition_window": int(getattr(args, "leg_lateral_transition_window", 10)),
        "leg_lateral_min_delta": float(getattr(args, "leg_lateral_min_delta", 0.12)),
        "leg_lateral_merge_gap": int(getattr(args, "leg_lateral_merge_gap", 4)),
        "body_low_threshold": float(getattr(args, "body_low_threshold", 0.52)),
        "body_high_threshold": float(getattr(args, "body_high_threshold", 0.58)),
        "body_level_min_run": int(getattr(args, "body_level_min_run", 5)),
        "body_long_low_min_duration": int(getattr(args, "body_long_low_min_duration", 30)),
        "body_high_state_min_duration": int(getattr(args, "body_high_state_min_duration", 8)),
        "body_transition_max_gap": int(getattr(args, "body_transition_max_gap", 40)),
        "body_emit_high_state": bool(getattr(args, "body_emit_high_state", False)),
        "body_support_min_run": int(getattr(args, "body_support_min_run", 6)),
        "body_support_merge_gap": int(getattr(args, "body_support_merge_gap", 4)),
        "body_prone_height_ratio": float(getattr(args, "body_prone_height_ratio", 0.34)),
        "body_horizontal_axis_threshold": float(getattr(args, "body_horizontal_axis_threshold", 0.58)),
        "body_inverted_head_margin": float(getattr(args, "body_inverted_head_margin", 0.10)),
        "body_hand_floor_ratio": float(getattr(args, "body_hand_floor_ratio", 0.12)),
        "body_foot_high_ratio": float(getattr(args, "body_foot_high_ratio", 0.75)),
    }


def _cache_matches(metadata: dict[str, Any], args: argparse.Namespace) -> bool:
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    return metadata.get("config") == _cache_config(args)


def _use_arm_trajectory_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_arm_trajectory_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) in {"v3", "v4"}


def _use_body_level_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_body_level_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) in {"v3", "v4"}


def _use_arm_reach_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_arm_reach_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) in {"v3", "v4"}


def _use_hand_proximity_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_hand_proximity_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) in {"v3", "v4"}


def _use_leg_lateral_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_leg_lateral_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) in {"v3", "v4"}


def _use_body_support_sidecar(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "disable_body_support_sidecar", False)):
        return False
    return str(getattr(args, "observable_refinement", "v1")) == "v4"


def _load_hml3d_joints_pack(hml3d_root: Path) -> dict[str, Any]:
    import torch

    return torch.load(hml3d_root / "joints3d.pth", map_location="cpu")


def _cache_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    channel_event_count = sum(len(record.get("channel_events") or []) for record in records)
    arm_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_trajectory" in set(event.get("observable_refinement_tags") or [])
    )
    body_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_body_level" in set(event.get("observable_refinement_tags") or [])
    )
    arm_reach_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_reach" in set(event.get("observable_refinement_tags") or [])
    )
    hand_proximity_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_hand_proximity" in set(event.get("observable_refinement_tags") or [])
    )
    leg_lateral_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_leg_lateral" in set(event.get("observable_refinement_tags") or [])
    )
    body_support_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_body_support" in set(event.get("observable_refinement_tags") or [])
    )
    packet_count = sum(len(record.get("packets") or []) for record in records)
    parallel_packet_count = sum(
        1
        for record in records
        for packet in (record.get("packets") or [])
        if packet.get("packet_type") == "parallel"
    )
    relation_type_counts: Counter[str] = Counter()
    for record in records:
        relation_type_counts.update({str(k): int(v) for k, v in (record.get("relation_type_counts") or {}).items()})
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "record_count": len(records),
        "channel_event_count": channel_event_count,
        "arm_trajectory_sidecar_event_count": arm_sidecar_event_count,
        "arm_reach_sidecar_event_count": arm_reach_sidecar_event_count,
        "hand_proximity_sidecar_event_count": hand_proximity_sidecar_event_count,
        "leg_lateral_sidecar_event_count": leg_lateral_sidecar_event_count,
        "body_level_sidecar_event_count": body_sidecar_event_count,
        "body_support_sidecar_event_count": body_support_sidecar_event_count,
        "packet_count": packet_count,
        "parallel_packet_count": parallel_packet_count,
        "relation_count": sum(int(record.get("relation_count") or 0) for record in records),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
    }


def _load_or_build_multichannel_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    source_path = Path(args.source_corpus)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        paths = _cache_paths(cache_dir)
        if not args.rebuild_cache and paths["metadata"].exists() and paths["records"].exists():
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if _cache_matches(metadata, args):
                records = _read_jsonl(paths["records"])
                return records, int(metadata.get("source_record_count", len(records))), {
                    "enabled": True,
                    "status": "hit",
                    "cache_dir": str(cache_dir),
                    "record_cache": str(paths["records"]),
                    "metadata": str(paths["metadata"]),
                }

    source_records = _read_jsonl(source_path, max_records=args.max_records)
    joints_pack = (
        _load_hml3d_joints_pack(Path(args.hml3d_root))
        if (
            _use_arm_trajectory_sidecar(args)
            or _use_arm_reach_sidecar(args)
            or _use_hand_proximity_sidecar(args)
            or _use_leg_lateral_sidecar(args)
            or _use_body_level_sidecar(args)
            or _use_body_support_sidecar(args)
        )
        else None
    )
    records = [build_multichannel_record(record, args, joints_pack=joints_pack) for record in source_records]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = _cache_paths(cache_dir)
        _write_jsonl(paths["records"], records)
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config": _cache_config(args),
            "source_record_count": len(source_records),
            "record_count": len(records),
            "files": {key: str(path) for key, path in paths.items()},
        }
        _write_json(paths["metadata"], metadata)
        _write_json(paths["summary"], _cache_summary(records))
        return records, len(source_records), {
            "enabled": True,
            "status": "rebuilt" if args.rebuild_cache else "miss_built",
            "cache_dir": str(cache_dir),
            "record_cache": str(paths["records"]),
            "metadata": str(paths["metadata"]),
        }
    return records, len(source_records), {
        "enabled": False,
        "status": "disabled",
    }


def _duration_bin(duration: int) -> str:
    if duration <= 3:
        return "xs"
    if duration <= 8:
        return "s"
    if duration <= 20:
        return "m"
    if duration <= 60:
        return "l"
    return "xl"


def _magnitude_bin(value: float, unit: str) -> str:
    if not math.isfinite(value) or value <= 0.0:
        return "none"
    if unit == "deg":
        if value < 30.0:
            return "deg_xs"
        if value < 90.0:
            return "deg_s"
        if value < 180.0:
            return "deg_m"
        if value < 360.0:
            return "deg_l"
        return "deg_xl"
    if unit == "m":
        if value < 0.06:
            return "m_xs"
        if value < 0.16:
            return "m_s"
        if value < 0.45:
            return "m_m"
        if value < 1.5:
            return "m_l"
        return "m_xl"
    if unit == "ratio":
        if value < 0.15:
            return "r_s"
        if value < 0.45:
            return "r_m"
        return "r_l"
    if value < 1.0:
        return "v_s"
    if value < 5.0:
        return "v_m"
    return "v_l"


def _count_bin(count: Any) -> str | None:
    try:
        value = int(count)
    except (TypeError, ValueError):
        return None
    if value <= 1:
        return "c1"
    if value <= 3:
        return "c2_3"
    if value <= 6:
        return "c4_6"
    return "c7p"


def _speed_bin(value: float, unit: str) -> str:
    if not math.isfinite(value) or value <= 0.0:
        return "speed_none"
    if unit == "deg_per_frame":
        if value < 2.0:
            return "omega_slow"
        if value < 6.0:
            return "omega_med"
        return "omega_fast"
    if unit == "count_per_frame":
        if value < 0.04:
            return "rate_slow"
        if value < 0.10:
            return "rate_med"
        return "rate_fast"
    if value < 0.010:
        return "speed_slow"
    if value < 0.035:
        return "speed_med"
    return "speed_fast"


def _channel_for_event(event: dict[str, Any]) -> str:
    super_family = str(event.get("super_family") or "")
    part = str(event.get("part") or "")
    cluster = str(event.get("cluster_id") or "")
    if super_family == "WHOLE_BODY_LOCOMOTION":
        return "root_rotation" if cluster.startswith("LOCO_TURN_") else "root_locomotion"
    if super_family == "WHOLE_BODY_ROTATION":
        return "root_rotation"
    if super_family == "WHOLE_BODY_VERTICAL":
        return "whole_body_vertical"
    if super_family in {"WHOLE_BODY_POSTURE", "WHOLE_BODY_STATE", "WHOLE_BODY_LEVEL"}:
        return "whole_body_state"
    if super_family == "WHOLE_BODY_SUPPORT":
        return "whole_body_support"
    if super_family.startswith("TORSO_") or part == "torso":
        return "torso"
    if super_family.startswith("LEFT_ARM_") or part == "left_arm":
        return "left_arm"
    if super_family.startswith("RIGHT_ARM_") or part == "right_arm":
        return "right_arm"
    if super_family.startswith("BIMANUAL_") or part == "bimanual":
        return "bimanual"
    if super_family.startswith("LEFT_LEG_") or part == "left_leg":
        return "left_leg"
    if super_family.startswith("RIGHT_LEG_") or part == "right_leg":
        return "right_leg"
    if "ACROBAT" in super_family or "INVERT" in cluster:
        return "acrobatics_or_inversion"
    return "other"


def _span(event: dict[str, Any]) -> list[int]:
    raw = event.get("span") or [0, 0]
    if len(raw) != 2:
        return [0, 0]
    start = int(raw[0])
    end = int(raw[1])
    if end < start:
        start, end = end, start
    return [start, end]


def _duration(event: dict[str, Any]) -> int:
    raw = event.get("duration")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    start, end = _span(event)
    return max(1, end - start + 1)


def _magnitude(event: dict[str, Any]) -> float:
    try:
        return abs(float(event.get("magnitude") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _channel_event_symbol(event: dict[str, Any], channel: str) -> str:
    geometry = str(event.get("geometry_cluster_id") or f"{event.get('super_family')}/{event.get('cluster_id')}")
    direction = str(event.get("direction") or "none")
    duration = _duration(event)
    magnitude = _magnitude(event)
    unit = str(event.get("unit") or "")
    dur_bin = _duration_bin(duration)
    mag_bin = _magnitude_bin(magnitude, unit)
    count_bin = _count_bin(event.get("count"))
    if unit == "deg":
        speed = magnitude / max(duration, 1)
        speed_unit = "deg_per_frame"
    elif event.get("count") is not None:
        try:
            speed = float(event.get("count") or 0.0) / max(duration, 1)
        except (TypeError, ValueError):
            speed = 0.0
        speed_unit = "count_per_frame"
    else:
        speed = magnitude / max(duration, 1)
        speed_unit = "m_per_frame" if unit == "m" else "value_per_frame"
    speed = _speed_bin(speed, speed_unit)
    parts = [f"{channel}/{geometry}", f"dir={direction}", f"dur={dur_bin}", f"mag={mag_bin}", f"speed={speed}"]
    if count_bin:
        parts.append(f"count={count_bin}")
    return "|".join(parts)


def _event_sort_key(event: dict[str, Any]) -> tuple[int, int, int, str]:
    start, end = _span(event)
    channel = str(event.get("channel") or "other")
    return (start, end, CHANNEL_RANK.get(channel, 999), str(event.get("event_id") or ""))


def _copy_event_with_refined_geometry(event: dict[str, Any], refined_cluster_id: str, tags: list[str]) -> dict[str, Any]:
    copied = dict(event)
    raw_geometry = str(event.get("geometry_cluster_id") or f"{event.get('super_family')}/{event.get('cluster_id')}")
    super_family = str(event.get("super_family") or "")
    merged_tags = list(dict.fromkeys([*(event.get("observable_refinement_tags") or []), *tags]))
    copied["raw_geometry_cluster_id"] = raw_geometry
    copied["raw_cluster_id"] = str(event.get("cluster_id") or "")
    copied["cluster_id"] = refined_cluster_id
    copied["geometry_cluster_id"] = f"{super_family}/{refined_cluster_id}"
    copied["observable_refinement_tags"] = merged_tags
    return copied


def _overlaps_any(event: dict[str, Any], others: list[dict[str, Any]], *, min_ratio: float = 0.15, max_gap: int | None = None) -> bool:
    for other in others:
        if _overlap_ratio(event, other) >= min_ratio:
            return True
        if max_gap is not None and _gap(event, other) <= max_gap:
            return True
    return False


def _event_signature(event: dict[str, Any]) -> dict[str, Any]:
    return event.get("motion_signature") or {}


def _events_with_super_family(events: list[dict[str, Any]], super_family: str) -> list[dict[str, Any]]:
    return [event for event in events if str(event.get("super_family") or "") == super_family]


def _events_with_cluster_prefix(events: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    return [event for event in events if str(event.get("cluster_id") or "").startswith(prefixes)]


def _events_with_clusters(events: list[dict[str, Any]], cluster_ids: set[str]) -> list[dict[str, Any]]:
    return [event for event in events if str(event.get("cluster_id") or "") in cluster_ids]


def _near_events(
    event: dict[str, Any],
    others: list[dict[str, Any]],
    *,
    min_ratio: float = 0.10,
    max_gap: int = 4,
) -> list[dict[str, Any]]:
    return [other for other in others if _overlap_ratio(event, other) >= min_ratio or _gap(event, other) <= max_gap]


def _has_before(event: dict[str, Any], others: list[dict[str, Any]], max_gap: int = 10) -> bool:
    start, _ = _span(event)
    for other in others:
        _, other_end = _span(other)
        if other_end <= start and _gap(event, other) <= max_gap:
            return True
    return False


def _has_after(event: dict[str, Any], others: list[dict[str, Any]], max_gap: int = 10) -> bool:
    _, end = _span(event)
    for other in others:
        other_start, _ = _span(other)
        if other_start >= end and _gap(event, other) <= max_gap:
            return True
    return False


def _cluster_suffix_after(prefix: str, cluster: str) -> str:
    if cluster.startswith(prefix):
        return cluster[len(prefix) :]
    return cluster


def _loco_direction(cluster: str, direction: str) -> str:
    if "FORWARD" in cluster or direction == "forward":
        return "FORWARD"
    if "BACKWARD" in cluster or direction == "backward":
        return "BACKWARD"
    if "LEFT" in cluster or direction == "left":
        return "LEFT"
    if "RIGHT" in cluster or direction == "right":
        return "RIGHT"
    if "MIXED" in cluster or direction == "mixed":
        return "MIXED"
    return "ACTIVE"


def _loco_speed(cluster: str, duration: int, magnitude: float) -> str:
    if "FAST" in cluster:
        return "FAST"
    if "MEDIUM" in cluster:
        return "MEDIUM"
    if "SLOW" in cluster:
        return "SLOW"
    speed = magnitude / max(duration, 1)
    if speed < 0.010:
        return "SLOW"
    if speed < 0.035:
        return "MEDIUM"
    return "FAST"


def _turn_side_and_angle(cluster: str, direction: str) -> tuple[str, str]:
    side = "LEFT" if ("LEFT" in cluster or direction == "left") else "RIGHT" if ("RIGHT" in cluster or direction == "right") else "MIXED"
    for angle in ("THREE_QUARTER", "QUARTER", "SMALL", "HALF", "FULL", "MULTI", "QTR"):
        if angle in cluster:
            return side, "QUARTER" if angle == "QTR" else angle
    return side, "GENERIC"


def _turn_tempo(duration: int, magnitude: float) -> str:
    speed = magnitude / max(duration, 1)
    if speed < 2.0:
        return "SLOW"
    if speed < 6.0:
        return "MEDIUM"
    return "FAST"


def _refine_root_locomotion_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    direction = _loco_direction(cluster, str(event.get("direction") or ""))
    duration = _duration(event)
    magnitude = _magnitude(event)
    speed = _loco_speed(cluster, duration, magnitude)
    tags: list[str] = []
    vertical_events = _events_with_super_family(all_events, "WHOLE_BODY_VERTICAL")
    leg_events = _events_with_cluster_prefix(all_events, ("LL_", "RL_"))
    turn_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_ROTATION"
        or str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    arm_periodic_events = [
        other for other in all_events
        if str(other.get("super_family") or "") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC", "BIMANUAL_PERIODIC"}
    ]
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.08, max_gap=4)
    coupled_leg = _overlaps_any(event, leg_events, min_ratio=0.08, max_gap=4)
    coupled_turn = _overlaps_any(event, turn_events, min_ratio=0.08, max_gap=8)
    coupled_arm = _overlaps_any(event, arm_periodic_events, min_ratio=0.08, max_gap=4)
    if coupled_vertical:
        tags.append("vertical_coupled")
    if coupled_leg:
        tags.append("leg_coupled")
    if coupled_turn:
        tags.append("turn_coupled")
    if coupled_arm:
        tags.append("arm_coupled")
    if magnitude < 0.18 or (direction in {"ACTIVE", "MIXED"} and magnitude < 0.35):
        tags.append("weak_root_drift")
        return f"LOCO_ROOT_DRIFT_{direction}_{speed}", tags
    if direction in {"ACTIVE", "MIXED"} or coupled_turn:
        tags.append("path_fragment")
        return f"LOCO_PATH_FRAGMENT_{direction}_{speed}", tags
    if coupled_vertical or coupled_leg:
        tags.append("gait_context")
        return f"LOCO_GAIT_CONTEXT_{direction}_{speed}", tags
    tags.append("translation_context")
    return f"LOCO_TRANSLATION_{direction}_{speed}", tags


def _refine_root_turn_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    super_family = str(event.get("super_family") or "")
    side, angle = _turn_side_and_angle(cluster, str(event.get("direction") or ""))
    tempo = _turn_tempo(_duration(event), _magnitude(event))
    tags = [f"turn_angle_{angle.lower()}", f"turn_tempo_{tempo.lower()}"]
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    vertical_events = _events_with_super_family(all_events, "WHOLE_BODY_VERTICAL")
    if _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=8):
        tags.append("path_turn_context")
        role = "PATH"
    elif _overlaps_any(event, vertical_events, min_ratio=0.08, max_gap=4):
        tags.append("vertical_turn_context")
        role = "VERTICAL"
    else:
        tags.append("isolated_turn")
        role = "ISOLATED"
    prefix = "LOCO_TURN" if super_family == "WHOLE_BODY_LOCOMOTION" else "WB_ROT"
    return f"{prefix}_{side}_{angle}_{tempo}_{role}", tags


def _refine_leg_forward_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    side = "LL" if cluster.startswith("LL_") else "RL"
    sig = _event_signature(event)
    tags: list[str] = []
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    vertical_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_VERTICAL"
    ]
    coupled_loco = bool(sig.get("coupled_with_locomotion")) or _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.12, max_gap=3)
    duration = _duration(event)
    magnitude = _magnitude(event)
    speed = magnitude / max(duration, 1)
    if coupled_loco:
        tags.append("locomotion_coupled")
    if coupled_vertical:
        tags.append("vertical_coupled")
    if cluster.endswith("LEG_FORWARD_POSE") or (duration >= 14 and speed < 0.025):
        tags.append("hold_like")
        return f"{side}_LEG_FORWARD_HOLD_POSE", tags
    if coupled_loco and magnitude < 0.55:
        tags.append("gait_phase")
        return f"{side}_LEG_FORWARD_GAIT_SWING", tags
    if coupled_vertical and duration <= 12 and speed >= 0.030:
        tags.append("impulse_like")
        return f"{side}_LEG_FORWARD_HOP_OR_KICK_IMPULSE", tags
    if magnitude >= 0.45 and duration <= 12:
        tags.append("impulse_like")
        return f"{side}_LEG_FORWARD_KICK_IMPULSE", tags
    tags.append("ambiguous_forward_leg")
    return f"{side}_LEG_FORWARD_UNRESOLVED", tags


def _refine_vertical_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    sig = _event_signature(event)
    tags: list[str] = []
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    low_body_events = [
        other for other in all_events
        if str(other.get("cluster_id") or "") in {"WB_SQUAT_HOLD", "WB_LOW_BODY_HOLD"}
    ]
    leg_events = [
        other for other in all_events
        if str(other.get("cluster_id") or "") in {"LL_KICK_FORWARD", "RL_KICK_FORWARD", "LL_LEG_FORWARD_POSE", "RL_LEG_FORWARD_POSE"}
    ]
    arm_raise_events = [
        other for other in all_events
        if str(other.get("cluster_id") or "") in {"BI_RAISE_SPREAD", "LA_HAND_HIGH", "RA_HAND_HIGH"}
    ]
    coupled_loco = bool(sig.get("coupled_with_locomotion")) or _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_low = _overlaps_any(event, low_body_events, min_ratio=0.12, max_gap=5)
    coupled_leg = _overlaps_any(event, leg_events, min_ratio=0.10, max_gap=4)
    coupled_arm_raise = _overlaps_any(event, arm_raise_events, min_ratio=0.12, max_gap=4)
    magnitude = _magnitude(event)
    direction = str(event.get("direction") or "")
    repeat_mode = str(sig.get("repeat_mode") or "")
    if magnitude < 0.08 and (coupled_loco or coupled_leg):
        tags.append("gait_bounce")
        return "WB_VERT_GAIT_BOUNCE", tags
    if coupled_low and direction == "down":
        tags.append("low_body_transition")
        return "WB_VERT_LOW_BODY_DESCENT", tags
    if coupled_low and direction == "up":
        tags.append("low_body_transition")
        return "WB_VERT_LOW_BODY_RISE", tags
    if coupled_arm_raise and repeat_mode in {"cycle", "repeated_cycle"}:
        tags.append("arm_raise_coupled")
        return "WB_VERT_ARM_RAISE_COORDINATED_CYCLE", tags
    if coupled_arm_raise and magnitude >= 0.12:
        tags.append("arm_raise_coupled")
        return "WB_VERT_ARM_RAISE_COUPLED", tags
    if magnitude >= 0.20 and direction == "up":
        tags.append("salient_jump_up")
        return "WB_VERT_JUMP_UP_IMPULSE", tags
    if magnitude >= 0.20 and direction == "down":
        tags.append("salient_descent")
        return "WB_VERT_SALIENT_DESCENT", tags
    if cluster in {"WB_VERT_CYCLE", "WB_VERT_REP", "WB_VERT_REP_ALT"}:
        tags.append("vertical_cycle")
        return "WB_VERT_GENERIC_CYCLE", tags
    tags.append("generic_vertical")
    return f"{cluster}_REFINED_GENERIC", tags


def _refine_low_body_posture_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    tags: list[str] = []
    vertical_down = [other for other in all_events if str(other.get("cluster_id") or "") == "WB_VERT_DOWN"]
    vertical_up = [other for other in all_events if str(other.get("cluster_id") or "") == "WB_VERT_UP"]
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    leg_events = _events_with_cluster_prefix(all_events, ("LL_", "RL_"))
    torso_events = [
        other for other in all_events
        if str(other.get("super_family") or "") in {"TORSO_POSTURE", "TORSO_PERIODIC"}
    ]
    down_near = _overlaps_any(event, vertical_down, min_ratio=0.08, max_gap=8) or _has_before(event, vertical_down, max_gap=10)
    up_near = _overlaps_any(event, vertical_up, min_ratio=0.08, max_gap=8) or _has_after(event, vertical_up, max_gap=10)
    coupled_loco = _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_leg = _overlaps_any(event, leg_events, min_ratio=0.10, max_gap=4)
    coupled_torso = _overlaps_any(event, torso_events, min_ratio=0.10, max_gap=4)
    if coupled_loco:
        tags.append("locomotion_coupled")
    if coupled_leg:
        tags.append("leg_coupled")
    if coupled_torso:
        tags.append("torso_coupled")
    if down_near and up_near:
        tags.append("low_body_down_up_cycle")
        return "WB_LOW_BODY_DOWN_UP_CYCLE", tags
    if down_near:
        tags.append("low_body_descent_hold")
        return "WB_LOW_BODY_DESCENT_HOLD", tags
    if up_near:
        tags.append("low_body_rise_from_low")
        return "WB_LOW_BODY_RISE_FROM_LOW", tags
    if _duration(event) >= 28 and not coupled_loco:
        tags.append("sustained_low_body")
        return f"{cluster}_SUSTAINED", tags
    if coupled_loco:
        tags.append("low_body_gait_context")
        return f"{cluster}_LOCO_CONTEXT", tags
    if coupled_leg:
        tags.append("low_body_leg_extension_context")
        return f"{cluster}_LEG_CONTEXT", tags
    tags.append("generic_low_body")
    return f"{cluster}_REFINED_GENERIC", tags


def _refine_torso_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    super_family = str(event.get("super_family") or "")
    direction = str(event.get("direction") or "none").upper()
    tags: list[str] = []
    low_body_events = _events_with_clusters(all_events, {"WB_SQUAT_HOLD", "WB_LOW_BODY_HOLD"})
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    vertical_events = _events_with_super_family(all_events, "WHOLE_BODY_VERTICAL")
    arm_events = [
        other for other in all_events
        if str(other.get("super_family") or "") in {"LEFT_ARM_PERIODIC", "RIGHT_ARM_PERIODIC", "BIMANUAL_PERIODIC", "LEFT_ARM_POSTURE", "RIGHT_ARM_POSTURE"}
    ]
    coupled_low = _overlaps_any(event, low_body_events, min_ratio=0.10, max_gap=5)
    coupled_loco = _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.08, max_gap=4)
    coupled_arm = _overlaps_any(event, arm_events, min_ratio=0.08, max_gap=4)
    if coupled_low:
        tags.append("low_body_coupled")
    if coupled_loco:
        tags.append("locomotion_coupled")
    if coupled_vertical:
        tags.append("vertical_coupled")
    if coupled_arm:
        tags.append("arm_coupled")
    if super_family == "TORSO_POSTURE" and _duration(event) >= 24 and not coupled_loco:
        tags.append("sustained_torso_posture")
        return f"{cluster}_SUSTAINED", tags
    if coupled_low:
        return f"{cluster}_LOW_BODY_CONTEXT", tags
    if coupled_loco:
        return f"{cluster}_LOCO_CONTEXT", tags
    if coupled_vertical:
        return f"{cluster}_VERTICAL_CONTEXT", tags
    if super_family == "TORSO_PERIODIC":
        repeat = str(_event_signature(event).get("repeat_mode") or "single").upper()
        tags.append("torso_periodic")
        return f"{cluster}_{repeat}_{direction}", tags
    tags.append("generic_torso")
    return f"{cluster}_REFINED_GENERIC", tags


def _refine_arm_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    side = "LA" if cluster.startswith("LA_") else "RA" if cluster.startswith("RA_") else "BI"
    sig = _event_signature(event)
    tags: list[str] = []
    opposite_prefix = "RA_" if side == "LA" else "LA_" if side == "RA" else ""
    opposite_events = [
        other for other in all_events
        if opposite_prefix and str(other.get("cluster_id") or "").startswith(opposite_prefix)
    ]
    vertical_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_VERTICAL"
    ]
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
    ]
    bimanual_events = [
        other for other in all_events
        if str(other.get("super_family") or "").startswith("BIMANUAL_")
    ]
    coupled_loco = bool(sig.get("coupled_with_locomotion")) or _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.12, max_gap=4)
    coupled_opposite = _overlaps_any(event, opposite_events, min_ratio=0.35, max_gap=2)
    coupled_bimanual = _overlaps_any(event, bimanual_events, min_ratio=0.20, max_gap=3)
    direction = str(event.get("direction") or "none")
    repeat_mode = str(sig.get("repeat_mode") or "")
    if side in {"LA", "RA"} and coupled_opposite and coupled_vertical:
        tags.extend(["bilateral", "vertical_coupled"])
        return f"{side}_BILATERAL_VERTICAL_ARM_CYCLE", tags
    if side in {"LA", "RA"} and coupled_opposite:
        tags.append("bilateral")
        return f"{side}_BILATERAL_ARM_PERIODIC_{direction.upper()}", tags
    if coupled_loco:
        tags.append("locomotion_coupled")
        return f"{side}_LOCO_ARM_SWING_{direction.upper()}", tags
    if coupled_vertical:
        tags.append("vertical_coupled")
        return f"{side}_VERTICAL_COUPLED_ARM_PERIODIC_{direction.upper()}", tags
    if coupled_bimanual:
        tags.append("bimanual_context")
        return f"{side}_BIMANUAL_CONTEXT_ARM_PERIODIC_{direction.upper()}", tags
    if repeat_mode in {"repeated_cycle", "cycle"}:
        tags.append("isolated_periodic")
        return f"{side}_ISOLATED_ARM_PERIODIC_{direction.upper()}", tags
    tags.append("generic_arm")
    return f"{cluster}_REFINED_GENERIC", tags


def _refine_arm_posture_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    side = "LA" if cluster.startswith("LA_") else "RA"
    tags: list[str] = []
    opposite = "RA_HAND_HIGH" if side == "LA" else "LA_HAND_HIGH"
    opposite_events = _events_with_clusters(all_events, {opposite})
    vertical_events = _events_with_super_family(all_events, "WHOLE_BODY_VERTICAL")
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
    ]
    bimanual_events = [
        other for other in all_events
        if str(other.get("super_family") or "").startswith("BIMANUAL_")
    ]
    coupled_opposite = _overlaps_any(event, opposite_events, min_ratio=0.25, max_gap=3)
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.10, max_gap=4)
    coupled_loco = _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_bimanual = _overlaps_any(event, bimanual_events, min_ratio=0.12, max_gap=3)
    if coupled_opposite:
        tags.append("bilateral_high_pose")
    if coupled_vertical:
        tags.append("vertical_coupled")
    if coupled_loco:
        tags.append("locomotion_coupled")
    if coupled_bimanual:
        tags.append("bimanual_context")
    if coupled_opposite and coupled_vertical:
        return f"{side}_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT", tags
    if coupled_opposite:
        return f"{side}_BILATERAL_HIGH_POSE", tags
    if coupled_loco:
        return f"{side}_HIGH_POSE_LOCO_CONTEXT", tags
    if _duration(event) >= 20:
        tags.append("hold_like")
        return f"{side}_HIGH_POSE_HOLD", tags
    tags.append("transient_high_pose")
    return f"{side}_HIGH_POSE_TRANSIENT", tags


def _refine_bimanual_event(event: dict[str, Any], all_events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cluster = str(event.get("cluster_id") or "")
    tags: list[str] = []
    vertical_events = _events_with_super_family(all_events, "WHOLE_BODY_VERTICAL")
    locomotion_events = [
        other for other in all_events
        if str(other.get("super_family") or "") == "WHOLE_BODY_LOCOMOTION"
        and not str(other.get("cluster_id") or "").startswith("LOCO_TURN_")
    ]
    low_body_events = _events_with_clusters(all_events, {"WB_SQUAT_HOLD", "WB_LOW_BODY_HOLD"})
    hand_high_events = _events_with_clusters(all_events, {"LA_HAND_HIGH", "RA_HAND_HIGH"})
    coupled_vertical = _overlaps_any(event, vertical_events, min_ratio=0.10, max_gap=4)
    coupled_loco = _overlaps_any(event, locomotion_events, min_ratio=0.08, max_gap=4)
    coupled_low = _overlaps_any(event, low_body_events, min_ratio=0.10, max_gap=5)
    coupled_high = _overlaps_any(event, hand_high_events, min_ratio=0.12, max_gap=3)
    repeat_mode = str(_event_signature(event).get("repeat_mode") or "")
    if coupled_vertical:
        tags.append("vertical_coupled")
    if coupled_loco:
        tags.append("locomotion_coupled")
    if coupled_low:
        tags.append("low_body_coupled")
    if coupled_high:
        tags.append("hand_high_coupled")
    if cluster == "BI_RAISE_SPREAD" and coupled_vertical:
        return "BI_RAISE_SPREAD_VERTICAL_CONTEXT", tags
    if cluster.startswith("BI_HANDS_CLOSE") and coupled_vertical:
        return "BI_HANDS_CLOSE_VERTICAL_CONTEXT", tags
    if coupled_low:
        return f"{cluster}_LOW_BODY_CONTEXT", tags
    if coupled_loco:
        return f"{cluster}_LOCO_CONTEXT", tags
    if repeat_mode in {"cycle", "repeated_cycle"}:
        tags.append("bimanual_cycle")
        return f"{cluster}_CYCLE_CONTEXT", tags
    tags.append("generic_bimanual")
    return f"{cluster}_REFINED_GENERIC", tags


def refine_events_for_motion_bpe(events: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "v1":
        return events
    if mode not in {"v2", "v3", "v4"}:
        raise ValueError(f"unsupported observable refinement mode: {mode}")
    refined: list[dict[str, Any]] = []
    for event in events:
        cluster = str(event.get("cluster_id") or "")
        super_family = str(event.get("super_family") or "")
        if super_family == "WHOLE_BODY_LOCOMOTION" and cluster.startswith("LOCO_TURN_"):
            new_cluster, tags = _refine_root_turn_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif super_family == "WHOLE_BODY_LOCOMOTION":
            new_cluster, tags = _refine_root_locomotion_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif super_family == "WHOLE_BODY_ROTATION":
            new_cluster, tags = _refine_root_turn_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif cluster in {"LL_KICK_FORWARD", "RL_KICK_FORWARD", "LL_LEG_FORWARD_POSE", "RL_LEG_FORWARD_POSE"}:
            new_cluster, tags = _refine_leg_forward_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif cluster in {"WB_VERT_UP", "WB_VERT_DOWN", "WB_VERT_CYCLE", "WB_VERT_REP", "WB_VERT_REP_ALT"}:
            new_cluster, tags = _refine_vertical_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif cluster in {"WB_SQUAT_HOLD", "WB_LOW_BODY_HOLD"}:
            new_cluster, tags = _refine_low_body_posture_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif super_family in {"TORSO_POSTURE", "TORSO_PERIODIC"}:
            new_cluster, tags = _refine_torso_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif cluster in {
            "LA_REPEAT",
            "LA_REPEAT_ALT",
            "LA_REPEAT_LOCO",
            "LA_REPEAT_ALT_LOCO",
            "LA_NEAR_FAR",
            "RA_REPEAT",
            "RA_REPEAT_ALT",
            "RA_REPEAT_LOCO",
            "RA_REPEAT_ALT_LOCO",
            "RA_NEAR_FAR",
        }:
            new_cluster, tags = _refine_arm_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif cluster in {"LA_HAND_HIGH", "RA_HAND_HIGH"}:
            new_cluster, tags = _refine_arm_posture_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        elif super_family == "BIMANUAL_PERIODIC":
            new_cluster, tags = _refine_bimanual_event(event, events)
            refined.append(_copy_event_with_refined_geometry(event, new_cluster, tags))
        else:
            refined.append(event)
    return refined


ARM_TRAJECTORY_SIDES = {
    "left": {
        "side_prefix": "LA",
        "super_family": "LEFT_ARM_TRAJECTORY",
        "part": "left_arm",
        "source_super_prefix": "LEFT_ARM_",
        "shoulder_joint": 16,
        "wrist_joint": 20,
    },
    "right": {
        "side_prefix": "RA",
        "super_family": "RIGHT_ARM_TRAJECTORY",
        "part": "right_arm",
        "source_super_prefix": "RIGHT_ARM_",
        "shoulder_joint": 17,
        "wrist_joint": 21,
    },
}

HAND_PROXIMITY_SIDES = {
    "left": {
        "side_prefix": "LA",
        "super_family": "LEFT_ARM_PROXIMITY",
        "part": "left_arm",
        "wrist_joint": 20,
    },
    "right": {
        "side_prefix": "RA",
        "super_family": "RIGHT_ARM_PROXIMITY",
        "part": "right_arm",
        "wrist_joint": 21,
    },
}

LEG_LATERAL_SIDES = {
    "left": {
        "side_prefix": "LL",
        "super_family": "LEFT_LEG_LATERAL",
        "part": "left_leg",
        "foot_joint": 10,
        "ankle_joint": 7,
    },
    "right": {
        "side_prefix": "RL",
        "super_family": "RIGHT_LEG_LATERAL",
        "part": "right_leg",
        "foot_joint": 11,
        "ankle_joint": 8,
    },
}


def _event_from_raw_signal(
    *,
    start_event_index: int,
    side_prefix: str,
    super_family: str,
    part: str,
    cluster_stem: str,
    direction: str,
    role: str,
    span: list[int],
    magnitude: float,
    unit: str,
    optional_semantic_name: str,
    motion_signature: dict[str, Any],
    tags: list[str],
) -> dict[str, Any]:
    return {
        "event_index": start_event_index,
        "token": "",
        "geometry_cluster_id": f"{super_family}/{side_prefix}_{cluster_stem}",
        "part": part,
        "super_family": super_family,
        "cluster_id": f"{side_prefix}_{cluster_stem}",
        "direction": direction,
        "role": role,
        "span": [int(span[0]), int(span[1])],
        "duration": int(span[1]) - int(span[0]) + 1,
        "magnitude": round(abs(float(magnitude)), 4),
        "unit": unit,
        "count": 1,
        "optional_semantic_name": optional_semantic_name,
        "motion_signature": motion_signature,
        "observable_refinement_tags": tags,
    }


def _as_numpy_joints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("joints3d")
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 22 or arr.shape[2] != 3:
        return None
    return arr


def _joints_for_case(joints_pack: dict[str, Any] | None, case_id: str) -> np.ndarray | None:
    if not joints_pack:
        return None
    key = f"{case_id}.npy"
    if key in joints_pack:
        return _as_numpy_joints(joints_pack[key])
    if case_id in joints_pack:
        return _as_numpy_joints(joints_pack[case_id])
    return None


def _merge_frame_spans(spans: list[tuple[int, int]], gap: int) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(spans):
        if not out or start - out[-1][1] > gap:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return out


def _arm_source_spans(events: list[dict[str, Any]], side: str, gap: int) -> list[list[int]]:
    spec = ARM_TRAJECTORY_SIDES[side]
    side_prefix = str(spec["source_super_prefix"])
    part = str(spec["part"])
    spans: list[tuple[int, int]] = []
    for event in events:
        super_family = str(event.get("super_family") or "")
        event_part = str(event.get("part") or "")
        if super_family.startswith(side_prefix) or event_part == part or super_family.startswith("BIMANUAL_"):
            start, end = _span(event)
            if end >= start:
                spans.append((start, end))
    return _merge_frame_spans(spans, gap=gap)


def _arm_trajectory_features(joints: np.ndarray, side: str, span: list[int], pad: int) -> dict[str, Any] | None:
    spec = ARM_TRAJECTORY_SIDES[side]
    start = max(0, int(span[0]) - pad)
    end = min(int(joints.shape[0]) - 1, int(span[1]) + pad)
    if end - start + 1 < 5:
        return None
    shoulder = int(spec["shoulder_joint"])
    wrist = int(spec["wrist_joint"])
    rel = joints[start : end + 1, wrist] - joints[start : end + 1, shoulder]
    if not np.isfinite(rel).all():
        return None

    centered = rel - rel.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    proj = centered @ vecs[:, :2]
    angle = np.unwrap(np.arctan2(proj[:, 1], proj[:, 0]))
    delta = np.diff(angle)
    radius = np.linalg.norm(proj, axis=1)
    path_length = float(np.linalg.norm(np.diff(rel, axis=0), axis=1).sum())
    ranges = np.ptp(rel, axis=0)
    eig_sum = float(vals.sum())
    planarity = float((vals[0] + vals[1]) / max(eig_sum, 1e-8))
    radius_mean = float(radius.mean())
    radius_cv = float(radius.std() / max(radius_mean, 1e-6))
    total_abs_deg = float(np.degrees(np.abs(delta).sum()))
    net_deg = float(np.degrees(angle[-1] - angle[0]))
    return {
        "span": [start, end],
        "duration": end - start + 1,
        "total_abs_deg": total_abs_deg,
        "net_deg": net_deg,
        "orbit_sign": "positive" if net_deg >= 0.0 else "negative",
        "radius_mean_m": radius_mean,
        "radius_cv": radius_cv,
        "path_length_m": path_length,
        "max_range_m": float(ranges.max()),
        "axis_ranges_m": [float(item) for item in ranges.tolist()],
        "planarity": planarity,
    }


def _arm_trajectory_kind(features: dict[str, Any], args: argparse.Namespace) -> str | None:
    radius_mean = float(features["radius_mean_m"])
    radius_cv = float(features["radius_cv"])
    total_abs_deg = float(features["total_abs_deg"])
    path_length = float(features["path_length_m"])
    max_range = float(features["max_range_m"])
    planarity = float(features["planarity"])

    min_radius = float(args.arm_min_radius)
    if (
        radius_mean >= min_radius
        and radius_cv <= float(args.arm_circle_max_radius_cv)
        and total_abs_deg >= float(args.arm_circle_min_abs_deg)
        and path_length >= float(args.arm_large_arc_min_path)
        and planarity >= float(args.arm_min_planarity)
    ):
        return "arm_orbit_cycle"
    if (
        radius_mean >= min_radius
        and radius_cv <= float(args.arm_large_arc_max_radius_cv)
        and max_range >= float(args.arm_large_arc_min_range)
        and total_abs_deg >= float(args.arm_large_arc_min_abs_deg)
        and path_length >= float(args.arm_large_arc_min_path)
        and planarity >= float(args.arm_min_planarity)
    ):
        return "large_arm_arc"
    return None


def _build_arm_trajectory_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    rows: list[dict[str, Any]] = []
    source_offset = len(events)
    for side, spec in ARM_TRAJECTORY_SIDES.items():
        spans = _arm_source_spans(events, side, gap=int(args.arm_span_gap))
        for span_idx, span in enumerate(spans):
            features = _arm_trajectory_features(joints, side, span, pad=int(args.arm_span_pad))
            if not features:
                continue
            kind = _arm_trajectory_kind(features, args)
            if kind is None:
                continue
            side_prefix = str(spec["side_prefix"])
            sign = str(features["orbit_sign"]).upper()
            cluster_stem = "ARM_ORBIT_CYCLE" if kind == "arm_orbit_cycle" else "LARGE_ARM_ARC"
            start, end = [int(item) for item in features["span"]]
            total_abs_deg = float(features["total_abs_deg"])
            rows.append(
                {
                    "event_index": source_offset + len(rows),
                    "token": "",
                    "geometry_cluster_id": f"{spec['super_family']}/{side_prefix}_{cluster_stem}_{sign}",
                    "part": str(spec["part"]),
                    "super_family": str(spec["super_family"]),
                    "cluster_id": f"{side_prefix}_{cluster_stem}_{sign}",
                    "direction": f"orbit_{str(features['orbit_sign'])}" if kind == "arm_orbit_cycle" else f"arc_{str(features['orbit_sign'])}",
                    "role": "composed",
                    "span": [start, end],
                    "duration": int(features["duration"]),
                    "magnitude": round(total_abs_deg, 4),
                    "unit": "deg",
                    "count": max(1, int(round(total_abs_deg / 360.0))) if kind == "arm_orbit_cycle" else None,
                    "optional_semantic_name": kind,
                    "motion_signature": {
                        "dominant_axis": "arm_wrist_relative_to_shoulder_orbit",
                        "repeat_mode": "cycle" if kind == "arm_orbit_cycle" else "large_arc",
                        "phase_template": kind,
                        "context_mode": "raw_joint_trajectory_sidecar",
                        "tempo_bucket": "medium",
                        "coupled_with_locomotion": False,
                        "radius_mean_m": round(float(features["radius_mean_m"]), 4),
                        "radius_cv": round(float(features["radius_cv"]), 4),
                        "path_length_m": round(float(features["path_length_m"]), 4),
                        "max_range_m": round(float(features["max_range_m"]), 4),
                        "axis_ranges_m": [round(float(item), 4) for item in features["axis_ranges_m"]],
                        "planarity": round(float(features["planarity"]), 4),
                        "net_degrees": round(float(features["net_deg"]), 4),
                        "total_abs_degrees": round(total_abs_deg, 4),
                    },
                    "observable_refinement_tags": ["raw_joint_trajectory", kind, "arm_trajectory_sidecar"],
                    "source_span_index": span_idx,
                }
            )
    return rows


def _body_forward_axes(joints: np.ndarray) -> np.ndarray:
    left_shoulder = joints[:, 16]
    right_shoulder = joints[:, 17]
    left_hip = joints[:, 1]
    right_hip = joints[:, 2]
    across = right_shoulder - left_shoulder
    up = ((left_shoulder + right_shoulder) * 0.5) - ((left_hip + right_hip) * 0.5)
    forward = np.cross(across, up)
    forward[:, 1] = 0.0
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    fallback = np.zeros_like(forward)
    fallback[:, 2] = 1.0
    return np.where(norm > 1e-6, forward / np.maximum(norm, 1e-6), fallback)


def _arm_reach_source_spans(events: list[dict[str, Any]], side: str, gap: int, total_frames: int) -> list[list[int]]:
    spans = _arm_source_spans(events, side, gap=gap)
    if spans:
        return spans
    if total_frames <= 0:
        return []
    return [[0, total_frames - 1]]


def _smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    if signal.size == 0 or window <= 1:
        return signal.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    return np.convolve(np.pad(signal, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def _reach_projection_stats(forward: np.ndarray) -> dict[str, Any]:
    min_idx = int(np.argmin(forward))
    max_idx = int(np.argmax(forward))
    min_value = float(forward[min_idx])
    max_value = float(forward[max_idx])
    before_min = float(forward[: max_idx + 1].min()) if max_idx >= 0 else min_value
    after_min = float(forward[max_idx:].min()) if max_idx < forward.size else min_value
    return {
        "forward": forward,
        "forward_min_m": min_value,
        "forward_max_m": max_value,
        "forward_delta_m": float(max_value - min_value),
        "forward_path_m": float(np.abs(np.diff(forward)).sum()),
        "extension_delta_m": float(max_value - before_min),
        "retraction_delta_m": float(max_value - after_min),
        "start_forward_m": float(forward[0]),
        "end_forward_m": float(forward[-1]),
        "peak_local_index": max_idx,
        "trough_local_index": min_idx,
    }


def _arm_reach_features(joints: np.ndarray, side: str, span: list[int], pad: int) -> dict[str, Any] | None:
    spec = ARM_TRAJECTORY_SIDES[side]
    start = max(0, int(span[0]) - pad)
    end = min(int(joints.shape[0]) - 1, int(span[1]) + pad)
    if end - start + 1 < 5:
        return None
    shoulder = int(spec["shoulder_joint"])
    wrist = int(spec["wrist_joint"])
    rel = joints[start : end + 1, wrist] - joints[start : end + 1, shoulder]
    axes = _body_forward_axes(joints)[start : end + 1]
    if not np.isfinite(rel).all() or not np.isfinite(axes).all():
        return None
    raw_forward = _smooth_signal(np.einsum("ij,ij->i", rel, axes), window=5)
    positive_stats = _reach_projection_stats(raw_forward)
    negative_stats = _reach_projection_stats(-raw_forward)
    orientation_sign = 1.0
    stats = positive_stats
    if float(negative_stats["forward_max_m"]) > float(positive_stats["forward_max_m"]):
        orientation_sign = -1.0
        stats = negative_stats
    axes_xz = axes[:, [0, 2]]
    rel_xz = rel[:, [0, 2]]
    lateral = np.linalg.norm(rel_xz - axes_xz * np.einsum("ij,ij->i", rel_xz, axes_xz)[:, None], axis=1)
    return {
        "span": [start, end],
        "duration": end - start + 1,
        "forward_min_m": float(stats["forward_min_m"]),
        "forward_max_m": float(stats["forward_max_m"]),
        "forward_delta_m": float(stats["forward_delta_m"]),
        "forward_path_m": float(stats["forward_path_m"]),
        "extension_delta_m": float(stats["extension_delta_m"]),
        "retraction_delta_m": float(stats["retraction_delta_m"]),
        "start_forward_m": float(stats["start_forward_m"]),
        "end_forward_m": float(stats["end_forward_m"]),
        "peak_frame": start + int(stats["peak_local_index"]),
        "trough_frame": start + int(stats["trough_local_index"]),
        "orientation_sign": orientation_sign,
        "mean_lateral_radius_m": float(lateral.mean()) if lateral.size else 0.0,
    }


def _arm_reach_kind(features: dict[str, Any], args: argparse.Namespace) -> str | None:
    delta = float(features["forward_delta_m"])
    path = float(features["forward_path_m"])
    peak = float(features["forward_max_m"])
    extension = float(features["extension_delta_m"])
    retraction = float(features["retraction_delta_m"])
    min_delta = float(args.arm_reach_min_delta)
    min_path = float(args.arm_reach_min_path)
    min_peak = float(args.arm_reach_min_peak)
    min_forward = float(args.arm_reach_min_forward_component)
    retract_ratio = float(args.arm_reach_retract_ratio)
    if peak < min_peak or delta < min_delta or path < min_path:
        return None
    has_extension = extension >= min_forward
    has_retraction = retraction >= min_forward
    if has_extension and retraction >= max(min_forward, extension * retract_ratio):
        return "arm_reach_retract"
    if has_extension:
        return "arm_reach_extend"
    if has_retraction:
        return "arm_retract"
    return None


def _build_arm_reach_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    rows: list[dict[str, Any]] = []
    source_offset = len(events)
    total_frames = int(record.get("num_frames") or joints.shape[0])
    for side, spec in ARM_TRAJECTORY_SIDES.items():
        spans = _arm_reach_source_spans(events, side, gap=int(args.arm_reach_span_gap), total_frames=total_frames)
        for span_idx, span in enumerate(spans):
            trajectory_features = _arm_trajectory_features(joints, side, span, pad=int(args.arm_span_pad))
            if trajectory_features and _arm_trajectory_kind(trajectory_features, args) is not None:
                continue
            features = _arm_reach_features(joints, side, span, pad=int(args.arm_reach_span_pad))
            if not features:
                continue
            kind = _arm_reach_kind(features, args)
            if kind is None:
                continue
            side_prefix = str(spec["side_prefix"])
            cluster_stem = {
                "arm_reach_retract": "ARM_REACH_RETRACT",
                "arm_reach_extend": "ARM_REACH_EXTEND",
                "arm_retract": "ARM_RETRACT",
            }[kind]
            start, end = [int(item) for item in features["span"]]
            rows.append(
                {
                    "event_index": source_offset + len(rows),
                    "token": "",
                    "geometry_cluster_id": f"{spec['super_family']}/{side_prefix}_{cluster_stem}",
                    "part": str(spec["part"]),
                    "super_family": str(spec["super_family"]),
                    "cluster_id": f"{side_prefix}_{cluster_stem}",
                    "direction": "forward_back" if kind == "arm_reach_retract" else "forward" if kind == "arm_reach_extend" else "backward",
                    "role": "composed",
                    "span": [start, end],
                    "duration": int(features["duration"]),
                    "magnitude": round(float(features["forward_delta_m"]), 4),
                    "unit": "m",
                    "count": 1,
                    "optional_semantic_name": kind,
                    "motion_signature": {
                        "dominant_axis": "arm_wrist_relative_to_shoulder_body_forward",
                        "repeat_mode": "reach_retract" if kind == "arm_reach_retract" else "reach_extend",
                        "phase_template": kind,
                        "context_mode": "raw_joint_reach_sidecar",
                        "tempo_bucket": "medium",
                        "coupled_with_locomotion": False,
                        "forward_min_m": round(float(features["forward_min_m"]), 4),
                        "forward_max_m": round(float(features["forward_max_m"]), 4),
                        "forward_delta_m": round(float(features["forward_delta_m"]), 4),
                        "forward_path_m": round(float(features["forward_path_m"]), 4),
                        "extension_delta_m": round(float(features["extension_delta_m"]), 4),
                        "retraction_delta_m": round(float(features["retraction_delta_m"]), 4),
                        "orientation_sign": int(float(features["orientation_sign"])),
                        "peak_frame": int(features["peak_frame"]),
                        "trough_frame": int(features["trough_frame"]),
                        "mean_lateral_radius_m": round(float(features["mean_lateral_radius_m"]), 4),
                    },
                    "observable_refinement_tags": ["raw_joint_reach", kind, "arm_reach_sidecar"],
                    "source_span_index": span_idx,
                }
            )
    return rows


def _body_height(joints: np.ndarray) -> np.ndarray:
    ankle_y = (joints[:, 7, 1] + joints[:, 8, 1]) * 0.5
    head_y = joints[:, 15, 1]
    return np.maximum(head_y - ankle_y, 1e-6)


def _hand_proximity_signal(joints: np.ndarray, side: str) -> np.ndarray:
    spec = HAND_PROXIMITY_SIDES[side]
    wrist = joints[:, int(spec["wrist_joint"])]
    head = joints[:, 15]
    normalized_distance = np.linalg.norm(wrist - head, axis=1) / _body_height(joints)
    return _smooth_signal(normalized_distance.astype(np.float32), window=5)


def _hand_proximity_transition_events(
    signal: np.ndarray,
    runs: list[list[int]],
    spec: dict[str, Any],
    args: argparse.Namespace,
    start_event_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window = int(args.hand_head_transition_window)
    min_delta = float(args.hand_head_min_delta)
    threshold = float(args.hand_head_proximity_threshold)
    hold_min = int(args.hand_head_hold_min_duration)
    for run in runs:
        run_start, run_end = [int(item) for item in run]
        local = signal[run_start : run_end + 1]
        if local.size == 0:
            continue
        close_value = float(local.min())
        hold_cluster = "HAND_NEAR_HEAD_HOLD" if run_end - run_start + 1 >= hold_min else "HAND_NEAR_HEAD_BRIEF"
        rows.append(
            _event_from_raw_signal(
                start_event_index=start_event_index + len(rows),
                side_prefix=str(spec["side_prefix"]),
                super_family=str(spec["super_family"]),
                part=str(spec["part"]),
                cluster_stem=hold_cluster,
                direction="near_head",
                role="state",
                span=[run_start, run_end],
                magnitude=max(0.0, threshold - close_value),
                unit="ratio",
                optional_semantic_name="hand_near_head",
                motion_signature={
                    "dominant_axis": "wrist_to_head_distance_normalized",
                    "repeat_mode": "state",
                    "phase_template": hold_cluster.lower(),
                    "context_mode": "raw_joint_hand_proximity_sidecar",
                    "tempo_bucket": "medium",
                    "coupled_with_locomotion": False,
                    "min_distance_ratio": round(close_value, 4),
                    "threshold_ratio": round(threshold, 4),
                },
                tags=["raw_joint_hand_proximity", "hand_proximity_sidecar", "hand_near_head_state"],
            )
        )
        before_start = max(0, run_start - window)
        if run_start > before_start:
            before_value = float(signal[before_start:run_start].max())
            delta = before_value - close_value
            if delta >= min_delta:
                rows.append(
                    _event_from_raw_signal(
                        start_event_index=start_event_index + len(rows),
                        side_prefix=str(spec["side_prefix"]),
                        super_family=str(spec["super_family"]),
                        part=str(spec["part"]),
                        cluster_stem="HAND_APPROACH_HEAD",
                        direction="toward_head",
                        role="transition",
                        span=[before_start, run_start],
                        magnitude=delta,
                        unit="ratio",
                        optional_semantic_name="hand_approach_head",
                        motion_signature={
                            "dominant_axis": "wrist_to_head_distance_normalized",
                            "repeat_mode": "transition",
                            "phase_template": "hand_approach_head",
                            "context_mode": "raw_joint_hand_proximity_sidecar",
                            "tempo_bucket": "medium",
                            "coupled_with_locomotion": False,
                            "start_distance_ratio": round(before_value, 4),
                            "end_distance_ratio": round(close_value, 4),
                        },
                        tags=["raw_joint_hand_proximity", "hand_proximity_sidecar", "hand_to_head_transition"],
                    )
                )
        after_end = min(signal.size - 1, run_end + window)
        if after_end > run_end:
            after_value = float(signal[run_end + 1 : after_end + 1].max())
            delta = after_value - close_value
            if delta >= min_delta:
                rows.append(
                    _event_from_raw_signal(
                        start_event_index=start_event_index + len(rows),
                        side_prefix=str(spec["side_prefix"]),
                        super_family=str(spec["super_family"]),
                        part=str(spec["part"]),
                        cluster_stem="HAND_LEAVE_HEAD",
                        direction="away_from_head",
                        role="transition",
                        span=[run_end, after_end],
                        magnitude=delta,
                        unit="ratio",
                        optional_semantic_name="hand_leave_head",
                        motion_signature={
                            "dominant_axis": "wrist_to_head_distance_normalized",
                            "repeat_mode": "transition",
                            "phase_template": "hand_leave_head",
                            "context_mode": "raw_joint_hand_proximity_sidecar",
                            "tempo_bucket": "medium",
                            "coupled_with_locomotion": False,
                            "start_distance_ratio": round(close_value, 4),
                            "end_distance_ratio": round(after_value, 4),
                        },
                        tags=["raw_joint_hand_proximity", "hand_proximity_sidecar", "hand_from_head_transition"],
                    )
                )
    return rows


def _build_hand_proximity_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    rows: list[dict[str, Any]] = []
    threshold = float(args.hand_head_proximity_threshold)
    min_run = int(args.hand_head_min_run)
    merge_gap = int(args.hand_head_merge_gap)
    min_delta = float(args.hand_head_min_delta)
    for side, spec in HAND_PROXIMITY_SIDES.items():
        signal = _hand_proximity_signal(joints, side)
        runs = _runs_from_mask(signal <= threshold, min_run=min_run)
        runs = _merge_frame_spans([(int(start), int(end)) for start, end in runs], gap=merge_gap)
        if not runs:
            continue
        if float(np.percentile(signal, 90) - np.percentile(signal, 10)) < min_delta:
            continue
        if len(runs) >= 2:
            span = [int(runs[0][0]), int(runs[-1][1])]
            local = signal[span[0] : span[1] + 1]
            rows.append(
                _event_from_raw_signal(
                    start_event_index=len(events) + len(rows),
                    side_prefix=str(spec["side_prefix"]),
                    super_family=str(spec["super_family"]),
                    part=str(spec["part"]),
                    cluster_stem="HAND_NEAR_HEAD_REPEATED",
                    direction="near_head_repeated",
                    role="composed",
                    span=span,
                    magnitude=float(np.percentile(signal, 90) - np.percentile(signal, 10)),
                    unit="ratio",
                    optional_semantic_name="repeated_hand_near_head",
                    motion_signature={
                        "dominant_axis": "wrist_to_head_distance_normalized",
                        "repeat_mode": "repeated_state",
                        "phase_template": "hand_near_head_repeated",
                        "context_mode": "raw_joint_hand_proximity_sidecar",
                        "tempo_bucket": "medium",
                        "coupled_with_locomotion": False,
                        "count": len(runs),
                        "min_distance_ratio": round(float(local.min()), 4),
                        "max_distance_ratio": round(float(local.max()), 4),
                        "threshold_ratio": round(threshold, 4),
                    },
                    tags=["raw_joint_hand_proximity", "hand_proximity_sidecar", "hand_near_head_repeated"],
                )
            )
            rows[-1]["count"] = len(runs)
            continue
        rows.extend(_hand_proximity_transition_events(signal, runs, spec, args, start_event_index=len(events) + len(rows)))
    return rows


def _body_lateral_axes(joints: np.ndarray) -> np.ndarray:
    left_hip = joints[:, 1]
    right_hip = joints[:, 2]
    across = right_hip - left_hip
    across[:, 1] = 0.0
    norm = np.linalg.norm(across, axis=1, keepdims=True)
    fallback = np.zeros_like(across)
    fallback[:, 0] = 1.0
    return np.where(norm > 1e-6, across / np.maximum(norm, 1e-6), fallback)


def _leg_lateral_signal(joints: np.ndarray, side: str) -> np.ndarray:
    spec = LEG_LATERAL_SIDES[side]
    pelvis = joints[:, 0]
    foot = (joints[:, int(spec["foot_joint"])] + joints[:, int(spec["ankle_joint"])]) * 0.5
    lateral_axis = _body_lateral_axes(joints)
    sign = -1.0 if side == "left" else 1.0
    lateral = np.einsum("ij,ij->i", foot - pelvis, lateral_axis) * sign
    smoothed = _smooth_signal(lateral.astype(np.float32), window=5)
    baseline = float(np.percentile(smoothed, 20))
    return np.maximum(smoothed - baseline, 0.0).astype(np.float32)


def _leg_lateral_events_from_signal(
    signal: np.ndarray,
    runs: list[list[int]],
    spec: dict[str, Any],
    args: argparse.Namespace,
    start_event_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window = int(args.leg_lateral_transition_window)
    min_delta = float(args.leg_lateral_min_delta)
    hold_min = int(args.leg_lateral_hold_min_duration)
    for run in runs:
        run_start, run_end = [int(item) for item in run]
        local = signal[run_start : run_end + 1]
        if local.size == 0:
            continue
        peak = float(local.max())
        cluster = "LEG_LATERAL_OUT_HOLD" if run_end - run_start + 1 >= hold_min else "LEG_LATERAL_OUT_BRIEF"
        rows.append(
            _event_from_raw_signal(
                start_event_index=start_event_index + len(rows),
                side_prefix=str(spec["side_prefix"]),
                super_family=str(spec["super_family"]),
                part=str(spec["part"]),
                cluster_stem=cluster,
                direction="outward",
                role="state",
                span=[run_start, run_end],
                magnitude=peak,
                unit="m",
                optional_semantic_name="leg_lateral_out",
                motion_signature={
                    "dominant_axis": "foot_relative_to_pelvis_body_lateral",
                    "repeat_mode": "state",
                    "phase_template": cluster.lower(),
                    "context_mode": "raw_joint_leg_lateral_sidecar",
                    "tempo_bucket": "medium",
                    "coupled_with_locomotion": False,
                    "peak_lateral_m": round(peak, 4),
                },
                tags=["raw_joint_leg_lateral", "leg_lateral_sidecar", "leg_lateral_state"],
            )
        )
        before_start = max(0, run_start - window)
        if run_start > before_start:
            before_value = float(signal[before_start:run_start].min())
            delta = peak - before_value
            if delta >= min_delta:
                rows.append(
                    _event_from_raw_signal(
                        start_event_index=start_event_index + len(rows),
                        side_prefix=str(spec["side_prefix"]),
                        super_family=str(spec["super_family"]),
                        part=str(spec["part"]),
                        cluster_stem="LEG_LATERAL_ABDUCT",
                        direction="outward",
                        role="transition",
                        span=[before_start, run_start],
                        magnitude=delta,
                        unit="m",
                        optional_semantic_name="leg_lateral_abduct",
                        motion_signature={
                            "dominant_axis": "foot_relative_to_pelvis_body_lateral",
                            "repeat_mode": "transition",
                            "phase_template": "leg_lateral_abduct",
                            "context_mode": "raw_joint_leg_lateral_sidecar",
                            "tempo_bucket": "medium",
                            "coupled_with_locomotion": False,
                            "start_lateral_m": round(before_value, 4),
                            "peak_lateral_m": round(peak, 4),
                        },
                        tags=["raw_joint_leg_lateral", "leg_lateral_sidecar", "leg_lateral_abduction"],
                    )
                )
        after_end = min(signal.size - 1, run_end + window)
        if after_end > run_end:
            after_value = float(signal[run_end + 1 : after_end + 1].min())
            delta = peak - after_value
            if delta >= min_delta:
                rows.append(
                    _event_from_raw_signal(
                        start_event_index=start_event_index + len(rows),
                        side_prefix=str(spec["side_prefix"]),
                        super_family=str(spec["super_family"]),
                        part=str(spec["part"]),
                        cluster_stem="LEG_LATERAL_ADDUCT",
                        direction="inward",
                        role="transition",
                        span=[run_end, after_end],
                        magnitude=delta,
                        unit="m",
                        optional_semantic_name="leg_lateral_adduct",
                        motion_signature={
                            "dominant_axis": "foot_relative_to_pelvis_body_lateral",
                            "repeat_mode": "transition",
                            "phase_template": "leg_lateral_adduct",
                            "context_mode": "raw_joint_leg_lateral_sidecar",
                            "tempo_bucket": "medium",
                            "coupled_with_locomotion": False,
                            "peak_lateral_m": round(peak, 4),
                            "end_lateral_m": round(after_value, 4),
                        },
                        tags=["raw_joint_leg_lateral", "leg_lateral_sidecar", "leg_lateral_adduction"],
                    )
                )
    return rows


def _build_leg_lateral_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    rows: list[dict[str, Any]] = []
    threshold = float(args.leg_lateral_threshold)
    min_run = int(args.leg_lateral_min_run)
    merge_gap = int(args.leg_lateral_merge_gap)
    min_delta = float(args.leg_lateral_min_delta)
    for side, spec in LEG_LATERAL_SIDES.items():
        signal = _leg_lateral_signal(joints, side)
        runs = _runs_from_mask(signal >= threshold, min_run=min_run)
        runs = _merge_frame_spans([(int(start), int(end)) for start, end in runs], gap=merge_gap)
        if not runs:
            continue
        if float(np.percentile(signal, 90) - np.percentile(signal, 10)) < min_delta:
            continue
        if len(runs) >= 2:
            span = [int(runs[0][0]), int(runs[-1][1])]
            local = signal[span[0] : span[1] + 1]
            rows.append(
                _event_from_raw_signal(
                    start_event_index=len(events) + len(rows),
                    side_prefix=str(spec["side_prefix"]),
                    super_family=str(spec["super_family"]),
                    part=str(spec["part"]),
                    cluster_stem="LEG_LATERAL_REPEAT",
                    direction="out_in_repeated",
                    role="composed",
                    span=span,
                    magnitude=float(local.max()),
                    unit="m",
                    optional_semantic_name="repeated_leg_lateral_motion",
                    motion_signature={
                        "dominant_axis": "foot_relative_to_pelvis_body_lateral",
                        "repeat_mode": "repeated_lateral",
                        "phase_template": "leg_lateral_repeat",
                        "context_mode": "raw_joint_leg_lateral_sidecar",
                        "tempo_bucket": "medium",
                        "coupled_with_locomotion": False,
                        "count": len(runs),
                        "peak_lateral_m": round(float(local.max()), 4),
                        "range_lateral_m": round(float(np.percentile(signal, 90) - np.percentile(signal, 10)), 4),
                    },
                    tags=["raw_joint_leg_lateral", "leg_lateral_sidecar", "leg_lateral_repeated"],
                )
            )
            rows[-1]["count"] = len(runs)
            continue
        rows.extend(_leg_lateral_events_from_signal(signal, runs, spec, args, start_event_index=len(events) + len(rows)))
    return rows


def _runs_from_mask(mask: np.ndarray, min_run: int) -> list[list[int]]:
    indices = np.where(mask)[0]
    if indices.size == 0:
        return []
    runs: list[list[int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        idx = int(raw)
        if idx == prev + 1:
            prev = idx
            continue
        if prev - start + 1 >= min_run:
            runs.append([start, prev])
        start = prev = idx
    if prev - start + 1 >= min_run:
        runs.append([start, prev])
    return runs


def _body_level_signal(joints: np.ndarray) -> dict[str, np.ndarray]:
    pelvis_y = joints[:, 0, 1]
    ankle_y = (joints[:, 7, 1] + joints[:, 8, 1]) * 0.5
    head_y = joints[:, 15, 1]
    body_height = np.maximum(head_y - ankle_y, 1e-6)
    pelvis_level = (pelvis_y - ankle_y) / body_height
    kernel = np.ones(7, dtype=np.float32) / 7.0
    pad = len(kernel) // 2
    smooth = np.convolve(np.pad(pelvis_level, (pad, pad), mode="edge"), kernel, mode="valid")
    return {
        "pelvis_level": pelvis_level.astype(np.float32),
        "pelvis_level_smooth": smooth.astype(np.float32),
    }


def _body_transition_events_from_runs(
    case_id: str,
    low_runs: list[list[int]],
    high_runs: list[list[int]],
    signal: np.ndarray,
    args: argparse.Namespace,
    start_event_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_gap = int(args.body_transition_max_gap)

    def add_event(cluster: str, direction: str, span: list[int], role: str, magnitude: float, tags: list[str]) -> None:
        rows.append(
            {
                "event_index": start_event_index + len(rows),
                "token": "",
                "geometry_cluster_id": f"WHOLE_BODY_LEVEL/{cluster}",
                "part": "whole_body",
                "super_family": "WHOLE_BODY_LEVEL",
                "cluster_id": cluster,
                "direction": direction,
                "role": role,
                "span": [int(span[0]), int(span[1])],
                "duration": int(span[1]) - int(span[0]) + 1,
                "magnitude": round(abs(float(magnitude)), 4),
                "unit": "ratio",
                "count": None,
                "optional_semantic_name": cluster.lower(),
                "motion_signature": {
                    "dominant_axis": "pelvis_height_normalized",
                    "repeat_mode": "state" if role == "state" else "transition",
                    "phase_template": cluster.lower(),
                    "context_mode": "raw_joint_body_level_sidecar",
                    "tempo_bucket": "medium",
                    "coupled_with_locomotion": False,
                    "start_level": round(float(signal[int(span[0])]), 4),
                    "end_level": round(float(signal[int(span[1])]), 4),
                    "min_level": round(float(signal[int(span[0]) : int(span[1]) + 1].min()), 4),
                    "max_level": round(float(signal[int(span[0]) : int(span[1]) + 1].max()), 4),
                },
                "observable_refinement_tags": ["raw_joint_body_level", "body_level_sidecar", *tags],
                "case_id_source": case_id,
            }
        )

    for run in low_runs:
        start, end = run
        min_level = float(signal[start : end + 1].min())
        max_level = float(signal[start : end + 1].max())
        cluster = "WB_LEVEL_LOW_SUSTAINED" if end - start + 1 >= int(args.body_long_low_min_duration) else "WB_LEVEL_LOW_BRIEF"
        add_event(cluster, "low", run, "state", max_level - min_level, ["low_body_level"])

    if bool(getattr(args, "body_emit_high_state", False)):
        for run in high_runs:
            start, end = run
            if end - start + 1 < int(args.body_high_state_min_duration):
                continue
            add_event("WB_LEVEL_HIGH_STANDLIKE", "high", run, "state", float(signal[start : end + 1].max() - signal[start : end + 1].min()), ["high_body_level"])

    for low in low_runs:
        low_start, low_end = low
        before = [run for run in high_runs if run[1] <= low_start and low_start - run[1] <= max_gap]
        after = [run for run in high_runs if run[0] >= low_end and run[0] - low_end <= max_gap]
        if before:
            prev_high = max(before, key=lambda run: run[1])
            span = [prev_high[1], low_start]
            add_event("WB_LEVEL_DESCEND_TO_LOW", "down", span, "transition", float(signal[span[1]] - signal[span[0]]), ["level_transition_down"])
        if after:
            next_high = min(after, key=lambda run: run[0])
            span = [low_end, next_high[0]]
            add_event("WB_LEVEL_RISE_FROM_LOW", "up", span, "transition", float(signal[span[1]] - signal[span[0]]), ["level_transition_up"])

    cycle_count = 0
    for low in low_runs:
        low_start, low_end = low
        next_high = [run for run in high_runs if run[0] >= low_end and run[0] - low_end <= max_gap]
        if not next_high:
            continue
        high = min(next_high, key=lambda run: run[0])
        next_low = [run for run in low_runs if run[0] >= high[1] and run[0] - high[1] <= max_gap]
        if not next_low:
            continue
        low2 = min(next_low, key=lambda run: run[0])
        span = [low_start, low2[1]]
        cycle_count += 1
        add_event(
            "WB_LEVEL_LOW_HIGH_LOW_CYCLE",
            "low_high_low",
            span,
            "composed",
            float(signal[span[0] : span[1] + 1].max() - signal[span[0] : span[1] + 1].min()),
            ["level_transition_cycle"],
        )

    high_low_high_count = 0
    for high in high_runs:
        high_start, high_end = high
        next_low = [run for run in low_runs if run[0] >= high_end and run[0] - high_end <= max_gap]
        if not next_low:
            continue
        low = min(next_low, key=lambda run: run[0])
        next_high = [run for run in high_runs if run[0] >= low[1] and run[0] - low[1] <= max_gap]
        if not next_high:
            continue
        high2 = min(next_high, key=lambda run: run[0])
        span = [high_start, high2[1]]
        high_low_high_count += 1
        add_event(
            "WB_LEVEL_HIGH_LOW_HIGH_CYCLE",
            "high_low_high",
            span,
            "composed",
            float(signal[span[0] : span[1] + 1].max() - signal[span[0] : span[1] + 1].min()),
            ["level_transition_cycle"],
        )

    if cycle_count or high_low_high_count:
        for event in rows:
            if event["cluster_id"].endswith("_CYCLE"):
                event["count"] = max(cycle_count, high_low_high_count)
                event["motion_signature"]["repeat_mode"] = "cycle"
    return rows


def _build_body_level_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    signal_dict = _body_level_signal(joints)
    signal = signal_dict["pelvis_level_smooth"]
    min_run = int(args.body_level_min_run)
    low_runs = _runs_from_mask(signal <= float(args.body_low_threshold), min_run=min_run)
    high_runs = _runs_from_mask(signal >= float(args.body_high_threshold), min_run=min_run)
    if not low_runs and not high_runs:
        return []
    return _body_transition_events_from_runs(
        str(record.get("case_id") or ""),
        low_runs,
        high_runs,
        signal,
        args,
        start_event_index=len(events),
    )


def _body_support_scale(joints: np.ndarray) -> float:
    torso = (
        np.linalg.norm(joints[:, 0] - joints[:, 3], axis=1)
        + np.linalg.norm(joints[:, 3] - joints[:, 6], axis=1)
        + np.linalg.norm(joints[:, 6] - joints[:, 9], axis=1)
        + np.linalg.norm(joints[:, 9] - joints[:, 12], axis=1)
        + np.linalg.norm(joints[:, 12] - joints[:, 15], axis=1)
    )
    left_leg = np.linalg.norm(joints[:, 1] - joints[:, 4], axis=1) + np.linalg.norm(joints[:, 4] - joints[:, 7], axis=1)
    right_leg = np.linalg.norm(joints[:, 2] - joints[:, 5], axis=1) + np.linalg.norm(joints[:, 5] - joints[:, 8], axis=1)
    scale = torso + 0.5 * (left_leg + right_leg)
    finite = scale[np.isfinite(scale) & (scale > 1e-4)]
    if finite.size == 0:
        return 1.0
    return max(0.25, float(np.median(finite)))


def _body_support_signals(joints: np.ndarray) -> dict[str, np.ndarray]:
    pelvis = joints[:, 0]
    head = joints[:, 15]
    left_shoulder = joints[:, 16]
    right_shoulder = joints[:, 17]
    left_hip = joints[:, 1]
    right_hip = joints[:, 2]
    left_wrist = joints[:, 20]
    right_wrist = joints[:, 21]
    left_foot = (joints[:, 7] + joints[:, 10]) * 0.5
    right_foot = (joints[:, 8] + joints[:, 11]) * 0.5
    feet = np.stack([left_foot, right_foot], axis=1)
    wrists = np.stack([left_wrist, right_wrist], axis=1)
    support_floor_y = np.min(np.concatenate([feet[:, :, 1], wrists[:, :, 1]], axis=1), axis=1)
    body_scale = _body_support_scale(joints)
    torso_axis = ((left_shoulder + right_shoulder) * 0.5) - ((left_hip + right_hip) * 0.5)
    torso_norm = np.linalg.norm(torso_axis, axis=1)
    vertical_component = np.abs(torso_axis[:, 1]) / np.maximum(torso_norm, 1e-6)
    hand_floor_ratio = (np.min(wrists[:, :, 1], axis=1) - support_floor_y) / body_scale
    foot_high_ratio = (np.max(feet[:, :, 1], axis=1) - support_floor_y) / body_scale
    pelvis_level = (pelvis[:, 1] - support_floor_y) / body_scale
    head_level = (head[:, 1] - support_floor_y) / body_scale
    head_minus_pelvis = (head[:, 1] - pelvis[:, 1]) / body_scale
    return {
        "vertical_component": _smooth_signal(vertical_component.astype(np.float32), window=5),
        "hand_floor_ratio": _smooth_signal(hand_floor_ratio.astype(np.float32), window=5),
        "foot_high_ratio": _smooth_signal(foot_high_ratio.astype(np.float32), window=5),
        "pelvis_level": _smooth_signal(pelvis_level.astype(np.float32), window=5),
        "head_level": _smooth_signal(head_level.astype(np.float32), window=5),
        "head_minus_pelvis": _smooth_signal(head_minus_pelvis.astype(np.float32), window=5),
    }


def _body_support_event(
    *,
    start_event_index: int,
    cluster: str,
    direction: str,
    role: str,
    span: list[int],
    magnitude: float,
    optional_semantic_name: str,
    signature: dict[str, Any],
    tags: list[str],
) -> dict[str, Any]:
    return {
        "event_index": start_event_index,
        "token": "",
        "geometry_cluster_id": f"WHOLE_BODY_SUPPORT/{cluster}",
        "part": "whole_body",
        "super_family": "WHOLE_BODY_SUPPORT",
        "cluster_id": cluster,
        "direction": direction,
        "role": role,
        "span": [int(span[0]), int(span[1])],
        "duration": int(span[1]) - int(span[0]) + 1,
        "magnitude": round(abs(float(magnitude)), 4),
        "unit": "ratio",
        "count": None,
        "optional_semantic_name": optional_semantic_name,
        "motion_signature": signature,
        "observable_refinement_tags": ["raw_joint_body_support", "body_support_sidecar", *tags],
    }


def _body_support_runs(signals: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, list[list[int]]]:
    min_run = int(args.body_support_min_run)
    merge_gap = int(args.body_support_merge_gap)
    horizontal = signals["vertical_component"] <= float(args.body_horizontal_axis_threshold)
    low_pelvis = signals["pelvis_level"] <= float(args.body_prone_height_ratio)
    hand_floor = signals["hand_floor_ratio"] <= float(args.body_hand_floor_ratio)
    foot_high = signals["foot_high_ratio"] >= float(args.body_foot_high_ratio)
    inverted = hand_floor & foot_high
    floor_low_horizontal = horizontal & low_pelvis & hand_floor & ~inverted
    hand_floor_low = hand_floor & low_pelvis & ~floor_low_horizontal & ~inverted
    return {
        "inverted_support": _merge_frame_spans([(int(a), int(b)) for a, b in _runs_from_mask(inverted, min_run=min_run)], gap=merge_gap),
        "floor_low_horizontal_support": _merge_frame_spans([(int(a), int(b)) for a, b in _runs_from_mask(floor_low_horizontal, min_run=min_run)], gap=merge_gap),
        "hand_floor_low_support": _merge_frame_spans([(int(a), int(b)) for a, b in _runs_from_mask(hand_floor_low, min_run=min_run)], gap=merge_gap),
    }


def _build_body_support_sidecar_events(
    record: dict[str, Any],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    joints = _joints_for_case(joints_pack, str(record.get("case_id") or ""))
    if joints is None:
        return []
    signals = _body_support_signals(joints)
    runs_by_kind = _body_support_runs(signals, args)
    specs = {
        "inverted_support": ("WB_SUPPORT_INVERTED", "inverted", "inverted_support"),
        "floor_low_horizontal_support": ("WB_SUPPORT_FLOOR_LOW_HORIZONTAL", "floor_low_horizontal", "floor_low_horizontal_support"),
        "hand_floor_low_support": ("WB_SUPPORT_HAND_FLOOR_LOW", "hand_floor_low", "hand_floor_low_support"),
    }
    rows: list[dict[str, Any]] = []
    for kind, runs in runs_by_kind.items():
        cluster, direction, tag = specs[kind]
        for span in runs:
            start, end = [int(item) for item in span]
            local = slice(start, end + 1)
            signature = {
                "dominant_axis": "whole_body_support_geometry",
                "repeat_mode": "state",
                "phase_template": kind,
                "context_mode": "raw_joint_body_support_sidecar",
                "tempo_bucket": "medium",
                "coupled_with_locomotion": False,
                "mean_vertical_component": round(float(signals["vertical_component"][local].mean()), 4),
                "min_hand_floor_ratio": round(float(signals["hand_floor_ratio"][local].min()), 4),
                "max_foot_high_ratio": round(float(signals["foot_high_ratio"][local].max()), 4),
                "mean_pelvis_level": round(float(signals["pelvis_level"][local].mean()), 4),
                "mean_head_minus_pelvis": round(float(signals["head_minus_pelvis"][local].mean()), 4),
            }
            magnitude = {
                "inverted_support": max(0.0, -float(signals["head_minus_pelvis"][local].min())),
                "floor_low_horizontal_support": max(0.0, float(args.body_horizontal_axis_threshold) - float(signals["vertical_component"][local].min())),
                "hand_floor_low_support": max(0.0, float(args.body_hand_floor_ratio) - float(signals["hand_floor_ratio"][local].min())),
            }[kind]
            rows.append(
                _body_support_event(
                    start_event_index=len(events) + len(rows),
                    cluster=cluster,
                    direction=direction,
                    role="state",
                    span=[start, end],
                    magnitude=magnitude,
                    optional_semantic_name=kind,
                    signature=signature,
                    tags=[tag],
                )
            )
    return rows


def build_channel_events(
    record: dict[str, Any],
    observable_refinement: str = "v1",
    args: argparse.Namespace | None = None,
    joints_pack: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_id = str(record.get("case_id") or "")
    source_events = list(record.get("events") or [])
    source_events = refine_events_for_motion_bpe(source_events, observable_refinement)
    if args is not None and _use_arm_trajectory_sidecar(args):
        source_events = source_events + _build_arm_trajectory_sidecar_events(record, source_events, args, joints_pack)
    if args is not None and _use_arm_reach_sidecar(args):
        source_events = source_events + _build_arm_reach_sidecar_events(record, source_events, args, joints_pack)
    if args is not None and _use_hand_proximity_sidecar(args):
        source_events = source_events + _build_hand_proximity_sidecar_events(record, source_events, args, joints_pack)
    if args is not None and _use_leg_lateral_sidecar(args):
        source_events = source_events + _build_leg_lateral_sidecar_events(record, source_events, args, joints_pack)
    if args is not None and _use_body_level_sidecar(args):
        source_events = source_events + _build_body_level_sidecar_events(record, source_events, args, joints_pack)
    if args is not None and _use_body_support_sidecar(args):
        source_events = source_events + _build_body_support_sidecar_events(record, source_events, args, joints_pack)
    for local_idx, event in enumerate(source_events):
        channel = _channel_for_event(event)
        start, end = _span(event)
        duration = _duration(event)
        magnitude = _magnitude(event)
        unit = str(event.get("unit") or "")
        if unit == "deg":
            speed_value = magnitude / max(duration, 1)
            speed_unit = "deg_per_frame"
        elif event.get("count") is not None:
            try:
                speed_value = float(event.get("count") or 0.0) / max(duration, 1)
            except (TypeError, ValueError):
                speed_value = 0.0
            speed_unit = "count_per_frame"
        else:
            speed_value = magnitude / max(duration, 1)
            speed_unit = "m_per_frame" if unit == "m" else "value_per_frame"
        count_bin = _count_bin(event.get("count"))
        copied = {
            "event_id": f"{case_id}:e{local_idx:04d}",
            "case_id": case_id,
            "source_event_index": int(event.get("event_index", local_idx)),
            "channel": channel,
            "part": str(event.get("part") or ""),
            "super_family": str(event.get("super_family") or ""),
            "cluster_id": str(event.get("cluster_id") or ""),
            "geometry_cluster_id": str(event.get("geometry_cluster_id") or ""),
            "raw_cluster_id": str(event.get("raw_cluster_id") or event.get("cluster_id") or ""),
            "raw_geometry_cluster_id": str(
                event.get("raw_geometry_cluster_id")
                or event.get("geometry_cluster_id")
                or f"{event.get('super_family')}/{event.get('cluster_id')}"
            ),
            "observable_refinement_tags": event.get("observable_refinement_tags") or [],
            "span": [start, end],
            "direction": str(event.get("direction") or "none"),
            "duration": duration,
            "duration_bin": _duration_bin(duration),
            "magnitude": round(magnitude, 4),
            "magnitude_unit": unit,
            "magnitude_bin": _magnitude_bin(magnitude, unit),
            "mean_speed": round(speed_value, 6),
            "speed_unit": speed_unit,
            "speed_bin": _speed_bin(speed_value, speed_unit),
            "count": event.get("count"),
            "count_bin": count_bin,
            "confidence": event.get("confidence"),
            "motion_signature": event.get("motion_signature") or {},
        }
        copied["symbol"] = _channel_event_symbol(event, channel)
        rows.append(copied)
    rows.sort(key=_event_sort_key)
    return rows


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a0, a1 = a.get("span") or [0, 0]
    b0, b1 = b.get("span") or [0, 0]
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a0, a1 = a.get("span") or [0, 0]
    b0, b1 = b.get("span") or [0, 0]
    if int(a1) < int(b0):
        return int(b0) - int(a1) - 1
    if int(b1) < int(a0):
        return int(a0) - int(b1) - 1
    return 0


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ov = _overlap(a, b)
    if ov <= 0:
        return 0.0
    da = max(1, int(a["span"][1]) - int(a["span"][0]) + 1)
    db = max(1, int(b["span"][1]) - int(b["span"][0]) + 1)
    return ov / max(1, min(da, db))


def build_relations(events: list[dict[str, Any]], *, parallel_overlap_min: float, lead_lag_gap_max: int) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for i, left in enumerate(events):
        for right in events[i + 1 :]:
            same_channel = left["channel"] == right["channel"]
            ratio = _overlap_ratio(left, right)
            gap = _gap(left, right)
            relation = ""
            if not same_channel and ratio >= parallel_overlap_min:
                relation = "parallel"
            elif ratio > 0.0 or gap <= lead_lag_gap_max:
                relation = "lead_lag" if not same_channel else "same_channel_adjacent"
            if not relation:
                continue
            relations.append(
                {
                    "left_event_id": left["event_id"],
                    "right_event_id": right["event_id"],
                    "left_channel": left["channel"],
                    "right_channel": right["channel"],
                    "relation": relation,
                    "overlap_ratio": round(ratio, 4),
                    "gap": gap,
                    "relative_order": "overlap" if ratio > 0.0 else ("left_before_right" if left["span"][1] < right["span"][0] else "right_before_left"),
                }
            )
    return relations


def _packet_symbol(members: list[dict[str, Any]], packet_type: str) -> str:
    parts = []
    for event in sorted(members, key=lambda item: (CHANNEL_RANK.get(item["channel"], 999), item["geometry_cluster_id"], item["symbol"])):
        parts.append(f"{event['channel']}:{event['geometry_cluster_id'].split('/', 1)[-1]}")
    prefix = "PAR" if packet_type == "parallel" else "SINGLE"
    return f"{prefix}[" + "+".join(parts) + "]"


def build_packets(events: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_by_id = {event["event_id"]: event for event in events}
    parent: dict[str, str] = {event["event_id"]: event["event_id"] for event in events}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    relation_counter_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    for rel in relations:
        if rel["relation"] != "parallel":
            continue
        union(str(rel["left_event_id"]), str(rel["right_event_id"]))
    for rel in relations:
        left = str(rel["left_event_id"])
        if left in parent:
            relation_counter_by_root[find(left)][str(rel["relation"])] += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[find(event["event_id"])].append(event)

    packets: list[dict[str, Any]] = []
    for idx, members in enumerate(sorted(groups.values(), key=lambda rows: min(int(e["span"][0]) for e in rows)), start=1):
        start = min(int(event["span"][0]) for event in members)
        end = max(int(event["span"][1]) for event in members)
        member_channels = sorted({event["channel"] for event in members}, key=lambda ch: CHANNEL_RANK.get(ch, 999))
        packet_type = "parallel" if len(member_channels) >= 2 else "single"
        packet = {
            "packet_id": f"{members[0]['case_id']}:p{idx:04d}",
            "case_id": members[0]["case_id"],
            "packet_type": packet_type,
            "span": [start, end],
            "members": [
                {
                    "event_id": event["event_id"],
                    "channel": event["channel"],
                    "symbol": event["symbol"],
                    "geometry_cluster_id": event["geometry_cluster_id"],
                    "raw_geometry_cluster_id": event.get("raw_geometry_cluster_id"),
                    "observable_refinement_tags": event.get("observable_refinement_tags") or [],
                    "span": event["span"],
                }
                for event in sorted(members, key=_event_sort_key)
            ],
            "member_channels": member_channels,
            "packet_symbol": _packet_symbol(members, packet_type),
            "relation_summary": dict(relation_counter_by_root.get(find(members[0]["event_id"]), Counter())),
        }
        packets.append(packet)
    packets.sort(key=lambda item: (int(item["span"][0]), int(item["span"][1]), str(item["packet_id"])))
    return packets


def _unit_from_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(event["symbol"]),
        "unit_type": "channel_event",
        "base_symbols": [str(event["symbol"])],
        "event_ids": [str(event["event_id"])],
        "packet_ids": [],
        "span": list(event["span"]),
        "channels": [str(event["channel"])],
        "geometry_clusters": [str(event["geometry_cluster_id"])],
        "raw_geometry_clusters": [str(event.get("raw_geometry_cluster_id") or event["geometry_cluster_id"])],
        "observable_refinement_tags": list(event.get("observable_refinement_tags") or []),
        "relation_types": [],
    }


def _unit_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(packet["packet_symbol"]),
        "unit_type": str(packet["packet_type"]) + "_packet",
        "base_symbols": [str(member["symbol"]) for member in packet.get("members") or []],
        "event_ids": [str(member["event_id"]) for member in packet.get("members") or []],
        "packet_ids": [str(packet["packet_id"])],
        "span": list(packet["span"]),
        "channels": list(packet.get("member_channels") or []),
        "geometry_clusters": sorted({str(member["geometry_cluster_id"]) for member in packet.get("members") or []}),
        "raw_geometry_clusters": sorted({str(member.get("raw_geometry_cluster_id") or member["geometry_cluster_id"]) for member in packet.get("members") or []}),
        "observable_refinement_tags": sorted({str(tag) for member in packet.get("members") or [] for tag in (member.get("observable_refinement_tags") or [])}),
        "relation_types": [str(packet.get("packet_type") or "")],
    }


def build_multichannel_record(
    record: dict[str, Any],
    args: argparse.Namespace,
    joints_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = build_channel_events(
        record,
        str(getattr(args, "observable_refinement", "v1")),
        args=args,
        joints_pack=joints_pack,
    )
    relations = build_relations(
        events,
        parallel_overlap_min=float(args.parallel_overlap_min),
        lead_lag_gap_max=int(args.lead_lag_gap_max),
    )
    relation_type_counts = Counter(str(rel.get("relation") or "") for rel in relations)
    packets = build_packets(events, relations)
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_channel[event["channel"]].append(_unit_from_event(event))
    for seq in by_channel.values():
        seq.sort(key=lambda unit: (int(unit["span"][0]), int(unit["span"][1]), unit["symbol"]))
    packet_units = [_unit_from_packet(packet) for packet in packets]
    views = {
        "channel_sequences": dict(sorted(by_channel.items(), key=lambda item: CHANNEL_RANK.get(item[0], 999))),
        "packet_sequence": packet_units,
    }
    return {
        "case_id": str(record.get("case_id") or ""),
        "num_frames": int(record.get("num_frames") or 0),
        "caption_texts": record.get("caption_texts") or [],
        "caption_alias_ids": record.get("caption_alias_ids") or [],
        "channel_events": events,
        "relations": [],
        "relation_count": len(relations),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "packets": packets,
        "views": views,
    }


def _merge_units(left: dict[str, Any], right: dict[str, Any], symbol: str, operator: str) -> dict[str, Any]:
    span = [min(int(left["span"][0]), int(right["span"][0])), max(int(left["span"][1]), int(right["span"][1]))]
    return {
        "symbol": symbol,
        "unit_type": "motif",
        "operator": operator,
        "base_symbols": list(left.get("base_symbols") or [left["symbol"]]) + list(right.get("base_symbols") or [right["symbol"]]),
        "event_ids": list(left.get("event_ids") or []) + list(right.get("event_ids") or []),
        "packet_ids": list(left.get("packet_ids") or []) + list(right.get("packet_ids") or []),
        "span": span,
        "channels": sorted(set(left.get("channels") or []) | set(right.get("channels") or []), key=lambda ch: CHANNEL_RANK.get(ch, 999)),
        "geometry_clusters": sorted(set(left.get("geometry_clusters") or []) | set(right.get("geometry_clusters") or [])),
        "raw_geometry_clusters": sorted(set(left.get("raw_geometry_clusters") or []) | set(right.get("raw_geometry_clusters") or [])),
        "observable_refinement_tags": sorted(set(left.get("observable_refinement_tags") or []) | set(right.get("observable_refinement_tags") or [])),
        "relation_types": sorted(set(left.get("relation_types") or []) | set(right.get("relation_types") or [])),
    }


def _channel_sequence_views(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_id = str(record["case_id"])
        for channel, seq in (record.get("views", {}).get("channel_sequences") or {}).items():
            if len(seq) >= 2:
                sequences[f"{case_id}::channel::{channel}"] = [dict(unit) for unit in seq]
    return sequences


def _channel_symbol_sequence_views(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for record in records:
        case_id = str(record["case_id"])
        for channel, seq in (record.get("views", {}).get("channel_sequences") or {}).items():
            if len(seq) >= 2:
                out[f"{case_id}::channel::{channel}"] = [str(unit.get("symbol") or "") for unit in seq]
    return out


def _pair_stats_symbols(sequences: dict[str, list[str]]) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sequence_id, seq in sequences.items():
        if len(seq) < 2:
            continue
        case_id = sequence_id.split("::", 1)[0]
        for idx in range(len(seq) - 1):
            pair = (seq[idx], seq[idx + 1])
            counts[pair] += 1
            cases[pair].add(case_id)
    return counts, cases


def _selected_pair_examples(sequences: dict[str, list[str]], pair: tuple[str, str], limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for sequence_id, seq in sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for idx in range(len(seq) - 1):
            if seq[idx] != pair[0] or seq[idx + 1] != pair[1]:
                continue
            examples.append(
                {
                    "case_id": case_id,
                    "sequence_id": sequence_id,
                    "left_symbol": pair[0],
                    "right_symbol": pair[1],
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _select_pair(
    counts: Counter[tuple[str, str]],
    cases: dict[tuple[str, str], set[str]],
    *,
    min_pair_count: int,
    min_pair_support: int,
    selection: str,
) -> tuple[str, str] | None:
    candidates = [
        pair
        for pair, count in counts.items()
        if count >= min_pair_count and len(cases[pair]) >= min_pair_support
    ]
    if not candidates:
        return None
    if selection == "support":
        return max(candidates, key=lambda pair: (len(cases[pair]), counts[pair], pair))
    return max(candidates, key=lambda pair: (counts[pair], len(cases[pair]), pair))


def _apply_merge(seq: list[dict[str, Any]], pair: tuple[str, str], merged_symbol: str, operator: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(seq):
        if idx < len(seq) - 1 and seq[idx]["symbol"] == pair[0] and seq[idx + 1]["symbol"] == pair[1]:
            out.append(_merge_units(seq[idx], seq[idx + 1], merged_symbol, operator))
            idx += 2
        else:
            out.append(dict(seq[idx]))
            idx += 1
    return out


def _apply_merge_symbols(seq: list[str], pair: tuple[str, str], merged_symbol: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(seq):
        if idx < len(seq) - 1 and seq[idx] == pair[0] and seq[idx + 1] == pair[1]:
            out.append(merged_symbol)
            idx += 2
        else:
            out.append(seq[idx])
            idx += 1
    return out


def _reconstruct_channel_motif_sequences(
    records: list[dict[str, Any]],
    merges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    sequences = _channel_sequence_views(records)
    for merge in merges:
        if str(merge.get("operator") or "") != "SEQ_CHANNEL_MERGE":
            continue
        pair = tuple(str(item) for item in (merge.get("parents") or []))
        if len(pair) != 2:
            continue
        merged_symbol = str(merge.get("merge_id") or "")
        operator = str(merge.get("operator") or "SEQ_MERGE")
        for sequence_id in list(sequences):
            sequences[sequence_id] = _apply_merge(sequences[sequence_id], pair, merged_symbol, operator)
    return sequences


def _learn_channel_motifs(records: list[dict[str, Any]], args: argparse.Namespace, *, budget: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    sequences = _channel_symbol_sequence_views(records)
    merges: list[dict[str, Any]] = []
    for step in range(1, budget + 1):
        counts, cases = _pair_stats_symbols(sequences)
        pair = _select_pair(
            counts,
            cases,
            min_pair_count=int(args.min_pair_count),
            min_pair_support=int(args.min_pair_support),
            selection=str(args.selection),
        )
        if pair is None:
            break
        merged_symbol = f"<CHM_{step:04d}>"
        examples = _selected_pair_examples(sequences, pair, int(args.examples_per_motif))
        merges.append(
            {
                "merge_id": merged_symbol,
                "step": step,
                "parents": list(pair),
                "operator": "SEQ_CHANNEL_MERGE",
                "operator_counts": {"SEQ_CHANNEL_MERGE": int(counts[pair])},
                "count": int(counts[pair]),
                "support_cases": len(cases[pair]),
                "example_case_ids": sorted(cases[pair])[: int(args.examples_per_motif)],
                "example_occurrences": examples,
            }
        )
        for sequence_id in list(sequences):
            sequences[sequence_id] = _apply_merge_symbols(sequences[sequence_id], pair, merged_symbol)
    return merges, _reconstruct_channel_motif_sequences(records, merges)


def _channel_units_by_case(channel_sequences: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, seq in channel_sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            by_case[case_id].append(dict(unit))
    for units in by_case.values():
        units.sort(key=lambda unit: (int((unit.get("span") or [0, 0])[0]), int((unit.get("span") or [0, 0])[1]), str(unit.get("symbol") or "")))
    return by_case


COORDINATION_SEED_TAGS = {
    "raw_joint_trajectory",
    "raw_joint_reach",
    "raw_joint_hand_proximity",
    "raw_joint_leg_lateral",
    "raw_joint_body_level",
    "raw_joint_body_support",
}

COORDINATION_SEED_CLUSTER_HINTS = {
    "RAISE_SPREAD_VERTICAL",
    "BI_RAISE_SPREAD",
    "BI_RAISE",
    "HANDS_CLOSE_VERTICAL",
    "BI_HANDS_CLOSE",
    "BILATERAL_VERTICAL_ARM_CYCLE",
    "ARM_ORBIT",
    "LARGE_ARM_ARC",
    "ARM_REACH",
    "ARM_RETRACT",
    "HAND_NEAR_HEAD",
    "HAND_APPROACH_HEAD",
    "HAND_LEAVE_HEAD",
    "LEG_LATERAL",
    "LOW_BODY",
    "SQUAT",
    "WB_LEVEL",
    "WB_SUPPORT",
    "LEG_FORWARD_HOLD_POSE",
    "LEG_FORWARD_KICK_IMPULSE",
    "LEG_FORWARD_HOP_OR_KICK_IMPULSE",
    "JUMP_UP_IMPULSE",
    "WB_VERT_UP",
    "WB_VERT_DOWN",
    "ACROBAT",
}


def _is_coordination_seed_unit(unit: dict[str, Any]) -> bool:
    tags = {str(tag) for tag in unit.get("observable_refinement_tags") or []}
    if tags & COORDINATION_SEED_TAGS:
        return True
    geometry = {str(item) for item in unit.get("geometry_clusters") or []}
    return any(any(hint in item for hint in COORDINATION_SEED_CLUSTER_HINTS) for item in geometry)


def _contains_any(items: set[str], needles: set[str]) -> bool:
    return any(any(needle in item for needle in needles) for item in items)


def _coordination_role(unit: dict[str, Any]) -> str:
    geometry = {str(item) for item in unit.get("geometry_clusters") or []}
    raw_geometry = {str(item) for item in unit.get("raw_geometry_clusters") or []}
    all_geometry = geometry | raw_geometry
    tags = {str(tag) for tag in unit.get("observable_refinement_tags") or []}
    channel = str((unit.get("channels") or ["other"])[0])

    if "raw_joint_body_support" in tags or _contains_any(all_geometry, {"WB_SUPPORT"}):
        if _contains_any(all_geometry, {"WB_SUPPORT_INVERTED"}):
            return "inverted_support"
        if _contains_any(all_geometry, {"WB_SUPPORT_FLOOR_LOW_HORIZONTAL"}):
            return "floor_low_horizontal_support"
        if _contains_any(all_geometry, {"WB_SUPPORT_HAND_FLOOR_LOW"}):
            return "hand_floor_low_support"
        return "body_support_state"
    if "raw_joint_body_level" in tags or _contains_any(all_geometry, {"WB_LEVEL"}):
        if _contains_any(all_geometry, {"HIGH_LOW_HIGH_CYCLE", "LOW_HIGH_LOW_CYCLE"}):
            return "body_level_cycle"
        if _contains_any(all_geometry, {"DESCEND_TO_LOW"}):
            return "body_level_down"
        if _contains_any(all_geometry, {"RISE_FROM_LOW"}):
            return "body_level_up"
        if _contains_any(all_geometry, {"LOW_BRIEF"}):
            return "body_level_low_hold"
        return "body_level_change"
    if "raw_joint_hand_proximity" in tags or _contains_any(all_geometry, {"HAND_NEAR_HEAD", "HAND_APPROACH_HEAD", "HAND_LEAVE_HEAD"}):
        if _contains_any(all_geometry, {"HAND_APPROACH_HEAD"}):
            return "hand_approach_head"
        if _contains_any(all_geometry, {"HAND_LEAVE_HEAD"}):
            return "hand_leave_head"
        if _contains_any(all_geometry, {"HAND_NEAR_HEAD_REPEATED"}):
            return "hand_near_head_repeat"
        return "hand_near_head"
    if "raw_joint_leg_lateral" in tags or _contains_any(all_geometry, {"LEG_LATERAL"}):
        if _contains_any(all_geometry, {"LEG_LATERAL_REPEAT"}):
            return "leg_lateral_repeat"
        if _contains_any(all_geometry, {"LEG_LATERAL_ADDUCT"}):
            return "leg_lateral_adduct"
        if _contains_any(all_geometry, {"LEG_LATERAL_ABDUCT"}):
            return "leg_lateral_abduct"
        return "leg_lateral"
    if "raw_joint_trajectory" in tags or _contains_any(all_geometry, {"ARM_ORBIT", "LARGE_ARM_ARC"}):
        if _contains_any(all_geometry, {"ARM_ORBIT"}):
            return "arm_orbit_cycle"
        if _contains_any(all_geometry, {"LARGE_ARM_ARC"}):
            return "arm_large_arc"
        return "arm_trajectory"
    if "raw_joint_reach" in tags or _contains_any(all_geometry, {"ARM_REACH", "ARM_RETRACT"}):
        if _contains_any(all_geometry, {"ARM_RETRACT"}):
            return "arm_retract"
        if _contains_any(all_geometry, {"ARM_REACH"}):
            return "arm_reach"
        return "arm_reach_change"
    if _contains_any(all_geometry, {"RAISE_SPREAD_VERTICAL", "BILATERAL_VERTICAL_ARM_CYCLE", "BI_RAISE_SPREAD", "BI_RAISE"}):
        return "bilateral_arm_vertical_cycle"
    if _contains_any(all_geometry, {"HANDS_CLOSE_VERTICAL", "BI_HANDS_CLOSE"}):
        return "hands_close"
    if _contains_any(all_geometry, {"LOW_BODY_DOWN_UP_CYCLE"}):
        return "low_body_cycle"
    if _contains_any(all_geometry, {"LOW_BODY_RISE_FROM_LOW", "WB_LEVEL_RISE_FROM_LOW"}):
        return "low_body_rise"
    if _contains_any(all_geometry, {"LOW_BODY", "SQUAT"}):
        return "low_body_posture"
    if _contains_any(all_geometry, {"LEG_FORWARD_KICK_IMPULSE", "LEG_FORWARD_HOP_OR_KICK_IMPULSE"}):
        return "leg_forward_impulse"
    if _contains_any(all_geometry, {"LEG_FORWARD_HOLD_POSE"}):
        return "leg_forward_hold"
    if _contains_any(all_geometry, {"JUMP_UP_IMPULSE"}):
        return "vertical_impulse"
    if _contains_any(all_geometry, {"VERT_GAIT_BOUNCE"}):
        return "vertical_gait_bounce"
    if _contains_any(all_geometry, {"VERT_LOW_BODY_DESCENT"}):
        return "vertical_low_body_descent"
    if _contains_any(all_geometry, {"VERT"}):
        return "vertical_change"
    if _contains_any(all_geometry, {"ACROBAT", "INVERSION"}):
        return "inversion_or_acrobatics"
    if channel == "root_rotation" or _contains_any(all_geometry, {"WB_ROT"}):
        return "root_turn"
    if channel == "root_locomotion" or _contains_any(all_geometry, {"LOCO"}):
        return "root_translation"
    if _contains_any(all_geometry, {"LOCO_ARM_SWING"}):
        return "arm_swing_context"
    return "generic_motion"


def _coordination_role_priority(unit: dict[str, Any]) -> float:
    role = _coordination_role(unit)
    if role in {"arm_orbit_cycle", "arm_large_arc", "bilateral_arm_vertical_cycle", "leg_lateral_repeat", "body_level_cycle", "inverted_support", "floor_low_horizontal_support"}:
        return 4.0
    if role in {"hand_near_head_repeat", "hand_approach_head", "hand_leave_head", "leg_forward_impulse", "vertical_impulse", "low_body_cycle", "hand_floor_low_support"}:
        return 3.0
    if role in {"hand_near_head", "arm_reach", "arm_retract", "leg_forward_hold", "body_level_down", "body_level_up", "low_body_posture"}:
        return 2.0
    if role in {"root_translation", "root_turn", "vertical_gait_bounce", "arm_swing_context"}:
        return 0.5
    return 1.0


def _coactivation_symbol(units: list[dict[str, Any]]) -> str:
    parts = []
    for unit in sorted(units, key=lambda item: (CHANNEL_RANK.get((item.get("channels") or ["other"])[0], 999), str(item.get("symbol") or ""))):
        channel = str((unit.get("channels") or ["other"])[0])
        symbol = str(unit.get("symbol") or "")
        parts.append(f"{channel}:{symbol}")
    return "COACT[" + "+".join(parts) + "]"


def _coactivation_signature(units: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for unit in sorted(units, key=lambda item: (CHANNEL_RANK.get((item.get("channels") or ["other"])[0], 999), str(item.get("symbol") or ""))):
        channel = str((unit.get("channels") or ["other"])[0])
        parts.append(f"{channel}:{_coordination_role(unit)}")
    return "COORD_SIG[" + "+".join(parts) + "]"


def _coactivation_pair_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    overlap = _overlap_ratio(anchor, candidate)
    if overlap <= 0.0:
        return -1.0
    span = candidate.get("span") or [0, 0]
    duration = max(1, int(span[1]) - int(span[0]) + 1)
    return overlap * 10.0 + _coordination_role_priority(candidate) - 0.001 * duration


def _representative_cross_channel_members(
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    parallel_overlap_min: float,
) -> list[dict[str, Any]]:
    anchor_channel = str((anchor.get("channels") or ["other"])[0])
    best_by_channel: dict[str, tuple[float, dict[str, Any]]] = {anchor_channel: (_coordination_role_priority(anchor), anchor)}
    for candidate in candidates:
        channel = str((candidate.get("channels") or ["other"])[0])
        if channel == anchor_channel:
            continue
        if _overlap_ratio(anchor, candidate) < parallel_overlap_min:
            continue
        score = _coactivation_pair_score(anchor, candidate)
        if score < 0.0:
            continue
        current = best_by_channel.get(channel)
        if current is None or score > current[0]:
            best_by_channel[channel] = (score, candidate)
    members = [item[1] for item in best_by_channel.values()]
    members.sort(key=lambda item: (CHANNEL_RANK.get((item.get("channels") or ["other"])[0], 999), int((item.get("span") or [0, 0])[0]), str(item.get("symbol") or "")))
    return members


def _build_coactivation_units(
    channel_sequences: dict[str, list[dict[str, Any]]],
    *,
    parallel_overlap_min: float,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for case_id, units in _channel_units_by_case(channel_sequences).items():
        candidates = [unit for unit in units if _is_coordination_seed_unit(unit)]
        coacts: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for anchor in sorted(candidates, key=lambda unit: (-_coordination_role_priority(unit), int((unit.get("span") or [0, 0])[0]), str(unit.get("symbol") or ""))):
            members = _representative_cross_channel_members(anchor, candidates, parallel_overlap_min=parallel_overlap_min)
            channels = sorted({channel for unit in members for channel in (unit.get("channels") or [])}, key=lambda ch: CHANNEL_RANK.get(ch, 999))
            if len(channels) < 2:
                continue
            span = [min(int((unit.get("span") or [0, 0])[0]) for unit in members), max(int((unit.get("span") or [0, 0])[1]) for unit in members)]
            member_symbol = _coactivation_symbol(members)
            signature = _coactivation_signature(members)
            dedupe_key = (signature, span[0] // 8, span[1] // 8)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            coacts.append(
                {
                    "symbol": signature,
                    "unit_type": "coactivation_packet",
                    "base_symbols": [str(symbol) for unit in members for symbol in (unit.get("base_symbols") or [unit.get("symbol")])],
                    "event_ids": [str(event_id) for unit in members for event_id in (unit.get("event_ids") or [])],
                    "packet_ids": [],
                    "span": span,
                    "channels": channels,
                    "geometry_clusters": sorted({str(cluster) for unit in members for cluster in (unit.get("geometry_clusters") or [])}),
                    "raw_geometry_clusters": sorted({str(cluster) for unit in members for cluster in (unit.get("raw_geometry_clusters") or [])}),
                    "observable_refinement_tags": sorted({str(tag) for unit in members for tag in (unit.get("observable_refinement_tags") or [])}),
                    "relation_types": ["coactivation"],
                    "member_symbols": [str(unit.get("symbol") or "") for unit in members],
                    "member_roles": [f"{str((unit.get('channels') or ['other'])[0])}:{_coordination_role(unit)}" for unit in members],
                    "member_coactivation_symbol": member_symbol,
                    "coactivation_id": f"{case_id}:coact{len(coacts) + 1:04d}",
                }
            )
        coacts.sort(key=lambda unit: (int(unit["span"][0]), int(unit["span"][1]), str(unit["symbol"])))
        if len(coacts) >= 1:
            out[f"{case_id}::coactivation"] = coacts
    return out


def _coactivation_stats(
    sequences: dict[str, list[dict[str, Any]]],
) -> tuple[Counter[str], dict[str, set[str]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Counter[str]]]]:
    counts: Counter[str] = Counter()
    cases: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "channels": Counter(),
            "geometry_clusters": Counter(),
            "observable_refinement_tags": Counter(),
            "relation_types": Counter(),
        }
    )
    for sequence_id, seq in sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            counts[symbol] += 1
            cases[symbol].add(case_id)
            meta[symbol]["channels"].update(str(item) for item in unit.get("channels") or [])
            meta[symbol]["geometry_clusters"].update(str(item) for item in unit.get("geometry_clusters") or [])
            meta[symbol]["observable_refinement_tags"].update(str(item) for item in unit.get("observable_refinement_tags") or [])
            meta[symbol]["relation_types"].update(str(item) for item in unit.get("relation_types") or [])
            if len(examples[symbol]) < 12:
                examples[symbol].append(
                    {
                        "case_id": case_id,
                        "sequence_id": sequence_id,
                        "span": unit.get("span"),
                        "member_symbols": unit.get("member_symbols") or [],
                        "member_coactivation_symbol": unit.get("member_coactivation_symbol"),
                    }
                )
    return counts, cases, examples, meta


def _channel_zone(channel: str) -> str:
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
    if channel == "whole_body_support":
        return "support"
    if channel == "acrobatics_or_inversion":
        return "inversion"
    return "other"


def _coordination_structure_features(
    symbol: str,
    *,
    count: int,
    support_cases: int,
    meta: dict[str, Counter[str]],
) -> dict[str, Any]:
    channels = sorted(meta.get("channels", Counter()), key=lambda ch: CHANNEL_RANK.get(ch, 999))
    zones = sorted({_channel_zone(channel) for channel in channels})
    geometry = set(meta.get("geometry_clusters", Counter()))
    tags = set(meta.get("observable_refinement_tags", Counter()))

    has_upper = "upper" in zones
    has_lower = "lower" in zones
    has_vertical = "vertical" in zones
    has_root = "root" in zones
    has_torso = "torso" in zones
    has_support = "support" in zones
    has_inversion = "inversion" in zones

    gait_like = any(tag in {"gait_phase", "gait_bounce"} for tag in tags) or any("GAIT_CONTEXT" in item or "LOCO_ARM_SWING" in item for item in geometry)
    root_context = any("ROOT_DRIFT" in item or "PATH_FRAGMENT" in item for item in geometry)
    generic_vertical = any("VERT_UP_REFINED_GENERIC" in item or "VERT_DOWN_REFINED_GENERIC" in item or "VERT_GENERIC" in item for item in geometry)
    low_body = any("LOW_BODY" in item or "SQUAT" in item for item in geometry) or "low_body_coupled" in tags
    leg_non_gait = any("LEG_FORWARD_HOLD_POSE" in item or "LEG_FORWARD_KICK_IMPULSE" in item or "LEG_FORWARD_HOP_OR_KICK_IMPULSE" in item for item in geometry)
    vertical_impulse = any("JUMP_UP_IMPULSE" in item or "SALIENT_DESCENT" in item for item in geometry)
    arm_vertical = "vertical_coupled" in tags and has_upper
    bimanual_vertical = any("RAISE_SPREAD_VERTICAL" in item or "HANDS_CLOSE_VERTICAL" in item for item in geometry)
    high_pose = any("HIGH_POSE" in item for item in geometry)
    arm_trajectory = "raw_joint_trajectory" in tags or any("ARM_ORBIT" in item or "LARGE_ARM_ARC" in item for item in geometry)
    arm_reach = "raw_joint_reach" in tags or any("ARM_REACH" in item or "ARM_RETRACT" in item for item in geometry)
    hand_proximity = "raw_joint_hand_proximity" in tags or any("HAND_NEAR_HEAD" in item or "HAND_APPROACH_HEAD" in item or "HAND_LEAVE_HEAD" in item for item in geometry)
    leg_lateral = "raw_joint_leg_lateral" in tags or any("LEG_LATERAL" in item for item in geometry)
    body_level = "raw_joint_body_level" in tags or any("WB_LEVEL" in item for item in geometry)
    body_support = "raw_joint_body_support" in tags or any("WB_SUPPORT" in item for item in geometry)
    inverted_support = any("WB_SUPPORT_INVERTED" in item for item in geometry)
    floor_support = any("WB_SUPPORT_FLOOR_LOW_HORIZONTAL" in item or "WB_SUPPORT_HAND_FLOOR_LOW" in item for item in geometry)

    score = math.log1p(max(0, support_cases))
    score += 0.35 * len(channels)
    score += 0.55 * len(zones)
    if has_upper and has_vertical:
        score += 1.25
    if has_lower and has_vertical and not gait_like:
        score += 1.00
    if has_upper and low_body:
        score += 1.00
    if has_torso and low_body:
        score += 0.80
    if bimanual_vertical:
        score += 1.50
    if arm_vertical:
        score += 0.75
    if leg_non_gait:
        score += 0.80
    if vertical_impulse:
        score += 0.80
    if high_pose and (has_vertical or low_body):
        score += 0.50
    if arm_trajectory and (has_lower or has_vertical or body_level):
        score += 1.20
    if arm_reach and (has_torso or has_root or has_lower or body_level):
        score += 0.90
    if hand_proximity and body_level:
        score += 1.30
    if leg_lateral and has_upper:
        score += 1.25
    if leg_lateral and has_vertical:
        score += 0.90
    if body_level and has_upper:
        score += 0.90
    if body_support:
        score += 1.30
    if inverted_support and (has_upper or has_lower or has_inversion):
        score += 1.30
    if floor_support and (has_upper or has_torso):
        score += 1.00
    if gait_like:
        score -= 2.00
    if gait_like and not (has_upper or has_torso or low_body or leg_non_gait or vertical_impulse or bimanual_vertical):
        score -= 2.50
    if root_context:
        score -= 1.00
    if generic_vertical and not (has_upper or low_body or leg_non_gait):
        score -= 0.80
    if len(zones) <= 1:
        score -= 1.50
    if has_root and not (has_upper or has_lower or has_vertical or has_torso):
        score -= 1.00

    return {
        "score": round(score, 6),
        "channels": channels,
        "zones": zones,
        "gait_like": gait_like,
        "root_context": root_context,
        "generic_vertical": generic_vertical,
        "low_body": low_body,
        "leg_non_gait": leg_non_gait,
        "vertical_impulse": vertical_impulse,
        "arm_vertical": arm_vertical,
        "bimanual_vertical": bimanual_vertical,
        "high_pose": high_pose,
        "arm_trajectory": arm_trajectory,
        "arm_reach": arm_reach,
        "hand_proximity": hand_proximity,
        "leg_lateral": leg_lateral,
        "body_level": body_level,
        "body_support": body_support,
        "inverted_support": inverted_support,
        "floor_support": floor_support,
        "count": int(count),
        "support_cases": int(support_cases),
        "symbol": symbol,
    }


def _select_coordination_symbols(
    counts: Counter[str],
    cases: dict[str, set[str]],
    meta: dict[str, dict[str, Counter[str]]],
    args: argparse.Namespace,
    budget: int,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    candidates = [
        symbol
        for symbol, count in counts.items()
        if count >= int(args.min_pair_count) and len(cases[symbol]) >= int(args.min_pair_support)
    ]
    features = {
        symbol: _coordination_structure_features(
            symbol,
            count=int(counts[symbol]),
            support_cases=len(cases[symbol]),
            meta=meta.get(symbol, {}),
        )
        for symbol in candidates
    }
    selection = str(getattr(args, "coordination_selection", "support"))
    if selection == "structure_score":
        min_score = float(getattr(args, "coordination_min_structure_score", 0.0))
        candidates = [symbol for symbol in candidates if float(features[symbol]["score"]) >= min_score]
        candidates.sort(key=lambda symbol: (-float(features[symbol]["score"]), -len(cases[symbol]), -counts[symbol], symbol))
    else:
        candidates.sort(key=lambda symbol: (-len(cases[symbol]), -counts[symbol], symbol))
    return candidates[:budget], features


def _coordination_unit_from_coactivation(unit: dict[str, Any], merge_id: str) -> dict[str, Any]:
    out = dict(unit)
    out["symbol"] = merge_id
    out["unit_type"] = "coordination_motif"
    out["operator"] = "COORDINATION_MERGE"
    out["relation_types"] = sorted(set(unit.get("relation_types") or []) | {"coordination"})
    return out


def _learn_coordination_motifs(
    coactivation_sequences: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    start_step: int,
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    counts, cases, examples, meta = _coactivation_stats(coactivation_sequences)
    selected, structure_features = _select_coordination_symbols(counts, cases, meta, args, budget)
    merges: list[dict[str, Any]] = []
    symbol_to_merge: dict[str, str] = {}
    for offset, symbol in enumerate(selected):
        step = start_step + offset
        merge_id = f"<COM_{offset + 1:04d}>"
        symbol_to_merge[symbol] = merge_id
        merges.append(
            {
                "merge_id": merge_id,
                "step": step,
                "parents": [symbol],
                "operator": "COORDINATION_MERGE",
                "operator_counts": {"COORDINATION_MERGE": int(counts[symbol])},
                "count": int(counts[symbol]),
                "support_cases": len(cases[symbol]),
                "selection_score": structure_features.get(symbol, {}).get("score"),
                "selection_features": structure_features.get(symbol, {}),
                "example_case_ids": sorted(cases[symbol])[: int(args.examples_per_motif)],
                "example_occurrences": examples[symbol][: int(args.examples_per_motif)],
            }
        )

    sequences: dict[str, list[dict[str, Any]]] = {}
    for sequence_id, seq in coactivation_sequences.items():
        replaced: list[dict[str, Any]] = []
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            if symbol in symbol_to_merge:
                replaced.append(_coordination_unit_from_coactivation(unit, symbol_to_merge[symbol]))
            else:
                replaced.append(dict(unit))
        if replaced:
            sequences[sequence_id] = replaced
    return merges, sequences


def learn_multichannel_bpe(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    total_budget = max(1, int(args.num_merges))
    channel_ratio = min(1.0, max(0.0, float(args.channel_merge_ratio)))
    channel_budget = min(total_budget, max(1, int(round(total_budget * channel_ratio))))
    channel_merges, channel_sequences = _learn_channel_motifs(records, args, budget=channel_budget)
    coordination_budget = 0 if channel_ratio >= 1.0 else max(0, total_budget - len(channel_merges))
    coactivation_sequences = _build_coactivation_units(
        channel_sequences,
        parallel_overlap_min=float(args.parallel_overlap_min),
    )
    coordination_merges, coordination_sequences = _learn_coordination_motifs(
        coactivation_sequences,
        args,
        start_step=len(channel_merges) + 1,
        budget=coordination_budget,
    )
    sequences = dict(channel_sequences)
    sequences.update(coordination_sequences)
    return channel_merges + coordination_merges, sequences


def _motif_occurrences(sequences: dict[str, list[dict[str, Any]]], motif_symbols: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, seq in sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            if symbol in motif_symbols:
                out[symbol].append({"case_id": case_id, "sequence_id": sequence_id, **dict(unit)})
    return dict(out)


def _top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"id": key, "count": int(value)} for key, value in counter.most_common(limit)]


def audit_motifs(records: list[dict[str, Any]], merges: list[dict[str, Any]], sequences: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    record_map = {str(record["case_id"]): record for record in records}
    merge_map = {str(merge["merge_id"]): merge for merge in merges}
    occurrences = _motif_occurrences(sequences, set(merge_map))
    rows: list[dict[str, Any]] = []
    for motif_id, motif_occs in occurrences.items():
        support_cases = sorted({str(occ["case_id"]) for occ in motif_occs})
        channel_counter: Counter[str] = Counter()
        geometry_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        refinement_counter: Counter[str] = Counter()
        alias_counter: Counter[str] = Counter()
        base_counter: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        alias_seen_cases: set[str] = set()
        for occ in motif_occs:
            case_id = str(occ["case_id"])
            channel_counter.update(str(item) for item in occ.get("channels") or [])
            geometry_counter.update(str(item) for item in occ.get("geometry_clusters") or [])
            refinement_counter.update(str(item) for item in occ.get("observable_refinement_tags") or [])
            relation_counter.update(str(item) for item in occ.get("relation_types") or [])
            base_counter.update(str(item) for item in occ.get("base_symbols") or [])
            if case_id not in alias_seen_cases:
                record = record_map.get(case_id, {})
                alias_ids = [str(item) for item in record.get("caption_alias_ids") or [] if item]
                alias_counter.update(alias_ids or ["__NO_CAPTION_ALIAS__"])
                alias_seen_cases.add(case_id)
            if len(examples) < int(args.examples_per_motif):
                record = record_map.get(case_id, {})
                examples.append(
                    {
                        "case_id": case_id,
                        "sequence_id": occ.get("sequence_id"),
                        "span": occ.get("span"),
                        "event_ids": occ.get("event_ids") or [],
                        "packet_ids": occ.get("packet_ids") or [],
                        "caption": (record.get("caption_texts") or [""])[0],
                    }
                )
        top_alias, top_alias_count = alias_counter.most_common(1)[0] if alias_counter else ("", 0)
        alias_purity = 0.0 if top_alias == "__NO_CAPTION_ALIAS__" else top_alias_count / max(1, len(support_cases))
        merge = merge_map[motif_id]
        parent_signature = str((merge.get("parents") or [""])[0])
        motion_role_signature = parent_signature if parent_signature.startswith("COORD_SIG[") else ""
        rows.append(
            {
                "motif_id": motif_id,
                "step": int(merge.get("step") or 0),
                "operator": merge.get("operator"),
                "parents": merge.get("parents") or [],
                "motion_role_signature": motion_role_signature,
                "selection_score": merge.get("selection_score"),
                "selection_features": merge.get("selection_features"),
                "occurrences": len(motif_occs),
                "support_cases": len(support_cases),
                "channels": _top_counter(channel_counter, 12),
                "relation_profile": _top_counter(relation_counter, 8),
                "top_geometry_clusters": _top_counter(geometry_counter, 12),
                "top_observable_refinement_tags": _top_counter(refinement_counter, 12),
                "top_base_symbols": _top_counter(base_counter, 12),
                "caption_alias_purity": round(alias_purity, 4),
                "top_caption_alias": "" if top_alias == "__NO_CAPTION_ALIAS__" else top_alias,
                "top_caption_aliases": [item for item in _top_counter(alias_counter, 8) if item["id"] != "__NO_CAPTION_ALIAS__"],
                "example_occurrences": examples,
            }
        )
    rows.sort(key=lambda item: (-int(item["support_cases"]), -int(item["occurrences"]), int(item["step"])))
    return rows


def _required_ids(items: list[dict[str, Any]], *, max_items: int = 4, min_relative_count: float = 0.50) -> list[str]:
    if not items:
        return []
    top = max(int(item.get("count") or 0) for item in items)
    cutoff = max(1, int(round(top * min_relative_count)))
    kept = [
        str(item.get("id") or "")
        for item in sorted(items, key=lambda row: (-int(row.get("count") or 0), str(row.get("id") or "")))
        if str(item.get("id") or "") and int(item.get("count") or 0) >= cutoff
    ]
    if not kept:
        kept = [str(max(items, key=lambda row: int(row.get("count") or 0)).get("id") or "")]
    return sorted(kept[:max_items])


def _family_motion_scope(operator: str, required_channels: list[str], role_signature: str) -> str:
    channels = set(required_channels)
    zones = {_channel_zone(channel) for channel in channels}
    if operator != "COORDINATION_MERGE":
        if len(channels) == 1:
            return f"local_{next(iter(zones or {'other'}))}_component"
        return "sequential_component"

    has_upper = "upper" in zones
    has_lower = "lower" in zones
    has_vertical = "vertical" in zones
    has_torso = "torso" in zones
    has_root = "root" in zones
    has_support = "support" in zones
    has_inversion = "inversion" in zones
    role = role_signature.lower()

    if has_support and ("floor_low_horizontal_support" in role or "hand_floor_low_support" in role):
        return "floor_support_coordination"
    if has_support and "inverted_support" in role:
        return "inverted_support_coordination"
    if has_inversion or "inversion_or_acrobatics" in role:
        return "inversion_acrobatic_coordination"
    if "whole_body_state" in channels and (has_torso or "body_level" in role or "low_body" in role):
        return "body_level_posture_transition"
    if has_upper and has_lower and (has_vertical or has_torso or has_root or "body_level" in role):
        return "whole_body_coordination"
    if has_upper and has_lower:
        return "upper_lower_coordination_component"
    if has_upper and has_vertical:
        return "upper_vertical_coordination_component"
    if has_lower and has_vertical:
        return "lower_vertical_coordination_component"
    if has_upper:
        return "upper_body_coordination_component"
    if has_lower:
        return "lower_body_coordination_component"
    return "coordination_component"


def _family_status(motion_scope: str, support_sum: int) -> str:
    if support_sum < 24:
        return "diagnostic_family"
    if motion_scope in {"whole_body_coordination", "body_level_posture_transition"}:
        return "composition_candidate"
    if support_sum >= 80:
        return "stable_component_candidate"
    return "component_candidate"


def build_motif_families(motif_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_meta: dict[str, dict[str, Any]] = {}
    for motif in motif_rows:
        required_clusters = _required_ids(motif.get("top_geometry_clusters") or [])
        required_channels = _required_ids(motif.get("channels") or [], max_items=6, min_relative_count=0.35)
        relation_types = _required_ids(motif.get("relation_profile") or [], max_items=3, min_relative_count=0.35)
        operator = str(motif.get("operator") or "")
        role_signature = str(motif.get("motion_role_signature") or "")
        structure_key = role_signature if role_signature else "+".join(required_clusters or ["none"])
        key = "|".join(
            [
                f"op={operator}",
                "channels=" + "+".join(required_channels or ["none"]),
                "relations=" + "+".join(relation_types or ["none"]),
                "structure=" + structure_key,
            ]
        )
        grouped[key].append(motif)
        group_meta[key] = {
            "operator": operator,
            "required_channels": required_channels,
            "required_relation_types": relation_types,
            "required_geometry_clusters": required_clusters,
            "motion_role_signature": role_signature,
        }

    families: list[dict[str, Any]] = []
    for idx, (key, rows) in enumerate(
        sorted(grouped.items(), key=lambda item: (-sum(int(row.get("support_cases") or 0) for row in item[1]), item[0])),
        start=1,
    ):
        support_sum = sum(int(row.get("support_cases") or 0) for row in rows)
        occurrence_sum = sum(int(row.get("occurrences") or 0) for row in rows)
        channel_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        geometry_counter: Counter[str] = Counter()
        alias_counter: Counter[str] = Counter()
        for row in rows:
            channel_counter.update({str(item["id"]): int(item["count"]) for item in row.get("channels") or []})
            relation_counter.update({str(item["id"]): int(item["count"]) for item in row.get("relation_profile") or []})
            geometry_counter.update({str(item["id"]): int(item["count"]) for item in row.get("top_geometry_clusters") or []})
            alias = str(row.get("top_caption_alias") or "")
            if alias:
                alias_counter[alias] += int(row.get("support_cases") or 0)
        motion_scope = _family_motion_scope(
            str(group_meta[key].get("operator") or ""),
            list(group_meta[key].get("required_channels") or []),
            str(group_meta[key].get("motion_role_signature") or ""),
        )
        status = _family_status(motion_scope, support_sum)
        families.append(
            {
                "family_id": f"multichannel_motif_family_{idx:04d}",
                "schema_version": "multichannel_motif_family_candidate_v1",
                "motion_family_key": key,
                "status": status,
                "motion_scope": motion_scope,
                "motif_count": len(rows),
                "support_cases_sum": support_sum,
                "occurrences_sum": occurrence_sum,
                "motion_definition": {
                    **group_meta[key],
                    "top_channels": _top_counter(channel_counter, 8),
                    "top_relation_types": _top_counter(relation_counter, 6),
                    "top_geometry_clusters": _top_counter(geometry_counter, 10),
                },
                "source_motifs": [
                    {
                        "motif_id": row.get("motif_id"),
                        "operator": row.get("operator"),
                        "support_cases": row.get("support_cases"),
                        "occurrences": row.get("occurrences"),
                        "top_caption_alias": row.get("top_caption_alias"),
                    }
                    for row in sorted(rows, key=lambda item: (-int(item.get("support_cases") or 0), str(item.get("motif_id") or "")))
                ],
                "naming_diagnostics": {
                    "top_caption_aliases": _top_counter(alias_counter, 8),
                    "policy": "diagnostic only; not used to create the family",
                },
            }
        )
    return {
        "schema_version": "multichannel_motif_family_candidates_v1",
        "family_count": len(families),
        "families": families,
    }


def build_pattern_forest_candidates(family_payload: dict[str, Any], motif_rows: list[dict[str, Any]]) -> dict[str, Any]:
    motif_by_id = {str(row.get("motif_id") or ""): row for row in motif_rows}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for family in family_payload.get("families") or []:
        family_id = str(family.get("family_id") or "")
        nodes.append(
            {
                "node_id": family_id,
                "node_kind": "motif_family",
                "status": family.get("status"),
                "motion_scope": family.get("motion_scope"),
                "motion_family_key": family.get("motion_family_key"),
                "support_cases_sum": family.get("support_cases_sum"),
                "occurrences_sum": family.get("occurrences_sum"),
                "motion_definition": family.get("motion_definition"),
                "naming_diagnostics": family.get("naming_diagnostics"),
            }
        )
        for source in family.get("source_motifs") or []:
            motif_id = str(source.get("motif_id") or "")
            if not motif_id:
                continue
            motif_node_id = "motif_" + motif_id.strip("<>").lower()
            motif = motif_by_id.get(motif_id, {})
            nodes.append(
                {
                    "node_id": motif_node_id,
                    "node_kind": "motif_leaf",
                    "status": "candidate" if int(motif.get("support_cases") or 0) >= 80 else "diagnostic",
                    "motion_role_signature": motif.get("motion_role_signature"),
                    "motif_id": motif_id,
                    "operator": motif.get("operator"),
                    "support_cases": motif.get("support_cases"),
                    "occurrences": motif.get("occurrences"),
                    "channels": motif.get("channels"),
                    "relation_profile": motif.get("relation_profile"),
                    "top_geometry_clusters": motif.get("top_geometry_clusters"),
                    "naming_diagnostics": {
                        "top_caption_alias": motif.get("top_caption_alias"),
                        "caption_alias_purity": motif.get("caption_alias_purity"),
                    },
                }
            )
            edges.append(
                {
                    "parent_node_id": family_id,
                    "child_node_id": motif_node_id,
                    "edge_type": "family_member",
                    "policy": "family membership is grouped from operator, channels, relation profile, and required geometry clusters",
                }
            )
    # Motifs can be duplicated if future grouping changes; keep the first stable occurrence.
    deduped_nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if node_id in seen:
            continue
        deduped_nodes.append(node)
        seen.add(node_id)
    return {
        "schema_version": "multichannel_motion_pattern_forest_candidates_v1",
        "runtime_policy": "offline candidate forest only; not the runtime AML tree",
        "summary": {
            "family_count": len(family_payload.get("families") or []),
            "motif_count": len(motif_rows),
            "node_count": len(deduped_nodes),
            "edge_count": len(edges),
        },
        "nodes": deduped_nodes,
        "edges": edges,
    }


def _base_vocab(records: list[dict[str, Any]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        for event in record.get("channel_events") or []:
            out[str(event["symbol"])] += 1
    return out


def _packet_vocab(records: list[dict[str, Any]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        for packet in record.get("packets") or []:
            out[str(packet["packet_symbol"])] += 1
    return out


def _final_vocab(sequences: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for seq in sequences.values():
        for unit in seq:
            out[str(unit.get("symbol") or "")] += 1
    return out


def _channel_input_token_count(records: list[dict[str, Any]]) -> int:
    return sum(
        len(seq)
        for record in records
        for seq in (record.get("views", {}).get("channel_sequences") or {}).values()
    )


def _channel_sequence_count(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        for seq in (record.get("views", {}).get("channel_sequences") or {}).values()
        if len(seq) >= 2
    )


def _packet_diagnostic_token_count(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        total += len(record.get("views", {}).get("packet_sequence") or [])
    return total


def _summary(records: list[dict[str, Any]], merges: list[dict[str, Any]], sequences: dict[str, list[dict[str, Any]]], args: argparse.Namespace, source_records: int) -> dict[str, Any]:
    channel_event_count = sum(len(record.get("channel_events") or []) for record in records)
    arm_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_trajectory" in set(event.get("observable_refinement_tags") or [])
    )
    body_level_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_body_level" in set(event.get("observable_refinement_tags") or [])
    )
    arm_reach_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_reach" in set(event.get("observable_refinement_tags") or [])
    )
    hand_proximity_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_hand_proximity" in set(event.get("observable_refinement_tags") or [])
    )
    leg_lateral_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_leg_lateral" in set(event.get("observable_refinement_tags") or [])
    )
    body_support_sidecar_event_count = sum(
        1
        for record in records
        for event in (record.get("channel_events") or [])
        if "raw_joint_body_support" in set(event.get("observable_refinement_tags") or [])
    )
    packet_count = sum(len(record.get("packets") or []) for record in records)
    parallel_packet_count = sum(1 for record in records for packet in (record.get("packets") or []) if packet.get("packet_type") == "parallel")
    relation_count = sum(int(record.get("relation_count", len(record.get("relations") or []))) for record in records)
    relation_type_counts: Counter[str] = Counter()
    for record in records:
        if record.get("relation_type_counts"):
            relation_type_counts.update({str(key): int(value) for key, value in (record.get("relation_type_counts") or {}).items()})
        else:
            relation_type_counts.update(str(rel.get("relation") or "") for rel in (record.get("relations") or []))
    base_vocab = _base_vocab(records)
    packet_vocab = _packet_vocab(records)
    final_vocab = _final_vocab(sequences)
    channel_input_tokens = _channel_input_token_count(records)
    packet_diagnostic_tokens = _packet_diagnostic_token_count(records)
    channel_output_token_count = sum(len(seq) for sequence_id, seq in sequences.items() if "::channel::" in sequence_id)
    coordination_output_token_count = sum(len(seq) for sequence_id, seq in sequences.items() if sequence_id.endswith("::coactivation"))
    final_token_count = channel_output_token_count + coordination_output_token_count
    operator_counts = Counter(str(merge.get("operator") or "") for merge in merges)
    channel_motif_count = sum(1 for merge in merges if str(merge.get("operator") or "") == "SEQ_CHANNEL_MERGE")
    coordination_motif_count = sum(1 for merge in merges if str(merge.get("operator") or "") == "COORDINATION_MERGE")
    covered_cases = {
        seq_id.split("::", 1)[0]
        for seq_id, seq in sequences.items()
        if any(str(unit.get("symbol") or "").startswith(("<CHM_", "<COM_")) for unit in seq)
    }
    return {
        "version": f"hml3d_multichannel_motion_bpe_{str(getattr(args, 'observable_refinement', 'v1'))}",
        "source_corpus": str(args.source_corpus),
        "observable_refinement": str(getattr(args, "observable_refinement", "v1")),
        "source_record_count": source_records,
        "num_records": len(records),
        "channel_event_count": channel_event_count,
        "arm_trajectory_sidecar_event_count": arm_sidecar_event_count,
        "arm_reach_sidecar_event_count": arm_reach_sidecar_event_count,
        "hand_proximity_sidecar_event_count": hand_proximity_sidecar_event_count,
        "leg_lateral_sidecar_event_count": leg_lateral_sidecar_event_count,
        "body_level_sidecar_event_count": body_level_sidecar_event_count,
        "body_support_sidecar_event_count": body_support_sidecar_event_count,
        "channel_event_type_count": len(base_vocab),
        "packet_count": packet_count,
        "packet_type_count": len(packet_vocab),
        "single_member_packet_count": packet_count - parallel_packet_count,
        "parallel_packet_count": parallel_packet_count,
        "relation_count": relation_count,
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "channel_sequence_count": _channel_sequence_count(records),
        "coordination_sequence_count": sum(1 for sequence_id in sequences if sequence_id.endswith("::coactivation")),
        "channel_input_token_count": channel_input_tokens,
        "channel_output_token_count": channel_output_token_count,
        "coordination_output_token_count": coordination_output_token_count,
        "packet_diagnostic_token_count": packet_diagnostic_tokens,
        "learned_motif_count": len(merges),
        "final_token_count": final_token_count,
        "final_vocab_size": len(final_vocab),
        "channel_bpe_output_ratio": round(channel_output_token_count / max(1, channel_input_tokens), 6),
        "all_output_view_ratio": round(final_token_count / max(1, channel_input_tokens), 6),
        "coordination_motif_ratio": round(coordination_motif_count / max(1, len(merges)), 6),
        "case_coverage": round(len(covered_cases) / max(1, len(records)), 6),
        "covered_case_count": len(covered_cases),
        "base_vocab_size": len(base_vocab),
        "packet_vocab_size": len(packet_vocab),
        "operator_counts": dict(sorted(operator_counts.items())),
        "channel_motif_count": channel_motif_count,
        "coordination_motif_count": coordination_motif_count,
        "coordination_selection": str(getattr(args, "coordination_selection", "support")),
        "coordination_min_structure_score": float(getattr(args, "coordination_min_structure_score", 0.0)),
        "num_merges_requested": int(args.num_merges),
        "channel_merge_ratio": float(args.channel_merge_ratio),
        "min_pair_count": int(args.min_pair_count),
        "min_pair_support": int(args.min_pair_support),
        "selection": str(args.selection),
        "parallel_overlap_min": float(args.parallel_overlap_min),
        "lead_lag_gap_max": int(args.lead_lag_gap_max),
        "heavy_corpora_written": bool(args.write_heavy_corpora),
    }


class MotionBpeSelfTest:
    def _args(self, **overrides: Any) -> argparse.Namespace:
        defaults: dict[str, Any] = {
            "source_corpus": "",
            "output_dir": "",
            "max_records": None,
            "parallel_overlap_min": 0.30,
            "lead_lag_gap_max": 6,
            "num_merges": 8,
            "channel_merge_ratio": 0.5,
            "min_pair_count": 1,
            "min_pair_support": 1,
            "selection": "count",
            "coordination_selection": "support",
            "coordination_min_structure_score": 0.0,
            "examples_per_motif": 4,
            "write_heavy_corpora": False,
            "cache_dir": None,
            "rebuild_cache": False,
            "observable_refinement": "v1",
            "hml3d_root": str(DEFAULT_HML3D_ROOT),
            "disable_arm_trajectory_sidecar": False,
            "arm_span_gap": 6,
            "arm_span_pad": 3,
            "arm_min_radius": 0.18,
            "arm_circle_min_abs_deg": 540.0,
            "arm_circle_max_radius_cv": 0.38,
            "arm_large_arc_min_abs_deg": 180.0,
            "arm_large_arc_min_path": 1.25,
            "arm_large_arc_min_range": 0.40,
            "arm_large_arc_max_radius_cv": 0.55,
            "arm_min_planarity": 0.80,
            "disable_arm_reach_sidecar": False,
            "arm_reach_span_gap": 4,
            "arm_reach_span_pad": 3,
            "arm_reach_min_delta": 0.16,
            "arm_reach_min_path": 0.22,
            "arm_reach_min_peak": 0.24,
            "arm_reach_retract_ratio": 0.55,
            "arm_reach_min_forward_component": 0.10,
            "disable_hand_proximity_sidecar": False,
            "hand_head_proximity_threshold": 0.30,
            "hand_head_min_run": 6,
            "hand_head_hold_min_duration": 10,
            "hand_head_transition_window": 14,
            "hand_head_min_delta": 0.14,
            "hand_head_merge_gap": 4,
            "disable_leg_lateral_sidecar": False,
            "leg_lateral_threshold": 0.16,
            "leg_lateral_min_run": 5,
            "leg_lateral_hold_min_duration": 10,
            "leg_lateral_transition_window": 10,
            "leg_lateral_min_delta": 0.12,
            "leg_lateral_merge_gap": 4,
            "disable_body_level_sidecar": False,
            "body_low_threshold": 0.52,
            "body_high_threshold": 0.58,
            "body_level_min_run": 5,
            "body_long_low_min_duration": 30,
            "body_high_state_min_duration": 8,
            "body_transition_max_gap": 40,
            "body_emit_high_state": False,
            "disable_body_support_sidecar": False,
            "body_support_min_run": 6,
            "body_support_merge_gap": 4,
            "body_prone_height_ratio": 0.34,
            "body_horizontal_axis_threshold": 0.58,
            "body_inverted_head_margin": 0.10,
            "body_hand_floor_ratio": 0.12,
            "body_foot_high_ratio": 0.75,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _toy_record(self, caption: str = "diagnostic text only", aliases: list[str] | None = None) -> dict[str, Any]:
        return {
            "case_id": "toy_0001",
            "num_frames": 20,
            "caption_texts": [caption],
            "caption_alias_ids": aliases or [],
            "events": [
                {
                    "event_index": 0,
                    "span": [0, 9],
                    "duration": 10,
                    "super_family": "BIMANUAL_PERIODIC",
                    "part": "bimanual",
                    "cluster_id": "BI_RAISE_SPREAD",
                    "geometry_cluster_id": "BIMANUAL_PERIODIC/BI_RAISE_SPREAD",
                    "direction": "up_out",
                    "magnitude": 0.40,
                    "unit": "m",
                    "count": 2,
                },
                {
                    "event_index": 1,
                    "span": [1, 10],
                    "duration": 10,
                    "super_family": "WHOLE_BODY_VERTICAL",
                    "part": "whole_body",
                    "cluster_id": "WB_VERT_UP",
                    "geometry_cluster_id": "WHOLE_BODY_VERTICAL/WB_VERT_UP",
                    "direction": "up",
                    "magnitude": 0.30,
                    "unit": "m",
                    "count": 2,
                },
                {
                    "event_index": 2,
                    "span": [11, 19],
                    "duration": 9,
                    "super_family": "BIMANUAL_PERIODIC",
                    "part": "bimanual",
                    "cluster_id": "BI_RAISE_SPREAD",
                    "geometry_cluster_id": "BIMANUAL_PERIODIC/BI_RAISE_SPREAD",
                    "direction": "up_out",
                    "magnitude": 0.40,
                    "unit": "m",
                    "count": 2,
                },
                {
                    "event_index": 3,
                    "span": [11, 19],
                    "duration": 9,
                    "super_family": "WHOLE_BODY_VERTICAL",
                    "part": "whole_body",
                    "cluster_id": "WB_VERT_UP",
                    "geometry_cluster_id": "WHOLE_BODY_VERTICAL/WB_VERT_UP",
                    "direction": "up",
                    "magnitude": 0.30,
                    "unit": "m",
                    "count": 2,
                },
            ],
        }

    def test_motion_bpe_smoke(self) -> None:
        args = self._args()
        record = build_multichannel_record(self._toy_record(), args)
        channels = {str(event["channel"]) for event in record["channel_events"]}
        required = {"bimanual", "whole_body_vertical"}
        assert required.issubset(channels), channels
        vocab = _base_vocab([record])
        assert vocab, "base vocab should not be empty"
        parallel_packets = [packet for packet in record["packets"] if packet.get("packet_type") == "parallel"]
        assert parallel_packets, "expected at least one parallel packet"
        records = [record]
        merges, sequences = learn_multichannel_bpe(records, args)
        assert merges, "expected at least one toy BPE merge"
        assert any(str(merge.get("operator") or "") == "COORDINATION_MERGE" for merge in merges), merges
        motif_rows = audit_motifs(records, merges, sequences, args)
        assert motif_rows, "expected toy motif rows"
        assert all(not any(str(key).endswith("_keywords") for key in row) for row in motif_rows), motif_rows[0]

    def test_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "toy_source.jsonl"
            source_path.write_text(json.dumps(self._toy_record(), ensure_ascii=True) + "\n", encoding="utf-8")
            args = self._args(source_corpus=str(source_path), cache_dir=str(tmp_path / "cache"))
            records_1, source_count_1, cache_info_1 = _load_or_build_multichannel_records(args)
            records_2, source_count_2, cache_info_2 = _load_or_build_multichannel_records(args)
            assert source_count_1 == source_count_2 == 1
            assert len(records_1) == len(records_2) == 1
            assert cache_info_1["status"] == "miss_built", cache_info_1
            assert cache_info_2["status"] == "hit", cache_info_2
            assert Path(cache_info_2["record_cache"]).exists()

    def run(self) -> dict[str, Any]:
        tests = [
            self.test_motion_bpe_smoke,
            self.test_cache_roundtrip,
        ]
        failures: list[dict[str, str]] = []
        for test in tests:
            try:
                test()
            except Exception as exc:
                failures.append({"test": test.__name__, "error": repr(exc)})
        return {
            "ok": not failures,
            "passed": len(tests) - len(failures),
            "failed": len(failures),
            "failures": failures,
        }


def write_report(
    path: Path,
    summary: dict[str, Any],
    motif_rows: list[dict[str, Any]],
    family_payload: dict[str, Any],
    forest_payload: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# HML3D Multi-Channel Motion-BPE Audit")
    lines.append("")
    lines.append("This audit derives channel events and overlap diagnostics from the existing Layer3 event corpus, learns per-channel temporal motifs first, then promotes frequent cross-channel motif coactivations into coordination motifs.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Candidate Forest")
    lines.append("")
    lines.append(f"- `family_count`: `{family_payload.get('family_count', 0)}`")
    lines.append(f"- `node_count`: `{(forest_payload.get('summary') or {}).get('node_count', 0)}`")
    lines.append(f"- `edge_count`: `{(forest_payload.get('summary') or {}).get('edge_count', 0)}`")
    lines.append("")
    lines.append("| family | status | motion scope | support sum | motifs | required channels | required relations | required geometry | top aliases |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for family in (family_payload.get("families") or [])[:80]:
        definition = family.get("motion_definition") or {}
        channels = ", ".join(definition.get("required_channels") or [])
        relations = ", ".join(definition.get("required_relation_types") or [])
        geometry = "<br>".join(definition.get("required_geometry_clusters") or [])
        aliases = ", ".join(
            f"{item['id']}:{item['count']}"
            for item in ((family.get("naming_diagnostics") or {}).get("top_caption_aliases") or [])[:4]
        )
        lines.append(
            f"| `{family['family_id']}` | `{family.get('status')}` | `{family.get('motion_scope')}` | {family.get('support_cases_sum')} | {family.get('motif_count')} | {channels} | {relations} | {geometry} | {aliases} |"
        )
    lines.append("")
    lines.append("## Top Motifs")
    lines.append("")
    lines.append("| motif | operator | support | occurrences | channels | relations | alias | geometry |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for row in motif_rows[:80]:
        channels = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("channels", [])[:5])
        relations = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("relation_profile", [])[:5])
        geometry = "<br>".join(f"{item['id']}:{item['count']}" for item in row.get("top_geometry_clusters", [])[:4])
        lines.append(
            f"| `{row['motif_id']}` | `{row.get('operator')}` | {row['support_cases']} | {row['occurrences']} | {channels} | {relations} | `{row.get('top_caption_alias') or ''}` | {geometry} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_coordination_review(path: Path, motif_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in motif_rows if str(row.get("operator") or "") == "COORDINATION_MERGE"]
    lines: list[str] = []
    lines.append("# Coordination Motif Review")
    lines.append("")
    lines.append("Motion-only review table for cross-channel coordination motifs. Caption aliases are diagnostic only.")
    lines.append("")
    lines.append(f"- coordination motifs: `{len(rows)}`")
    lines.append("")
    for row in rows:
        lines.append(f"## {row.get('motif_id')}")
        lines.append("")
        lines.append(f"- support cases: `{row.get('support_cases')}`")
        lines.append(f"- occurrences: `{row.get('occurrences')}`")
        lines.append(f"- top caption alias: `{row.get('top_caption_alias') or ''}`")
        lines.append(f"- caption alias purity: `{row.get('caption_alias_purity')}`")
        lines.append(f"- parents: `{row.get('parents')}`")
        channels = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("channels", [])[:8])
        geometry = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("top_geometry_clusters", [])[:10])
        lines.append(f"- channels: {channels}")
        lines.append(f"- geometry: {geometry}")
        aliases = ", ".join(f"{item['id']}:{item['count']}" for item in row.get("top_caption_aliases", [])[:6])
        lines.append(f"- caption aliases: {aliases}")
        lines.append("")
        lines.append("| case | span | caption |")
        lines.append("| --- | --- | --- |")
        for example in row.get("example_occurrences", [])[:8]:
            caption = str(example.get("caption") or "").replace("|", "\\|")
            lines.append(f"| `{example.get('case_id')}` | `{example.get('span')}` | {caption} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def output_paths(output_dir: Path, *, write_heavy_corpora: bool) -> dict[str, str]:
    paths = {
        "channel_event_vocab": str(output_dir / "channel_event_vocab.json"),
        "packet_vocab": str(output_dir / "packet_vocab.json"),
        "multichannel_motion_bpe_vocab": str(output_dir / "multichannel_motion_bpe_vocab.json"),
        "case_multichannel_bpe_sequences": str(output_dir / "case_multichannel_bpe_sequences.jsonl"),
        "motif_audit": str(output_dir / "motif_audit.json"),
        "motif_family_candidates": str(output_dir / "motif_family_candidates.json"),
        "motion_pattern_forest_candidates": str(output_dir / "motion_pattern_forest_candidates.json"),
        "summary": str(output_dir / "summary.json"),
        "audit_report": str(output_dir / "audit_report.md"),
        "coordination_review": str(output_dir / "coordination_review.md"),
    }
    if write_heavy_corpora:
        paths["multi_channel_event_corpus"] = str(output_dir / "multi_channel_event_corpus.jsonl")
        paths["overlap_packet_corpus"] = str(output_dir / "overlap_packet_corpus.jsonl")
    return paths


def run_multichannel_motion_bpe(args: argparse.Namespace) -> dict[str, Any]:
    records, source_record_count, cache_info = _load_or_build_multichannel_records(args)
    merges, sequences = learn_multichannel_bpe(records, args)
    motif_rows = audit_motifs(records, merges, sequences, args)
    family_payload = build_motif_families(motif_rows)
    forest_payload = build_pattern_forest_candidates(family_payload, motif_rows)
    summary = _summary(records, merges, sequences, args, source_records=source_record_count)
    summary["record_cache"] = cache_info
    summary["motif_family_count"] = int(family_payload.get("family_count") or 0)
    summary["forest_node_count"] = int((forest_payload.get("summary") or {}).get("node_count") or 0)
    summary["forest_edge_count"] = int((forest_payload.get("summary") or {}).get("edge_count") or 0)
    return {
        "records": records,
        "merges": merges,
        "sequences": sequences,
        "motif_rows": motif_rows,
        "family_payload": family_payload,
        "forest_payload": forest_payload,
        "summary": summary,
    }


def write_multichannel_motion_bpe_outputs(output_dir: Path, result: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = result["records"]
    merges = result["merges"]
    sequences = result["sequences"]
    motif_rows = result["motif_rows"]
    family_payload = result["family_payload"]
    forest_payload = result["forest_payload"]
    summary = dict(result["summary"])
    paths = output_paths(output_dir, write_heavy_corpora=bool(args.write_heavy_corpora))
    summary["outputs"] = paths
    if not args.write_heavy_corpora:
        summary["optional_outputs_not_written"] = {
            "multi_channel_event_corpus": str(output_dir / "multi_channel_event_corpus.jsonl"),
            "overlap_packet_corpus": str(output_dir / "overlap_packet_corpus.jsonl"),
        }

    if args.write_heavy_corpora:
        _write_jsonl(
            output_dir / "multi_channel_event_corpus.jsonl",
            [
                {
                    "case_id": record["case_id"],
                    "num_frames": record["num_frames"],
                    "caption_texts": record.get("caption_texts") or [],
                    "caption_alias_ids": record.get("caption_alias_ids") or [],
                    "channel_events": record.get("channel_events") or [],
                    "relation_count": int(record.get("relation_count") or 0),
                    "relation_type_counts": record.get("relation_type_counts") or {},
                }
                for record in records
            ],
        )
        _write_jsonl(
            output_dir / "overlap_packet_corpus.jsonl",
            [
                {
                    "case_id": record["case_id"],
                    "num_frames": record["num_frames"],
                    "packets": record.get("packets") or [],
                }
                for record in records
            ],
        )

    _write_json(output_dir / "channel_event_vocab.json", {"version": "channel_event_vocab_v1", "items": _top_counter(_base_vocab(records), 5000)})
    _write_json(output_dir / "packet_vocab.json", {"version": "packet_vocab_v1", "items": _top_counter(_packet_vocab(records), 5000)})
    _write_json(
        output_dir / "multichannel_motion_bpe_vocab.json",
        {"version": f"multichannel_motion_bpe_vocab_{str(getattr(args, 'observable_refinement', 'v1'))}", "merges": merges},
    )
    _write_jsonl(
        output_dir / "case_multichannel_bpe_sequences.jsonl",
        [
            {
                "sequence_id": sequence_id,
                "case_id": sequence_id.split("::", 1)[0],
                "view": sequence_id.split("::", 1)[1] if "::" in sequence_id else "",
                "tokens": seq,
            }
            for sequence_id, seq in sorted(sequences.items())
        ],
    )
    _write_json(output_dir / "motif_audit.json", {"summary": summary, "motifs": motif_rows})
    _write_json(output_dir / "motif_family_candidates.json", family_payload)
    _write_json(output_dir / "motion_pattern_forest_candidates.json", forest_payload)
    _write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "audit_report.md", summary, motif_rows, family_payload, forest_payload)
    write_coordination_review(output_dir / "coordination_review.md", motif_rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit concurrency-aware multi-channel Motion-BPE over HML3D Layer3 events.")
    parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--self-test", action="store_true", help="Run built-in smoke tests and exit. Command: python scripts/audit_hml3d_multichannel_motion_bpe.py --self-test")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for built multi-channel records; BPE thresholds can then be tuned without rescanning Layer3 events.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore an existing multi-channel record cache and rebuild it.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--parallel-overlap-min", type=float, default=0.30)
    parser.add_argument("--lead-lag-gap-max", type=int, default=6)
    parser.add_argument("--observable-refinement", choices=["v1", "v2", "v3", "v4"], default="v1", help="Refine coarse Layer3 geometry labels for Motion-BPE tokenization without modifying the source corpus. v4 adds raw-joint whole-body support sidecar events.")
    parser.add_argument("--hml3d-root", default=str(DEFAULT_HML3D_ROOT), help="HumanML3D root used by raw-joint sidecar events.")
    parser.add_argument("--disable-arm-trajectory-sidecar", action="store_true", help="Disable v3 raw-joint arm trajectory sidecar events.")
    parser.add_argument("--arm-span-gap", type=int, default=6, help="Merge same-side arm source spans separated by at most this many frames before trajectory scoring.")
    parser.add_argument("--arm-span-pad", type=int, default=3, help="Pad merged arm spans before trajectory scoring.")
    parser.add_argument("--arm-min-radius", type=float, default=0.18, help="Minimum mean wrist-to-shoulder trajectory radius for arm orbit candidates.")
    parser.add_argument("--arm-circle-min-abs-deg", type=float, default=540.0, help="Minimum accumulated in-plane wrist angle for arm orbit candidates.")
    parser.add_argument("--arm-circle-max-radius-cv", type=float, default=0.38, help="Maximum radius coefficient of variation for arm orbit candidates.")
    parser.add_argument("--arm-large-arc-min-abs-deg", type=float, default=180.0, help="Minimum accumulated in-plane wrist angle for large arm arc candidates.")
    parser.add_argument("--arm-large-arc-min-path", type=float, default=1.25, help="Minimum wrist path length for arm trajectory sidecar candidates.")
    parser.add_argument("--arm-large-arc-min-range", type=float, default=0.40, help="Minimum per-axis wrist trajectory range for large arm arc candidates.")
    parser.add_argument("--arm-large-arc-max-radius-cv", type=float, default=0.55, help="Maximum radius coefficient of variation for large arm arc candidates.")
    parser.add_argument("--arm-min-planarity", type=float, default=0.80, help="Minimum PCA planarity for arm trajectory sidecar candidates.")
    parser.add_argument("--disable-arm-reach-sidecar", action="store_true", help="Disable v3 raw-joint arm reach/retract sidecar events.")
    parser.add_argument("--arm-reach-span-gap", type=int, default=4, help="Merge same-side arm source spans separated by at most this many frames before reach/retract scoring.")
    parser.add_argument("--arm-reach-span-pad", type=int, default=3, help="Pad merged arm spans before reach/retract scoring.")
    parser.add_argument("--arm-reach-min-delta", type=float, default=0.16, help="Minimum body-forward wrist extension range for arm reach/retract candidates.")
    parser.add_argument("--arm-reach-min-path", type=float, default=0.22, help="Minimum accumulated body-forward wrist path for arm reach/retract candidates.")
    parser.add_argument("--arm-reach-min-peak", type=float, default=0.24, help="Minimum peak body-forward wrist offset for arm reach/retract candidates.")
    parser.add_argument("--arm-reach-retract-ratio", type=float, default=0.55, help="Minimum retract/extend ratio to classify an arm reach as reach-retract.")
    parser.add_argument("--arm-reach-min-forward-component", type=float, default=0.10, help="Minimum forward extension and retraction component for arm reach/retract candidates.")
    parser.add_argument("--disable-hand-proximity-sidecar", action="store_true", help="Disable v3 raw-joint hand-to-head proximity sidecar events.")
    parser.add_argument("--hand-head-proximity-threshold", type=float, default=0.30, help="Normalized wrist-to-head distance threshold for hand-near-head states.")
    parser.add_argument("--hand-head-min-run", type=int, default=6, help="Minimum consecutive frames for hand-near-head state.")
    parser.add_argument("--hand-head-hold-min-duration", type=int, default=10, help="Minimum duration for a sustained hand-near-head hold.")
    parser.add_argument("--hand-head-transition-window", type=int, default=14, help="Frames before/after a hand-near-head state used to emit approach/leave transitions.")
    parser.add_argument("--hand-head-min-delta", type=float, default=0.14, help="Minimum normalized distance change for hand approach/leave-head transitions.")
    parser.add_argument("--hand-head-merge-gap", type=int, default=4, help="Merge hand-near-head runs separated by at most this many frames.")
    parser.add_argument("--disable-leg-lateral-sidecar", action="store_true", help="Disable v3 raw-joint leg lateral spread/adduction sidecar events.")
    parser.add_argument("--leg-lateral-threshold", type=float, default=0.16, help="Minimum baseline-relative body-lateral foot displacement for leg-out state.")
    parser.add_argument("--leg-lateral-min-run", type=int, default=5, help="Minimum consecutive frames for a leg-lateral state.")
    parser.add_argument("--leg-lateral-hold-min-duration", type=int, default=10, help="Minimum duration for a sustained leg-lateral hold.")
    parser.add_argument("--leg-lateral-transition-window", type=int, default=10, help="Frames before/after leg-out state used to emit abduct/adduct transitions.")
    parser.add_argument("--leg-lateral-min-delta", type=float, default=0.12, help="Minimum lateral displacement change for leg abduct/adduct transitions.")
    parser.add_argument("--leg-lateral-merge-gap", type=int, default=4, help="Merge leg-lateral runs separated by at most this many frames.")
    parser.add_argument("--disable-body-level-sidecar", action="store_true", help="Disable v3 raw-joint body-level sidecar events.")
    parser.add_argument("--body-low-threshold", type=float, default=0.52, help="Normalized pelvis-height threshold for low body-level state.")
    parser.add_argument("--body-high-threshold", type=float, default=0.58, help="Normalized pelvis-height threshold for high stand-like body-level state.")
    parser.add_argument("--body-level-min-run", type=int, default=5, help="Minimum consecutive frames for a body-level state run.")
    parser.add_argument("--body-long-low-min-duration", type=int, default=30, help="Minimum low-run duration for sustained low-body-level state.")
    parser.add_argument("--body-high-state-min-duration", type=int, default=8, help="Minimum high-run duration for stand-like body-level state.")
    parser.add_argument("--body-transition-max-gap", type=int, default=40, help="Maximum gap between low/high body-level runs to create transition events.")
    parser.add_argument("--body-emit-high-state", action="store_true", help="Emit generic high stand-like body-level states. Disabled by default because they are ubiquitous context.")
    parser.add_argument("--disable-body-support-sidecar", action="store_true", help="Disable v4 raw-joint whole-body support sidecar events.")
    parser.add_argument("--body-support-min-run", type=int, default=6, help="Minimum consecutive frames for body-support state events.")
    parser.add_argument("--body-support-merge-gap", type=int, default=4, help="Merge body-support runs separated by at most this many frames.")
    parser.add_argument("--body-prone-height-ratio", type=float, default=0.34, help="Normalized pelvis height threshold for low floor-support candidates.")
    parser.add_argument("--body-horizontal-axis-threshold", type=float, default=0.58, help="Maximum torso vertical-axis component for low horizontal floor-support candidates.")
    parser.add_argument("--body-inverted-head-margin", type=float, default=0.10, help="Normalized head-below-pelvis margin for inverted support candidates.")
    parser.add_argument("--body-hand-floor-ratio", type=float, default=0.12, help="Normalized wrist-to-floor height threshold for hand-floor support.")
    parser.add_argument("--body-foot-high-ratio", type=float, default=0.75, help="Normalized foot height threshold used with hand-floor evidence for inverted support.")
    parser.add_argument("--num-merges", type=int, default=256)
    parser.add_argument("--channel-merge-ratio", type=float, default=0.5, help="Fraction of merge budget used for per-channel sequential motifs before cross-channel coordination mining.")
    parser.add_argument("--min-pair-count", type=int, default=80)
    parser.add_argument("--min-pair-support", type=int, default=40)
    parser.add_argument("--selection", choices=["count", "support"], default="count")
    parser.add_argument("--coordination-selection", choices=["support", "structure_score"], default="support", help="How to rank cross-channel coactivation motifs after channel BPE.")
    parser.add_argument("--coordination-min-structure-score", type=float, default=0.0, help="Minimum structure score when --coordination-selection=structure_score.")
    parser.add_argument("--examples-per-motif", type=int, default=8)
    parser.add_argument("--write-heavy-corpora", action="store_true", help="Write full event/packet corpora; summaries and vocab are always written.")
    args = parser.parse_args()

    if args.self_test:
        result = MotionBpeSelfTest().run()
        print(json.dumps(result, ensure_ascii=True, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
        return

    output_dir = Path(args.output_dir)
    result = run_multichannel_motion_bpe(args)
    paths = write_multichannel_motion_bpe_outputs(output_dir, result, args)
    print(paths["summary"])
    print(paths["audit_report"])


if __name__ == "__main__":
    main()
