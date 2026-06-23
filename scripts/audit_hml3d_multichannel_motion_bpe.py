"""Audit HumanML3D multi-channel Motion-BPE.

What this script does:
1. Read the Layer3 event corpus.
2. Convert each case into channel events, overlap relations, and packets.
3. Learn per-channel temporal motifs first.
4. Promote frequently overlapping cross-channel motifs into coordination motifs.
5. Write motif/family/forest artifacts for manual inspection.

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
- Changing source/max-records/overlap/gap/relation flags rebuilds the cache.
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


DEFAULT_SOURCE_CORPUS = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1")
CACHE_SCHEMA_VERSION = "hml3d_multichannel_record_cache_v1"

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
    return {
        "source_signature": _source_signature(Path(args.source_corpus)),
        "max_records": args.max_records,
        "parallel_overlap_min": float(args.parallel_overlap_min),
        "lead_lag_gap_max": int(args.lead_lag_gap_max),
    }


def _cache_matches(metadata: dict[str, Any], args: argparse.Namespace) -> bool:
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    return metadata.get("config") == _cache_config(args)


def _cache_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    channel_event_count = sum(len(record.get("channel_events") or []) for record in records)
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
    records = [build_multichannel_record(record, args) for record in source_records]
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
    if super_family in {"WHOLE_BODY_POSTURE", "WHOLE_BODY_STATE"}:
        return "whole_body_state"
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


def build_channel_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_id = str(record.get("case_id") or "")
    for local_idx, event in enumerate(record.get("events") or []):
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
        "relation_types": [str(packet.get("packet_type") or "")],
    }


def build_multichannel_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    events = build_channel_events(record)
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
        clusters = sorted(str(cluster) for cluster in (unit.get("geometry_clusters") or []) if cluster)
        cluster_key = "&".join(clusters[:4]) if clusters else "unknown_geometry"
        parts.append(f"{channel}:{cluster_key}")
    return "COORD_SIG[" + "+".join(parts) + "]"


def _build_coactivation_units(
    channel_sequences: dict[str, list[dict[str, Any]]],
    *,
    parallel_overlap_min: float,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for case_id, units in _channel_units_by_case(channel_sequences).items():
        candidates = [unit for unit in units if str(unit.get("symbol") or "").startswith("<CHM_")]
        parent: dict[int, int] = {idx: idx for idx in range(len(candidates))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, left in enumerate(candidates):
            left_channels = set(left.get("channels") or [])
            for j in range(i + 1, len(candidates)):
                right = candidates[j]
                if left_channels & set(right.get("channels") or []):
                    continue
                if _overlap_ratio(left, right) >= parallel_overlap_min:
                    union(i, j)

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for idx, unit in enumerate(candidates):
            groups[find(idx)].append(unit)

        coacts: list[dict[str, Any]] = []
        for idx, members in enumerate(groups.values(), start=1):
            channels = sorted({channel for unit in members for channel in (unit.get("channels") or [])}, key=lambda ch: CHANNEL_RANK.get(ch, 999))
            if len(channels) < 2:
                continue
            span = [min(int((unit.get("span") or [0, 0])[0]) for unit in members), max(int((unit.get("span") or [0, 0])[1]) for unit in members)]
            member_symbol = _coactivation_symbol(members)
            signature = _coactivation_signature(members)
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
                    "relation_types": ["coactivation"],
                    "member_symbols": [str(unit.get("symbol") or "") for unit in members],
                    "member_coactivation_symbol": member_symbol,
                    "coactivation_id": f"{case_id}:coact{idx:04d}",
                }
            )
        coacts.sort(key=lambda unit: (int(unit["span"][0]), int(unit["span"][1]), str(unit["symbol"])))
        if len(coacts) >= 1:
            out[f"{case_id}::coactivation"] = coacts
    return out


def _coactivation_stats(sequences: dict[str, list[dict[str, Any]]]) -> tuple[Counter[str], dict[str, set[str]], dict[str, list[dict[str, Any]]]]:
    counts: Counter[str] = Counter()
    cases: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence_id, seq in sequences.items():
        case_id = sequence_id.split("::", 1)[0]
        for unit in seq:
            symbol = str(unit.get("symbol") or "")
            counts[symbol] += 1
            cases[symbol].add(case_id)
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
    return counts, cases, examples


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
    counts, cases, examples = _coactivation_stats(coactivation_sequences)
    candidates = [
        symbol
        for symbol, count in counts.items()
        if count >= int(args.min_pair_count) and len(cases[symbol]) >= int(args.min_pair_support)
    ]
    candidates.sort(key=lambda symbol: (-len(cases[symbol]), -counts[symbol], symbol))
    selected = candidates[:budget]
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
        alias_counter: Counter[str] = Counter()
        base_counter: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        alias_seen_cases: set[str] = set()
        for occ in motif_occs:
            case_id = str(occ["case_id"])
            channel_counter.update(str(item) for item in occ.get("channels") or [])
            geometry_counter.update(str(item) for item in occ.get("geometry_clusters") or [])
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
        rows.append(
            {
                "motif_id": motif_id,
                "step": int(merge.get("step") or 0),
                "operator": merge.get("operator"),
                "parents": merge.get("parents") or [],
                "occurrences": len(motif_occs),
                "support_cases": len(support_cases),
                "channels": _top_counter(channel_counter, 12),
                "relation_profile": _top_counter(relation_counter, 8),
                "top_geometry_clusters": _top_counter(geometry_counter, 12),
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


def build_motif_families(motif_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_meta: dict[str, dict[str, Any]] = {}
    for motif in motif_rows:
        required_clusters = _required_ids(motif.get("top_geometry_clusters") or [])
        required_channels = _required_ids(motif.get("channels") or [], max_items=6, min_relative_count=0.35)
        relation_types = _required_ids(motif.get("relation_profile") or [], max_items=3, min_relative_count=0.35)
        operator = str(motif.get("operator") or "")
        key = "|".join(
            [
                f"op={operator}",
                "channels=" + "+".join(required_channels or ["none"]),
                "relations=" + "+".join(relation_types or ["none"]),
                "clusters=" + "+".join(required_clusters or ["none"]),
            ]
        )
        grouped[key].append(motif)
        group_meta[key] = {
            "operator": operator,
            "required_channels": required_channels,
            "required_relation_types": relation_types,
            "required_geometry_clusters": required_clusters,
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
        families.append(
            {
                "family_id": f"multichannel_motif_family_{idx:04d}",
                "schema_version": "multichannel_motif_family_candidate_v1",
                "motion_family_key": key,
                "status": "candidate_family" if support_sum >= 80 else "diagnostic_family",
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
        "version": "hml3d_multichannel_motion_bpe_v1",
        "source_corpus": str(args.source_corpus),
        "source_record_count": source_records,
        "num_records": len(records),
        "channel_event_count": channel_event_count,
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
            "examples_per_motif": 4,
            "write_heavy_corpora": False,
            "cache_dir": None,
            "rebuild_cache": False,
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
    lines.append("| family | status | support sum | motifs | required channels | required relations | required geometry | top aliases |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- |")
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
            f"| `{family['family_id']}` | `{family.get('status')}` | {family.get('support_cases_sum')} | {family.get('motif_count')} | {channels} | {relations} | {geometry} | {aliases} |"
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
    _write_json(output_dir / "multichannel_motion_bpe_vocab.json", {"version": "multichannel_motion_bpe_vocab_v1", "merges": merges})
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
    parser.add_argument("--num-merges", type=int, default=256)
    parser.add_argument("--channel-merge-ratio", type=float, default=0.5, help="Fraction of merge budget used for per-channel sequential motifs before cross-channel coordination mining.")
    parser.add_argument("--min-pair-count", type=int, default=80)
    parser.add_argument("--min-pair-support", type=int, default=40)
    parser.add_argument("--selection", choices=["count", "support"], default="count")
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
