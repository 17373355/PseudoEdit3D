from __future__ import annotations

from typing import Any


def _event_sort_key(evt: dict[str, Any]) -> tuple[int, int, str, str, int]:
    return (
        int(evt.get("start_frame", -1)),
        int(evt.get("end_frame", -1)),
        str(evt.get("super_family", "")),
        str(evt.get("cluster_id", "")),
        int(evt.get("event_index", -1)),
    )


def _span(evt: dict[str, Any]) -> tuple[int, int]:
    return int(evt.get("start_frame", -1)), int(evt.get("end_frame", -1))


def _duration(evt: dict[str, Any]) -> int:
    s, e = _span(evt)
    return max(0, e - s + 1)


def _magnitude(evt: dict[str, Any] | None) -> float:
    if not evt:
        return 0.0
    value = evt.get("magnitude")
    if value is None:
        value = evt.get("signed_delta")
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _mean_magnitude(events: list[dict[str, Any]]) -> float:
    values = [_magnitude(evt) for evt in events if _magnitude(evt) > 0.0]
    return sum(values) / max(1, len(values))


def _overlap_frames(a: dict[str, Any], b: dict[str, Any]) -> int:
    s1, e1 = _span(a)
    s2, e2 = _span(b)
    return max(0, min(e1, e2) - max(s1, s2) + 1)


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = _overlap_frames(a, b)
    if inter <= 0:
        return 0.0
    return inter / max(1, min(_duration(a), _duration(b)))


def _gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    s1, e1 = _span(a)
    s2, e2 = _span(b)
    if e1 < s2:
        return s2 - e1
    if e2 < s1:
        return s1 - e2
    return 0


def _is_after(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a.get("start_frame", -1)) > int(b.get("end_frame", -1))


def _indexed_events(program_or_events: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if isinstance(program_or_events, dict):
        raw_events = list(program_or_events.get("events") or [])
    else:
        raw_events = list(program_or_events or [])
    out: list[dict[str, Any]] = []
    for idx, evt in enumerate(raw_events):
        copied = dict(evt)
        copied["event_index"] = int(copied.get("event_index", idx))
        out.append(copied)
    return sorted(out, key=_event_sort_key)


def _total_frames(events: list[dict[str, Any]], total_frames: int | None) -> int:
    if total_frames:
        return int(total_frames)
    return max((int(evt.get("end_frame", 0)) for evt in events), default=0)


def _coverage(events: list[dict[str, Any]], total_frames: int) -> float:
    if not events or total_frames <= 0:
        return 0.0
    spans = sorted((_span(evt) for evt in events if _duration(evt) > 0))
    if not spans:
        return 0.0
    merged: list[list[int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    frames = sum(e - s + 1 for s, e in merged)
    return frames / max(1, total_frames)


def _count_peaks(events: list[dict[str, Any]], *, min_gap: int = 6) -> int:
    starts = sorted(int(evt.get("start_frame", -1)) for evt in events if int(evt.get("start_frame", -1)) >= 0)
    if not starts:
        return 0
    picked = [starts[0]]
    for start in starts[1:]:
        if start - picked[-1] >= min_gap:
            picked.append(start)
    return len(picked)


def _speed_from_event(evt: dict[str, Any] | None) -> str:
    if not evt:
        return "unknown"
    cluster = str(evt.get("cluster_id", ""))
    if "FAST" in cluster:
        return "fast"
    if "SLOW" in cluster:
        return "slow"
    if "MEDIUM" in cluster:
        return "medium"
    meta = evt.get("metadata") or {}
    try:
        mean_speed = float(meta.get("mean_speed"))
    except (TypeError, ValueError):
        return "unknown"
    if mean_speed >= 0.04:
        return "fast"
    if mean_speed <= 0.022:
        return "slow"
    return "medium"


def _event_indices_for_families(events: list[dict[str, Any]], families: set[str]) -> list[int]:
    return [
        int(evt["event_index"])
        for evt in events
        if str(evt.get("super_family", "")) in families
    ]


def _indices_by_family(events: list[dict[str, Any]], family: str) -> set[int]:
    return {int(evt["event_index"]) for evt in events if evt.get("super_family") == family}


def _event_by_index(events: list[dict[str, Any]], idx: int | None) -> dict[str, Any] | None:
    if idx is None:
        return None
    for evt in events:
        if int(evt.get("event_index", -1)) == int(idx):
            return evt
    return None


def _event_ref(evt: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_index": int(evt.get("event_index", -1)),
        "family": str(evt.get("super_family", "")),
        "cluster": str(evt.get("cluster_id", "")),
        "direction": str(evt.get("direction", "")),
        "span": list(_span(evt)),
        "magnitude": evt.get("magnitude"),
        "count": evt.get("count"),
    }


def _event_family_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evt in events:
        key = str(evt.get("super_family", ""))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _event_cluster_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evt in events:
        family = str(evt.get("super_family", ""))
        cluster = str(evt.get("cluster_id", ""))
        if not family or not cluster:
            continue
        key = f"{family}/{cluster}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
