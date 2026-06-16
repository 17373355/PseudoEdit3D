from __future__ import annotations

from typing import Any


def geometry_record_span(event: dict[str, Any]) -> tuple[int, int]:
    span = event.get("span")
    if isinstance(span, list) and len(span) == 2:
        return int(span[0]), int(span[1])
    return int(event.get("start_frame", -1)), int(event.get("end_frame", -1))


def geometry_record_duration(event: dict[str, Any]) -> int:
    if event.get("duration") is not None:
        return max(0, int(event["duration"]))
    start, end = geometry_record_span(event)
    return max(0, end - start + 1)


def geometry_record_magnitude(event: dict[str, Any]) -> float:
    try:
        return abs(float(event.get("magnitude") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def geometry_record_gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a_start, a_end = geometry_record_span(a)
    b_start, b_end = geometry_record_span(b)
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def geometry_record_overlap_frames(a: dict[str, Any], b: dict[str, Any]) -> int:
    a_start, a_end = geometry_record_span(a)
    b_start, b_end = geometry_record_span(b)
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def geometry_record_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = geometry_record_overlap_frames(a, b)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(geometry_record_duration(a), geometry_record_duration(b)))


def direct_family_ids(event: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for link in event.get("aml_links") or []:
        if str(link.get("directness")) != "source":
            continue
        family_id = str(link.get("family_id") or "")
        if family_id:
            out.append(family_id)
    return sorted(set(out))


def context_family_ids(event: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for link in event.get("aml_links") or []:
        if str(link.get("directness")) == "source":
            continue
        family_id = str(link.get("family_id") or "")
        if family_id:
            out.append(family_id)
    return sorted(set(out))


def nearest_linked_geometry_event(
    event: dict[str, Any],
    linked_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not linked_events:
        return None
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for other in linked_events:
        overlap = geometry_record_overlap_ratio(event, other)
        gap = geometry_record_gap(event, other)
        ranked.append((overlap, -gap, other))
    overlap, neg_gap, nearest = max(ranked, key=lambda item: (item[0], item[1]))
    return {
        "event_index": nearest.get("event_index"),
        "geometry_cluster_id": str(
            nearest.get("geometry_cluster_id")
            or f"{nearest.get('super_family')}/{nearest.get('cluster_id')}"
        ),
        "span": list(geometry_record_span(nearest)),
        "gap": int(-neg_gap),
        "overlap_ratio": round(overlap, 4),
        "direct_family_ids": direct_family_ids(nearest),
        "context_family_ids": context_family_ids(nearest),
    }


def unmapped_geometry_disposition(classification: str) -> str:
    if classification.endswith("_noise"):
        return "ignore_noise"
    if classification.endswith("_context_fragment"):
        return "mark_context_only"
    if classification.endswith("_coverage_gap"):
        return "fix_coverage_or_tree"
    if classification.endswith("_candidate"):
        return "review_before_new_proxy"
    return "review"


def classify_unmapped_geometry_event(
    event: dict[str, Any],
    nearest: dict[str, Any] | None,
) -> tuple[str, str]:
    super_family = str(event.get("super_family") or "")
    cluster_id = str(event.get("cluster_id") or "")
    direction = str(event.get("direction") or "")
    duration = geometry_record_duration(event)
    magnitude = geometry_record_magnitude(event)
    near_overlap = float((nearest or {}).get("overlap_ratio") or 0.0)
    near_gap = int((nearest or {}).get("gap") or 999999)
    near_tight = near_overlap >= 0.15 or near_gap <= 8
    near_loose = near_overlap >= 0.05 or near_gap <= 16

    if super_family == "WHOLE_BODY_LOCOMOTION":
        if cluster_id.startswith("LOCO_TURN_"):
            if near_loose:
                return (
                    "turn_context_fragment",
                    "turn-like root segment is temporally close to an already named AML action",
                )
            return (
                "turn_sparse_coverage_gap",
                "salient root-yaw segment has no AML link and should be checked against sparse turn coverage",
            )
        if direction in {"active", "mixed"} and magnitude < 0.30 and duration <= 16:
            return (
                "root_jitter_noise",
                "short low-distance active/mixed root motion is more likely segmentation drift than a nameable action",
            )
        if direction in {"active", "mixed"} and magnitude < 0.35 and duration <= 18:
            return (
                "root_jitter_noise",
                "short ambiguous active/mixed root motion stays below residual gait distance threshold",
            )
        if near_tight:
            return (
                "root_context_fragment",
                "root drift is close to named action evidence and should remain context rather than a separate action",
            )
        if magnitude >= 0.35:
            return (
                "root_sparse_coverage_gap",
                "directed residual root motion exceeds sparse gait threshold but has no AML link",
            )
        return (
            "weak_root_drift_candidate",
            "directed but sub-threshold root drift may need a context/noise policy before adding a proxy",
        )

    if super_family == "WHOLE_BODY_VERTICAL":
        if magnitude < 0.08 and duration <= 8:
            return (
                "vertical_micro_noise",
                "very small vertical cycle is below semantic vertical-motion evidence scale",
            )
        if near_loose:
            return (
                "vertical_context_fragment",
                "vertical residual is close to a named action and is better treated as context",
            )
        return (
            "vertical_proxy_coverage_gap",
            "vertical event has a tree proxy but no AML link, so residual emission/coverage should be inspected",
        )

    if super_family == "WHOLE_BODY_POSTURE":
        if near_loose:
            return (
                "posture_context_fragment",
                "posture residual is near a named action and may be context-only",
            )
        return (
            "posture_proxy_coverage_gap",
            "posture event has a tree proxy but no AML link, so residual emission/coverage should be inspected",
        )

    return (
        "unclassified_unmapped_geometry",
        "unmapped geometry family is outside the current audit policy",
    )


def classify_unlinked_geometry_record(
    event: dict[str, Any],
    linked_events: list[dict[str, Any]],
) -> dict[str, Any]:
    nearest = nearest_linked_geometry_event(event, linked_events)
    classification, reason = classify_unmapped_geometry_event(event, nearest)
    return {
        "classification": classification,
        "recommended_disposition": unmapped_geometry_disposition(classification),
        "reason": reason,
        "nearest_linked_event": nearest,
    }
