from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .coarse_event_utils import _duration, _event_sort_key, _magnitude, _span
from .unmapped_geometry_policy import classify_unlinked_geometry_record


UNMAPPED_FAMILY_ID = "__AML_UNMAPPED_GEOMETRY__"
CONTEXT_ONLY_FAMILY_ID = "__AML_CONTEXT_ONLY_GEOMETRY__"
CONTEXT_FRAGMENT_FAMILY_ID = "__AML_CONTEXT_FRAGMENT_GEOMETRY__"
IGNORED_NOISE_FAMILY_ID = "__AML_IGNORED_NOISE_GEOMETRY__"
PSEUDO_FAMILY_IDS = frozenset(
    {
        UNMAPPED_FAMILY_ID,
        CONTEXT_ONLY_FAMILY_ID,
        CONTEXT_FRAGMENT_FAMILY_ID,
        IGNORED_NOISE_FAMILY_ID,
    }
)


def geometry_cluster_id(event: dict[str, Any]) -> str:
    family = str(event.get("super_family", "UNKNOWN_FAMILY"))
    cluster = str(event.get("cluster_id", "UNKNOWN_CLUSTER"))
    return f"{family}/{cluster}"


def _layer3_events(layer3: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = layer3.get("events") if isinstance(layer3, dict) else layer3
    out: list[dict[str, Any]] = []
    for idx, event in enumerate(events or []):
        copied = dict(event)
        copied["event_index"] = int(copied.get("event_index", idx))
        out.append(copied)
    return sorted(out, key=_event_sort_key)


def _action_links_by_event(coarse_action_program: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    links: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action_index, action in enumerate(coarse_action_program.get("canonical_actions") or []):
        family = dict(action.get("semantic_family") or {})
        slots = dict(action.get("slots") or {})
        covered = slots.get("covered_event_indices") or action.get("covered_event_indices") or []
        source = set(int(idx) for idx in (slots.get("source_event_indices") or action.get("source_event_indices") or []) if idx is not None)
        for raw_idx in covered:
            try:
                event_index = int(raw_idx)
            except (TypeError, ValueError):
                continue
            links[event_index].append(
                {
                    "action_index": int(action_index),
                    "canonical_id": str(action.get("canonical_id") or family.get("family_id") or "UNKNOWN"),
                    "family_id": str(family.get("family_id") or action.get("canonical_id") or "UNKNOWN"),
                    "status": str(family.get("status") or slots.get("semantic_family_status") or "unknown"),
                    "taxonomy_parent_id": family.get("taxonomy_parent_id"),
                    "probe_visible": family.get("probe_visible", True) is not False,
                    "probe_alias": action.get("probe_alias"),
                    "directness": "source" if event_index in source or not source else "covered_context",
                }
            )
    return dict(links)


def _motion_signature(event: dict[str, Any]) -> dict[str, Any]:
    signature = dict(event.get("motion_signature") or {})
    keys = (
        "dominant_axis",
        "repeat_mode",
        "phase_template",
        "contact_mode",
        "support_mode",
        "bilateral_symmetry",
        "alternation",
        "tempo_bucket",
        "coupled_with_locomotion",
        "context_mode",
    )
    return {key: signature.get(key) for key in keys if key in signature}


def _geometry_event_record(event: dict[str, Any], links: list[dict[str, Any]]) -> dict[str, Any]:
    record = {
        "event_index": int(event["event_index"]),
        "geometry_cluster_id": geometry_cluster_id(event),
        "super_family": str(event.get("super_family", "")),
        "cluster_id": str(event.get("cluster_id", "")),
        "part": str(event.get("part", "")),
        "span": list(_span(event)),
        "duration": _duration(event),
        "direction": str(event.get("direction", "")),
        "role": str(event.get("role", "")),
        "confidence": event.get("confidence"),
        "magnitude": round(_magnitude(event), 4),
        "unit": event.get("unit"),
        "count": event.get("count"),
        "motion_signature": _motion_signature(event),
        "aml_links": links,
        "named_by_aml": bool(links),
    }
    optional_name = event.get("optional_semantic_name")
    if optional_name:
        record["layer3_optional_semantic_name"] = str(optional_name)
    return record


def _attach_unmapped_policies(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked_events = [event for event in events if event.get("aml_links")]
    out: list[dict[str, Any]] = []
    for event in events:
        copied = dict(event)
        if not copied.get("aml_links"):
            copied["unmapped_policy"] = classify_unlinked_geometry_record(copied, linked_events)
        out.append(copied)
    return out


def _unmapped_disposition(event: dict[str, Any]) -> str | None:
    policy = event.get("unmapped_policy")
    if not isinstance(policy, dict):
        return None
    return str(policy.get("recommended_disposition") or "")


def _unmapped_classification(event: dict[str, Any]) -> str | None:
    policy = event.get("unmapped_policy")
    if not isinstance(policy, dict):
        return None
    return str(policy.get("classification") or "")


def _event_link_profile(event: dict[str, Any]) -> dict[str, Any]:
    links = list(event.get("aml_links") or [])
    direct_links = [link for link in links if str(link.get("directness")) == "source"]
    context_links = [link for link in links if str(link.get("directness")) != "source"]
    if not links:
        classification = _unmapped_classification(event)
        disposition = _unmapped_disposition(event)
        if disposition == "ignore_noise":
            return {
                "bucket": "ignored_noise",
                "pseudo_family_id": IGNORED_NOISE_FAMILY_ID,
                "classification": classification,
                "links": links,
                "direct_links": direct_links,
                "context_links": context_links,
            }
        if disposition == "mark_context_only":
            return {
                "bucket": "context_fragment",
                "pseudo_family_id": CONTEXT_FRAGMENT_FAMILY_ID,
                "classification": classification,
                "links": links,
                "direct_links": direct_links,
                "context_links": context_links,
            }
        return {
            "bucket": "unresolved_unmapped",
            "pseudo_family_id": UNMAPPED_FAMILY_ID,
            "classification": classification,
            "links": links,
            "direct_links": direct_links,
            "context_links": context_links,
        }
    if not direct_links:
        return {
            "bucket": "context_only",
            "pseudo_family_id": CONTEXT_ONLY_FAMILY_ID,
            "classification": None,
            "links": links,
            "direct_links": direct_links,
            "context_links": context_links,
        }
    return {
        "bucket": "direct",
        "pseudo_family_id": None,
        "classification": None,
        "links": links,
        "direct_links": direct_links,
        "context_links": context_links,
    }


def _semantic_family_counter(family_counter: Counter[str]) -> Counter[str]:
    return Counter({family: count for family, count in family_counter.items() if family not in PSEUDO_FAMILY_IDS})


def _pseudo_family_counter(family_counter: Counter[str]) -> Counter[str]:
    return Counter({family: count for family, count in family_counter.items() if family in PSEUDO_FAMILY_IDS})


def _cluster_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["geometry_cluster_id"])].append(event)
    summaries: list[dict[str, Any]] = []
    for cluster_id, group in grouped.items():
        family_counter: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()
        unlinked = 0
        context_only = 0
        context_fragment = 0
        ignored_noise = 0
        unresolved_unmapped = 0
        unmapped_class_counter: Counter[str] = Counter()
        for event in group:
            profile = _event_link_profile(event)
            bucket = str(profile["bucket"])
            classification = profile.get("classification")
            if classification:
                unmapped_class_counter[str(classification)] += 1
            if bucket != "direct":
                family_counter[str(profile["pseudo_family_id"])] += 1
            if bucket in {"ignored_noise", "context_fragment", "unresolved_unmapped"}:
                unlinked += 1
                if bucket == "ignored_noise":
                    ignored_noise += 1
                elif bucket == "context_fragment":
                    context_fragment += 1
                else:
                    unresolved_unmapped += 1
                continue
            if bucket == "context_only":
                context_only += 1
                continue
            for link in profile["direct_links"]:
                family_counter[str(link.get("family_id") or "UNKNOWN")] += 1
                status_counter[str(link.get("status") or "unknown")] += 1
        semantic_counter = _semantic_family_counter(family_counter)
        pseudo_counter = _pseudo_family_counter(family_counter)
        summaries.append(
            {
                "geometry_cluster_id": cluster_id,
                "event_count": len(group),
                "event_indices": [int(event["event_index"]) for event in group],
                "unlinked_event_count": int(unlinked),
                "unresolved_unmapped_event_count": int(unresolved_unmapped),
                "context_fragment_event_count": int(context_fragment),
                "ignored_noise_event_count": int(ignored_noise),
                "context_only_event_count": int(context_only),
                "unmapped_event_count": int(unresolved_unmapped),
                "unmapped_classification_counts": unmapped_class_counter.most_common(),
                "named_event_count": int(len(group) - unlinked),
                "direct_named_event_count": int(len(group) - unlinked - context_only),
                "family_counts": family_counter.most_common(),
                "semantic_family_counts": semantic_counter.most_common(),
                "pseudo_family_counts": pseudo_counter.most_common(),
                "status_counts": status_counter.most_common(),
                "stable_family_ids": sorted(
                    family
                    for family in semantic_counter
                    if any(
                        str(link.get("family_id")) == family and str(link.get("status")) == "stable"
                        for event in group
                        for link in (event.get("aml_links") or [])
                        if str(link.get("directness")) == "source"
                    )
                ),
            }
        )
    return sorted(summaries, key=lambda item: (-int(item["event_count"]), str(item["geometry_cluster_id"])))


def build_geometry_signature(
    layer3: dict[str, Any] | list[dict[str, Any]],
    coarse_action_program: dict[str, Any],
) -> dict[str, Any]:
    links_by_event = _action_links_by_event(coarse_action_program)
    events = [
        _geometry_event_record(event, links_by_event.get(int(event["event_index"]), []))
        for event in _layer3_events(layer3)
    ]
    events = _attach_unmapped_policies(events)
    cluster_ids = sorted({str(event["geometry_cluster_id"]) for event in events})
    return {
        "schema_version": "aml_geometry_signature_v2",
        "event_count": len(events),
        "cluster_count": len(cluster_ids),
        "cluster_ids": cluster_ids,
        "events": events,
        "cluster_summaries": _cluster_summaries(events),
    }


def _case_examples(items: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        case_id = str(item.get("case_id") or "")
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        out.append(case_id)
        if len(out) >= limit:
            break
    return out


def summarize_geometry_sidecars(records: list[dict[str, Any]], *, top_n: int = 40) -> dict[str, Any]:
    cluster_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_event_counts: Counter[str] = Counter()
    cluster_case_support: dict[str, set[str]] = defaultdict(set)
    cluster_unlinked_events: Counter[str] = Counter()
    cluster_unresolved_unmapped_events: Counter[str] = Counter()
    cluster_context_only_events: Counter[str] = Counter()
    cluster_context_fragment_events: Counter[str] = Counter()
    cluster_ignored_noise_events: Counter[str] = Counter()
    cluster_unmapped_classifications: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_unknown_links: Counter[str] = Counter()
    cluster_context_links: Counter[str] = Counter()
    cluster_context_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_family_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_events = 0

    for record in records:
        case_id = str(record.get("case_id") or "")
        signature = record.get("geometry_signature") or {}
        events = _attach_unmapped_policies(list(signature.get("events") or []))
        for event in events:
            cluster_id = str(event.get("geometry_cluster_id") or "UNKNOWN")
            total_events += 1
            cluster_event_counts[cluster_id] += 1
            cluster_case_support[cluster_id].add(case_id)
            cluster_examples[cluster_id].append({"case_id": case_id, "event_index": event.get("event_index")})
            profile = _event_link_profile(event)
            cluster_context_links[cluster_id] += len(profile["context_links"])
            for link in profile["context_links"]:
                family_id = str(link.get("family_id") or "UNKNOWN")
                cluster_context_family_counts[cluster_id][family_id] += 1
                context_family_cluster_counts[family_id][cluster_id] += 1
            classification = profile.get("classification")
            if classification:
                cluster_unmapped_classifications[cluster_id][str(classification)] += 1
            bucket = str(profile["bucket"])
            if bucket != "direct":
                cluster_family_counts[cluster_id][str(profile["pseudo_family_id"])] += 1
            if bucket in {"ignored_noise", "context_fragment", "unresolved_unmapped"}:
                cluster_unlinked_events[cluster_id] += 1
                if bucket == "ignored_noise":
                    cluster_ignored_noise_events[cluster_id] += 1
                elif bucket == "context_fragment":
                    cluster_context_fragment_events[cluster_id] += 1
                else:
                    cluster_unresolved_unmapped_events[cluster_id] += 1
                continue
            if bucket == "context_only":
                cluster_context_only_events[cluster_id] += 1
                continue
            for link in profile["direct_links"]:
                family_id = str(link.get("family_id") or "UNKNOWN")
                status = str(link.get("status") or "unknown")
                cluster_family_counts[cluster_id][family_id] += 1
                cluster_status_counts[cluster_id][status] += 1
                family_cluster_counts[family_id][cluster_id] += 1
                if status == "unknown":
                    cluster_unknown_links[cluster_id] += 1

    stable_geometry_cluster_mappings: list[dict[str, Any]] = []
    one_to_many_geometry_clusters: list[dict[str, Any]] = []
    unmapped_geometry_clusters: list[dict[str, Any]] = []
    context_only_geometry_clusters: list[dict[str, Any]] = []
    context_fragment_geometry_clusters: list[dict[str, Any]] = []
    ignored_noise_geometry_clusters: list[dict[str, Any]] = []
    aml_unable_to_name_clusters: list[dict[str, Any]] = []
    for cluster_id in cluster_event_counts:
        family_counter = cluster_family_counts.get(cluster_id, Counter())
        semantic_counter = _semantic_family_counter(family_counter)
        pseudo_counter = _pseudo_family_counter(family_counter)
        named_families = list(semantic_counter)
        status_counter = cluster_status_counts.get(cluster_id, Counter())
        event_count = int(cluster_event_counts[cluster_id])
        support = len(cluster_case_support[cluster_id])
        unlinked_count = int(cluster_unlinked_events[cluster_id])
        unresolved_unmapped_count = int(cluster_unresolved_unmapped_events[cluster_id])
        context_only_count = int(cluster_context_only_events[cluster_id])
        context_fragment_count = int(cluster_context_fragment_events[cluster_id])
        ignored_noise_count = int(cluster_ignored_noise_events[cluster_id])
        unknown_count = int(cluster_unknown_links[cluster_id])
        base = {
            "geometry_cluster_id": cluster_id,
            "event_count": event_count,
            "case_support": support,
            "family_counts": family_counter.most_common(),
            "semantic_family_counts": semantic_counter.most_common(),
            "pseudo_family_counts": pseudo_counter.most_common(),
            "status_counts": status_counter.most_common(),
            "covered_context_link_count": int(cluster_context_links[cluster_id]),
            "context_family_counts": cluster_context_family_counts[cluster_id].most_common(),
            "unmapped_classification_counts": cluster_unmapped_classifications[cluster_id].most_common(),
            "example_case_ids": _case_examples(cluster_examples[cluster_id]),
        }
        if unresolved_unmapped_count:
            unmapped_geometry_clusters.append(
                {
                    **base,
                    "unlinked_event_count": unlinked_count,
                    "unresolved_unmapped_event_count": unresolved_unmapped_count,
                    "unmapped_event_count": unresolved_unmapped_count,
                    "unmapped_share": round(unresolved_unmapped_count / max(1, event_count), 4),
                }
            )
        if context_only_count:
            context_only_geometry_clusters.append(
                {
                    **base,
                    "context_only_event_count": context_only_count,
                    "context_only_share": round(context_only_count / max(1, event_count), 4),
                }
            )
        if context_fragment_count:
            context_fragment_geometry_clusters.append(
                {
                    **base,
                    "unlinked_event_count": unlinked_count,
                    "context_fragment_event_count": context_fragment_count,
                    "context_fragment_share": round(context_fragment_count / max(1, event_count), 4),
                }
            )
        if ignored_noise_count:
            ignored_noise_geometry_clusters.append(
                {
                    **base,
                    "unlinked_event_count": unlinked_count,
                    "ignored_noise_event_count": ignored_noise_count,
                    "ignored_noise_share": round(ignored_noise_count / max(1, event_count), 4),
                }
            )
        if unresolved_unmapped_count or unknown_count:
            aml_unable_to_name_clusters.append(
                {
                    **base,
                    "unlinked_event_count": unlinked_count,
                    "unresolved_unmapped_event_count": unresolved_unmapped_count,
                    "unmapped_event_count": unresolved_unmapped_count,
                    "unknown_link_count": unknown_count,
                    "unable_to_name_count": unresolved_unmapped_count + unknown_count,
                    "unable_to_name_share": round((unresolved_unmapped_count + unknown_count) / max(1, event_count), 4),
                }
            )
        if len(named_families) > 1:
            one_to_many_geometry_clusters.append(base)
        if (
            len(named_families) == 1
            and not unlinked_count
            and not context_only_count
            and not context_fragment_count
            and not ignored_noise_count
            and status_counter
            and set(status_counter) == {"stable"}
        ):
            stable_geometry_cluster_mappings.append(
                {
                    **base,
                    "stable_family_id": named_families[0],
                }
            )

    multi_cluster_semantic_families = [
        {
            "family_id": family_id,
            "cluster_count": len(counter),
            "total_links": int(sum(counter.values())),
            "cluster_counts": counter.most_common(top_n),
        }
        for family_id, counter in family_cluster_counts.items()
        if len(counter) > 1
    ]
    context_multi_cluster_semantic_families = [
        {
            "family_id": family_id,
            "cluster_count": len(counter),
            "total_context_links": int(sum(counter.values())),
            "cluster_counts": counter.most_common(top_n),
        }
        for family_id, counter in context_family_cluster_counts.items()
        if len(counter) > 1
    ]

    return {
        "schema_version": "aml_geometry_sidecar_summary_v3",
        "num_cases": len(records),
        "total_geometry_events": int(total_events),
        "geometry_cluster_counts": cluster_event_counts.most_common(top_n),
        "geometry_cluster_case_support": [
            [cluster_id, len(cases)]
            for cluster_id, cases in sorted(
                cluster_case_support.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )[:top_n]
        ],
        "stable_geometry_cluster_mappings": sorted(
            stable_geometry_cluster_mappings,
            key=lambda item: (-int(item["event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "one_to_many_geometry_clusters": sorted(
            one_to_many_geometry_clusters,
            key=lambda item: (-len(item["family_counts"]), -int(item["event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "multi_cluster_semantic_families": sorted(
            multi_cluster_semantic_families,
            key=lambda item: (-int(item["cluster_count"]), -int(item["total_links"]), str(item["family_id"])),
        )[:top_n],
        "covered_context_multi_cluster_semantic_families": sorted(
            context_multi_cluster_semantic_families,
            key=lambda item: (-int(item["cluster_count"]), -int(item["total_context_links"]), str(item["family_id"])),
        )[:top_n],
        "unmapped_geometry_clusters": sorted(
            unmapped_geometry_clusters,
            key=lambda item: (-float(item["unmapped_share"]), -int(item["unmapped_event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "context_only_geometry_clusters": sorted(
            context_only_geometry_clusters,
            key=lambda item: (-float(item["context_only_share"]), -int(item["context_only_event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "context_fragment_geometry_clusters": sorted(
            context_fragment_geometry_clusters,
            key=lambda item: (-float(item["context_fragment_share"]), -int(item["context_fragment_event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "ignored_noise_geometry_clusters": sorted(
            ignored_noise_geometry_clusters,
            key=lambda item: (-float(item["ignored_noise_share"]), -int(item["ignored_noise_event_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
        "aml_unable_to_name_clusters": sorted(
            aml_unable_to_name_clusters,
            key=lambda item: (-float(item["unable_to_name_share"]), -int(item["unable_to_name_count"]), str(item["geometry_cluster_id"])),
        )[:top_n],
    }
