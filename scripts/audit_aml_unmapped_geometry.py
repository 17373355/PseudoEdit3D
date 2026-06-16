from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_policy_module():
    path = ROOT_DIR / "pseudoedit3d" / "edit" / "unmapped_geometry_policy.py"
    spec = importlib.util.spec_from_file_location("unmapped_geometry_policy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["unmapped_geometry_policy"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_policy = _load_policy_module()
classify_unlinked_geometry_record = _policy.classify_unlinked_geometry_record
geometry_record_duration = _policy.geometry_record_duration
geometry_record_magnitude = _policy.geometry_record_magnitude
geometry_record_span = _policy.geometry_record_span
unmapped_geometry_disposition = _policy.unmapped_geometry_disposition


def _geometry_cluster(event: dict[str, Any]) -> str:
    return str(event.get("geometry_cluster_id") or f"{event.get('super_family')}/{event.get('cluster_id')}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def audit_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audited_events: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    case_counts: Counter[str] = Counter()

    for record in records:
        case_id = str(record.get("case_id") or "")
        events = list((record.get("geometry_signature") or {}).get("events") or [])
        linked_events = [event for event in events if event.get("aml_links")]
        for event in events:
            if event.get("aml_links"):
                continue
            policy = classify_unlinked_geometry_record(event, linked_events)
            classification = str(policy["classification"])
            disposition = str(policy["recommended_disposition"])
            cluster_id = _geometry_cluster(event)
            item = {
                "case_id": case_id,
                "event_index": event.get("event_index"),
                "geometry_cluster_id": cluster_id,
                "super_family": event.get("super_family"),
                "cluster_id": event.get("cluster_id"),
                "span": list(geometry_record_span(event)),
                "duration": geometry_record_duration(event),
                "direction": event.get("direction"),
                "magnitude": round(geometry_record_magnitude(event), 4),
                "classification": classification,
                "recommended_disposition": disposition,
                "reason": policy["reason"],
                "nearest_linked_event": policy["nearest_linked_event"],
            }
            audited_events.append(item)
            classification_counts[classification] += 1
            disposition_counts[disposition] += 1
            cluster_counts[cluster_id][classification] += 1
            case_counts[case_id] += 1

    cluster_summary = [
        {
            "geometry_cluster_id": cluster_id,
            "unmapped_event_count": int(sum(counter.values())),
            "classification_counts": counter.most_common(),
            "dominant_classification": counter.most_common(1)[0][0],
            "dominant_disposition": unmapped_geometry_disposition(counter.most_common(1)[0][0]),
        }
        for cluster_id, counter in cluster_counts.items()
    ]
    summary = {
        "schema_version": "aml_unmapped_geometry_audit_v1",
        "num_cases": len(records),
        "unmapped_event_count": len(audited_events),
        "classification_counts": classification_counts.most_common(),
        "disposition_counts": disposition_counts.most_common(),
        "cluster_summary": sorted(
            cluster_summary,
            key=lambda item: (-int(item["unmapped_event_count"]), str(item["geometry_cluster_id"])),
        ),
        "case_summary": case_counts.most_common(),
    }
    return audited_events, summary


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(out)


def _write_markdown(events: list[dict[str, Any]], summary: dict[str, Any], output: Path, top_n: int) -> None:
    class_rows = [[name, count] for name, count in summary.get("classification_counts") or []]
    disposition_rows = [[name, count] for name, count in summary.get("disposition_counts") or []]
    cluster_rows = [
        [
            item["geometry_cluster_id"],
            item["unmapped_event_count"],
            item["dominant_classification"],
            item["dominant_disposition"],
            ", ".join(f"{name}:{count}" for name, count in item.get("classification_counts") or []),
        ]
        for item in (summary.get("cluster_summary") or [])[:top_n]
    ]
    event_rows = [
        [
            item["case_id"],
            item["event_index"],
            item["geometry_cluster_id"],
            item["span"],
            item["magnitude"],
            item["classification"],
            item["recommended_disposition"],
            (item.get("nearest_linked_event") or {}).get("geometry_cluster_id", ""),
            (item.get("nearest_linked_event") or {}).get("direct_family_ids", []),
        ]
        for item in events[:top_n]
    ]
    lines = [
        "# AML Unmapped Geometry Audit",
        "",
        f"- cases: {summary.get('num_cases')}",
        f"- unmapped events: {summary.get('unmapped_event_count')}",
        "",
        "## Classification Counts",
        "",
        _table(["classification", "count"], class_rows),
        "",
        "## Recommended Dispositions",
        "",
        _table(["disposition", "count"], disposition_rows),
        "",
        "## Cluster Summary",
        "",
        _table(["geometry_cluster", "unmapped", "dominant_class", "dominant_disposition", "class_counts"], cluster_rows),
        "",
        "## Event Examples",
        "",
        _table(
            [
                "case",
                "event",
                "geometry_cluster",
                "span",
                "magnitude",
                "classification",
                "disposition",
                "nearest_cluster",
                "nearest_families",
            ],
            event_rows,
        ),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-jsonl", required=True)
    parser.add_argument("--output-events-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    records = _load_records(Path(args.geometry_jsonl))
    events, summary = audit_records(records)

    events_path = Path(args.output_events_jsonl)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")

    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(events, summary, md_path, args.top_n)

    print(f"saved_events_jsonl={events_path}")
    print(f"saved_summary_json={summary_path}")
    print(f"saved_md={md_path}")
    print(f"unmapped_event_count={summary['unmapped_event_count']}")
    print(f"classification_counts={summary['classification_counts']}")
    print(f"disposition_counts={summary['disposition_counts']}")


if __name__ == "__main__":
    main()
