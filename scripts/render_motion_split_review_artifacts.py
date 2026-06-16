from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUEUE = Path("outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue.json")
DEFAULT_OUTPUT_DIR = Path("outputs/aml_regression_testset_v2/motion_split_proposals_v1")
DEFAULT_HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")
DEFAULT_BPE_SEQUENCES = Path("outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _read_hml3d_captions(hml_root: Path, case_id: str) -> list[str]:
    path = hml_root / "texts" / f"{case_id}.txt"
    if not path.exists():
        return []
    captions: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        caption = line.split("#", 1)[0].strip()
        if caption and caption not in seen:
            captions.append(caption)
            seen.add(caption)
    return captions


def _read_bpe_caption_lookup(path: Path) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return lookup
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            caption = str(row.get("caption") or "").strip()
            if case_id and caption and caption not in seen[case_id]:
                lookup[case_id].append(caption)
                seen[case_id].add(caption)
    return lookup


def _caption_lookup_for_examples(
    queue: dict[str, Any],
    *,
    hml_root: Path,
    bpe_sequences: Path,
) -> dict[str, list[str]]:
    case_ids: set[str] = set()
    for item in queue.get("queue") or []:
        for ref in item.get("example_refs") or []:
            case_id = str(ref.get("case_id") or "")
            if case_id:
                case_ids.add(case_id)
    fallback = _read_bpe_caption_lookup(bpe_sequences)
    lookup: dict[str, list[str]] = {}
    for case_id in sorted(case_ids):
        captions = _read_hml3d_captions(hml_root, case_id)
        if not captions:
            captions = fallback.get(case_id, [])
        lookup[case_id] = captions
    return lookup


def _join_ids(items: list[dict[str, Any]], *, limit: int = 6) -> str:
    return "; ".join(f"{item.get('id')}:{item.get('count')}" for item in items[:limit])


def _join_list(items: list[Any], *, limit: int = 8) -> str:
    return "; ".join(str(item) for item in items[:limit])


def _example_refs(item: dict[str, Any], *, limit: int = 6) -> str:
    refs = []
    for ref in (item.get("example_refs") or [])[:limit]:
        refs.append(f"{ref.get('case_id')}@{ref.get('span')}")
    return "; ".join(refs)


def _example_caption_rows(
    item: dict[str, Any],
    caption_lookup: dict[str, list[str]],
    *,
    limit: int = 6,
    captions_per_case: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in (item.get("example_refs") or [])[:limit]:
        case_id = str(ref.get("case_id") or "")
        rows.append(
            {
                "case_id": case_id,
                "span": ref.get("span"),
                "captions": caption_lookup.get(case_id, [])[:captions_per_case],
            }
        )
    return rows


def _example_captions_text(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        caption_text = " | ".join(str(item) for item in row.get("captions") or [])
        parts.append(f"{row.get('case_id')}@{row.get('span')}: {caption_text}")
    return "; ".join(parts)


def _review_decision(item: dict[str, Any], accepted_readiness: set[str]) -> str:
    readiness = str(item.get("promotion_readiness") or "")
    if readiness in accepted_readiness:
        return "promote_candidate"
    return "review_pending"


def build_rows(
    queue: dict[str, Any],
    accepted_readiness: set[str],
    caption_lookup: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in queue.get("queue") or []:
        axis = item.get("structural_axis") or {}
        support = item.get("support") or {}
        example_caption_rows = _example_caption_rows(item, caption_lookup)
        rows.append(
            {
                "proposal_node_id": item.get("proposal_node_id"),
                "parent_node_id": item.get("parent_node_id"),
                "review_decision": _review_decision(item, accepted_readiness),
                "promotion_readiness": item.get("promotion_readiness"),
                "axis_id": axis.get("axis_id"),
                "axis_role": axis.get("axis_role"),
                "context_bucket": axis.get("context_bucket"),
                "cluster_id": axis.get("cluster_id"),
                "support_cases": support.get("support_cases"),
                "support_ratio": support.get("support_ratio"),
                "required_geometry_clusters": _join_list(item.get("required_geometry_clusters") or []),
                "top_cooccurring_clusters": _join_ids(item.get("top_cooccurring_clusters") or []),
                "example_refs": _example_refs(item),
                "example_captions": _example_captions_text(example_caption_rows),
                "example_caption_rows": example_caption_rows,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "proposal_node_id",
        "parent_node_id",
        "review_decision",
        "promotion_readiness",
        "axis_id",
        "axis_role",
        "context_bucket",
        "cluster_id",
        "support_cases",
        "support_ratio",
        "required_geometry_clusters",
        "top_cooccurring_clusters",
        "example_refs",
        "example_captions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_review_markdown(path: Path, rows: list[dict[str, Any]], source: str) -> None:
    lines: list[str] = []
    lines.append("# Motion Split Promotion Queue Review")
    lines.append("")
    lines.append(f"- source: `{source}`")
    lines.append("- structure policy: review decisions use motion evidence only.")
    lines.append("- HML3D captions are shown for human filtering only and are not copied into the manual seed.")
    lines.append("")
    counts = Counter(str(row["review_decision"]) for row in rows)
    readiness = Counter(str(row["promotion_readiness"]) for row in rows)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- queue rows: `{len(rows)}`")
    lines.append(f"- review decisions: `{dict(sorted(counts.items()))}`")
    lines.append(f"- readiness: `{dict(sorted(readiness.items()))}`")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["review_decision"])].append(row)
    for decision, group_rows in sorted(grouped.items()):
        lines.append("")
        lines.append(f"## {decision}")
        lines.append("")
        lines.append("| candidate | cluster | support | ratio | geometry | example captions |")
        lines.append("| --- | --- | ---: | ---: | --- | --- |")
        for row in group_rows:
            geometry = str(row["required_geometry_clusters"]).replace("; ", "<br>")
            examples = str(row["example_captions"]).replace("; ", "<br>")
            lines.append(
                "| `{candidate}` | `{cluster}` | {support} | {ratio} | {geometry} | {examples} |".format(
                    candidate=row["proposal_node_id"],
                    cluster=row["cluster_id"],
                    support=row["support_cases"],
                    ratio=row["support_ratio"],
                    geometry=geometry,
                    examples=examples,
                )
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_seed(path: Path, rows: list[dict[str, Any]], source: str, accepted_readiness: set[str]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        decision = str(row["review_decision"])
        decisions.append(
            {
                "proposal_node_id": row["proposal_node_id"],
                "parent_node_id": row["parent_node_id"],
                "decision": decision,
                "structural_axis": {
                    "axis_id": row["axis_id"],
                    "axis_role": row["axis_role"],
                    "context_bucket": row["context_bucket"],
                    "cluster_id": row["cluster_id"],
                },
                "support": {
                    "support_cases": row["support_cases"],
                    "support_ratio": row["support_ratio"],
                },
                "review_note": (
                    "accepted from current manual review of the review_for_promotion group"
                    if decision == "promote_candidate"
                    else "left pending for later manual review"
                ),
            }
        )
    payload = {
        "schema_version": "motion_split_manual_review_seed_v1",
        "source": {
            "promotion_queue": source,
        },
        "accepted_readiness": sorted(accepted_readiness),
        "policy": "manual seed records promotion decisions without language labels; tree insertion remains a later explicit step",
        "summary": {
            "decision_counts": dict(sorted(Counter(row["decision"] for row in decisions).items())),
            "decision_count": len(decisions),
        },
        "decisions": decisions,
    }
    _write_json(path, payload)
    return payload


def write_manual_seed_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Motion Split Manual Review Seed V1")
    lines.append("")
    lines.append("This file records the current manual review state for split-child candidates.")
    lines.append("It does not insert the candidates into the runtime tree.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in (payload.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("| candidate | decision | cluster | support | note |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for item in payload.get("decisions") or []:
        axis = item.get("structural_axis") or {}
        support = item.get("support") or {}
        lines.append(
            f"| `{item['proposal_node_id']}` | `{item['decision']}` | `{axis.get('cluster_id')}` | {support.get('support_cases')} | {item.get('review_note')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render human-readable review artifacts for motion split promotion queue.")
    parser.add_argument("--promotion-queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hml-root", default=str(DEFAULT_HML_ROOT))
    parser.add_argument("--bpe-sequences", default=str(DEFAULT_BPE_SEQUENCES))
    parser.add_argument(
        "--accept-readiness",
        action="append",
        default=["review_for_promotion"],
        help="Readiness class to mark as promote_candidate in the manual seed. Repeatable.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = Path(args.promotion_queue)
    accepted_readiness = {str(item) for item in args.accept_readiness if str(item)}
    queue = _read_json(queue_path)
    caption_lookup = _caption_lookup_for_examples(
        queue,
        hml_root=Path(args.hml_root),
        bpe_sequences=Path(args.bpe_sequences),
    )
    rows = build_rows(queue, accepted_readiness, caption_lookup)
    write_csv(output_dir / "motion_split_promotion_queue_review.csv", rows)
    write_review_markdown(output_dir / "motion_split_promotion_queue_review.md", rows, str(queue_path))
    manual_seed = write_manual_seed(
        output_dir / "manual_split_review_seed_v1.json",
        rows,
        str(queue_path),
        accepted_readiness,
    )
    write_manual_seed_markdown(output_dir / "manual_split_review_seed_v1.md", manual_seed)
    print(output_dir)


if __name__ == "__main__":
    main()
