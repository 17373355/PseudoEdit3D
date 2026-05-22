from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


FRAME_SUFFIX_RE = re.compile(r"_frame_\d+_\d+_(contact|neutral|non_contact)$")


def derive_clip_group_id(record: dict) -> str:
    path = Path(record["path"])
    stem = path.stem
    stem = FRAME_SUFFIX_RE.sub("", stem)
    return f"{path.parent.name}::{stem}"


def load_jsonl_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def dump_jsonl_records(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _stable_group_sort_key(seed: int, group_id: str) -> str:
    return hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def build_group_split(
    records: list[dict],
    seed: int = 42,
    test_ratio: float = 0.2,
) -> dict:
    bucket_to_groups = {}
    for record in records:
        bucket = record.get("contact_bucket", "unknown")
        group_id = derive_clip_group_id(record)
        bucket_to_groups.setdefault(bucket, set()).add(group_id)

    test_groups = set()
    train_groups = set()
    for bucket, groups in bucket_to_groups.items():
        groups_sorted = sorted(groups, key=lambda gid: _stable_group_sort_key(seed, gid))
        n_test = max(1, int(math.ceil(len(groups_sorted) * test_ratio))) if len(groups_sorted) > 1 else 0
        bucket_test = set(groups_sorted[:n_test])
        bucket_train = set(groups_sorted[n_test:])
        test_groups.update(bucket_test)
        train_groups.update(bucket_train)

    train_records = []
    test_records = []
    for record in records:
        group_id = derive_clip_group_id(record)
        augmented = dict(record)
        augmented["split_group_id"] = group_id
        if group_id in test_groups:
            augmented["split_name"] = "test"
            test_records.append(augmented)
        else:
            augmented["split_name"] = "train"
            train_records.append(augmented)

    return {
        "train_records": train_records,
        "test_records": test_records,
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
    }


def split_report(split_result: dict) -> dict:
    train_records = split_result["train_records"]
    test_records = split_result["test_records"]

    def summarize(records: list[dict]) -> dict:
        bucket_counts = {}
        groups = set()
        for record in records:
            bucket = record.get("contact_bucket", "unknown")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            groups.add(record["split_group_id"])
        return {
            "num_records": len(records),
            "num_groups": len(groups),
            "bucket_counts": bucket_counts,
        }

    return {
        "train": summarize(train_records),
        "test": summarize(test_records),
    }
