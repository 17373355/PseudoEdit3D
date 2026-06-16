from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SMPLX_FEATURES = [
    "smplx_mesh_betas",
    "smplx_mesh_body_pose",
    "smplx_mesh_global_orient",
    "smplx_mesh_left_hand_pose",
    "smplx_mesh_right_hand_pose",
    "smplx_mesh_transl",
]

OPTIONAL_FEATURES = ["missing"]

TEXT_FIELDS = [
    "describe_person_movement",
    "describe_person_action",
    "describe_person_posture_free_form",
    "describe_person_mood_free_form",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a reproducible Embody3D person-level movement chunk manifest "
            "with explicit filters and machine-readable summaries."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted subset root containing dataset.json.")
    parser.add_argument("--subset-name", required=True, help="Stable subset name, e.g. embody3d_daylife.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-zip", type=Path, default=None, help="Optional text zip used as authoritative text source.")
    parser.add_argument("--smplx-zip", type=Path, default=None, help="Optional SMPL-X zip used as authoritative feature source.")
    parser.add_argument(
        "--zip-subdir",
        default=None,
        help="Top-level directory inside Embody3D zips. Defaults to data-root basename.",
    )
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--min-target-frames", type=int, default=40)
    parser.add_argument(
        "--max-target-frames",
        type=int,
        default=0,
        help="Optional max target-frame filter. 0 disables this filter.",
    )
    parser.add_argument(
        "--min-mask-valid-ratio",
        type=float,
        default=0.95,
        help="Minimum fraction of chunk frames with mask-valid-value. Set <0 to disable mask-ratio filtering.",
    )
    parser.add_argument(
        "--mask-valid-value",
        type=float,
        default=1.0,
        help="Observed Embody3D valid tracking mask value. Mask semantics are recorded in summary.",
    )
    parser.add_argument(
        "--no-scan-missing-mask",
        action="store_true",
        help="Do not load missing masks. Records will keep mask paths but skip mask-ratio filtering.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-person-sequences",
        type=int,
        default=0,
        help="Debug limit over sorted person-sequences. 0 exports all.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sequence_span(sequence_name: str) -> tuple[int | None, int | None]:
    tail = sequence_name.split("--")[-1]
    if "-" not in tail:
        return None, None
    start, end = tail.split("-", 1)
    try:
        return int(start), int(end)
    except ValueError:
        return None, None


def scenario_label(sequence_name: str) -> str:
    if "--MotionPrior--" in sequence_name:
        label = sequence_name.split("--MotionPrior--", 1)[1]
    else:
        label = sequence_name
    if "--" in label:
        label = label.rsplit("--", 1)[0]
    return label


def stable_id(*parts: str, length: int = 12) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "item"


def text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [str(value)]
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def open_zip(path: Path | None) -> zipfile.ZipFile | None:
    if path is None:
        return None
    return zipfile.ZipFile(path)


def zip_member(zip_subdir: str, sequence_name: str, subject_id: str, feature: str, suffix: str) -> str:
    return f"{zip_subdir}/{sequence_name}/{subject_id}/{feature}/{sequence_name}{suffix}"


def local_member(data_root: Path, sequence_name: str, subject_id: str, feature: str, suffix: str) -> Path:
    return data_root / sequence_name / subject_id / feature / f"{sequence_name}{suffix}"


def resolve_blob(
    data_root: Path,
    sequence_name: str,
    subject_id: str,
    feature: str,
    suffix: str,
    zip_subdir: str,
    zf: zipfile.ZipFile | None,
    zip_names: set[str],
) -> dict[str, Any]:
    local_path = local_member(data_root, sequence_name, subject_id, feature, suffix)
    member = zip_member(zip_subdir, sequence_name, subject_id, feature, suffix)
    if zf is not None:
        return {
            "exists": member in zip_names,
            "storage": "zip",
            "zip_path": str(Path(zf.filename).resolve()),
            "zip_member": member,
            "local_path": str(local_path),
        }
    return {
        "exists": local_path.exists(),
        "storage": "local",
        "path": str(local_path),
    }


def read_json_blob(info: dict[str, Any], zf: zipfile.ZipFile | None) -> tuple[dict[str, Any] | None, str | None]:
    if not info.get("exists"):
        return None, "missing"
    try:
        if info.get("storage") == "zip":
            assert zf is not None
            return json.loads(zf.read(info["zip_member"]).decode("utf-8")), None
        return load_json(Path(info["path"])), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_npy_blob(info: dict[str, Any], zf: zipfile.ZipFile | None) -> tuple[np.ndarray | None, str | None]:
    if not info.get("exists"):
        return None, "missing"
    try:
        if info.get("storage") == "zip":
            assert zf is not None
            return np.load(io.BytesIO(zf.read(info["zip_member"]))), None
        return np.load(Path(info["path"]), mmap_mode="r"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parse_chunks(
    raw_text: dict[str, Any],
    seq_start: int | None,
    motion_len: int | None,
    chunk_frames: int,
) -> list[dict[str, Any]]:
    chunks = []
    for key in sorted(raw_text.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        anno = raw_text[key]
        if not isinstance(anno, dict):
            continue
        try:
            abs_start = int(key)
        except ValueError:
            abs_start = None
        local_start = None
        local_end = None
        if abs_start is not None and seq_start is not None:
            local_start = abs_start - seq_start
            local_end = local_start + chunk_frames
            if motion_len is not None:
                local_start = max(0, min(local_start, motion_len))
                local_end = max(local_start, min(local_end, motion_len))
        fields = {field: text_list(anno.get(field)) for field in TEXT_FIELDS}
        fields = {k: v for k, v in fields.items() if v}
        chunks.append(
            {
                "text_key": str(key),
                "abs_start_frame": abs_start,
                "local_start_frame": local_start,
                "local_end_frame": local_end,
                "fields": fields,
                "available_fields": sorted([k for k, v in anno.items() if v]),
            }
        )
    return chunks


def mask_stats_for_chunk(
    mask_info: dict[str, Any],
    mask_zip: zipfile.ZipFile | None,
    local_start: int | None,
    local_end: int | None,
    valid_value: float,
) -> dict[str, Any]:
    stats = {
        "available": bool(mask_info.get("exists")),
        "storage": mask_info.get("storage"),
        "valid_value_assumption": valid_value,
        "chunk_valid_ratio": None,
        "load_error": None,
    }
    if local_start is None or local_end is None or local_end <= local_start:
        return stats
    arr, err = read_npy_blob(mask_info, mask_zip)
    if err is not None:
        stats["load_error"] = err
        return stats
    assert arr is not None
    chunk = np.asarray(arr[local_start:local_end])
    if chunk.size == 0:
        stats["chunk_valid_ratio"] = 0.0
        return stats
    valid = np.isclose(chunk.astype(float), valid_value)
    stats["chunk_valid_ratio"] = float(valid.mean())
    stats["chunk_frames_scanned"] = int(chunk.shape[0])
    return stats


def assign_splits(records: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    captures = sorted({record["sequence_name"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(captures)
    n = len(captures)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    else:
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
    split_by_capture = {}
    for i, capture in enumerate(captures):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_by_capture[capture] = split
    return split_by_capture


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root
    dataset_json = data_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")
    zip_subdir = args.zip_subdir or data_root.name

    dataset_info = load_json(dataset_json)
    text_zip = open_zip(args.text_zip)
    smplx_zip = open_zip(args.smplx_zip)
    text_names = set(text_zip.namelist()) if text_zip is not None else set()
    smplx_names = set(smplx_zip.namelist()) if smplx_zip is not None else set()

    chunk_records: list[dict[str, Any]] = []
    caption_records: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    per_chunk_movement_strings: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()

    items = [
        (str(subject_id), str(sequence_name), meta)
        for subject_id, sequences in dataset_info.items()
        for sequence_name, meta in sequences.items()
        if isinstance(meta, dict)
    ]
    items.sort(key=lambda x: (x[1], x[0]))
    if args.max_person_sequences > 0:
        items = items[: args.max_person_sequences]

    scan_missing_mask = not args.no_scan_missing_mask
    for subject_id, sequence_name, meta in items:
        counters["person_sequences"] += 1
        seq_start, seq_end = sequence_span(sequence_name)
        motion_len = int(meta["length"]) if str(meta.get("length", "")).isdigit() else None
        other_subject_ids = [str(x) for x in (meta.get("multiperson") or [])]
        if other_subject_ids:
            counters["multiperson_person_sequences"] += 1
        counters["captures_seen"] += 0

        feature_infos = {}
        missing_features = []
        for feature in SMPLX_FEATURES:
            info = resolve_blob(
                data_root, sequence_name, subject_id, feature, ".npy", zip_subdir, smplx_zip, smplx_names
            )
            feature_infos[feature] = info
            if not info.get("exists"):
                missing_features.append(feature)
        missing_info = resolve_blob(
            data_root, sequence_name, subject_id, "missing", ".npy", zip_subdir, smplx_zip, smplx_names
        )
        complete_smplx = not missing_features
        if complete_smplx:
            counters["complete_smplx_person_sequences"] += 1
        if missing_info.get("exists"):
            counters["with_missing_mask"] += 1

        text_info = resolve_blob(
            data_root, sequence_name, subject_id, "text_annotations", ".json", zip_subdir, text_zip, text_names
        )
        raw_text, text_error = read_json_blob(text_info, text_zip)
        if raw_text is not None:
            counters["with_person_text"] += 1
        sequence_reasons = []
        if seq_start is None or seq_end is None:
            sequence_reasons.append("invalid_sequence_span")
        if motion_len is None or motion_len <= 0:
            sequence_reasons.append("invalid_motion_len")
        if not complete_smplx:
            sequence_reasons.append("missing_smplx_features")
        if not missing_info.get("exists"):
            sequence_reasons.append("missing_tracking_mask")
        if raw_text is None:
            sequence_reasons.append("missing_or_unreadable_person_text")
            if text_error and text_error != "missing":
                sequence_reasons.append("person_text_parse_error")

        sequence_records.append(
            {
                "schema_version": "embody3d_sequence_filter_report_v1",
                "subset_name": args.subset_name,
                "sequence_name": sequence_name,
                "subject_id": subject_id,
                "other_subject_ids": other_subject_ids,
                "scenario_label": scenario_label(sequence_name),
                "sequence_start_frame": seq_start,
                "sequence_end_frame": seq_end,
                "motion_len_frames": motion_len,
                "complete_smplx": complete_smplx,
                "missing_features": missing_features,
                "has_missing_mask": bool(missing_info.get("exists")),
                "has_person_text": raw_text is not None,
                "text_error": text_error,
                "status": "accepted_for_chunking" if not sequence_reasons else "sequence_has_filter_reasons",
                "filter_reasons": sequence_reasons,
            }
        )

        if raw_text is None:
            reject_reasons["missing_or_unreadable_person_text"] += 1
            continue

        chunks = parse_chunks(raw_text, seq_start, motion_len, args.chunk_frames)
        counters["text_chunks"] += len(chunks)
        for chunk in chunks:
            reasons = list(sequence_reasons)
            movement = chunk.get("fields", {}).get("describe_person_movement", [])
            if not movement:
                reasons.append("missing_movement_text")
            local_start = chunk.get("local_start_frame")
            local_end = chunk.get("local_end_frame")
            if local_start is None or local_end is None or local_end <= local_start:
                reasons.append("invalid_chunk_frame_range")
                duration_frames = 0
            else:
                duration_frames = int(local_end - local_start)
            target_frames = int(math.floor(duration_frames * args.target_fps / args.source_fps))
            if target_frames < args.min_target_frames:
                reasons.append("target_too_short")
            if args.max_target_frames > 0 and target_frames > args.max_target_frames:
                reasons.append("target_too_long")

            mask_stats = {
                "available": bool(missing_info.get("exists")),
                "storage": missing_info.get("storage"),
                "valid_value_assumption": args.mask_valid_value,
                "chunk_valid_ratio": None,
                "scan_disabled": not scan_missing_mask,
            }
            if scan_missing_mask and missing_info.get("exists"):
                mask_stats = mask_stats_for_chunk(
                    missing_info, smplx_zip, local_start, local_end, args.mask_valid_value
                )
                mask_stats["scan_disabled"] = False
                if mask_stats.get("load_error"):
                    reasons.append("missing_mask_load_error")
                elif (
                    args.min_mask_valid_ratio >= 0
                    and mask_stats.get("chunk_valid_ratio") is not None
                    and float(mask_stats["chunk_valid_ratio"]) < args.min_mask_valid_ratio
                ):
                    reasons.append("low_tracking_mask_valid_ratio")

            status = "accepted" if not reasons else "rejected"
            for reason in reasons:
                reject_reasons[reason] += 1
            if status == "accepted":
                counters["accepted_chunks"] += 1
                counters["accepted_raw_movement_strings"] += len(movement)
                per_chunk_movement_strings[str(len(movement))] += 1
            for field in chunk.get("available_fields", []):
                field_counts[field] += 1

            rec_id = (
                f"{slugify(args.subset_name, 32)}__{slugify(scenario_label(sequence_name), 40)}__"
                f"{subject_id}__{chunk['text_key']}__{stable_id(sequence_name, subject_id, chunk['text_key'])}"
            )
            record = {
                "schema_version": "embody3d_motion_text_chunk_manifest_v1",
                "record_id": rec_id,
                "subset_name": args.subset_name,
                "sequence_name": sequence_name,
                "subject_id": subject_id,
                "other_subject_ids": other_subject_ids,
                "scenario_label": scenario_label(sequence_name),
                "fps": args.source_fps,
                "target_fps": args.target_fps,
                "sequence_start_frame": seq_start,
                "sequence_end_frame": seq_end,
                "motion_len_frames": motion_len,
                "chunk": {
                    "text_key": chunk["text_key"],
                    "abs_start_frame": chunk["abs_start_frame"],
                    "local_start_frame": local_start,
                    "local_end_frame": local_end,
                    "duration_frames": duration_frames,
                    "duration_seconds": duration_frames / args.source_fps if args.source_fps else None,
                    "target_frames_at_target_fps": target_frames,
                },
                "smplx": {
                    "complete": complete_smplx,
                    "features": feature_infos,
                    "missing_mask": missing_info,
                },
                "missing_mask_stats": mask_stats,
                "text": {
                    "source": text_info,
                    "movement": movement,
                    "action": chunk.get("fields", {}).get("describe_person_action", []),
                    "posture": chunk.get("fields", {}).get("describe_person_posture_free_form", []),
                    "mood": chunk.get("fields", {}).get("describe_person_mood_free_form", []),
                    "available_fields": chunk.get("available_fields", []),
                    "selected_movement": movement[0] if movement else None,
                },
                "filter": {
                    "status": status,
                    "reasons": reasons,
                    "min_target_frames": args.min_target_frames,
                    "max_target_frames": args.max_target_frames,
                    "min_mask_valid_ratio": args.min_mask_valid_ratio,
                    "mask_valid_value_assumption": args.mask_valid_value,
                },
                "conversion_notes": {
                    "direct_momask_compatible": False,
                    "requires_smplx_to_hml3d": True,
                    "source_fps": args.source_fps,
                    "target_fps": args.target_fps,
                },
            }
            chunk_records.append(record)
            if status == "accepted":
                for i, caption in enumerate(movement):
                    caption_records.append(
                        {
                            "schema_version": "embody3d_caption_inventory_v1",
                            "caption_id": f"{rec_id}__m{i:02d}",
                            "record_id": rec_id,
                            "subset_name": args.subset_name,
                            "sequence_name": sequence_name,
                            "subject_id": subject_id,
                            "scenario_label": scenario_label(sequence_name),
                            "caption_type": "describe_person_movement",
                            "caption_index": i,
                            "caption": caption,
                        }
                    )

    accepted = [r for r in chunk_records if r["filter"]["status"] == "accepted"]
    split_by_capture = assign_splits(accepted, args.train_ratio, args.val_ratio, args.seed)
    for record in accepted:
        record["split"] = split_by_capture.get(record["sequence_name"], "unsplit")
    for record in chunk_records:
        record.setdefault("split", None)
    split_records = {
        split: [r for r in accepted if r.get("split") == split]
        for split in ["train", "val", "test", "unsplit"]
    }

    unique_captures = {seq for _, seq, _ in items}
    multiperson_captures = {seq for _, seq, meta in items if meta.get("multiperson")}
    summary = {
        "schema_version": "embody3d_chunk_manifest_summary_v1",
        "subset_name": args.subset_name,
        "data_root": str(data_root),
        "dataset_json": str(dataset_json),
        "text_zip": str(args.text_zip) if args.text_zip else None,
        "smplx_zip": str(args.smplx_zip) if args.smplx_zip else None,
        "zip_subdir": zip_subdir,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "filter_config": {
            "chunk_frames": args.chunk_frames,
            "min_target_frames": args.min_target_frames,
            "max_target_frames": args.max_target_frames,
            "min_mask_valid_ratio": args.min_mask_valid_ratio,
            "mask_valid_value_assumption": args.mask_valid_value,
            "scan_missing_mask": scan_missing_mask,
            "split_seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "counts": {
            "subjects": len(dataset_info),
            "captures": len(unique_captures),
            "multiperson_captures": len(multiperson_captures),
            "person_sequences": counters["person_sequences"],
            "complete_smplx_person_sequences": counters["complete_smplx_person_sequences"],
            "with_missing_mask": counters["with_missing_mask"],
            "with_person_text": counters["with_person_text"],
            "text_chunks": counters["text_chunks"],
            "chunk_records": len(chunk_records),
            "accepted_chunks": counters["accepted_chunks"],
            "rejected_chunks": len(chunk_records) - counters["accepted_chunks"],
            "accepted_raw_movement_strings": counters["accepted_raw_movement_strings"],
            "caption_records": len(caption_records),
        },
        "split_counts": {
            split: {
                "chunks": len(rows),
                "captures": len({r["sequence_name"] for r in rows}),
                "raw_movement_strings": sum(len(r["text"]["movement"]) for r in rows),
            }
            for split, rows in split_records.items()
            if rows
        },
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "accepted_per_chunk_movement_string_counts": dict(sorted(per_chunk_movement_strings.items())),
        "nonempty_text_field_counts": dict(field_counts.most_common()),
    }

    output_dir = args.output_dir
    write_jsonl(output_dir / "manifest_all.jsonl", chunk_records)
    write_jsonl(output_dir / "manifest_filtered.jsonl", accepted)
    write_jsonl(output_dir / "sequence_filter_report.jsonl", sequence_records)
    write_jsonl(output_dir / "caption_inventory.jsonl", caption_records)
    for split, rows in split_records.items():
        if not rows:
            continue
        write_jsonl(output_dir / "splits" / f"manifest_{split}.jsonl", rows)
        ids = "\n".join(r["record_id"] for r in rows) + "\n"
        (output_dir / "splits").mkdir(parents=True, exist_ok=True)
        (output_dir / "splits" / f"{split}.txt").write_text(ids, encoding="utf-8")
    write_json(output_dir / "summary.json", summary)

    if text_zip is not None:
        text_zip.close()
    if smplx_zip is not None:
        smplx_zip.close()
    return summary


def main() -> None:
    args = parse_args()
    summary = build_manifest(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
