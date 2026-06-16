from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from collections import Counter
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

OPTIONAL_FEATURES = [
    "missing",
]

TEXT_FIELDS = [
    "describe_person_movement",
    "describe_person_action",
    "describe_person_posture_free_form",
    "describe_person_mood_free_form",
]

BODY_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Embody3D SMPL-X/text layout and export person-level QC visualizations."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/home/guoruoxi/code/embody-3d/datasets/daylife"),
        help="Extracted Embody3D subset root containing dataset.json and capture folders.",
    )
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("/mnt/data/home/guoruoxi/code/CharRet_multi/body_models"),
        help="Root passed to smplx.create(); expected to contain smplx/SMPLX_NEUTRAL.*.",
    )
    parser.add_argument(
        "--momask-hml-root",
        type=Path,
        default=Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D"),
        help="MoMask HumanML3D dataset root used for format-contract inspection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs/embody3d_daylife_audit"),
        help="Directory for summary files and visualization outputs.",
    )
    parser.add_argument("--samples", type=int, default=4, help="Number of person-level text/SMPL-X samples to render.")
    parser.add_argument(
        "--max-scan-captures",
        type=int,
        default=0,
        help="Debug limit for capture directories. 0 scans all captures.",
    )
    parser.add_argument(
        "--shape-scan",
        action="store_true",
        help="Load every .npy header to record shapes. Disabled by default because this is slow on NFS.",
    )
    parser.add_argument(
        "--render-frames",
        type=int,
        default=72,
        help="Frames sampled from each 10s text chunk for skeleton GIF rendering.",
    )
    parser.add_argument(
        "--mesh-frames",
        type=int,
        default=24,
        help="Frames sampled from each 10s text chunk for optional mesh GIF rendering.",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate.")
    parser.add_argument("--device", default="cpu", help="Torch device for SMPL-X forward.")
    parser.add_argument(
        "--render-mesh",
        action="store_true",
        help="Also try pyrender mesh GIFs. Skeleton GIFs are always exported.",
    )
    parser.add_argument(
        "--include-mesh-vertices",
        action="store_true",
        help="Keep full vertices in memory for mesh rendering. Set automatically by --render-mesh.",
    )
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sequence_span(sequence_name: str) -> tuple[int | None, int | None]:
    tail = sequence_name.split("--")[-1]
    if "-" not in tail:
        return None, None
    start, end = tail.split("-", 1)
    try:
        return int(start), int(end)
    except ValueError:
        return None, None


def feature_file(person_dir: Path, sequence_name: str, feature: str) -> Path:
    return person_dir / feature / f"{sequence_name}.npy"


def is_person_dir(path: Path, sequence_name: str) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("text_annotations"):
        return False
    if (path / "text_annotations" / f"{sequence_name}.json").exists():
        return True
    return any((path / feature).exists() for feature in SMPLX_FEATURES + OPTIONAL_FEATURES)


def npy_shape(path: Path) -> dict[str, Any]:
    try:
        arr = np.load(path, mmap_mode="r")
        return {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as exc:
        return {"exists": True, "load_error": f"{type(exc).__name__}: {exc}"}


def text_list(value: Any, max_items: int = 2) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value[:max_items] if str(v).strip()]
    return [str(value)]


def parse_text_chunks(text_path: Path, seq_start: int | None, motion_len: int | None) -> list[dict[str, Any]]:
    if not text_path.exists():
        return []
    try:
        raw = safe_load_json(text_path)
    except Exception as exc:
        return [{"parse_error": f"{type(exc).__name__}: {exc}"}]
    chunks: list[dict[str, Any]] = []
    for key in sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        anno = raw[key]
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
            local_end = local_start + 300
            if motion_len is not None:
                local_start = max(0, min(local_start, motion_len))
                local_end = max(local_start, min(local_end, motion_len))
        fields: dict[str, list[str]] = {}
        for field in TEXT_FIELDS:
            vals = text_list(anno.get(field), max_items=2)
            if vals:
                fields[field] = vals
        chunks.append(
            {
                "text_key": key,
                "abs_start_frame": abs_start,
                "local_start_frame": local_start,
                "local_end_frame": local_end,
                "fields": fields,
                "available_fields": sorted([k for k, v in anno.items() if v]),
            }
        )
    return chunks


def scan_daylife_from_dataset_json(
    data_root: Path,
    dataset_info: dict[str, Any],
    max_scan_captures: int = 0,
    shape_scan: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    missing_feature_counter: Counter[str] = Counter()
    shape_error_counter: Counter[str] = Counter()
    unique_captures: set[str] = set()
    multiperson_captures: set[str] = set()

    sequence_names = sorted({seq for sequences in dataset_info.values() for seq in sequences.keys()})
    if max_scan_captures > 0:
        allowed_sequences = set(sequence_names[:max_scan_captures])
    else:
        allowed_sequences = set(sequence_names)

    for subject_id in sorted(dataset_info.keys()):
        for sequence_name in sorted(dataset_info[subject_id].keys()):
            if sequence_name not in allowed_sequences:
                continue
            meta = dataset_info[subject_id][sequence_name]
            if not isinstance(meta, dict):
                continue
            capture_dir = data_root / sequence_name
            person_dir = capture_dir / subject_id
            seq_start, seq_end = sequence_span(sequence_name)
            motion_len = int(meta["length"]) if str(meta.get("length", "")).isdigit() else None
            other_subject_ids = [str(x) for x in (meta.get("multiperson") or [])]
            unique_captures.add(sequence_name)
            if other_subject_ids:
                multiperson_captures.add(sequence_name)
            counters["person_sequences"] += 1

            feature_shapes: dict[str, Any] = {}
            missing_features: list[str] = []
            for feature in SMPLX_FEATURES + OPTIONAL_FEATURES:
                path = feature_file(person_dir, sequence_name, feature)
                if not path.exists():
                    missing_features.append(feature)
                    missing_feature_counter[feature] += 1
                    continue
                if shape_scan:
                    shape_info = npy_shape(path)
                    feature_shapes[feature] = shape_info
                    if "load_error" in shape_info:
                        shape_error_counter[feature] += 1

            complete_smplx = all(feature not in missing_features for feature in SMPLX_FEATURES)
            if complete_smplx:
                counters["complete_smplx_person_sequences"] += 1
            if "missing" not in missing_features:
                counters["with_missing_mask"] += 1

            text_path = person_dir / "text_annotations" / f"{sequence_name}.json"
            has_text = text_path.exists()
            if has_text:
                counters["with_person_text"] += 1
            chunks = parse_text_chunks(text_path, seq_start, motion_len) if has_text else []
            valid_text_chunks = [
                c
                for c in chunks
                if c.get("local_start_frame") is not None
                and c.get("local_end_frame") is not None
                and c.get("local_end_frame", 0) > c.get("local_start_frame", 0)
                and c.get("fields", {}).get("describe_person_movement")
            ]
            counters["text_chunks"] += len(chunks)
            counters["valid_movement_text_chunks"] += len(valid_text_chunks)

            records.append(
                {
                    "sequence_name": sequence_name,
                    "sequence_dir": str(capture_dir),
                    "subject_id": subject_id,
                    "other_subject_ids": other_subject_ids,
                    "sequence_start_frame": seq_start,
                    "sequence_end_frame": seq_end,
                    "motion_len_frames": motion_len,
                    "fps": 30,
                    "complete_smplx": complete_smplx,
                    "missing_features": missing_features,
                    "feature_shapes": feature_shapes,
                    "text_path": str(text_path) if has_text else None,
                    "dataset_json_text_path": meta.get("text"),
                    "dataset_json_audio_path": meta.get("audio"),
                    "text_chunks": chunks,
                    "valid_movement_text_chunks": valid_text_chunks,
                }
            )

    counters["captures"] = len(unique_captures)
    counters["multiperson_captures"] = len(multiperson_captures)
    summary = {
        "data_root": str(data_root),
        "scan_mode": "dataset_json_fast",
        "shape_scan": shape_scan,
        "scan_limits": {"max_scan_captures": max_scan_captures},
        "counts": dict(counters),
        "missing_feature_counts": dict(missing_feature_counter),
        "shape_load_error_counts": dict(shape_error_counter),
    }
    return records, summary


def scan_daylife_by_dirs(
    data_root: Path,
    max_scan_captures: int = 0,
    shape_scan: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    missing_feature_counter: Counter[str] = Counter()
    shape_error_counter: Counter[str] = Counter()

    capture_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if max_scan_captures > 0:
        capture_dirs = capture_dirs[:max_scan_captures]

    for capture_dir in capture_dirs:
        sequence_name = capture_dir.name
        seq_start, seq_end = sequence_span(sequence_name)
        person_dirs = sorted([p for p in capture_dir.iterdir() if is_person_dir(p, sequence_name)])
        person_ids = [p.name for p in person_dirs]
        counters["captures"] += 1
        counters["person_sequences"] += len(person_dirs)
        if len(person_dirs) > 1:
            counters["multiperson_captures"] += 1

        for person_dir in person_dirs:
            feature_shapes: dict[str, Any] = {}
            missing_features: list[str] = []
            motion_len = None
            for feature in SMPLX_FEATURES + OPTIONAL_FEATURES:
                path = feature_file(person_dir, sequence_name, feature)
                if not path.exists():
                    missing_features.append(feature)
                    missing_feature_counter[feature] += 1
                    continue
                if shape_scan:
                    shape_info = npy_shape(path)
                    feature_shapes[feature] = shape_info
                    if "load_error" in shape_info:
                        shape_error_counter[feature] += 1
                    elif feature == "smplx_mesh_body_pose" and shape_info.get("shape"):
                        motion_len = int(shape_info["shape"][0])

            complete_smplx = all(feature not in missing_features for feature in SMPLX_FEATURES)
            if complete_smplx:
                counters["complete_smplx_person_sequences"] += 1
            if "missing" not in missing_features:
                counters["with_missing_mask"] += 1

            text_path = person_dir / "text_annotations" / f"{sequence_name}.json"
            has_text = text_path.exists()
            if has_text:
                counters["with_person_text"] += 1
            chunks = parse_text_chunks(text_path, seq_start, motion_len) if has_text else []
            valid_text_chunks = [
                c
                for c in chunks
                if c.get("local_start_frame") is not None
                and c.get("local_end_frame") is not None
                and c.get("local_end_frame", 0) > c.get("local_start_frame", 0)
                and c.get("fields", {}).get("describe_person_movement")
            ]
            counters["text_chunks"] += len(chunks)
            counters["valid_movement_text_chunks"] += len(valid_text_chunks)

            records.append(
                {
                    "sequence_name": sequence_name,
                    "sequence_dir": str(capture_dir),
                    "subject_id": person_dir.name,
                    "other_subject_ids": [pid for pid in person_ids if pid != person_dir.name],
                    "sequence_start_frame": seq_start,
                    "sequence_end_frame": seq_end,
                    "motion_len_frames": motion_len,
                    "fps": 30,
                    "complete_smplx": complete_smplx,
                    "missing_features": missing_features,
                    "feature_shapes": feature_shapes,
                    "text_path": str(text_path) if has_text else None,
                    "text_chunks": chunks,
                    "valid_movement_text_chunks": valid_text_chunks,
                }
            )

    summary = {
        "data_root": str(data_root),
        "scan_mode": "directory_fallback",
        "shape_scan": shape_scan,
        "scan_limits": {"max_scan_captures": max_scan_captures},
        "counts": dict(counters),
        "missing_feature_counts": dict(missing_feature_counter),
        "shape_load_error_counts": dict(shape_error_counter),
    }
    return records, summary


def scan_daylife(data_root: Path, max_scan_captures: int = 0, shape_scan: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_json = data_root / "dataset.json"
    if dataset_json.exists():
        return scan_daylife_from_dataset_json(
            data_root=data_root,
            dataset_info=safe_load_json(dataset_json),
            max_scan_captures=max_scan_captures,
            shape_scan=shape_scan,
        )
    return scan_daylife_by_dirs(data_root=data_root, max_scan_captures=max_scan_captures, shape_scan=shape_scan)


def inspect_momask_humanml3d(hml_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hml_root": str(hml_root),
        "exists": hml_root.exists(),
        "format_contract": {
            "motion_dir": "new_joint_vecs/<id>.npy",
            "motion_feature": "HumanML3D processed vector, usually F x 263 for 22 joints",
            "text_dir": "texts/<id>.txt",
            "text_line_format": "caption#tokens#start_seconds#end_seconds",
            "raw_joint_optional": "new_joints/<id>.npy or packed joints3d.pth, usually F x 22 x 3",
        },
        "direct_compatible_with_embody_smplx": False,
        "required_conversion": [
            "forward Embody SMPL-X parameters with the SMPL-X model to recover body joints",
            "select/reorder the body joints to the HumanML3D 22-joint skeleton",
            "filter bad tracking with missing masks before accepting chunks",
            "downsample DAYLIFE 30fps chunks to the HumanML3D/MoMask 20fps convention",
            "run the HumanML3D feature extractor (MoMask utils.motion_process.process_file) to create 263-dim vectors",
            "write MoMask-style new_joint_vecs, texts, and split files without using same-case text as AML labels",
        ],
    }
    if not hml_root.exists():
        return info

    for subdir in ["new_joint_vecs", "new_joints", "texts"]:
        path = hml_root / subdir
        info[f"{subdir}_exists"] = path.exists()
        if path.exists():
            files = sorted(path.glob("*"))
            info[f"{subdir}_file_count"] = len(files)
            first = next((p for p in files if p.is_file()), None)
            if first is not None:
                info[f"{subdir}_example"] = str(first)
                if first.suffix == ".npy":
                    try:
                        arr = np.load(first, mmap_mode="r")
                        info[f"{subdir}_example_shape"] = list(arr.shape)
                        info[f"{subdir}_example_dtype"] = str(arr.dtype)
                    except Exception as exc:
                        info[f"{subdir}_example_load_error"] = f"{type(exc).__name__}: {exc}"
                elif first.suffix == ".txt":
                    try:
                        first_line = first.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
                    except Exception as exc:
                        first_line = f"{type(exc).__name__}: {exc}"
                    info[f"{subdir}_example_first_line"] = first_line

    joints_pack = hml_root / "joints3d.pth"
    info["joints3d_pth_exists"] = joints_pack.exists()
    return info


def select_samples(records: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    candidates = [
        r
        for r in records
        if r["complete_smplx"] and r["valid_movement_text_chunks"] and r.get("motion_len_frames")
    ]
    candidates.sort(
        key=lambda r: (
            -len(r.get("other_subject_ids") or []),
            -len(r.get("valid_movement_text_chunks") or []),
            r["sequence_name"],
            r["subject_id"],
        )
    )
    return candidates[: max(0, samples)]


def frame_indices_for_chunk(chunk: dict[str, Any], max_frames: int) -> np.ndarray:
    start = int(chunk["local_start_frame"])
    end = int(chunk["local_end_frame"])
    if end <= start:
        raise ValueError(f"Invalid chunk frame range: {start}-{end}")
    count = min(max_frames, end - start)
    if count <= 1:
        return np.array([start], dtype=np.int64)
    return np.linspace(start, end - 1, count, dtype=np.int64)


def load_feature_frames(record: dict[str, Any], feature: str, frame_indices: np.ndarray) -> np.ndarray:
    path = feature_file(Path(record["sequence_dir"]) / record["subject_id"], record["sequence_name"], feature)
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr[frame_indices], dtype=np.float32)


def smplx_forward(
    record: dict[str, Any],
    chunk: dict[str, Any],
    body_model_root: Path,
    frame_count: int,
    device: str,
    return_verts: bool,
) -> dict[str, Any]:
    import torch
    import smplx

    frame_indices = frame_indices_for_chunk(chunk, frame_count)
    batch_size = int(len(frame_indices))
    model = smplx.create(
        str(body_model_root),
        model_type="smplx",
        gender="neutral",
        flat_hand_mean=True,
        num_betas=300,
        num_expression_coeffs=100,
        use_pca=False,
        batch_size=batch_size,
    ).to(device)
    inputs = {
        "betas": load_feature_frames(record, "smplx_mesh_betas", frame_indices),
        "body_pose": load_feature_frames(record, "smplx_mesh_body_pose", frame_indices),
        "global_orient": load_feature_frames(record, "smplx_mesh_global_orient", frame_indices),
        "left_hand_pose": load_feature_frames(record, "smplx_mesh_left_hand_pose", frame_indices),
        "right_hand_pose": load_feature_frames(record, "smplx_mesh_right_hand_pose", frame_indices),
        "transl": load_feature_frames(record, "smplx_mesh_transl", frame_indices),
    }
    tensor_inputs = {k: torch.from_numpy(v).to(device=device, dtype=torch.float32) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**tensor_inputs, return_verts=return_verts)
    result = {
        "frame_indices": frame_indices,
        "joints": out.joints.detach().cpu().numpy().astype(np.float32),
        "faces": np.asarray(model.faces, dtype=np.int32),
    }
    if return_verts:
        result["vertices"] = out.vertices.detach().cpu().numpy().astype(np.float32)
    return result


def missing_valid_ratio(record: dict[str, Any], chunk: dict[str, Any]) -> float | None:
    person_dir = Path(record["sequence_dir"]) / record["subject_id"]
    path = feature_file(person_dir, record["sequence_name"], "missing")
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    start = int(chunk["local_start_frame"])
    end = int(chunk["local_end_frame"])
    values = np.asarray(arr[start:end])
    if values.size == 0:
        return None
    return float(np.mean(values > 0))


def load_font(size: int):
    from PIL import ImageFont

    for name in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def project_joints(joints: np.ndarray) -> np.ndarray:
    body = joints[:, :22].copy()
    x = body[..., 0]
    y = body[..., 1]
    z = body[..., 2]
    u = 0.72 * x + 0.45 * z
    v = y
    pts = np.stack([u, v], axis=-1)
    mins = pts.reshape(-1, 2).min(axis=0)
    maxs = pts.reshape(-1, 2).max(axis=0)
    center = (mins + maxs) * 0.5
    span = np.maximum(maxs - mins, 1e-4)
    panel_w, panel_h, margin = 560, 560, 36
    scale = min((panel_w - 2 * margin) / span[0], (panel_h - 2 * margin) / span[1])
    norm = (pts - center) * scale
    norm[..., 0] += panel_w * 0.5
    norm[..., 1] = panel_h * 0.5 - norm[..., 1]
    return norm


def wrap_lines(text: str, width: int = 72, max_lines: int = 8) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    if len(lines) > max_lines:
        return lines[: max_lines - 1] + ["..."]
    return lines


def representative_text(chunk: dict[str, Any]) -> dict[str, str]:
    fields = chunk.get("fields") or {}
    out: dict[str, str] = {}
    for key in TEXT_FIELDS:
        vals = fields.get(key) or []
        if vals:
            out[key] = vals[0]
    return out


def render_skeleton_gif(
    record: dict[str, Any],
    chunk: dict[str, Any],
    joints: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Path,
    fps: int,
    valid_ratio: float | None,
) -> None:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected = project_joints(joints)
    font_title = load_font(22)
    font_body = load_font(14)
    font_small = load_font(12)
    frames = []
    text = representative_text(chunk)
    movement = text.get("describe_person_movement", "")
    action = text.get("describe_person_action", "")
    posture = text.get("describe_person_posture_free_form", "")

    root_xz = joints[:, 0, [0, 2]]
    traj_min = root_xz.min(axis=0)
    traj_max = root_xz.max(axis=0)
    traj_span = np.maximum(traj_max - traj_min, 1e-4)

    for i, joints2d in enumerate(projected):
        img = Image.new("RGB", (1280, 720), (248, 248, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle((24, 24, 612, 690), outline=(205, 207, 214), width=2, fill=(255, 255, 255))
        draw.rectangle((636, 24, 1256, 690), outline=(205, 207, 214), width=2, fill=(255, 255, 255))

        offset = np.array([38.0, 72.0])
        pts = joints2d + offset
        for parent, child in BODY_EDGES:
            draw.line([tuple(pts[parent]), tuple(pts[child])], fill=(95, 102, 120), width=4)
        for j, (x, y) in enumerate(pts):
            color = (34, 116, 181) if j in {0, 10, 11, 20, 21} else (70, 75, 90)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

        draw.text((42, 38), "SMPL-X forward joints (single person)", fill=(25, 28, 34), font=font_title)
        draw.text(
            (42, 640),
            f"frame {int(frame_indices[i])}  chunk {chunk['local_start_frame']}-{chunk['local_end_frame']}  fps 30",
            fill=(70, 74, 84),
            font=font_body,
        )

        inset = (410, 500, 590, 640)
        draw.rectangle(inset, outline=(214, 216, 222), fill=(250, 250, 252))
        draw.text((inset[0] + 8, inset[1] + 8), "root x/z", fill=(90, 94, 105), font=font_small)
        traj = (root_xz - traj_min) / traj_span
        traj[:, 0] = inset[0] + 14 + traj[:, 0] * (inset[2] - inset[0] - 28)
        traj[:, 1] = inset[3] - 14 - traj[:, 1] * (inset[3] - inset[1] - 34)
        if len(traj) > 1:
            draw.line([tuple(x) for x in traj], fill=(154, 166, 181), width=2)
        x, y = traj[i]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(220, 80, 60))

        y = 44
        meta = [
            f"subject: {record['subject_id']}",
            f"sequence: {record['sequence_name']}",
            f"other people in scene: {', '.join(record.get('other_subject_ids') or [])}",
            f"text key: {chunk['text_key']}",
            "missing-valid ratio: " + ("n/a" if valid_ratio is None else f"{valid_ratio:.3f}"),
        ]
        for line in meta:
            draw.text((660, y), line, fill=(45, 49, 58), font=font_body)
            y += 23
        y += 12

        sections = [
            ("movement", movement),
            ("action", action),
            ("posture", posture),
        ]
        for label, value in sections:
            if not value:
                continue
            draw.text((660, y), label, fill=(28, 92, 148), font=font_body)
            y += 21
            for line in wrap_lines(value, width=68, max_lines=7):
                draw.text((680, y), line, fill=(45, 49, 58), font=font_small)
                y += 17
                if y > 660:
                    break
            y += 10
            if y > 660:
                break

        frames.append(img)

    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    frames[0].save(str(output_path.with_suffix(".png")))


def render_mesh_gif(vertices: np.ndarray, faces: np.ndarray, output_path: Path, fps: int) -> dict[str, Any]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw

    def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return pose

    output_path.parent.mkdir(parents=True, exist_ok=True)
    verts = vertices.astype(np.float32).copy()
    center = np.median(verts.reshape(-1, 3), axis=0)
    verts -= center[None, None, :]
    low = verts.reshape(-1, 3).min(axis=0)
    high = verts.reshape(-1, 3).max(axis=0)
    span = high - low
    radius = max(float(np.linalg.norm(span)), 1.0)
    target = np.array([0.0, 0.65, 0.0], dtype=np.float32)
    eye = np.array([0.0, 0.9, radius * 1.4], dtype=np.float32)
    camera_pose = look_at(eye, target)
    renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=720)
    frames = []
    try:
        for idx, frame_vertices in enumerate(verts):
            mesh = trimesh.Trimesh(vertices=frame_vertices, faces=faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.75,
                baseColorFactor=(0.48, 0.68, 0.86, 1.0),
            )
            scene = pyrender.Scene(bg_color=(248, 248, 250, 255), ambient_light=(0.45, 0.45, 0.45))
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=camera_pose)
            light_pose = np.eye(4, dtype=np.float32)
            light_pose[:3, 3] = np.array([0.0, 3.0, 3.0], dtype=np.float32)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=light_pose)
            color, _ = renderer.render(scene)
            img = Image.fromarray(color)
            draw = ImageDraw.Draw(img)
            draw.rectangle((12, 12, 160, 40), fill=(255, 255, 255))
            draw.text((20, 20), f"mesh frame {idx + 1}/{len(verts)}", fill=(30, 30, 35))
            frames.append(np.asarray(img))
    finally:
        renderer.delete()
    imageio.mimsave(str(output_path), frames, fps=fps)
    return {"mesh_gif_path": str(output_path), "mesh_frames": len(frames)}


def render_samples(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = select_samples(records, args.samples)
    results: list[dict[str, Any]] = []
    for sample_idx, record in enumerate(samples, start=1):
        chunk = record["valid_movement_text_chunks"][0]
        stem = f"{sample_idx:02d}__{record['subject_id']}__{record['sequence_name']}"
        sample_dir = args.output_dir / "visualizations" / stem
        valid_ratio = missing_valid_ratio(record, chunk)
        result: dict[str, Any] = {
            "sample_index": sample_idx,
            "subject_id": record["subject_id"],
            "sequence_name": record["sequence_name"],
            "text_key": chunk["text_key"],
            "text_chunk": chunk,
            "missing_valid_ratio": valid_ratio,
            "status": "pending",
        }
        try:
            forward = smplx_forward(
                record=record,
                chunk=chunk,
                body_model_root=args.body_model_root,
                frame_count=args.render_frames,
                device=args.device,
                return_verts=False,
            )
            skeleton_path = sample_dir / f"{stem}__smplx_joints_text.gif"
            render_skeleton_gif(
                record=record,
                chunk=chunk,
                joints=forward["joints"],
                frame_indices=forward["frame_indices"],
                output_path=skeleton_path,
                fps=args.fps,
                valid_ratio=valid_ratio,
            )
            result["skeleton_gif_path"] = str(skeleton_path)
            result["skeleton_png_path"] = str(skeleton_path.with_suffix(".png"))
            result["skeleton_frames"] = int(len(forward["frame_indices"]))
            result["status"] = "skeleton_rendered"
        except Exception as exc:
            result["status"] = "skeleton_failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue

        if args.render_mesh:
            try:
                mesh_forward = smplx_forward(
                    record=record,
                    chunk=chunk,
                    body_model_root=args.body_model_root,
                    frame_count=args.mesh_frames,
                    device=args.device,
                    return_verts=True,
                )
                mesh_path = sample_dir / f"{stem}__smplx_mesh.gif"
                result.update(render_mesh_gif(mesh_forward["vertices"], mesh_forward["faces"], mesh_path, args.fps))
                result["status"] = "skeleton_and_mesh_rendered"
            except Exception as exc:
                result["mesh_render_error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    momask_info: dict[str, Any],
    visualizations: list[dict[str, Any]],
) -> str:
    counts = summary.get("counts", {})
    subset_name = args.data_root.name or "subset"
    lines = [
        f"# Embody3D {subset_name} Audit",
        "",
        "## Inputs",
        f"- Embody3D subset root: `{args.data_root}`",
        f"- SMPL-X body model root: `{args.body_model_root}`",
        f"- MoMask HumanML3D root: `{args.momask_hml_root}`",
        "",
        "## Dataset Layout Metrics",
        f"- captures scanned: {counts.get('captures', 0)}",
        f"- person-sequences scanned: {counts.get('person_sequences', 0)}",
        f"- complete SMPL-X person-sequences: {counts.get('complete_smplx_person_sequences', 0)}",
        f"- person-sequences with person text: {counts.get('with_person_text', 0)}",
        f"- text chunks: {counts.get('text_chunks', 0)}",
        f"- valid movement text chunks: {counts.get('valid_movement_text_chunks', 0)}",
        f"- person-sequences with missing mask: {counts.get('with_missing_mask', 0)}",
        "",
        "## MoMask/HumanML3D Contract",
        "- Direct compatibility: no.",
        "- Embody stores SMPL-X parameter streams at 30fps; MoMask HumanML3D consumes per-clip HumanML3D motion vectors under `new_joint_vecs/` and text files under `texts/`.",
        "- Required bridge: SMPL-X forward -> 22-joint HumanML3D skeleton mapping -> 30fps to 20fps conversion -> `utils.motion_process.process_file()` -> MoMask split/text export.",
        "",
        "## Visualization Results",
    ]
    if not visualizations:
        lines.append("- No samples rendered.")
    for item in visualizations:
        lines.append(
            f"- sample {item.get('sample_index')}: subject `{item.get('subject_id')}`, status `{item.get('status')}`"
        )
        if item.get("skeleton_gif_path"):
            lines.append(f"  - skeleton/text GIF: `{item['skeleton_gif_path']}`")
        if item.get("mesh_gif_path"):
            lines.append(f"  - mesh GIF: `{item['mesh_gif_path']}`")
        if item.get("mesh_render_error"):
            lines.append(f"  - mesh render error: `{item['mesh_render_error']}`")
        if item.get("error"):
            lines.append(f"  - error: `{item['error']}`")
    lines.extend(
        [
            "",
            "## Notes",
            "- Text annotations are already person-level in multi-person captures; this script uses each subject directory's own `text_annotations/<sequence>.json`.",
            "- The official Embody loader may fail on the extracted local copy if it uses the original absolute text paths in `dataset.json`; this audit reads local files directly.",
            "- Captions/text are suitable as noisy person-level descriptions or retrieval references, not as same-case AML ground-truth labels.",
        ]
    )
    if summary.get("missing_feature_counts"):
        lines.extend(["", "## Missing Feature Counts", ""])
        for key, value in sorted(summary["missing_feature_counts"].items()):
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Machine-Readable Outputs", ""])
    lines.append(f"- summary: `{args.output_dir / 'summary.json'}`")
    lines.append(f"- records: `{args.output_dir / 'records.jsonl'}`")
    lines.append(f"- visualizations: `{args.output_dir / 'visualizations.json'}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.render_mesh:
        args.include_mesh_vertices = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, summary = scan_daylife(
        args.data_root,
        max_scan_captures=args.max_scan_captures,
        shape_scan=args.shape_scan,
    )
    momask_info = inspect_momask_humanml3d(args.momask_hml_root)
    visualizations = render_samples(args, records) if args.samples > 0 else []

    summary.update(
        {
            "smplx_body_model_root": str(args.body_model_root),
            "momask_humanml3d": momask_info,
            "visualization_count": len(visualizations),
        }
    )
    write_records(args.output_dir / "records.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "visualizations.json", visualizations)
    (args.output_dir / "audit_report.md").write_text(
        make_report(args, summary, momask_info, visualizations), encoding="utf-8"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(f"records: {args.output_dir / 'records.jsonl'}")
    print(f"visualizations: {args.output_dir / 'visualizations.json'}")
    print(f"report: {args.output_dir / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
