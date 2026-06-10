from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_HML_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _as_numpy_joints(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints shape [T,22,3], got {arr.shape}")
    return arr


def _load_joints_pack(hml_root: Path) -> dict[str, Any]:
    return torch.load(hml_root / "joints3d.pth", map_location="cpu")


def export_motion_batch(
    condition_batch_dir: Path,
    hml_root: Path,
    max_frames: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    schema = _load_json(condition_batch_dir / "condition_batch_schema.json")
    index_rows = _load_jsonl(condition_batch_dir / "condition_batch_index.jsonl")
    condition_npz = np.load(condition_batch_dir / "condition_batch.npz")
    condition_case_index = condition_npz["case_index"]
    condition_num_frames = condition_npz["num_frames"]
    condition_mask = condition_npz["condition_mask"]

    if len(index_rows) != int(schema["num_cases"]):
        raise ValueError("index row count does not match condition schema")
    if len(index_rows) != int(condition_case_index.shape[0]):
        raise ValueError("index row count does not match condition npz")

    packed = _load_joints_pack(hml_root)
    missing_cases: list[str] = []
    joints_by_case: list[np.ndarray] = []
    observed_lengths: list[int] = []
    for row in index_rows:
        case_id = str(row["case_id"])
        key = f"{case_id}.npy"
        if key not in packed:
            missing_cases.append(case_id)
            continue
        joints = _as_numpy_joints(packed[key]["joints3d"])
        joints_by_case.append(joints)
        observed_lengths.append(int(joints.shape[0]))

    if missing_cases:
        raise ValueError(f"Missing {len(missing_cases)} cases from joints3d pack: {missing_cases[:20]}")
    if not joints_by_case:
        raise ValueError("No joints loaded")

    target_frames = int(max_frames or max(observed_lengths))
    num_cases = len(index_rows)
    joints = np.zeros((num_cases, target_frames, 22, 3), dtype=np.float32)
    frame_mask = np.zeros((num_cases, target_frames), dtype=np.float32)
    source_num_frames = np.zeros((num_cases,), dtype=np.int64)
    truncated = np.zeros((num_cases,), dtype=np.float32)
    alignment_rows: list[dict[str, Any]] = []

    mismatched_frame_count = 0
    truncated_count = 0
    for case_idx, (row, case_joints) in enumerate(zip(index_rows, joints_by_case)):
        observed = int(case_joints.shape[0])
        expected = int(row.get("num_frames") or 0)
        source_num_frames[case_idx] = observed
        if observed != expected or observed != int(condition_num_frames[case_idx]):
            mismatched_frame_count += 1
        keep = min(observed, target_frames)
        joints[case_idx, :keep] = case_joints[:keep]
        frame_mask[case_idx, :keep] = 1.0
        if observed > target_frames:
            truncated[case_idx] = 1.0
            truncated_count += 1
        alignment_rows.append(
            {
                "case_index": int(case_idx),
                "case_id": row["case_id"],
                "condition_case_index": int(condition_case_index[case_idx]),
                "condition_num_frames": int(condition_num_frames[case_idx]),
                "index_num_frames": expected,
                "joints_num_frames": observed,
                "stored_num_frames": keep,
                "num_selected": int(row.get("num_selected") or 0),
                "condition_mask_sum": float(condition_mask[case_idx].sum()),
                "truncated": bool(observed > target_frames),
            }
        )

    arrays = {
        "joints": joints,
        "frame_mask": frame_mask,
        "source_num_frames": source_num_frames,
        "truncated": truncated,
        "case_index": condition_case_index.astype(np.int64),
    }
    report = {
        "schema_version": "aml_condition_motion_batch_v1",
        "condition_batch_dir": str(condition_batch_dir),
        "hml_root": str(hml_root),
        "num_cases": num_cases,
        "target_frames": target_frames,
        "joints_shape": list(joints.shape),
        "frame_mask_shape": list(frame_mask.shape),
        "source_num_frames_min": int(source_num_frames.min()),
        "source_num_frames_max": int(source_num_frames.max()),
        "source_num_frames_mean": float(source_num_frames.mean()),
        "total_valid_frames": int(frame_mask.sum()),
        "mismatched_frame_count": mismatched_frame_count,
        "truncated_count": truncated_count,
        "condition_mask_shape": list(condition_mask.shape),
        "condition_real_count": int(condition_mask.sum()),
        "alignment_status": "pass" if mismatched_frame_count == 0 and truncated_count == 0 else "warn",
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
    }
    return arrays, report, alignment_rows


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AML Condition + Motion Batch Smoke",
        "",
        "## Source",
        "",
        f"- condition batch dir: `{report['condition_batch_dir']}`",
        f"- HumanML3D root: `{report['hml_root']}`",
        "",
        "## Status",
        "",
        f"- alignment status: `{report['alignment_status']}`",
        f"- cases: `{report['num_cases']}`",
        f"- target frames: `{report['target_frames']}`",
        f"- joints shape: `{report['joints_shape']}`",
        f"- frame mask shape: `{report['frame_mask_shape']}`",
        f"- source frames: min `{report['source_num_frames_min']}`, mean `{report['source_num_frames_mean']:.4f}`, max `{report['source_num_frames_max']}`",
        f"- total valid frames: `{report['total_valid_frames']}`",
        f"- mismatched frame count: `{report['mismatched_frame_count']}`",
        f"- truncated count: `{report['truncated_count']}`",
        f"- condition mask shape: `{report['condition_mask_shape']}`",
        f"- condition real count: `{report['condition_real_count']}`",
        "",
        "## Arrays",
        "",
        "| name | shape | dtype |",
        "| --- | --- | --- |",
    ]
    for name, shape in report["array_shapes"].items():
        lines.append(f"| {name} | {shape} | {report['array_dtypes'][name]} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `condition_motion_batch.npz`: padded HumanML3D joints and frame masks.",
            "- `condition_motion_alignment.json`: machine-readable alignment summary.",
            "- `condition_motion_alignment.jsonl`: per-case alignment rows.",
            "- `condition_motion_report.md`: this report.",
            "",
            "This is a smoke artifact only. It does not train a model and does not alter the condition batch schema.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hml-root", default=str(DEFAULT_HML_ROOT))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    condition_batch_dir = Path(args.condition_batch_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, report, alignment_rows = export_motion_batch(
        condition_batch_dir=condition_batch_dir,
        hml_root=Path(args.hml_root),
        max_frames=args.max_frames,
    )

    np.savez_compressed(out_dir / "condition_motion_batch.npz", **arrays)
    (out_dir / "condition_motion_alignment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "condition_motion_alignment.jsonl", alignment_rows)
    write_report(out_dir / "condition_motion_report.md", report)

    print(f"saved={out_dir}")
    print(
        "cases={num_cases} joints_shape={joints_shape} valid_frames={total_valid_frames} "
        "mismatch={mismatched_frame_count} truncated={truncated_count} status={alignment_status}".format(**report)
    )


if __name__ == "__main__":
    main()
