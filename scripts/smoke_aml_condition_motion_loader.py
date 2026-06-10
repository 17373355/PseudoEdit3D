from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.data import AMLConditionMotionDataset, collate_aml_condition_motion_samples


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _batch_tensor_summary(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _tensor_summary(value)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AML Condition + Motion DataLoader Smoke",
        "",
        "## Inputs",
        "",
        f"- condition batch dir: `{report['condition_batch_dir']}`",
        f"- motion batch dir: `{report['motion_batch_dir']}`",
        "",
        "## Status",
        "",
        f"- status: `{report['status']}`",
        f"- dataset length: `{report['dataset_length']}`",
        f"- batch size: `{report['batch_size']}`",
        f"- checked batches: `{report['checked_batches']}`",
        f"- checked samples: `{report['checked_samples']}`",
        f"- condition count from masks: `{report['condition_count_from_masks']}`",
        f"- valid frame count from masks: `{report['valid_frame_count_from_masks']}`",
        f"- mask mismatches: `{report['mask_mismatches']}`",
        "",
        "## First Batch Tensor Shapes",
        "",
        "| key | shape | dtype |",
        "| --- | --- | --- |",
    ]
    for key, item in report.get("first_batch_tensors", {}).items():
        lines.append(f"| {key} | {item['shape']} | {item['dtype']} |")
    lines.extend(["", "## Example Cases", ""])
    lines.append("| case_id | source_num_frames | num_selected | selected_families |")
    lines.append("| --- | --- | --- | --- |")
    for item in report.get("example_cases", []):
        lines.append(
            f"| {item['case_id']} | {item['source_num_frames']} | {item['num_selected']} | {', '.join(item['selected_families'])} |"
        )
    lines.extend(
        [
            "",
            "This smoke only checks dataset and DataLoader behavior. It does not instantiate or train a model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dataset = AMLConditionMotionDataset(
        condition_batch_dir=args.condition_batch_dir,
        motion_batch_dir=args.motion_batch_dir,
        max_cases=args.max_cases,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_aml_condition_motion_samples,
    )
    checked_batches = 0
    checked_samples = 0
    condition_count = 0
    valid_frame_count = 0
    mask_mismatches: list[dict[str, Any]] = []
    first_batch_tensors: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        if first_batch_tensors is None:
            first_batch_tensors = _batch_tensor_summary(batch)
        batch_size = int(batch["case_index"].shape[0])
        checked_batches += 1
        checked_samples += batch_size
        condition_count += int(batch["condition_mask"].sum().item())
        valid_frame_count += int(batch["frame_mask"].sum().item())

        num_selected = batch["num_selected"].long()
        condition_mask_sum = batch["condition_mask"].sum(dim=1).long()
        source_num_frames = batch["source_num_frames"].long()
        frame_mask_sum = batch["frame_mask"].sum(dim=1).long()
        for row_idx in range(batch_size):
            if int(num_selected[row_idx]) != int(condition_mask_sum[row_idx]):
                mask_mismatches.append(
                    {
                        "case_id": batch["case_id"][row_idx],
                        "issue": "num_selected_vs_condition_mask",
                        "num_selected": int(num_selected[row_idx]),
                        "condition_mask_sum": int(condition_mask_sum[row_idx]),
                    }
                )
            if int(source_num_frames[row_idx]) != int(frame_mask_sum[row_idx]):
                mask_mismatches.append(
                    {
                        "case_id": batch["case_id"][row_idx],
                        "issue": "source_num_frames_vs_frame_mask",
                        "source_num_frames": int(source_num_frames[row_idx]),
                        "frame_mask_sum": int(frame_mask_sum[row_idx]),
                    }
                )
            if len(examples) < args.example_cases:
                examples.append(
                    {
                        "case_id": batch["case_id"][row_idx],
                        "source_num_frames": int(source_num_frames[row_idx]),
                        "num_selected": int(num_selected[row_idx]),
                        "selected_families": list(batch["selected_families"][row_idx]),
                    }
                )
        if args.max_batches and checked_batches >= args.max_batches:
            break

    status = "pass" if not mask_mismatches else "warn"
    return {
        "schema_version": "aml_condition_motion_loader_smoke_v1",
        "condition_batch_dir": args.condition_batch_dir,
        "motion_batch_dir": args.motion_batch_dir,
        "dataset_length": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "checked_batches": checked_batches,
        "checked_samples": checked_samples,
        "condition_count_from_masks": condition_count,
        "valid_frame_count_from_masks": valid_frame_count,
        "mask_mismatches": mask_mismatches,
        "status": status,
        "first_batch_tensors": first_batch_tensors or {},
        "example_cases": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-batch-dir", required=True)
    parser.add_argument("--motion-batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--example-cases", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_smoke(args)
    _write_json(out_dir / "loader_smoke.json", report)
    _write_report(out_dir / "loader_smoke.md", report)
    print(f"saved={out_dir}")
    print(
        "status={status} dataset={dataset_length} batches={checked_batches} samples={checked_samples} "
        "conditions={condition_count_from_masks} frames={valid_frame_count_from_masks}".format(**report)
    )


if __name__ == "__main__":
    main()
