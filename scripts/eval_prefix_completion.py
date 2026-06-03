import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.inference.predict import load_model_for_inference
from pseudoedit3d.training.train_stage1 import _build_condition_inputs, build_dataset


def _step_velocity_deg(seq: torch.Tensor) -> torch.Tensor:
    step = seq[:, 1:] - seq[:, :-1]
    return torch.linalg.vector_norm(step, dim=-1) * (180.0 / math.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--abs-freeze-deg", type=float, default=0.75)
    parser.add_argument("--rel-freeze-ratio", type=float, default=0.15)
    parser.add_argument("--worst-k", type=int, default=12)
    args = parser.parse_args()

    cfg, model, torch_device, _ = load_model_for_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if args.manifest:
        cfg.manifest_path = args.manifest
    if args.max_clips > 0:
        cfg.max_clips = args.max_clips

    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    records = []
    total_future_vel = 0.0
    total_target_vel = 0.0
    total_active_vel = 0.0
    total_active_target_vel = 0.0
    freeze_count = 0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            source_pose = batch["source_pose"].to(torch_device).reshape(batch["source_pose"].shape[0], batch["source_pose"].shape[1], -1)
            target_pose = batch["target_pose"].to(torch_device).reshape(batch["target_pose"].shape[0], batch["target_pose"].shape[1], -1)
            condition = _build_condition_inputs(batch, torch_device, cfg.condition_mode)
            pred_pose = model(source_pose, **condition)

            pred_pose = pred_pose.view(pred_pose.shape[0], pred_pose.shape[1], -1, 3)
            target_pose = target_pose.view(target_pose.shape[0], target_pose.shape[1], -1, 3)

            future_step_mask = (1.0 - batch["conditioning_frame_mask"][:, 1:].to(torch_device)).float()
            active_step_mask = batch["time_mask"][:, 1:].to(torch_device).float()

            pred_vel = _step_velocity_deg(pred_pose)
            target_vel = _step_velocity_deg(target_pose)

            future_den = future_step_mask.sum(dim=1).clamp_min(1.0)
            active_den = active_step_mask.sum(dim=1).clamp_min(1.0)

            pred_future_mean = (pred_vel.mean(dim=-1) * future_step_mask).sum(dim=1) / future_den
            target_future_mean = (target_vel.mean(dim=-1) * future_step_mask).sum(dim=1) / future_den
            pred_active_mean = (pred_vel.mean(dim=-1) * active_step_mask).sum(dim=1) / active_den
            target_active_mean = (target_vel.mean(dim=-1) * active_step_mask).sum(dim=1) / active_den

            freeze_thr = torch.maximum(
                torch.full_like(target_future_mean, args.abs_freeze_deg),
                target_future_mean * args.rel_freeze_ratio,
            )
            freeze_flags = pred_future_mean < freeze_thr

            batch_size = pred_pose.shape[0]
            for i in range(batch_size):
                case_idx = batch_idx * args.batch_size + i
                future_vel = float(pred_future_mean[i].item())
                target_vel_mean = float(target_future_mean[i].item())
                active_vel = float(pred_active_mean[i].item())
                active_target_vel = float(target_active_mean[i].item())
                vel_ratio = future_vel / max(target_vel_mean, 1e-6)
                active_ratio = active_vel / max(active_target_vel, 1e-6)
                frozen = bool(freeze_flags[i].item())

                total_future_vel += future_vel
                total_target_vel += target_vel_mean
                total_active_vel += active_vel
                total_active_target_vel += active_target_vel
                freeze_count += int(frozen)
                sample_count += 1

                records.append(
                    {
                        "case_idx": case_idx,
                        "source_path": batch["source_path"][i],
                        "prompt_text": batch["prompt_text"][i],
                        "future_vel_deg": future_vel,
                        "target_future_vel_deg": target_vel_mean,
                        "future_vel_ratio": vel_ratio,
                        "active_future_vel_deg": active_vel,
                        "active_target_future_vel_deg": active_target_vel,
                        "active_future_vel_ratio": active_ratio,
                        "freeze_threshold_deg": float(freeze_thr[i].item()),
                        "is_frozen": frozen,
                        "program": json.loads(batch["program_json"][i]),
                    }
                )

    records.sort(key=lambda item: item["future_vel_ratio"])
    worst_cases = records[: args.worst_k]
    frozen_cases = [item for item in records if item["is_frozen"]][: args.worst_k]

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "manifest": cfg.manifest_path,
        "num_samples": sample_count,
        "mean_future_vel_deg": total_future_vel / max(sample_count, 1),
        "mean_target_future_vel_deg": total_target_vel / max(sample_count, 1),
        "mean_future_vel_ratio": sum(item["future_vel_ratio"] for item in records) / max(sample_count, 1),
        "median_future_vel_ratio": records[sample_count // 2]["future_vel_ratio"] if records else 0.0,
        "mean_active_future_vel_deg": total_active_vel / max(sample_count, 1),
        "mean_active_target_future_vel_deg": total_active_target_vel / max(sample_count, 1),
        "mean_active_future_vel_ratio": sum(item["active_future_vel_ratio"] for item in records) / max(sample_count, 1),
        "freeze_frac": freeze_count / max(sample_count, 1),
        "freeze_count": freeze_count,
        "abs_freeze_deg": args.abs_freeze_deg,
        "rel_freeze_ratio": args.rel_freeze_ratio,
        "worst_case_indices_by_future_ratio": [item["case_idx"] for item in worst_cases],
        "frozen_case_indices": [item["case_idx"] for item in frozen_cases],
    }

    output = {
        "summary": summary,
        "worst_cases": worst_cases,
        "frozen_cases": frozen_cases,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"saved_eval={output_path}")


if __name__ == "__main__":
    main()
