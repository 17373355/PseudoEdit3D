from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.constants import SMPLH_POSE_DIM
from pseudoedit3d.data import MinedMotionEditDataset, MotionEditDataset
from pseudoedit3d.edit.schema import load_label_schema
from pseudoedit3d.models import MaskedMotionEditor
from pseudoedit3d.text import CharTokenizer
from pseudoedit3d.training.goal_losses import compute_goal_satisfaction_losses


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_error = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_error.sum() / denom


def build_dataset(cfg):
    if cfg.data_mode == "mined":
        return MinedMotionEditDataset(
            pair_manifest_path=cfg.pair_manifest_path,
            max_pairs=cfg.max_clips,
            label_schema_path=cfg.label_schema_path,
            prompt_style=cfg.prompt_style,
            prompt_max_length=cfg.prompt_max_length,
            input_source_mode=cfg.input_source_mode,
            source_prefix_frames=cfg.source_prefix_frames,
        )
    return MotionEditDataset(
        dataset_root=cfg.dataset_root,
        manifest_path=cfg.manifest_path,
        contact_filter=cfg.contact_filter,
        max_clips=cfg.max_clips,
        delta_scale_deg=cfg.delta_scale_deg,
        label_schema_path=cfg.label_schema_path,
        prompt_style=cfg.prompt_style,
        prompt_max_length=cfg.prompt_max_length,
    )


def build_model(cfg, device):
    tokenizer = CharTokenizer(max_length=cfg.prompt_max_length)
    schema = load_label_schema(cfg.label_schema_path) if cfg.label_schema_path else load_label_schema()
    return MaskedMotionEditor(
        pose_dim=SMPLH_POSE_DIM,
        edit_dim=schema.vector_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        text_vocab_size=tokenizer.vocab_size if cfg.condition_mode in {"text", "hybrid"} else 0,
    ).to(device)


def _build_condition_inputs(batch, device, condition_mode: str) -> dict:
    condition = {}
    if condition_mode in {"program", "hybrid"}:
        condition["edit_vector"] = batch["edit_vector"].to(device)
    else:
        condition["edit_vector"] = None
    if condition_mode in {"text", "hybrid"}:
        condition["prompt_token_ids"] = batch["prompt_token_ids"].to(device)
        condition["prompt_attention_mask"] = batch["prompt_attention_mask"].to(device)
    else:
        condition["prompt_token_ids"] = None
        condition["prompt_attention_mask"] = None
    return condition


def train_one_epoch(model, loader, optimizer, device, condition_mode: str, cfg, writer=None, global_step_start: int = 0):
    model.train()
    sums = {
        "loss": 0.0,
        "edit_loss": 0.0,
        "keep_loss": 0.0,
        "smooth_loss": 0.0,
        "condition_loss": 0.0,
        "goal_delta_loss": 0.0,
        "goal_direction_loss": 0.0,
        "goal_tolerance_loss": 0.0,
        "goal_span_consistency_loss": 0.0,
        "goal_offset_loss": 0.0,
        "goal_amplitude_preserve_loss": 0.0,
        "goal_preserve_attr_loss": 0.0,
    }
    global_step = global_step_start
    for batch_idx, batch in enumerate(loader):
        source_pose = batch["source_pose"].to(device).reshape(batch["source_pose"].shape[0], batch["source_pose"].shape[1], -1)
        target_pose = batch["target_pose"].to(device).reshape(batch["target_pose"].shape[0], batch["target_pose"].shape[1], -1)
        joint_mask = batch["joint_mask"].to(device).unsqueeze(-1).expand(-1, -1, -1, 3).reshape_as(source_pose)
        conditioning_frame_mask = batch["conditioning_frame_mask"].to(device).unsqueeze(-1).expand_as(source_pose)
        if cfg.input_source_mode == "source_motion":
            keep_mask = 1.0 - joint_mask
        else:
            keep_mask = (1.0 - joint_mask) * conditioning_frame_mask
        condition = _build_condition_inputs(batch, device, condition_mode)

        pred_pose = model(source_pose, **condition)
        edit_loss = _masked_l1(pred_pose, target_pose, joint_mask)
        keep_loss = _masked_l1(pred_pose, source_pose, keep_mask)
        condition_loss = _masked_l1(pred_pose, source_pose, conditioning_frame_mask)
        smooth_loss = (pred_pose[:, 1:] - pred_pose[:, :-1]).abs().mean()
        loss = edit_loss + cfg.lambda_keep * keep_loss + cfg.lambda_condition * condition_loss + cfg.lambda_smooth * smooth_loss

        goal_loss_terms = None
        if cfg.use_goal_satisfaction_loss:
            goal_loss_terms = compute_goal_satisfaction_losses(source_pose, pred_pose, batch)
            loss = (
                loss
                + cfg.lambda_goal_delta * goal_loss_terms["goal_delta_loss"]
                + cfg.lambda_goal_direction * goal_loss_terms["goal_direction_loss"]
                + cfg.lambda_goal_tolerance * goal_loss_terms["goal_tolerance_loss"]
                + cfg.lambda_goal_span * goal_loss_terms["goal_span_consistency_loss"]
                + cfg.lambda_goal_offset * goal_loss_terms["goal_offset_loss"]
                + cfg.lambda_goal_amplitude * goal_loss_terms["goal_amplitude_preserve_loss"]
                + cfg.lambda_goal_preserve_attr * goal_loss_terms["goal_preserve_attr_loss"]
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        edit_loss_value = float(edit_loss.item())
        keep_loss_value = float(keep_loss.item())
        condition_loss_value = float(condition_loss.item())
        smooth_loss_value = float(smooth_loss.item())
        sums["loss"] += loss_value
        sums["edit_loss"] += edit_loss_value
        sums["keep_loss"] += keep_loss_value
        sums["condition_loss"] += condition_loss_value
        sums["smooth_loss"] += smooth_loss_value
        if goal_loss_terms is not None:
            for key in [
                "goal_delta_loss",
                "goal_direction_loss",
                "goal_tolerance_loss",
                "goal_span_consistency_loss",
                "goal_offset_loss",
                "goal_amplitude_preserve_loss",
                "goal_preserve_attr_loss",
            ]:
                value = float(goal_loss_terms[key].item())
                sums[key] += value
        if writer is not None:
            writer.add_scalar("train_step/loss", loss_value, global_step)
            writer.add_scalar("train_step/edit_loss", edit_loss_value, global_step)
            writer.add_scalar("train_step/keep_loss", keep_loss_value, global_step)
            writer.add_scalar("train_step/condition_loss", condition_loss_value, global_step)
            writer.add_scalar("train_step/smooth_loss", smooth_loss_value, global_step)
            if goal_loss_terms is not None:
                writer.add_scalar("train_step/goal_delta_loss", float(goal_loss_terms["goal_delta_loss"].item()), global_step)
                writer.add_scalar(
                    "train_step/goal_direction_loss", float(goal_loss_terms["goal_direction_loss"].item()), global_step
                )
                writer.add_scalar(
                    "train_step/goal_tolerance_loss", float(goal_loss_terms["goal_tolerance_loss"].item()), global_step
                )
                writer.add_scalar(
                    "train_step/goal_span_consistency_loss",
                    float(goal_loss_terms["goal_span_consistency_loss"].item()),
                    global_step,
                )
                writer.add_scalar("train_step/goal_offset_loss", float(goal_loss_terms["goal_offset_loss"].item()), global_step)
                writer.add_scalar(
                    "train_step/goal_amplitude_preserve_loss",
                    float(goal_loss_terms["goal_amplitude_preserve_loss"].item()),
                    global_step,
                )
                writer.add_scalar(
                    "train_step/goal_preserve_attr_loss",
                    float(goal_loss_terms["goal_preserve_attr_loss"].item()),
                    global_step,
                )
        global_step += 1
    denom = max(len(loader), 1)
    averages = {key: value / denom for key, value in sums.items()}
    return averages, global_step


def train_from_config(cfg, checkpoint_name: str = "stage1_last.pt") -> dict:
    _set_seed(cfg.seed)
    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.txt"
    tensorboard_dir = save_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    last_loss = None
    global_step = 0

    for epoch in range(cfg.epochs):
        epoch_metrics, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            cfg.condition_mode,
            cfg,
            writer=writer,
            global_step_start=global_step,
        )
        last_loss = epoch_metrics["loss"]
        line = (
            f"epoch={epoch} loss={epoch_metrics['loss']:.6f} "
            f"edit_loss={epoch_metrics['edit_loss']:.6f} "
            f"keep_loss={epoch_metrics['keep_loss']:.6f} "
            f"condition_loss={epoch_metrics['condition_loss']:.6f} "
            f"smooth_loss={epoch_metrics['smooth_loss']:.6f}"
        )
        if cfg.use_goal_satisfaction_loss:
            line += (
                f" goal_delta_loss={epoch_metrics['goal_delta_loss']:.6f}"
                f" goal_direction_loss={epoch_metrics['goal_direction_loss']:.6f}"
                f" goal_tolerance_loss={epoch_metrics['goal_tolerance_loss']:.6f}"
                f" goal_span_consistency_loss={epoch_metrics['goal_span_consistency_loss']:.6f}"
                f" goal_offset_loss={epoch_metrics['goal_offset_loss']:.6f}"
                f" goal_amplitude_preserve_loss={epoch_metrics['goal_amplitude_preserve_loss']:.6f}"
                f" goal_preserve_attr_loss={epoch_metrics['goal_preserve_attr_loss']:.6f}"
            )
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        writer.add_scalar("train_epoch/loss", epoch_metrics["loss"], epoch)
        writer.add_scalar("train_epoch/edit_loss", epoch_metrics["edit_loss"], epoch)
        writer.add_scalar("train_epoch/keep_loss", epoch_metrics["keep_loss"], epoch)
        writer.add_scalar("train_epoch/condition_loss", epoch_metrics["condition_loss"], epoch)
        writer.add_scalar("train_epoch/smooth_loss", epoch_metrics["smooth_loss"], epoch)
        if cfg.use_goal_satisfaction_loss:
            writer.add_scalar("train_epoch/goal_delta_loss", epoch_metrics["goal_delta_loss"], epoch)
            writer.add_scalar("train_epoch/goal_direction_loss", epoch_metrics["goal_direction_loss"], epoch)
            writer.add_scalar("train_epoch/goal_tolerance_loss", epoch_metrics["goal_tolerance_loss"], epoch)
            writer.add_scalar(
                "train_epoch/goal_span_consistency_loss", epoch_metrics["goal_span_consistency_loss"], epoch
            )
            writer.add_scalar("train_epoch/goal_offset_loss", epoch_metrics["goal_offset_loss"], epoch)
            writer.add_scalar(
                "train_epoch/goal_amplitude_preserve_loss", epoch_metrics["goal_amplitude_preserve_loss"], epoch
            )
            writer.add_scalar("train_epoch/goal_preserve_attr_loss", epoch_metrics["goal_preserve_attr_loss"], epoch)

    checkpoint_path = save_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    writer.flush()
    writer.close()
    return {
        "checkpoint_path": str(checkpoint_path),
        "save_dir": str(save_dir),
        "last_loss": float(last_loss) if last_loss is not None else None,
        "num_samples": len(dataset),
        "tensorboard_dir": str(tensorboard_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
