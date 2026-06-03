from __future__ import annotations

import torch

from pseudoedit3d.edit.attributes import extract_upper_body_proxy_attributes_torch, stack_proxy_attributes_torch


def _span_mask(start_frame: torch.Tensor, end_frame: torch.Tensor, num_frames: int) -> torch.Tensor:
    frame_ids = torch.arange(num_frames, device=start_frame.device).unsqueeze(0)
    return ((frame_ids >= start_frame.unsqueeze(1)) & (frame_ids <= end_frame.unsqueeze(1))).float()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom


def compute_goal_satisfaction_losses(
    source_pose: torch.Tensor,
    pred_pose: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source_pose = source_pose.view(source_pose.shape[0], source_pose.shape[1], -1, 3)
    pred_pose = pred_pose.view(pred_pose.shape[0], pred_pose.shape[1], -1, 3)

    source_trans = torch.zeros((source_pose.shape[0], source_pose.shape[1], 3), device=source_pose.device, dtype=source_pose.dtype)
    pred_trans = torch.zeros((pred_pose.shape[0], pred_pose.shape[1], 3), device=pred_pose.device, dtype=pred_pose.dtype)
    source_attrs = stack_proxy_attributes_torch(extract_upper_body_proxy_attributes_torch(source_pose, trans=source_trans))
    pred_attrs = stack_proxy_attributes_torch(extract_upper_body_proxy_attributes_torch(pred_pose, trans=pred_trans))

    goal_attr_idx = batch["goal_attr_idx"].long().to(pred_pose.device)
    goal_operator_idx = batch["goal_operator_idx"].long().to(pred_pose.device)
    goal_start = batch["goal_start_frame"].long().to(pred_pose.device)
    goal_end = batch["goal_end_frame"].long().to(pred_pose.device)
    goal_delta_deg = batch["goal_delta_deg"].float().to(pred_pose.device)
    goal_target_value_deg = batch["goal_target_value_deg"].float().to(pred_pose.device)
    goal_source_attr_amplitude_deg = batch["goal_source_attr_amplitude_deg"].float().to(pred_pose.device)
    goal_target_offset_deg = batch["goal_target_offset_deg"].float().to(pred_pose.device)
    goal_preserve_amplitude = batch["goal_preserve_amplitude"].float().to(pred_pose.device)
    goal_direction_sign = batch["goal_direction_sign"].float().to(pred_pose.device)
    goal_tolerance_deg = batch["goal_tolerance_deg"].float().to(pred_pose.device)

    gather_idx = goal_attr_idx.view(-1, 1, 1).expand(-1, source_attrs.shape[1], 1)
    source_goal = torch.gather(source_attrs, dim=2, index=gather_idx).squeeze(-1)
    pred_goal = torch.gather(pred_attrs, dim=2, index=gather_idx).squeeze(-1)

    mask = _span_mask(goal_start, goal_end, source_goal.shape[1])
    source_mean = _masked_mean(source_goal, mask)
    pred_mean = _masked_mean(pred_goal, mask)
    pred_delta = pred_mean - source_mean
    source_max = torch.where(mask > 0.0, source_goal, torch.full_like(source_goal, float("-inf"))).max(dim=1).values
    source_min = torch.where(mask > 0.0, source_goal, torch.full_like(source_goal, float("inf"))).min(dim=1).values
    pred_max = torch.where(mask > 0.0, pred_goal, torch.full_like(pred_goal, float("-inf"))).max(dim=1).values
    pred_min = torch.where(mask > 0.0, pred_goal, torch.full_like(pred_goal, float("inf"))).min(dim=1).values
    source_amp = 0.5 * (source_max - source_min)
    pred_amp = 0.5 * (pred_max - pred_min)

    is_set = (goal_operator_idx == 1).float()
    target_absolute = torch.where(torch.isfinite(goal_target_value_deg), goal_target_value_deg, source_mean + goal_delta_deg)
    target_delta = torch.where(is_set > 0.5, target_absolute - source_mean, goal_delta_deg)
    target_mean = torch.where(torch.isfinite(goal_target_offset_deg), goal_target_offset_deg, source_mean + target_delta)
    scale = torch.maximum(target_delta.abs(), goal_tolerance_deg).clamp_min(1.0)

    delta_loss = (torch.abs(pred_delta - target_delta) / scale).mean()
    direction_margin = 1.0
    direction_progress = goal_direction_sign * pred_delta / scale
    direction_loss = torch.relu(direction_margin - direction_progress).mean()
    tolerance_loss = (torch.relu(torch.abs(pred_delta - target_delta) - goal_tolerance_deg) / goal_tolerance_deg.clamp_min(1.0)).mean()

    pred_delta_seq = pred_goal - source_goal
    span_center_loss = _masked_mean(torch.abs(pred_delta_seq - target_delta.unsqueeze(1)) / scale.unsqueeze(1), mask).mean()
    offset_loss = (torch.abs(pred_mean - target_mean) / scale).mean()
    safe_source_amp = torch.nan_to_num(goal_source_attr_amplitude_deg, nan=0.0, posinf=0.0, neginf=0.0)
    amp_scale = torch.maximum(safe_source_amp.abs(), goal_tolerance_deg).clamp_min(1.0)
    amplitude_mask = goal_preserve_amplitude * torch.isfinite(goal_source_attr_amplitude_deg).float()
    amplitude_error = torch.abs(pred_amp - safe_source_amp) / amp_scale
    amplitude_preserve_loss = (amplitude_mask * amplitude_error).sum() / amplitude_mask.sum().clamp_min(1.0)

    other_source = source_attrs.clone()
    other_pred = pred_attrs.clone()
    other_mask = torch.ones((source_attrs.shape[0], source_attrs.shape[2]), device=pred_pose.device, dtype=pred_pose.dtype)
    other_mask.scatter_(1, goal_attr_idx.unsqueeze(1), 0.0)
    other_delta = (other_pred - other_source).abs().mean(dim=1)
    preserve_attr_loss = (other_delta * other_mask).sum(dim=1) / other_mask.sum(dim=1).clamp_min(1.0)
    preserve_attr_loss = preserve_attr_loss / scale
    preserve_attr_loss = preserve_attr_loss.mean()

    return {
        "goal_delta_loss": delta_loss,
        "goal_direction_loss": direction_loss,
        "goal_tolerance_loss": tolerance_loss,
        "goal_span_consistency_loss": span_center_loss,
        "goal_offset_loss": offset_loss,
        "goal_amplitude_preserve_loss": amplitude_preserve_loss,
        "goal_preserve_attr_loss": preserve_attr_loss,
    }
