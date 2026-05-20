from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.constants import SMPLH_POSE_DIM
from pseudoedit3d.data import MinedMotionEditDataset, MotionEditDataset
from pseudoedit3d.models import MaskedMotionEditor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_error = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return masked_error.sum() / denom


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        source_pose = batch["source_pose"].to(device).reshape(batch["source_pose"].shape[0], batch["source_pose"].shape[1], -1)
        target_pose = batch["target_pose"].to(device).reshape(batch["target_pose"].shape[0], batch["target_pose"].shape[1], -1)
        edit_vector = batch["edit_vector"].to(device)
        joint_mask = batch["joint_mask"].to(device).unsqueeze(-1).expand(-1, -1, -1, 3).reshape_as(source_pose)
        keep_mask = 1.0 - joint_mask

        pred_pose = model(source_pose, edit_vector)
        edit_loss = _masked_l1(pred_pose, target_pose, joint_mask)
        keep_loss = _masked_l1(pred_pose, source_pose, keep_mask)
        smooth_loss = (pred_pose[:, 1:] - pred_pose[:, :-1]).abs().mean()
        loss = edit_loss + 0.5 * keep_loss + 0.01 * smooth_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    _set_seed(cfg.seed)

    if cfg.data_mode == "mined":
        dataset = MinedMotionEditDataset(
            pair_manifest_path=cfg.pair_manifest_path,
            max_pairs=cfg.max_clips,
        )
    else:
        dataset = MotionEditDataset(
            dataset_root=cfg.dataset_root,
            manifest_path=cfg.manifest_path,
            contact_filter=cfg.contact_filter,
            max_clips=cfg.max_clips,
            delta_scale_deg=cfg.delta_scale_deg,
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedMotionEditor(
        pose_dim=SMPLH_POSE_DIM,
        edit_dim=19,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_log.txt"

    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        line = f"epoch={epoch} loss={loss:.6f}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.save(model.state_dict(), save_dir / "stage1_last.pt")


if __name__ == "__main__":
    main()
