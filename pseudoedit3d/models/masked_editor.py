from __future__ import annotations

import torch
from torch import nn


class MaskedMotionEditor(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        edit_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.edit_proj = nn.Linear(edit_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(self, source_pose: torch.Tensor, edit_vector: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, pose_dim = source_pose.shape
        pose_feat = self.pose_proj(source_pose)
        edit_feat = self.edit_proj(edit_vector).unsqueeze(1).expand(batch_size, num_frames, -1)
        hidden = self.encoder(pose_feat + edit_feat)
        residual = self.out_proj(hidden)
        return source_pose + residual
