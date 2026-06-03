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
        text_vocab_size: int = 0,
        max_frames: int = 60,
    ) -> None:
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.edit_proj = nn.Linear(edit_dim, hidden_dim)
        self.text_embed = nn.Embedding(text_vocab_size, hidden_dim) if text_vocab_size > 0 else None
        self.text_proj = nn.Linear(hidden_dim, hidden_dim) if text_vocab_size > 0 else None
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.frame_type_embed = nn.Embedding(2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, pose_dim)

    def forward(
        self,
        source_pose: torch.Tensor,
        edit_vector: torch.Tensor | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        conditioning_frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_frames, pose_dim = source_pose.shape
        pose_feat = self.pose_proj(source_pose)
        if num_frames > self.pos_embed.shape[1]:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.pos_embed.shape[1]}")

        cond_feat = self.pos_embed[:, :num_frames].expand(batch_size, -1, -1)

        if conditioning_frame_mask is not None:
            frame_mask = conditioning_frame_mask
            if frame_mask.dim() == 3:
                frame_mask = frame_mask[..., 0]
            frame_type_ids = (frame_mask > 0.5).long()
            cond_feat = cond_feat + self.frame_type_embed(frame_type_ids)

        if edit_vector is not None:
            cond_feat = cond_feat + self.edit_proj(edit_vector).unsqueeze(1).expand(batch_size, num_frames, -1)

        if prompt_token_ids is not None and self.text_embed is not None:
            text_feat = self.text_embed(prompt_token_ids)
            if prompt_attention_mask is not None:
                mask = prompt_attention_mask.unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = (text_feat * mask).sum(dim=1) / denom
            else:
                pooled = text_feat.mean(dim=1)
            pooled = self.text_proj(pooled).unsqueeze(1).expand(batch_size, num_frames, -1)
            cond_feat = cond_feat + pooled

        if edit_vector is None and prompt_token_ids is None:
            raise ValueError("At least one conditioning input must be provided")

        hidden = self.encoder(pose_feat + cond_feat)
        residual = self.out_proj(hidden)
        pred = source_pose + residual
        if conditioning_frame_mask is not None:
            mask = conditioning_frame_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            pred = pred * (1.0 - mask) + source_pose * mask
        return pred
