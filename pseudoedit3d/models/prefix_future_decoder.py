from __future__ import annotations

import torch
from torch import nn


class PrefixFutureDecoder(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        edit_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        text_vocab_size: int = 0,
        max_frames: int = 60,
        base_step_scale: float = 0.01,
        active_step_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.base_step_scale = float(base_step_scale)
        self.active_step_scale = float(active_step_scale)
        self.pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.last_pose_proj = nn.Linear(pose_dim, hidden_dim)
        self.edit_proj = nn.Linear(edit_dim, hidden_dim)
        self.seq_edit_proj = nn.Linear(edit_dim, hidden_dim)
        self.time_proj = nn.Linear(6, hidden_dim)
        self.text_embed = nn.Embedding(text_vocab_size, hidden_dim) if text_vocab_size > 0 else None
        self.text_proj = nn.Linear(hidden_dim, hidden_dim) if text_vocab_size > 0 else None
        self.prefix_pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.future_pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.query_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.frame_type_embed = nn.Embedding(2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.delta_out = nn.Linear(hidden_dim, pose_dim)
        nn.init.normal_(self.prefix_pos_embed, std=0.02)
        nn.init.normal_(self.future_pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)

    def _pooled_text_condition(self, prompt_token_ids: torch.Tensor | None, prompt_attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if prompt_token_ids is None or self.text_embed is None:
            return None
        text_feat = self.text_embed(prompt_token_ids)
        if prompt_attention_mask is not None:
            mask = prompt_attention_mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (text_feat * mask).sum(dim=1) / denom
        else:
            pooled = text_feat.mean(dim=1)
        return self.text_proj(pooled)

    def forward(
        self,
        source_pose: torch.Tensor,
        edit_vector: torch.Tensor | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        conditioning_frame_mask: torch.Tensor | None = None,
        goal_start_frame: torch.Tensor | None = None,
        goal_end_frame: torch.Tensor | None = None,
        goal_delta_deg: torch.Tensor | None = None,
        goal_direction_sign: torch.Tensor | None = None,
        joint_mask: torch.Tensor | None = None,
        seq_edit_vectors: torch.Tensor | None = None,
        seq_start_frames: torch.Tensor | None = None,
        seq_end_frames: torch.Tensor | None = None,
        seq_delta_deg: torch.Tensor | None = None,
        seq_direction_sign: torch.Tensor | None = None,
        seq_num_edits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditioning_frame_mask is None:
            raise ValueError('PrefixFutureDecoder requires conditioning_frame_mask')
        if edit_vector is None and prompt_token_ids is None:
            raise ValueError('At least one conditioning input must be provided')

        batch_size, num_frames, _ = source_pose.shape
        if num_frames > self.max_frames:
            raise ValueError(f'num_frames={num_frames} exceeds max_frames={self.max_frames}')

        frame_mask = conditioning_frame_mask
        if frame_mask.dim() == 3:
            frame_mask = frame_mask[..., 0]
        prefix_lens = frame_mask.sum(dim=1).round().long()
        if not torch.all(prefix_lens == prefix_lens[0]):
            raise ValueError('PrefixFutureDecoder expects consistent prefix length in batch')
        prefix_len = int(prefix_lens[0].item())
        future_len = num_frames - prefix_len
        if future_len <= 0:
            return source_pose

        cond_global = torch.zeros((batch_size, self.edit_proj.out_features), device=source_pose.device, dtype=source_pose.dtype)
        if edit_vector is not None:
            cond_global = cond_global + self.edit_proj(edit_vector)
        pooled_text = self._pooled_text_condition(prompt_token_ids, prompt_attention_mask)
        if pooled_text is not None:
            cond_global = cond_global + pooled_text

        prefix_pose = source_pose[:, :prefix_len]
        prefix_feat = self.pose_proj(prefix_pose)
        prefix_feat = prefix_feat + self.prefix_pos_embed[:, :prefix_len]
        prefix_feat = prefix_feat + self.frame_type_embed(torch.ones((batch_size, prefix_len), device=source_pose.device, dtype=torch.long))
        prefix_feat = prefix_feat + cond_global.unsqueeze(1)
        memory = self.encoder(prefix_feat)

        last_prefix_pose = prefix_pose[:, prefix_len - 1]
        future_query = self.query_embed[:, :future_len].expand(batch_size, -1, -1)
        future_query = future_query + self.future_pos_embed[:, :future_len]
        future_query = future_query + self.frame_type_embed(torch.zeros((batch_size, future_len), device=source_pose.device, dtype=torch.long))
        future_query = future_query + cond_global.unsqueeze(1)
        future_query = future_query + self.last_pose_proj(last_prefix_pose).unsqueeze(1)

        future_frame_ids = torch.arange(prefix_len, num_frames, device=source_pose.device, dtype=source_pose.dtype).view(1, future_len, 1)
        part_active = torch.ones((batch_size, future_len, 1), device=source_pose.device, dtype=source_pose.dtype)
        if joint_mask is not None:
            future_joint_mask = joint_mask[:, prefix_len:].float().to(source_pose.device)
            part_active = future_joint_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)

        if seq_edit_vectors is not None and seq_start_frames is not None and seq_end_frames is not None and seq_num_edits is not None:
            max_seq = seq_edit_vectors.shape[1]
            seq_presence = (torch.arange(max_seq, device=source_pose.device).view(1, max_seq, 1, 1) < seq_num_edits.view(batch_size, 1, 1, 1)).to(source_pose.dtype)
            seq_start = seq_start_frames.float().view(batch_size, max_seq, 1, 1)
            seq_end = seq_end_frames.float().view(batch_size, max_seq, 1, 1)
            seq_delta_mag = seq_delta_deg.abs().view(batch_size, max_seq, 1, 1) / 45.0 if seq_delta_deg is not None else torch.zeros((batch_size, max_seq, 1, 1), device=source_pose.device, dtype=source_pose.dtype)
            seq_delta_sign = seq_direction_sign.view(batch_size, max_seq, 1, 1) if seq_direction_sign is not None else torch.zeros((batch_size, max_seq, 1, 1), device=source_pose.device, dtype=source_pose.dtype)
            frame_ids = future_frame_ids.view(1, 1, future_len, 1)
            seq_active = ((frame_ids >= seq_start) & (frame_ids <= seq_end)).to(source_pose.dtype)
            start_norm = (frame_ids - seq_start) / max(num_frames - 1, 1)
            end_norm = (frame_ids - seq_end) / max(num_frames - 1, 1)
            seq_part_active = part_active.unsqueeze(1).expand(-1, max_seq, -1, -1)
            seq_time_feat = torch.cat([start_norm, end_norm, seq_active, seq_delta_mag.expand(-1, -1, future_len, -1), seq_delta_sign.expand(-1, -1, future_len, -1), seq_part_active], dim=-1)
            seq_time_emb = self.time_proj(seq_time_feat.reshape(batch_size * max_seq * future_len, -1)).view(batch_size, max_seq, future_len, -1)
            seq_edit_emb = self.seq_edit_proj(seq_edit_vectors).unsqueeze(2)
            future_query = future_query + ((seq_time_emb + seq_edit_emb) * seq_presence).sum(dim=1)
            active_gate = (seq_active * seq_presence).amax(dim=1)
        else:
            if goal_start_frame is not None and goal_end_frame is not None:
                start = goal_start_frame.float().view(batch_size, 1)
                end = goal_end_frame.float().view(batch_size, 1)
            elif edit_vector is not None:
                start = edit_vector[:, -2] * 59.0
                end = edit_vector[:, -1] * 59.0
                start = start.view(batch_size, 1)
                end = end.view(batch_size, 1)
            else:
                start = torch.full((batch_size, 1), float(prefix_len), device=source_pose.device, dtype=source_pose.dtype)
                end = torch.full((batch_size, 1), float(num_frames - 1), device=source_pose.device, dtype=source_pose.dtype)
            start_norm = (future_frame_ids - start.unsqueeze(1)) / max(num_frames - 1, 1)
            end_norm = (future_frame_ids - end.unsqueeze(1)) / max(num_frames - 1, 1)
            active_gate = ((future_frame_ids >= start.unsqueeze(1)) & (future_frame_ids <= end.unsqueeze(1))).to(source_pose.dtype)
            delta_mag = torch.zeros((batch_size, 1, 1), device=source_pose.device, dtype=source_pose.dtype) if goal_delta_deg is None else goal_delta_deg.abs().view(batch_size, 1, 1) / 45.0
            delta_sign = torch.zeros((batch_size, 1, 1), device=source_pose.device, dtype=source_pose.dtype) if goal_direction_sign is None else goal_direction_sign.view(batch_size, 1, 1)
            time_feat = torch.cat([start_norm, end_norm, active_gate, delta_mag.expand(-1, future_len, -1), delta_sign.expand(-1, future_len, -1), part_active], dim=-1)
            future_query = future_query + self.time_proj(time_feat)

        future_hidden = self.decoder(tgt=future_query, memory=memory)
        raw_delta = torch.tanh(self.delta_out(future_hidden))
        step_scale = self.base_step_scale + (self.active_step_scale - self.base_step_scale) * active_gate * part_active
        future_delta = step_scale * raw_delta
        pred_future = last_prefix_pose.unsqueeze(1) + torch.cumsum(future_delta, dim=1)
        return torch.cat([prefix_pose, pred_future], dim=1)
