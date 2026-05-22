from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import SMPLH_NUM_JOINTS, SMPLH_POSE_DIM
from pseudoedit3d.edit.action_program import build_goal_spec, goal_spec_to_numpy
from pseudoedit3d.edit import verbalize_program
from pseudoedit3d.edit.schema import EditProgram, load_label_schema
from pseudoedit3d.edit.synthetic import build_synthetic_edit_sample
from pseudoedit3d.text import CharTokenizer


def _iter_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class MotionEditDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        manifest_path: str,
        contact_filter: str = "any",
        max_clips: int = 0,
        delta_scale_deg: float = 15.0,
        label_schema_path: str = "",
        prompt_style: str = "template",
        prompt_max_length: int = 96,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.contact_filter = contact_filter
        self.delta_scale_deg = delta_scale_deg
        self.schema = load_label_schema(label_schema_path) if label_schema_path else load_label_schema()
        self.prompt_style = prompt_style
        self.tokenizer = CharTokenizer(max_length=prompt_max_length)
        self.records = self._load_records()
        if max_clips > 0:
            self.records = self.records[:max_clips]

    def _load_records(self) -> list[dict]:
        records = []
        for record in _iter_manifest(self.manifest_path):
            if "error" in record:
                continue
            if record.get("poses_shape") != [60, SMPLH_POSE_DIM]:
                continue
            if record.get("trans_shape") != [60, 3]:
                continue
            bucket = record.get("contact_bucket", "unknown")
            if self.contact_filter != "any" and bucket != self.contact_filter:
                continue
            records.append(record)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        npz_path = Path(record["path"])
        data = np.load(npz_path, allow_pickle=True)
        poses = data["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
        trans = data["trans"].astype(np.float32)
        contact_mask = data["ground_contact_mask"].astype(np.float32) if "ground_contact_mask" in data.files else None
        sample = build_synthetic_edit_sample(
            poses=poses,
            trans=trans,
            contact_mask=contact_mask,
            delta_scale_deg=self.delta_scale_deg,
            label_schema_path=self.schema.path,
        )
        sample["source_path"] = str(npz_path)
        program = EditProgram.from_dict(sample["program"])
        goal_spec = sample.get("goal_spec", goal_spec_to_numpy(build_goal_spec(program, schema=self.schema)))
        prompt_text = verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index)
        prompt_token_ids, prompt_attention_mask = self.tokenizer.encode(prompt_text)
        output = {
            "source_pose": torch.from_numpy(sample["source_pose"]),
            "target_pose": torch.from_numpy(sample["target_pose"]),
            "source_trans": torch.from_numpy(sample["source_trans"]),
            "target_trans": torch.from_numpy(sample["target_trans"]),
            "joint_mask": torch.from_numpy(sample["joint_mask"]),
            "time_mask": torch.from_numpy(sample["time_mask"]),
            "edit_vector": torch.from_numpy(sample["edit_vector"]),
            "prompt_token_ids": torch.from_numpy(prompt_token_ids),
            "prompt_attention_mask": torch.from_numpy(prompt_attention_mask),
            "prompt_text": prompt_text,
        }
        for key, value in goal_spec.items():
            output[key] = torch.from_numpy(value)
        return output
