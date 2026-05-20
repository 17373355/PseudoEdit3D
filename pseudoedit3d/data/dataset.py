from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import SMPLH_NUM_JOINTS, SMPLH_POSE_DIM
from pseudoedit3d.edit.synthetic import build_synthetic_edit_sample


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
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        self.contact_filter = contact_filter
        self.delta_scale_deg = delta_scale_deg
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
        )
        sample["source_path"] = str(npz_path)
        return {
            "source_pose": torch.from_numpy(sample["source_pose"]),
            "target_pose": torch.from_numpy(sample["target_pose"]),
            "source_trans": torch.from_numpy(sample["source_trans"]),
            "target_trans": torch.from_numpy(sample["target_trans"]),
            "joint_mask": torch.from_numpy(sample["joint_mask"]),
            "time_mask": torch.from_numpy(sample["time_mask"]),
            "edit_vector": torch.from_numpy(sample["edit_vector"]),
        }
