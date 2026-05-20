from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS
from pseudoedit3d.edit.schema import EditProgram


class MinedMotionEditDataset(Dataset):
    def __init__(self, pair_manifest_path: str, max_pairs: int = 0) -> None:
        self.records = []
        manifest_path = Path(pair_manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        if max_pairs > 0:
            self.records = self.records[:max_pairs]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        program = EditProgram.from_dict(record["program"])
        source = np.load(record["source_path"], allow_pickle=True)
        target = np.load(record["target_path"], allow_pickle=True)

        source_pose = source["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
        target_pose = target["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
        source_trans = source["trans"].astype(np.float32)
        target_trans = target["trans"].astype(np.float32)

        joint_mask = np.zeros((source_pose.shape[0], SMPLH_NUM_JOINTS), dtype=np.float32)
        time_mask = np.zeros((source_pose.shape[0],), dtype=np.float32)
        joint_ids = BODY_PART_TO_JOINTS[program.part]
        time_mask[program.start_frame:program.end_frame + 1] = 1.0
        joint_mask[program.start_frame:program.end_frame + 1, joint_ids] = 1.0

        return {
            "source_pose": torch.from_numpy(source_pose),
            "target_pose": torch.from_numpy(target_pose),
            "source_trans": torch.from_numpy(source_trans),
            "target_trans": torch.from_numpy(target_trans),
            "joint_mask": torch.from_numpy(joint_mask),
            "time_mask": torch.from_numpy(time_mask),
            "edit_vector": torch.from_numpy(np.asarray(program.to_vector(), dtype=np.float32)),
        }
