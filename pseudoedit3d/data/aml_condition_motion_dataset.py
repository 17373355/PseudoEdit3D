from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class AMLConditionMotionDataset(Dataset):
    """Read fixed AML condition arrays aligned to padded HumanML3D joints."""

    def __init__(
        self,
        condition_batch_dir: str | Path,
        motion_batch_dir: str | Path,
        max_cases: int = 0,
    ) -> None:
        self.condition_batch_dir = Path(condition_batch_dir)
        self.motion_batch_dir = Path(motion_batch_dir)
        self.condition_schema = _load_json(self.condition_batch_dir / "condition_batch_schema.json")
        self.motion_alignment = _load_json(self.motion_batch_dir / "condition_motion_alignment.json")
        self.index_rows = _load_jsonl(self.condition_batch_dir / "condition_batch_index.jsonl")

        self.condition_npz = np.load(self.condition_batch_dir / "condition_batch.npz")
        self.motion_npz = np.load(self.motion_batch_dir / "condition_motion_batch.npz")
        self.num_cases = int(self.condition_schema["num_cases"])
        if self.num_cases != int(self.motion_alignment["num_cases"]):
            raise ValueError("condition and motion batch case counts differ")
        if self.num_cases != len(self.index_rows):
            raise ValueError("condition index row count differs from schema")
        if self.condition_npz["case_index"].shape[0] != self.motion_npz["case_index"].shape[0]:
            raise ValueError("condition and motion case_index lengths differ")
        if not np.array_equal(self.condition_npz["case_index"], self.motion_npz["case_index"]):
            raise ValueError("condition and motion case_index arrays differ")
        if not np.array_equal(self.condition_npz["num_frames"], self.motion_npz["source_num_frames"]):
            raise ValueError("condition num_frames and motion source_num_frames differ")
        self.length = self.num_cases if max_cases <= 0 else min(int(max_cases), self.num_cases)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.index_rows[index]
        sample: dict[str, Any] = {
            "case_id": str(row.get("case_id") or ""),
            "case_index": torch.tensor(int(self.condition_npz["case_index"][index]), dtype=torch.long),
            "joints": torch.from_numpy(self.motion_npz["joints"][index].astype(np.float32, copy=False)),
            "frame_mask": torch.from_numpy(self.motion_npz["frame_mask"][index].astype(np.float32, copy=False)),
            "source_num_frames": torch.tensor(int(self.motion_npz["source_num_frames"][index]), dtype=torch.long),
            "num_selected": torch.tensor(int(self.condition_npz["num_selected"][index]), dtype=torch.long),
            "condition_mask": torch.from_numpy(self.condition_npz["condition_mask"][index].astype(np.float32, copy=False)),
            "condition_action_index": torch.from_numpy(self.condition_npz["action_index"][index].astype(np.int64, copy=False)),
            "condition_family_id": torch.from_numpy(self.condition_npz["family_id"][index].astype(np.int64, copy=False)),
            "condition_status_id": torch.from_numpy(self.condition_npz["status_id"][index].astype(np.int64, copy=False)),
            "condition_score": torch.from_numpy(self.condition_npz["score"][index].astype(np.float32, copy=False)),
            "condition_weight": torch.from_numpy(self.condition_npz["condition_weight"][index].astype(np.float32, copy=False)),
            "condition_span": torch.from_numpy(self.condition_npz["span"][index].astype(np.int64, copy=False)),
            "condition_span_mask": torch.from_numpy(self.condition_npz["span_mask"][index].astype(np.float32, copy=False)),
            "condition_span_norm": torch.from_numpy(self.condition_npz["span_norm"][index].astype(np.float32, copy=False)),
            "condition_numeric_slots": torch.from_numpy(self.condition_npz["numeric_slots"][index].astype(np.float32, copy=False)),
            "condition_numeric_slot_mask": torch.from_numpy(self.condition_npz["numeric_slot_mask"][index].astype(np.float32, copy=False)),
            "condition_categorical_slots": torch.from_numpy(self.condition_npz["categorical_slots"][index].astype(np.int64, copy=False)),
            "condition_categorical_slot_mask": torch.from_numpy(self.condition_npz["categorical_slot_mask"][index].astype(np.float32, copy=False)),
            "reference_prompt": str(row.get("reference_prompt") or ""),
            "selected_families": list(row.get("selected_families") or []),
        }
        return sample


def collate_aml_condition_motion_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    tensor_keys = [
        key
        for key, value in samples[0].items()
        if isinstance(value, torch.Tensor)
    ]
    batch: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in tensor_keys
    }
    metadata_keys = [
        "case_id",
        "reference_prompt",
        "selected_families",
    ]
    for key in metadata_keys:
        batch[key] = [sample[key] for sample in samples]
    return batch
