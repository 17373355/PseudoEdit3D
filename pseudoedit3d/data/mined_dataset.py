from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS
from pseudoedit3d.edit.action_program import build_goal_spec, goal_spec_to_numpy
from pseudoedit3d.edit import verbalize_program
from pseudoedit3d.edit.attributes import compute_motion_statistics, extract_upper_body_proxy_attributes
from pseudoedit3d.edit.schema import EditProgram, LabelSchema, load_label_schema
from pseudoedit3d.edit.skill_context import infer_skill_context
from pseudoedit3d.text import CharTokenizer


def build_masks_from_program(program: EditProgram, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    joint_mask = np.zeros((num_frames, SMPLH_NUM_JOINTS), dtype=np.float32)
    time_mask = np.zeros((num_frames,), dtype=np.float32)
    joint_ids = BODY_PART_TO_JOINTS[program.part]
    time_mask[program.start_frame:program.end_frame + 1] = 1.0
    joint_mask[program.start_frame:program.end_frame + 1, joint_ids] = 1.0
    return joint_mask, time_mask


def load_mined_pair_arrays(record: dict, schema: LabelSchema | None = None) -> dict[str, np.ndarray]:
    schema = schema or load_label_schema()
    program = EditProgram.from_dict(record["program"])
    source = np.load(record["source_path"], allow_pickle=True)
    target = np.load(record["target_path"], allow_pickle=True)

    source_pose = source["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
    target_pose = target["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
    source_trans = source["trans"].astype(np.float32)
    target_trans = target["trans"].astype(np.float32)

    if (program.skill_label or "unknown") == "unknown":
        source_attrs = extract_upper_body_proxy_attributes(source_pose)
        skill_context = infer_skill_context(
            source_attrs,
            motion_stats=compute_motion_statistics(source_pose, source_trans),
            num_frames=source_pose.shape[0],
            anchor_frame=program.start_frame,
        )
        program.skill_label = skill_context["skill_label"]
        program.skill_phase = skill_context["skill_phase"]
        if skill_context.get("is_relative_friendly", False) and program.preserve_mode == "all_non_target":
            program.preserve_mode = "skill_structure"

    joint_mask, time_mask = build_masks_from_program(program, source_pose.shape[0])

    return {
        "source_pose": source_pose,
        "target_pose": target_pose,
        "source_trans": source_trans,
        "target_trans": target_trans,
        "joint_mask": joint_mask,
        "time_mask": time_mask,
        "edit_vector": np.asarray(program.to_vector(schema), dtype=np.float32),
        "goal_spec": goal_spec_to_numpy(build_goal_spec(program, schema=schema)),
        "program": program.to_dict(),
    }


class MinedMotionEditDataset(Dataset):
    def __init__(
        self,
        pair_manifest_path: str,
        max_pairs: int = 0,
        label_schema_path: str = "",
        prompt_style: str = "template",
        prompt_max_length: int = 96,
        input_source_mode: str = "source_motion",
        source_prefix_frames: int = 1,
    ) -> None:
        self.records = []
        self.schema = load_label_schema(label_schema_path) if label_schema_path else load_label_schema()
        self.prompt_style = prompt_style
        self.tokenizer = CharTokenizer(max_length=prompt_max_length)
        self.input_source_mode = input_source_mode
        self.source_prefix_frames = max(1, int(source_prefix_frames))
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
        arrays = load_mined_pair_arrays(record, schema=self.schema)
        program = EditProgram.from_dict(arrays["program"])
        prompt_text = verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index)
        prompt_token_ids, prompt_attention_mask = self.tokenizer.encode(prompt_text)

        num_frames = arrays["target_pose"].shape[0]
        conditioning_frame_mask = np.zeros((num_frames,), dtype=np.float32)
        if self.input_source_mode == "source_motion":
            conditioned_source_pose = arrays["source_pose"]
            conditioned_source_trans = arrays["source_trans"]
            conditioning_frame_mask[:] = 1.0
        elif self.input_source_mode == "target_start_pose":
            conditioned_source_pose = np.repeat(arrays["target_pose"][:1], num_frames, axis=0)
            conditioned_source_trans = np.repeat(arrays["target_trans"][:1], num_frames, axis=0)
            conditioning_frame_mask[0] = 1.0
        elif self.input_source_mode == "target_prefix":
            prefix = min(self.source_prefix_frames, num_frames)
            conditioned_source_pose = np.repeat(arrays["target_pose"][prefix - 1 : prefix], num_frames, axis=0)
            conditioned_source_pose[:prefix] = arrays["target_pose"][:prefix]
            conditioned_source_trans = np.repeat(arrays["target_trans"][prefix - 1 : prefix], num_frames, axis=0)
            conditioned_source_trans[:prefix] = arrays["target_trans"][:prefix]
            conditioning_frame_mask[:prefix] = 1.0
        else:
            raise ValueError(f"Unsupported input_source_mode={self.input_source_mode}")

        output = {
            "source_pose": torch.from_numpy(conditioned_source_pose),
            "target_pose": torch.from_numpy(arrays["target_pose"]),
            "source_trans": torch.from_numpy(conditioned_source_trans),
            "target_trans": torch.from_numpy(arrays["target_trans"]),
            "joint_mask": torch.from_numpy(arrays["joint_mask"]),
            "time_mask": torch.from_numpy(arrays["time_mask"]),
            "conditioning_frame_mask": torch.from_numpy(conditioning_frame_mask),
            "edit_vector": torch.from_numpy(arrays["edit_vector"]),
            "prompt_token_ids": torch.from_numpy(prompt_token_ids),
            "prompt_attention_mask": torch.from_numpy(prompt_attention_mask),
            "prompt_text": prompt_text,
        }
        for key, value in arrays["goal_spec"].items():
            output[key] = torch.from_numpy(value)
        return output
