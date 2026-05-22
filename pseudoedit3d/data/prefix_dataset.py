from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS, SMPLH_POSE_DIM
from pseudoedit3d.edit.action_program import build_goal_spec, goal_spec_to_numpy
from pseudoedit3d.edit.attributes import compute_motion_statistics, extract_upper_body_proxy_attributes, resolve_proxy_attribute_key
from pseudoedit3d.edit.schema import EditProgram, LabelSchema, load_label_schema
from pseudoedit3d.edit.skill_context import SKILL_LABEL_TO_PROMPT_PHRASE, infer_skill_context, summarize_skill_attribute
from pseudoedit3d.edit.verbalizer import verbalize_program
from pseudoedit3d.text import CharTokenizer


def _iter_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _make_prefix_condition(poses: np.ndarray, trans: np.ndarray, prefix_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = poses.shape[0]
    prefix = min(max(1, prefix_frames), num_frames)
    conditioned_pose = np.repeat(poses[prefix - 1 : prefix], num_frames, axis=0)
    conditioned_pose[:prefix] = poses[:prefix]
    conditioned_trans = np.repeat(trans[prefix - 1 : prefix], num_frames, axis=0)
    conditioned_trans[:prefix] = trans[:prefix]
    conditioning_mask = np.zeros((num_frames,), dtype=np.float32)
    conditioning_mask[:prefix] = 1.0
    return conditioned_pose, conditioned_trans, conditioning_mask


def _build_future_mask(num_frames: int, prefix_frames: int) -> np.ndarray:
    mask = np.zeros((num_frames,), dtype=np.float32)
    mask[min(prefix_frames, num_frames) :] = 1.0
    return mask


def _sample_delta_bin(schema: LabelSchema) -> str:
    return random.choice(schema.delta_bin_keys)


def _attribute_from_proxy_key(proxy_key: str) -> tuple[str, str, str]:
    if proxy_key == "left_shoulder_pitch_proxy_deg":
        return "left_arm", "raise", "lower"
    if proxy_key == "right_shoulder_pitch_proxy_deg":
        return "right_arm", "raise", "lower"
    if proxy_key == "both_shoulder_pitch_proxy_deg":
        return "both_arms", "raise", "lower"
    if proxy_key == "left_elbow_flex_proxy_deg":
        return "left_arm", "bend", "extend"
    if proxy_key == "right_elbow_flex_proxy_deg":
        return "right_arm", "bend", "extend"
    if proxy_key == "both_elbow_flex_proxy_deg":
        return "both_arms", "bend", "extend"
    if proxy_key == "torso_pitch_proxy_deg":
        return "torso", "lean_forward", "lean_backward"
    if proxy_key == "torso_roll_proxy_deg":
        return "torso", "lean_left", "lean_right"
    raise KeyError(f"Unsupported proxy key: {proxy_key}")


def _build_relative_program(
    schema: LabelSchema,
    skill_context: dict,
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
) -> EditProgram:
    skill_label = skill_context["skill_label"]
    if skill_label == "periodic_arm_motion":
        part = skill_context.get("periodic_limb", "both_arms")
        periodic_state = skill_context.get("periodic_states", {}).get(part, {})
        attr_key = periodic_state.get("attr_key", skill_context["dominant_attr_key"])
        part, positive_attr, negative_attr = _attribute_from_proxy_key(attr_key)
        direction = random.choice(["increase", "decrease"])
        attribute = positive_attr if direction == "increase" else negative_attr
        preserve_mode = "skill_structure"
        metadata = {
            "source_attr_mean_deg": float(periodic_state.get("mean_deg", np.mean(proxy_attributes[attr_key]))),
            "source_attr_amplitude_deg": float(periodic_state.get("amplitude_deg", summarize_skill_attribute(proxy_attributes[attr_key])["amplitude_deg"])),
            "relative_skill_parameter": "offset_deg",
            "preserve_amplitude": True,
            "periodic_limb": part,
            "periodic_state": periodic_state,
        }
    elif skill_label == "locomotion":
        attr_key = random.choice(["both_shoulder_pitch_proxy_deg", "torso_pitch_proxy_deg"])
        part, positive_attr, negative_attr = _attribute_from_proxy_key(attr_key)
        direction = random.choice(["increase", "decrease"])
        attribute = positive_attr if direction == "increase" else negative_attr
        preserve_mode = "skill_structure"
        metadata = {
            "source_attr_mean_deg": float(np.mean(proxy_attributes[attr_key])),
            "source_attr_amplitude_deg": float(summarize_skill_attribute(proxy_attributes[attr_key])["amplitude_deg"]),
            "relative_skill_parameter": "attribute_delta_deg",
            "preserve_amplitude": False,
        }
    else:
        attr_key = skill_context["dominant_attr_key"]
        part, positive_attr, negative_attr = _attribute_from_proxy_key(attr_key)
        direction = random.choice(["increase", "decrease"])
        attribute = positive_attr if direction == "increase" else negative_attr
        preserve_mode = "all_non_target"
        metadata = {
            "source_attr_mean_deg": float(np.mean(proxy_attributes[attr_key])),
            "source_attr_amplitude_deg": float(summarize_skill_attribute(proxy_attributes[attr_key])["amplitude_deg"]),
            "relative_skill_parameter": "attribute_delta_deg",
            "preserve_amplitude": False,
        }

    delta_bin = _sample_delta_bin(schema)
    delta_value_deg = float(schema.delta_bin(delta_bin).default_degrees or 0.0)
    signed_delta = delta_value_deg if direction == "increase" else -delta_value_deg
    if metadata.get("relative_skill_parameter") == "offset_deg":
        metadata["target_offset_deg"] = float(metadata["source_attr_mean_deg"] + signed_delta)
    else:
        metadata["target_offset_deg"] = float("nan")

    start_frame = min(prefix_frames, num_frames - 1)
    end_frame = num_frames - 1
    if num_frames - start_frame > 10:
        start_frame = random.randint(prefix_frames, max(prefix_frames, num_frames - 10))
        end_frame = random.randint(start_frame + 4, num_frames - 1)

    return EditProgram(
        part=part,
        attribute=attribute,
        delta_bin=delta_bin,
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="ignore",
        attribute_key=attr_key,
        direction=direction,
        delta_value_deg=signed_delta,
        source_type="same_clip_prefix",
        schema_version=schema.schema_version,
        input_mode="motion_prefix",
        operator="add",
        reference="current_state",
        preserve_parts=[],
        preserve_mode=preserve_mode,
        skill_label=skill_label,
        skill_phase=float(skill_context["skill_phase"]),
        tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
        metadata=metadata,
    )


def _apply_program_to_future(
    poses: np.ndarray,
    program: EditProgram,
    schema: LabelSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pose = poses.copy()
    num_frames = poses.shape[0]
    joint_mask = np.zeros((num_frames, SMPLH_NUM_JOINTS), dtype=np.float32)
    time_mask = np.zeros((num_frames,), dtype=np.float32)
    affected_joints = BODY_PART_TO_JOINTS[program.part]
    attr_entry = schema.attribute(program.attribute)
    axis_idx = int(attr_entry.synthetic_axis)
    axis_sign = float(attr_entry.synthetic_sign)
    delta = math.radians(abs(program.delta_value_deg or 0.0)) * axis_sign
    if program.direction == "decrease":
        delta = -abs(delta)
    else:
        delta = abs(delta)

    for frame_idx in range(program.start_frame, program.end_frame + 1):
        time_mask[frame_idx] = 1.0
        for joint_idx in affected_joints:
            joint_mask[frame_idx, joint_idx] = 1.0
            target_pose[frame_idx, joint_idx, axis_idx] += delta
    return target_pose.astype(np.float32), joint_mask, time_mask


def _continuation_prompt(skill_context: dict) -> str:
    phrase = SKILL_LABEL_TO_PROMPT_PHRASE.get(skill_context["skill_label"], "the current motion")
    return f"continue {phrase}"


def _dummy_goal_spec() -> dict[str, np.ndarray]:
    return {
        "goal_attr_idx": np.asarray(0, dtype=np.int64),
        "goal_operator_idx": np.asarray(0, dtype=np.int64),
        "goal_reference_idx": np.asarray(0, dtype=np.int64),
        "goal_preserve_mode_idx": np.asarray(0, dtype=np.int64),
        "goal_skill_label_idx": np.asarray(0, dtype=np.int64),
        "goal_start_frame": np.asarray(0, dtype=np.int64),
        "goal_end_frame": np.asarray(0, dtype=np.int64),
        "goal_delta_deg": np.asarray(0.0, dtype=np.float32),
        "goal_target_value_deg": np.asarray(np.nan, dtype=np.float32),
        "goal_source_attr_mean_deg": np.asarray(np.nan, dtype=np.float32),
        "goal_source_attr_amplitude_deg": np.asarray(np.nan, dtype=np.float32),
        "goal_target_offset_deg": np.asarray(np.nan, dtype=np.float32),
        "goal_preserve_amplitude": np.asarray(0.0, dtype=np.float32),
        "goal_direction_sign": np.asarray(1.0, dtype=np.float32),
        "goal_tolerance_deg": np.asarray(1.0, dtype=np.float32),
        "goal_skill_phase": np.asarray(np.nan, dtype=np.float32),
    }


class PrefixMotionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        max_clips: int = 0,
        label_schema_path: str = "",
        prompt_style: str = "template",
        prompt_max_length: int = 96,
        prefix_frames: int = 8,
        task_mode: str = "relative_edit",
        contact_filter: str = "any",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.schema = load_label_schema(label_schema_path) if label_schema_path else load_label_schema()
        self.prompt_style = prompt_style
        self.tokenizer = CharTokenizer(max_length=prompt_max_length)
        self.prefix_frames = max(1, int(prefix_frames))
        self.task_mode = task_mode
        self.contact_filter = contact_filter
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
        data = np.load(record["path"], allow_pickle=True)
        poses = data["poses"].reshape(-1, SMPLH_NUM_JOINTS, 3).astype(np.float32)
        trans = data["trans"].astype(np.float32)
        betas = np.asarray(data.get("betas", np.zeros((1, 16), dtype=np.float32)), dtype=np.float32)

        proxy_attributes = extract_upper_body_proxy_attributes(poses)
        motion_stats = compute_motion_statistics(poses, trans)
        prefix_pose, prefix_trans, conditioning_frame_mask = _make_prefix_condition(poses, trans, self.prefix_frames)
        future_mask = _build_future_mask(poses.shape[0], self.prefix_frames)
        skill_context = infer_skill_context(
            proxy_attributes,
            motion_stats=motion_stats,
            num_frames=poses.shape[0],
            anchor_frame=max(0, self.prefix_frames - 1),
        )

        if self.task_mode == "continue":
            prompt_text = _continuation_prompt(skill_context)
            edit_vector = np.zeros((self.schema.vector_dim,), dtype=np.float32)
            target_pose = poses
            target_trans = trans
            joint_mask = np.ones((poses.shape[0], SMPLH_NUM_JOINTS), dtype=np.float32) * future_mask[:, None]
            time_mask = future_mask
            goal_spec = _dummy_goal_spec()
            program_dict = {
                "task_mode": "continue",
                "skill_context": skill_context,
            }
        elif self.task_mode == "relative_edit":
            program = _build_relative_program(
                schema=self.schema,
                skill_context=skill_context,
                proxy_attributes=proxy_attributes,
                prefix_frames=self.prefix_frames,
                num_frames=poses.shape[0],
            )
            program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
            target_pose, joint_mask, time_mask = _apply_program_to_future(poses, program, self.schema)
            target_trans = trans.copy()
            edit_vector = np.asarray(program.to_vector(self.schema), dtype=np.float32)
            goal_spec = goal_spec_to_numpy(build_goal_spec(program, schema=self.schema))
            prompt_text = verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index)
            program_dict = program.to_dict()
        else:
            raise ValueError(f"Unsupported prefix task mode: {self.task_mode}")

        prompt_token_ids, prompt_attention_mask = self.tokenizer.encode(prompt_text)
        output = {
            "source_pose": torch.from_numpy(prefix_pose.astype(np.float32)),
            "target_pose": torch.from_numpy(target_pose.astype(np.float32)),
            "source_trans": torch.from_numpy(prefix_trans.astype(np.float32)),
            "target_trans": torch.from_numpy(target_trans.astype(np.float32)),
            "joint_mask": torch.from_numpy(joint_mask.astype(np.float32)),
            "time_mask": torch.from_numpy(time_mask.astype(np.float32)),
            "conditioning_frame_mask": torch.from_numpy(conditioning_frame_mask.astype(np.float32)),
            "edit_vector": torch.from_numpy(edit_vector),
            "prompt_token_ids": torch.from_numpy(prompt_token_ids),
            "prompt_attention_mask": torch.from_numpy(prompt_attention_mask),
            "prompt_text": prompt_text,
            "program_json": json.dumps(program_dict, ensure_ascii=True),
            "betas": torch.from_numpy(betas.astype(np.float32)),
        }
        for key, value in goal_spec.items():
            output[key] = torch.from_numpy(value)
        return output
