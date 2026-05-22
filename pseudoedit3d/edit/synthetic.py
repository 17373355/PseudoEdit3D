from __future__ import annotations

import math
import random

import numpy as np

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS
from pseudoedit3d.edit.action_program import build_goal_spec, goal_spec_to_numpy
from pseudoedit3d.edit.attributes import compute_motion_statistics, extract_upper_body_proxy_attributes
from pseudoedit3d.edit.schema import EditProgram, LabelSchema, get_default_schema
from pseudoedit3d.edit.skill_context import infer_skill_context, summarize_skill_attribute


def _sample_program(num_frames: int, schema: LabelSchema) -> EditProgram:
    part = random.choice(schema.part_keys)
    attribute = random.choice(schema.attributes_for_part(part))
    delta_bin = random.choice(schema.delta_bin_keys)
    start_frame = random.randint(0, max(0, num_frames - 20))
    end_frame = min(num_frames - 1, start_frame + random.randint(10, 24))
    delta_value_deg = schema.delta_bin(delta_bin).default_degrees
    return EditProgram(
        part=part,
        attribute=attribute,
        delta_bin=delta_bin,
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="ignore",
        delta_value_deg=delta_value_deg,
        direction="increase" if schema.attribute(attribute).synthetic_sign and schema.attribute(attribute).synthetic_sign > 0 else "decrease",
        schema_version=schema.schema_version,
        operator="add",
        reference="current_state",
        tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
    )


def _delta_radians(delta_bin: str, delta_scale_deg: float) -> float:
    scale = {"small": 0.5, "medium": 1.0, "large": 1.5}[delta_bin]
    return math.radians(delta_scale_deg * scale)


def build_synthetic_edit_sample(
    poses: np.ndarray,
    trans: np.ndarray,
    contact_mask: np.ndarray | None,
    delta_scale_deg: float,
    label_schema_path: str | None = None,
) -> dict:
    if not label_schema_path:
        schema = get_default_schema()
    else:
        from pseudoedit3d.edit.schema import load_label_schema
        schema = load_label_schema(label_schema_path)
    num_frames = poses.shape[0]
    program = _sample_program(num_frames, schema=schema)
    from pseudoedit3d.edit.attributes import resolve_proxy_attribute_key
    program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
    source_pose = poses.copy()
    target_pose = poses.copy()
    source_trans = trans.copy()
    target_trans = trans.copy()

    joint_mask = np.zeros((num_frames, SMPLH_NUM_JOINTS), dtype=np.float32)
    time_mask = np.zeros((num_frames,), dtype=np.float32)
    affected_joints = BODY_PART_TO_JOINTS[program.part]
    attribute_entry = schema.attribute(program.attribute)
    axis_idx = int(attribute_entry.synthetic_axis)
    axis_sign = float(attribute_entry.synthetic_sign)
    delta = _delta_radians(program.delta_bin, delta_scale_deg) * axis_sign

    for frame_idx in range(program.start_frame, program.end_frame + 1):
        time_mask[frame_idx] = 1.0
        for joint_idx in affected_joints:
            joint_mask[frame_idx, joint_idx] = 1.0
            target_pose[frame_idx, joint_idx, axis_idx] += delta

    edit_vector = np.asarray(program.to_vector(schema), dtype=np.float32)
    proxy_attributes = extract_upper_body_proxy_attributes(source_pose)
    skill_context = infer_skill_context(
        proxy_attributes,
        motion_stats=compute_motion_statistics(source_pose, source_trans),
        num_frames=num_frames,
        anchor_frame=program.start_frame,
    )
    program.skill_label = skill_context["skill_label"]
    program.skill_phase = skill_context["skill_phase"]
    if skill_context.get("is_relative_friendly", False):
        program.preserve_mode = "skill_structure"
    attr_state = summarize_skill_attribute(proxy_attributes[program.attribute_key], frame_idx=program.start_frame)
    program.metadata["source_attr_mean_deg"] = float(attr_state["mean_deg"])
    program.metadata["source_attr_amplitude_deg"] = float(attr_state["amplitude_deg"])
    if program.skill_label == "periodic_arm_motion" and program.operator == "add" and program.reference == "current_state":
        program.metadata["relative_skill_parameter"] = "offset_deg"
        program.metadata["target_offset_deg"] = float(attr_state["mean_deg"] + program.delta_value_deg * (1.0 if program.direction == "increase" else -1.0))
        program.metadata["preserve_amplitude"] = True
        program.metadata["periodic_limb"] = skill_context.get("periodic_limb")
        program.metadata["periodic_state"] = skill_context.get("periodic_states", {}).get(skill_context.get("periodic_limb", ""), {})
    goal_spec = build_goal_spec(program, schema=schema)
    return {
        "source_pose": source_pose.astype(np.float32),
        "target_pose": target_pose.astype(np.float32),
        "source_trans": source_trans.astype(np.float32),
        "target_trans": target_trans.astype(np.float32),
        "joint_mask": joint_mask,
        "time_mask": time_mask,
        "edit_vector": edit_vector,
        "program": program.to_dict(),
        "goal_spec": goal_spec_to_numpy(goal_spec),
        "skill_context": skill_context,
        "proxy_attributes": {key: values.astype(np.float32) for key, values in proxy_attributes.items()},
    }
