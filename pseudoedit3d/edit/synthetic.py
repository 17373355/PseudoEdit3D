from __future__ import annotations

import math
import random

import numpy as np

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS
from pseudoedit3d.edit.attributes import extract_upper_body_proxy_attributes
from pseudoedit3d.edit.schema import DELTA_BINS, EDIT_ATTRIBUTES, EDIT_PARTS, EditProgram


ATTRIBUTE_TO_AXIS = {
    "raise": (0, 1.0),
    "lower": (0, -1.0),
    "bend": (2, 1.0),
    "extend": (2, -1.0),
    "lean_left": (2, 1.0),
    "lean_right": (2, -1.0),
    "lean_forward": (0, 1.0),
    "lean_backward": (0, -1.0),
}


def _sample_program(num_frames: int) -> EditProgram:
    part = random.choice(EDIT_PARTS)
    if part == "torso":
        attribute = random.choice(["lean_left", "lean_right", "lean_forward", "lean_backward"])
    else:
        attribute = random.choice(["raise", "lower", "bend", "extend"])
    delta_bin = random.choice(DELTA_BINS)
    start_frame = random.randint(0, max(0, num_frames - 20))
    end_frame = min(num_frames - 1, start_frame + random.randint(10, 24))
    return EditProgram(
        part=part,
        attribute=attribute,
        delta_bin=delta_bin,
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="ignore",
    )


def _delta_radians(delta_bin: str, delta_scale_deg: float) -> float:
    scale = {"small": 0.5, "medium": 1.0, "large": 1.5}[delta_bin]
    return math.radians(delta_scale_deg * scale)


def build_synthetic_edit_sample(
    poses: np.ndarray,
    trans: np.ndarray,
    contact_mask: np.ndarray | None,
    delta_scale_deg: float,
) -> dict:
    num_frames = poses.shape[0]
    program = _sample_program(num_frames)
    source_pose = poses.copy()
    target_pose = poses.copy()
    source_trans = trans.copy()
    target_trans = trans.copy()

    joint_mask = np.zeros((num_frames, SMPLH_NUM_JOINTS), dtype=np.float32)
    time_mask = np.zeros((num_frames,), dtype=np.float32)
    affected_joints = BODY_PART_TO_JOINTS[program.part]
    axis_idx, axis_sign = ATTRIBUTE_TO_AXIS[program.attribute]
    delta = _delta_radians(program.delta_bin, delta_scale_deg) * axis_sign

    for frame_idx in range(program.start_frame, program.end_frame + 1):
        time_mask[frame_idx] = 1.0
        for joint_idx in affected_joints:
            joint_mask[frame_idx, joint_idx] = 1.0
            target_pose[frame_idx, joint_idx, axis_idx] += delta

    edit_vector = np.asarray(program.to_vector(), dtype=np.float32)
    proxy_attributes = extract_upper_body_proxy_attributes(source_pose)
    return {
        "source_pose": source_pose.astype(np.float32),
        "target_pose": target_pose.astype(np.float32),
        "source_trans": source_trans.astype(np.float32),
        "target_trans": target_trans.astype(np.float32),
        "joint_mask": joint_mask,
        "time_mask": time_mask,
        "edit_vector": edit_vector,
        "program": program.to_dict(),
        "proxy_attributes": {key: values.astype(np.float32) for key, values in proxy_attributes.items()},
    }
