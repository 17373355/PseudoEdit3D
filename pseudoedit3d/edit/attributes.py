from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pseudoedit3d.constants import JOINT_INDEX


PROXY_ATTRIBUTE_SPECS = {
    "left_shoulder_pitch_proxy_deg": [(JOINT_INDEX["L_Collar"], 0, 0.35), (JOINT_INDEX["L_Shoulder"], 0, 1.0)],
    "right_shoulder_pitch_proxy_deg": [(JOINT_INDEX["R_Collar"], 0, 0.35), (JOINT_INDEX["R_Shoulder"], 0, 1.0)],
    "left_shoulder_roll_proxy_deg": [(JOINT_INDEX["L_Shoulder"], 2, 1.0)],
    "right_shoulder_roll_proxy_deg": [(JOINT_INDEX["R_Shoulder"], 2, 1.0)],
    "left_elbow_flex_proxy_deg": [(JOINT_INDEX["L_Elbow"], 2, 1.0)],
    "right_elbow_flex_proxy_deg": [(JOINT_INDEX["R_Elbow"], 2, 1.0)],
    "torso_pitch_proxy_deg": [
        (JOINT_INDEX["Spine1"], 0, 0.2),
        (JOINT_INDEX["Spine2"], 0, 0.3),
        (JOINT_INDEX["Spine3"], 0, 0.5),
    ],
    "torso_roll_proxy_deg": [
        (JOINT_INDEX["Spine1"], 2, 0.2),
        (JOINT_INDEX["Spine2"], 2, 0.3),
        (JOINT_INDEX["Spine3"], 2, 0.5),
    ],
}

PROXY_ATTRIBUTE_ORDER = [
    "left_shoulder_pitch_proxy_deg",
    "right_shoulder_pitch_proxy_deg",
    "both_shoulder_pitch_proxy_deg",
    "left_elbow_flex_proxy_deg",
    "right_elbow_flex_proxy_deg",
    "both_elbow_flex_proxy_deg",
    "torso_pitch_proxy_deg",
    "torso_roll_proxy_deg",
]

ACTION_TO_PROXY_ATTRIBUTE = {
    ("left_arm", "raise"): "left_shoulder_pitch_proxy_deg",
    ("left_arm", "lower"): "left_shoulder_pitch_proxy_deg",
    ("right_arm", "raise"): "right_shoulder_pitch_proxy_deg",
    ("right_arm", "lower"): "right_shoulder_pitch_proxy_deg",
    ("both_arms", "raise"): "both_shoulder_pitch_proxy_deg",
    ("both_arms", "lower"): "both_shoulder_pitch_proxy_deg",
    ("left_arm", "bend"): "left_elbow_flex_proxy_deg",
    ("left_arm", "extend"): "left_elbow_flex_proxy_deg",
    ("right_arm", "bend"): "right_elbow_flex_proxy_deg",
    ("right_arm", "extend"): "right_elbow_flex_proxy_deg",
    ("both_arms", "bend"): "both_elbow_flex_proxy_deg",
    ("both_arms", "extend"): "both_elbow_flex_proxy_deg",
    ("torso", "lean_forward"): "torso_pitch_proxy_deg",
    ("torso", "lean_backward"): "torso_pitch_proxy_deg",
    ("torso", "lean_left"): "torso_roll_proxy_deg",
    ("torso", "lean_right"): "torso_roll_proxy_deg",
}

PROXY_ATTRIBUTE_INDEX = {key: idx for idx, key in enumerate(PROXY_ATTRIBUTE_ORDER)}


def _smooth_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _to_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(values.astype(np.float32))


def _to_deg_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(values.float())


def extract_upper_body_proxy_attributes(poses: np.ndarray, smooth_window: int = 5) -> dict[str, np.ndarray]:
    if poses.ndim == 2:
        poses = poses.reshape(poses.shape[0], -1, 3)
    attributes = {}
    for key, terms in PROXY_ATTRIBUTE_SPECS.items():
        values = np.zeros((poses.shape[0],), dtype=np.float32)
        for joint_idx, axis_idx, weight in terms:
            values += weight * _to_deg(poses[:, joint_idx, axis_idx])
        attributes[key] = _smooth_1d(values, window=smooth_window)

    attributes["both_shoulder_pitch_proxy_deg"] = 0.5 * (
        attributes["left_shoulder_pitch_proxy_deg"] + attributes["right_shoulder_pitch_proxy_deg"]
    )
    attributes["both_elbow_flex_proxy_deg"] = 0.5 * (
        np.abs(attributes["left_elbow_flex_proxy_deg"]) + np.abs(attributes["right_elbow_flex_proxy_deg"])
    )
    return attributes


def resolve_proxy_attribute_key(part: str, attribute: str, fallback_attribute_key: str | None = None) -> str:
    if fallback_attribute_key:
        return fallback_attribute_key
    key = ACTION_TO_PROXY_ATTRIBUTE.get((part, attribute))
    if key is None:
        raise KeyError(f"No proxy attribute mapping for part={part}, attribute={attribute}")
    return key


def _smooth_torch(values: torch.Tensor, window: int = 5) -> torch.Tensor:
    if window <= 1:
        return values
    if values.ndim == 1:
        values = values.unsqueeze(0)
    values = values.float()
    pad = window // 2
    padded = F.pad(values.unsqueeze(1), (pad, pad), mode="replicate")
    kernel = torch.ones((1, 1, window), device=values.device, dtype=values.dtype) / float(window)
    smoothed = F.conv1d(padded, kernel).squeeze(1)
    return smoothed


def extract_upper_body_proxy_attributes_torch(poses: torch.Tensor, smooth_window: int = 5) -> dict[str, torch.Tensor]:
    if poses.ndim == 3:
        poses = poses.view(poses.shape[0], poses.shape[1], -1, 3)
    elif poses.ndim == 2:
        poses = poses.view(1, poses.shape[0], -1, 3)

    attributes = {}
    for key, terms in PROXY_ATTRIBUTE_SPECS.items():
        values = torch.zeros((poses.shape[0], poses.shape[1]), dtype=poses.dtype, device=poses.device)
        for joint_idx, axis_idx, weight in terms:
            values = values + weight * _to_deg_torch(poses[:, :, joint_idx, axis_idx])
        attributes[key] = _smooth_torch(values, window=smooth_window)

    attributes["both_shoulder_pitch_proxy_deg"] = 0.5 * (
        attributes["left_shoulder_pitch_proxy_deg"] + attributes["right_shoulder_pitch_proxy_deg"]
    )
    attributes["both_elbow_flex_proxy_deg"] = 0.5 * (
        attributes["left_elbow_flex_proxy_deg"].abs() + attributes["right_elbow_flex_proxy_deg"].abs()
    )
    return attributes


def stack_proxy_attributes_torch(attributes: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([attributes[key] for key in PROXY_ATTRIBUTE_ORDER], dim=-1)


def compute_motion_statistics(poses: np.ndarray, trans: np.ndarray) -> dict[str, float]:
    pose_vel = np.linalg.norm(np.diff(poses, axis=0), axis=-1).mean()
    root_vel = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
    return {
        "pose_velocity_mean": float(np.rad2deg(pose_vel)),
        "root_speed_mean": float(root_vel.mean()) if len(root_vel) > 0 else 0.0,
        "root_speed_std": float(root_vel.std()) if len(root_vel) > 0 else 0.0,
        "root_displacement": float(np.linalg.norm(trans[-1] - trans[0])) if len(trans) > 1 else 0.0,
    }


def summarize_attributes(attributes: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {}
    for key, values in attributes.items():
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
        summary[f"{key}_range"] = float(values.max() - values.min())
    return summary
