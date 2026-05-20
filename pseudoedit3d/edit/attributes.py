from __future__ import annotations

import numpy as np

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


def _smooth_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _to_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(values.astype(np.float32))


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
