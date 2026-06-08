from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from pseudoedit3d.constants import JOINT_INDEX
from pseudoedit3d.edit.attributes import extract_upper_body_proxy_attributes


@dataclass
class ObservableSequence:
    name: str
    values: np.ndarray
    unit: str
    level: str = 'frame'
    part: str = 'whole_body'
    source: str = 'motion'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameObservables:
    num_frames: int
    sequences: Dict[str, ObservableSequence]

    def names(self) -> list[str]:
        return list(self.sequences.keys())

    def values(self, name: str) -> np.ndarray:
        return self.sequences[name].values

    def get(self, name: str) -> ObservableSequence:
        return self.sequences[name]

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        return {name: seq.values for name, seq in self.sequences.items()}


def _pelvis_to_ankle_height(joints: np.ndarray) -> np.ndarray:
    ankles = 0.5 * (joints[:, 7, 1] + joints[:, 8, 1])
    pelvis = joints[:, 0, 1]
    return (pelvis - ankles).astype(np.float32)


def _torso_bend_drop_signal(joints: np.ndarray) -> np.ndarray:
    pelvis_y = joints[:, 0, 1]
    neck_rel = joints[:, 12, 1] - pelvis_y
    head_rel = joints[:, 15, 1] - pelvis_y
    shoulder_rel = 0.5 * (joints[:, 16, 1] + joints[:, 17, 1]) - pelvis_y
    return ((neck_rel + head_rel + shoulder_rel) / 3.0).astype(np.float32)


def _torso_forward_extent(joints: np.ndarray) -> np.ndarray:
    pelvis_xz = joints[:, 0][:, [0, 2]]
    neck_xz = np.linalg.norm(joints[:, 12][:, [0, 2]] - pelvis_xz, axis=-1)
    head_xz = np.linalg.norm(joints[:, 15][:, [0, 2]] - pelvis_xz, axis=-1)
    shoulder_center = 0.5 * (joints[:, 16][:, [0, 2]] + joints[:, 17][:, [0, 2]])
    shoulder_xz = np.linalg.norm(shoulder_center - pelvis_xz, axis=-1)
    return ((neck_xz + head_xz + shoulder_xz) / 3.0).astype(np.float32)


def _left_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    return ((joints[:, 20, 1] - joints[:, 16, 1]) * 180.0).astype(np.float32)


def _right_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    return ((joints[:, 21, 1] - joints[:, 17, 1]) * 180.0).astype(np.float32)


def _left_elbow_lift_deg(joints: np.ndarray) -> np.ndarray:
    return ((joints[:, 18, 1] - joints[:, 16, 1]) * 180.0).astype(np.float32)


def _right_elbow_lift_deg(joints: np.ndarray) -> np.ndarray:
    return ((joints[:, 19, 1] - joints[:, 17, 1]) * 180.0).astype(np.float32)


def _wrist_chest_distance(joints: np.ndarray, side: str) -> np.ndarray:
    chest = 0.5 * (joints[:, JOINT_INDEX['L_Shoulder']] + joints[:, JOINT_INDEX['R_Shoulder']])
    wrist_idx = JOINT_INDEX['L_Wrist'] if side == 'left' else JOINT_INDEX['R_Wrist']
    return np.linalg.norm(joints[:, wrist_idx] - chest, axis=-1).astype(np.float32)


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or len(values) < 3:
        return values
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode='valid').astype(np.float32)


def _root_heading_vectors_from_joints(joints: np.ndarray, smooth_window: int) -> tuple[np.ndarray, np.ndarray]:
    if joints.shape[1] <= max(JOINT_INDEX['R_Shoulder'], JOINT_INDEX['R_Hip']):
        forward = np.zeros((joints.shape[0], 2), dtype=np.float32)
        forward[:, 1] = 1.0
        right = np.zeros((joints.shape[0], 2), dtype=np.float32)
        right[:, 0] = 1.0
        return forward, right
    left_hip = joints[:, JOINT_INDEX['L_Hip']][:, [0, 2]]
    right_hip = joints[:, JOINT_INDEX['R_Hip']][:, [0, 2]]
    left_shoulder = joints[:, JOINT_INDEX['L_Shoulder']][:, [0, 2]]
    right_shoulder = joints[:, JOINT_INDEX['R_Shoulder']][:, [0, 2]]
    across = (right_hip - left_hip) + (right_shoulder - left_shoulder)
    norm = np.linalg.norm(across, axis=-1, keepdims=True)
    right = across / np.clip(norm, 1e-8, None)
    right[:, 0] = _smooth_1d(right[:, 0], smooth_window)
    right[:, 1] = _smooth_1d(right[:, 1], smooth_window)
    right = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8, None)
    # HumanML3D joint convention points the raw cross-product backward for walking clips.
    forward = -np.stack([-right[:, 1], right[:, 0]], axis=-1)
    return forward.astype(np.float32), right.astype(np.float32)


def _root_yaw_proxy_from_joints(joints: np.ndarray, smooth_window: int) -> np.ndarray:
    forward, _ = _root_heading_vectors_from_joints(joints, smooth_window=smooth_window)
    yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1]))).astype(np.float32)
    yaw = yaw - yaw[0]
    return _smooth_1d(yaw, smooth_window)


def _root_body_frame_velocity(joints: np.ndarray, smooth_window: int) -> tuple[np.ndarray, np.ndarray]:
    root = joints[:, 0][:, [0, 2]].astype(np.float32)
    delta = np.zeros_like(root, dtype=np.float32)
    if len(root) > 1:
        delta[1:] = root[1:] - root[:-1]
    forward, right = _root_heading_vectors_from_joints(joints, smooth_window=smooth_window)
    forward_vel = np.sum(delta * forward, axis=-1).astype(np.float32)
    lateral_vel = np.sum(delta * right, axis=-1).astype(np.float32)
    return _smooth_1d(forward_vel, smooth_window), _smooth_1d(lateral_vel, smooth_window)


def extract_layer0_frame_observables(
    poses: np.ndarray,
    joints: np.ndarray,
    trans: np.ndarray | None = None,
    smooth_window: int = 5,
) -> FrameObservables:
    proxy = extract_upper_body_proxy_attributes(poses=poses, trans=trans, smooth_window=smooth_window)
    # HumanML3D AML extraction often has joints but no reliable pose rotations.
    # Recompute heading from joints so spins/turns are not lost at Layer 0.
    proxy['root_yaw_proxy_deg'] = _root_yaw_proxy_from_joints(joints, smooth_window=smooth_window)
    root_forward_velocity, root_lateral_velocity = _root_body_frame_velocity(joints, smooth_window=smooth_window)
    num_frames = joints.shape[0]
    sequences: dict[str, ObservableSequence] = {}

    def add(name: str, values: np.ndarray, unit: str, part: str, source: str = 'motion', **metadata: Any) -> None:
        sequences[name] = ObservableSequence(
            name=name,
            values=np.asarray(values, dtype=np.float32),
            unit=unit,
            part=part,
            source=source,
            metadata=metadata,
        )

    for name, values in proxy.items():
        unit = 'deg' if 'deg' in name or 'yaw' in name else 'm'
        part = 'whole_body'
        if 'left_' in name:
            part = 'left_arm'
        elif 'right_' in name:
            part = 'right_arm'
        elif 'torso' in name:
            part = 'torso'
        if name == 'root_xz_speed_proxy':
            add(
                name,
                values,
                unit=unit,
                part=part,
                source='proxy',
                root_forward_velocity=root_forward_velocity,
                root_lateral_velocity=root_lateral_velocity,
            )
        else:
            add(name, values, unit=unit, part=part, source='proxy')

    add('pelvis_to_ankle_height', _pelvis_to_ankle_height(joints), unit='m', part='whole_body', source='geometry')
    add('torso_bend_drop_signal', _torso_bend_drop_signal(joints), unit='m', part='torso', source='geometry')
    add('torso_forward_extent', _torso_forward_extent(joints), unit='m', part='torso', source='geometry')
    add('left_arm_raise_deg', _left_arm_raise_deg(joints), unit='deg', part='left_arm', source='geometry')
    add('right_arm_raise_deg', _right_arm_raise_deg(joints), unit='deg', part='right_arm', source='geometry')
    add('left_elbow_lift_deg', _left_elbow_lift_deg(joints), unit='deg', part='left_arm', source='geometry')
    add('right_elbow_lift_deg', _right_elbow_lift_deg(joints), unit='deg', part='right_arm', source='geometry')
    add('left_wrist_chest_distance', _wrist_chest_distance(joints, 'left'), unit='m', part='left_arm', source='geometry')
    add('right_wrist_chest_distance', _wrist_chest_distance(joints, 'right'), unit='m', part='right_arm', source='geometry')

    return FrameObservables(num_frames=num_frames, sequences=sequences)
