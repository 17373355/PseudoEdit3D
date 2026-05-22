from __future__ import annotations

import math

import numpy as np


SKILL_CONTEXT_LABELS = [
    "unknown",
    "static_pose",
    "locomotion",
    "periodic_arm_motion",
    "torso_leaning",
    "arm_reaching_or_repositioning",
]


SKILL_LABEL_TO_PROMPT_PHRASE = {
    "unknown": "the current motion",
    "static_pose": "the standing pose",
    "locomotion": "the ongoing locomotion",
    "periodic_arm_motion": "the ongoing arm motion",
    "torso_leaning": "the torso leaning motion",
    "arm_reaching_or_repositioning": "the current arm movement",
}

PERIODIC_ARM_KEYS = {
    "left_arm": ["left_shoulder_pitch_proxy_deg", "left_elbow_flex_proxy_deg"],
    "right_arm": ["right_shoulder_pitch_proxy_deg", "right_elbow_flex_proxy_deg"],
    "both_arms": ["both_shoulder_pitch_proxy_deg", "both_elbow_flex_proxy_deg"],
}


def _dominant_attr(attributes: dict[str, np.ndarray], keys: list[str]) -> tuple[str, float]:
    best_key = keys[0]
    best_score = -1.0
    for key in keys:
        values = np.asarray(attributes[key], dtype=np.float32)
        score = float(values.max() - values.min())
        if score > best_score:
            best_key = key
            best_score = score
    return best_key, best_score


def estimate_phase(values: np.ndarray, frame_idx: int) -> float:
    values = np.asarray(values, dtype=np.float32)
    frame_idx = int(np.clip(frame_idx, 0, len(values) - 1))
    centered = values - float(np.mean(values))
    if len(values) > 1:
        velocity = np.gradient(centered)
    else:
        velocity = np.zeros_like(centered)
    phase = math.atan2(float(velocity[frame_idx]), float(centered[frame_idx] + 1e-6))
    return float((phase + math.pi) / (2.0 * math.pi))


def summarize_skill_attribute(values: np.ndarray, frame_idx: int | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    frame_idx = int(len(values) // 2 if frame_idx is None else np.clip(frame_idx, 0, len(values) - 1))
    mean_deg = float(values.mean())
    amplitude_deg = float(0.5 * (values.max() - values.min()))
    current_deg = float(values[frame_idx])
    phase = estimate_phase(values, frame_idx)
    return {
        "mean_deg": mean_deg,
        "amplitude_deg": amplitude_deg,
        "current_deg": current_deg,
        "phase": float(phase),
    }


def infer_skill_context(
    attributes: dict[str, np.ndarray],
    motion_stats: dict[str, float],
    num_frames: int,
    anchor_frame: int | None = None,
) -> dict:
    anchor_frame = int(num_frames // 2 if anchor_frame is None else anchor_frame)
    arm_keys = [
        "left_shoulder_pitch_proxy_deg",
        "right_shoulder_pitch_proxy_deg",
        "both_shoulder_pitch_proxy_deg",
        "left_elbow_flex_proxy_deg",
        "right_elbow_flex_proxy_deg",
        "both_elbow_flex_proxy_deg",
    ]
    torso_keys = ["torso_pitch_proxy_deg", "torso_roll_proxy_deg"]
    dominant_arm_key, dominant_arm_range = _dominant_attr(attributes, arm_keys)
    dominant_torso_key, dominant_torso_range = _dominant_attr(attributes, torso_keys)

    pose_velocity_mean = float(motion_stats.get("pose_velocity_mean", 0.0))
    root_speed_mean = float(motion_stats.get("root_speed_mean", 0.0))
    root_displacement = float(motion_stats.get("root_displacement", 0.0))

    if root_displacement > 0.35 or root_speed_mean > 0.05:
        skill_label = "locomotion"
        dominant_attr_key = dominant_arm_key if dominant_arm_range >= dominant_torso_range else dominant_torso_key
    elif dominant_torso_range > 10.0 and dominant_torso_range >= dominant_arm_range * 0.85:
        skill_label = "torso_leaning"
        dominant_attr_key = dominant_torso_key
    elif dominant_arm_range > 25.0:
        skill_label = "periodic_arm_motion"
        dominant_attr_key = dominant_arm_key
    elif dominant_arm_range > 8.0:
        skill_label = "arm_reaching_or_repositioning"
        dominant_attr_key = dominant_arm_key
    elif pose_velocity_mean < 1.5 and dominant_arm_range < 6.0 and dominant_torso_range < 5.0:
        skill_label = "static_pose"
        dominant_attr_key = dominant_arm_key
    else:
        skill_label = "unknown"
        dominant_attr_key = dominant_arm_key if dominant_arm_range >= dominant_torso_range else dominant_torso_key

    phase_attr_values = np.asarray(attributes[dominant_attr_key], dtype=np.float32)
    skill_phase = estimate_phase(phase_attr_values, anchor_frame)
    dominant_state = summarize_skill_attribute(phase_attr_values, frame_idx=anchor_frame)
    is_relative_friendly = skill_label in {"locomotion", "periodic_arm_motion", "torso_leaning"}

    periodic_states = {}
    for limb, keys in PERIODIC_ARM_KEYS.items():
        limb_attr_key, limb_range = _dominant_attr(attributes, keys)
        limb_state = summarize_skill_attribute(np.asarray(attributes[limb_attr_key], dtype=np.float32), frame_idx=anchor_frame)
        periodic_states[limb] = {
            "attr_key": limb_attr_key,
            "range_deg": float(limb_range),
            "mean_deg": float(limb_state["mean_deg"]),
            "amplitude_deg": float(limb_state["amplitude_deg"]),
            "current_deg": float(limb_state["current_deg"]),
            "phase": float(limb_state["phase"]),
        }
    periodic_limb = max(periodic_states.keys(), key=lambda key: periodic_states[key]["range_deg"])

    return {
        "skill_label": skill_label,
        "skill_phase": float(skill_phase),
        "dominant_attr_key": dominant_attr_key,
        "dominant_attr_range": float(max(dominant_arm_range, dominant_torso_range)),
        "dominant_attr_mean_deg": float(dominant_state["mean_deg"]),
        "dominant_attr_amplitude_deg": float(dominant_state["amplitude_deg"]),
        "dominant_attr_current_deg": float(dominant_state["current_deg"]),
        "periodic_limb": periodic_limb,
        "periodic_states": periodic_states,
        "anchor_frame": int(anchor_frame),
        "is_relative_friendly": bool(is_relative_friendly),
    }
