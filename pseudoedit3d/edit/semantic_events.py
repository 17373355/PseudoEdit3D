from __future__ import annotations

from typing import Any

import numpy as np

from pseudoedit3d.constants import JOINT_INDEX


def _event(
    part: str,
    super_family: str,
    cluster_id: str,
    start: int,
    end: int,
    *,
    direction: str,
    role: str,
    optional_semantic_name: str | None = None,
    magnitude: float | None = None,
    signed_delta: float | None = None,
    unit: str | None = None,
    count: int | None = None,
    confidence: float = 0.7,
    source: str = "semantic_joints",
    supporting_units: list[str] | None = None,
    motion_signature: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "part": part,
        "super_family": super_family,
        "cluster_id": cluster_id,
        "optional_semantic_name": optional_semantic_name,
        "direction": direction,
        "role": role,
        "start_frame": int(start),
        "end_frame": int(end),
        "confidence": float(confidence),
        "source": source,
        "source_span": [int(start), int(end)],
        "supporting_units": supporting_units or [],
        "motion_signature": motion_signature or {},
    }
    if magnitude is not None:
        out["magnitude"] = float(magnitude)
    if signed_delta is not None:
        out["signed_delta"] = float(signed_delta)
    if unit is not None:
        out["unit"] = unit
    if count is not None:
        out["count"] = int(count)
    if metadata:
        out["metadata"] = metadata
    return out


def _sig(
    dominant_axis: str,
    repeat_mode: str,
    phase_template: str,
    contact_mode: str,
    *,
    support_mode: str | None = None,
    bilateral_symmetry: str | None = None,
    alternation: bool = False,
    tempo_bucket: str | None = None,
) -> dict[str, Any]:
    return {
        "dominant_axis": dominant_axis,
        "repeat_mode": repeat_mode,
        "phase_template": phase_template,
        "contact_mode": contact_mode,
        "support_mode": support_mode or contact_mode,
        "bilateral_symmetry": bilateral_symmetry or ("bilateral" if "bi" in dominant_axis else "unilateral"),
        "alternation": bool(alternation),
        "tempo_bucket": tempo_bucket or "medium",
    }


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3 or window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _segments(mask: np.ndarray, *, min_len: int = 4, max_gap: int = 3) -> list[tuple[int, int]]:
    active = np.flatnonzero(mask)
    if active.size == 0:
        return []
    groups: list[list[int]] = [[int(active[0]), int(active[0])]]
    for idx in active[1:]:
        idx = int(idx)
        if idx - groups[-1][1] <= max_gap:
            groups[-1][1] = idx
        else:
            groups.append([idx, idx])
    return [(s, e) for s, e in groups if e - s + 1 >= min_len]


def _scale(joints: np.ndarray) -> float:
    left = joints[:, JOINT_INDEX["L_Shoulder"]]
    right = joints[:, JOINT_INDEX["R_Shoulder"]]
    width = np.linalg.norm(left - right, axis=-1)
    return float(np.nanmedian(width)) if len(width) else 0.35


def _heading_vectors(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_hip = joints[:, JOINT_INDEX["L_Hip"]][:, [0, 2]]
    right_hip = joints[:, JOINT_INDEX["R_Hip"]][:, [0, 2]]
    left_shoulder = joints[:, JOINT_INDEX["L_Shoulder"]][:, [0, 2]]
    right_shoulder = joints[:, JOINT_INDEX["R_Shoulder"]][:, [0, 2]]
    across = (right_hip - left_hip) + (right_shoulder - left_shoulder)
    right = across / np.clip(np.linalg.norm(across, axis=-1, keepdims=True), 1e-8, None)
    right[:, 0] = _smooth(right[:, 0])
    right[:, 1] = _smooth(right[:, 1])
    right = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8, None)
    forward = -np.stack([-right[:, 1], right[:, 0]], axis=-1)
    return forward.astype(np.float32), right.astype(np.float32)


def _root_path_features(joints: np.ndarray) -> dict[str, float]:
    root = np.asarray(joints[:, JOINT_INDEX["Pelvis"], [0, 2]], dtype=np.float32)
    if len(root) < 4:
        return {"path_length": 0.0, "net": 0.0, "circle_score": 0.0, "curvature": 0.0}
    diffs = np.diff(root, axis=0)
    step = np.linalg.norm(diffs, axis=-1)
    path = float(step.sum())
    net = float(np.linalg.norm(root[-1] - root[0]))
    centered = root - root.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=-1)
    radius_mean = float(radius.mean())
    radius_cv = float(radius.std() / max(radius_mean, 1e-6))
    if path <= 1e-6 or radius_mean <= 1e-6:
        circle_score = 0.0
    else:
        circle_score = float(np.clip((path / (2.0 * np.pi * radius_mean)) * (1.0 - radius_cv), 0.0, 2.0))
    angles = np.unwrap(np.arctan2(centered[:, 0], centered[:, 1]))
    curvature = float(abs(angles[-1] - angles[0]))
    return {"path_length": path, "net": net, "circle_score": circle_score, "curvature": curvature}


def _torso_pitch_signal(joints: np.ndarray) -> np.ndarray:
    pelvis = joints[:, JOINT_INDEX["Pelvis"]]
    neck = joints[:, JOINT_INDEX["Neck"]]
    vec = neck - pelvis
    forward, _ = _heading_vectors(joints)
    horizontal = np.sum(vec[:, [0, 2]] * forward, axis=-1)
    norm = np.linalg.norm(vec, axis=-1)
    # Use a bounded angle-like ratio instead of horizontal / vertical; the latter
    # explodes during inverted acrobatics when the torso vertical component flips.
    ratio = horizontal / np.clip(norm, 1e-6, None)
    ratio = np.where(vec[:, 1] > 0.12, ratio, 0.0)
    return _smooth(np.clip(ratio, -1.0, 1.0))


def _knee_flex_signal(joints: np.ndarray) -> np.ndarray:
    pelvis_y = joints[:, JOINT_INDEX["Pelvis"], 1]
    ankles_y = 0.5 * (joints[:, JOINT_INDEX["L_Ankle"], 1] + joints[:, JOINT_INDEX["R_Ankle"], 1])
    return _smooth(pelvis_y - ankles_y)


def _wrist_height(joints: np.ndarray, side: str) -> np.ndarray:
    wrist = JOINT_INDEX["L_Wrist"] if side == "left" else JOINT_INDEX["R_Wrist"]
    shoulder = JOINT_INDEX["L_Shoulder"] if side == "left" else JOINT_INDEX["R_Shoulder"]
    return _smooth(joints[:, wrist, 1] - joints[:, shoulder, 1])


def _leg_lift_signal(joints: np.ndarray, side: str) -> np.ndarray:
    ankle = JOINT_INDEX["L_Ankle"] if side == "left" else JOINT_INDEX["R_Ankle"]
    pelvis = JOINT_INDEX["Pelvis"]
    return _smooth(joints[:, ankle, 1] - joints[:, pelvis, 1])


def _leg_forward_signal(joints: np.ndarray, side: str) -> np.ndarray:
    ankle = JOINT_INDEX["L_Ankle"] if side == "left" else JOINT_INDEX["R_Ankle"]
    pelvis = JOINT_INDEX["Pelvis"]
    forward, _ = _heading_vectors(joints)
    rel = joints[:, ankle, [0, 2]] - joints[:, pelvis, [0, 2]]
    return _smooth(np.sum(rel * forward, axis=-1))


def _leg_forward_extension_signal(joints: np.ndarray, side: str) -> np.ndarray:
    ankle = JOINT_INDEX["L_Ankle"] if side == "left" else JOINT_INDEX["R_Ankle"]
    hip = JOINT_INDEX["L_Hip"] if side == "left" else JOINT_INDEX["R_Hip"]
    forward, _ = _heading_vectors(joints)
    rel = joints[:, ankle, [0, 2]] - joints[:, hip, [0, 2]]
    return _smooth(np.sum(rel * forward, axis=-1))


def _body_inversion_signal(joints: np.ndarray) -> np.ndarray:
    pelvis_y = joints[:, JOINT_INDEX["Pelvis"], 1]
    head_y = joints[:, JOINT_INDEX["Head"], 1]
    wrists_y = 0.5 * (joints[:, JOINT_INDEX["L_Wrist"], 1] + joints[:, JOINT_INDEX["R_Wrist"], 1])
    ankles_y = 0.5 * (joints[:, JOINT_INDEX["L_Ankle"], 1] + joints[:, JOINT_INDEX["R_Ankle"], 1])
    return _smooth((pelvis_y - head_y) + 0.5 * (ankles_y - wrists_y))


def _event_span_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.asarray(values[start : end + 1], dtype=np.float32)


def _add_state_events(out: list[dict[str, Any]], joints: np.ndarray) -> None:
    n = len(joints)
    scale = _scale(joints)

    torso = _torso_pitch_signal(joints)
    torso_threshold = max(0.26, 0.75 * scale)
    for start, end in _segments(torso > torso_threshold, min_len=8):
        vals = _event_span_values(torso, start, end)
        out.append(_event(
            "torso", "TORSO_POSTURE", "TORSO_HUNCHED_FORWARD", start, end,
            direction="forward", role="state", optional_semantic_name="torso_hunched_forward",
            magnitude=float(vals.max()), unit="ratio", confidence=0.64,
            source="semantic_joints", supporting_units=["torso_pitch_signal"],
            motion_signature=_sig("torso_pitch", "state", "hunched_forward", "free", support_mode="none", bilateral_symmetry="axial"),
            metadata={"rule": "bounded_torso_pitch>threshold", "threshold": float(torso_threshold)},
        ))

    knee = _knee_flex_signal(joints)
    base = float(np.percentile(knee, 85))
    low = base - knee
    for start, end in _segments(low > max(0.12, 0.35 * scale), min_len=8):
        vals = _event_span_values(low, start, end)
        out.append(_event(
            "whole_body", "WHOLE_BODY_POSTURE", "WB_SQUAT_HOLD", start, end,
            direction="low", role="state", optional_semantic_name="squat_hold",
            magnitude=float(vals.max()), unit="m", confidence=0.68,
            source="semantic_joints", supporting_units=["pelvis_ankle_height_drop"],
            motion_signature=_sig("knee_flexion", "state", "squat_hold", "grounded", support_mode="feet", bilateral_symmetry="axial"),
            metadata={"rule": "pelvis_to_ankle_height_drop>threshold", "threshold": float(max(0.12, 0.35 * scale))},
        ))

    for side in ("left", "right"):
        height = _wrist_height(joints, side)
        for start, end in _segments(height > max(0.10, 0.28 * scale), min_len=5):
            vals = _event_span_values(height, start, end)
            family = "LEFT_ARM_POSTURE" if side == "left" else "RIGHT_ARM_POSTURE"
            cluster = "LA_HAND_HIGH" if side == "left" else "RA_HAND_HIGH"
            out.append(_event(
                f"{side}_arm", family, cluster, start, end,
                direction="up", role="state", optional_semantic_name=f"{side}_hand_high",
                magnitude=float(vals.max()), unit="m", confidence=0.64,
                source="semantic_joints", supporting_units=[f"{side}_wrist_above_shoulder"],
                motion_signature=_sig("hand_height", "state", "hand_high", "free", support_mode="none", bilateral_symmetry="unilateral"),
                metadata={"rule": "wrist_y-shoulder_y>threshold", "threshold": float(max(0.10, 0.28 * scale))},
            ))

    for side in ("left", "right"):
        forward_ext = _leg_forward_extension_signal(joints, side)
        prominence = forward_ext - float(np.median(forward_ext))
        velocity = np.abs(np.diff(forward_ext, prepend=forward_ext[0]))
        threshold = max(0.16, 0.42 * scale)
        prominence_threshold = max(0.09, 0.28 * scale)
        candidate_segments = _segments(
            (forward_ext > threshold) & (prominence > prominence_threshold),
            min_len=5,
            max_gap=2,
        )
        for start, end in candidate_segments:
            vals = _event_span_values(forward_ext, start, end)
            prom_vals = _event_span_values(prominence, start, end)
            vel_vals = _event_span_values(velocity, start, end)
            family = "LEFT_LEG_ACTION" if side == "left" else "RIGHT_LEG_ACTION"
            mean_velocity = float(vel_vals.mean()) if len(vel_vals) else 0.0
            held_pose = (end - start + 1) >= 18 and mean_velocity < 0.055
            if side == "left":
                cluster = "LL_LEG_FORWARD_POSE" if held_pose else "LL_KICK_FORWARD"
            else:
                cluster = "RL_LEG_FORWARD_POSE" if held_pose else "RL_KICK_FORWARD"
            out.append(_event(
                f"{side}_leg", family, cluster, start, end,
                direction="forward", role="primitive", optional_semantic_name=f"{side}_leg_forward_pose" if held_pose else f"{side}_leg_kick_forward",
                magnitude=float(vals.max()), unit="m", confidence=0.64 if held_pose else 0.62,
                source="semantic_joints", supporting_units=[f"{side}_ankle_forward_from_hip"],
                motion_signature=_sig("leg_forward_extension", "state" if held_pose else "single", "forward_pose" if held_pose else "kick_forward", "grounded", support_mode="opposite_foot", bilateral_symmetry="unilateral"),
                metadata={
                    "rule": "ankle_forward_extension>threshold and prominence>threshold",
                    "threshold": float(threshold),
                    "prominence_threshold": float(prominence_threshold),
                    "prominence": float(prom_vals.max()),
                    "mean_velocity": mean_velocity,
                    "segment_count_for_side": len(candidate_segments),
                },
            ))

    inv = _body_inversion_signal(joints)
    for start, end in _segments(inv > max(0.18, 0.45 * scale), min_len=4):
        vals = _event_span_values(inv, start, end)
        path = _root_path_features(joints[start : end + 1])
        cluster = "WB_CARTWHEEL_CANDIDATE" if path["path_length"] >= 0.25 else "WB_INVERTED_ROTATION_CANDIDATE"
        out.append(_event(
            "whole_body", "WHOLE_BODY_ACROBATICS", cluster, start, end,
            direction="inverted", role="composed", optional_semantic_name="inverted_acrobatic_rotation",
            magnitude=float(vals.max()), unit="m", confidence=0.58,
            source="semantic_joints", supporting_units=["body_inversion_signal"],
            motion_signature=_sig("body_inversion", "single", "inverted_rotation", "hand_or_air_support", support_mode="hands_or_air", bilateral_symmetry="axial"),
            metadata={"rule": "pelvis/head+ankle/wrist inversion proxy", **path},
        ))

    path = _root_path_features(joints)
    if path["path_length"] >= 2.0 and path["circle_score"] >= 0.70 and path["curvature"] >= 2.4:
        out.append(_event(
            "whole_body", "WHOLE_BODY_PATH", "ROOT_CIRCULAR_PATH", 0, n - 1,
            direction="circular", role="state", optional_semantic_name="circular_root_path",
            magnitude=path["path_length"], signed_delta=path["curvature"], unit="m", confidence=0.64,
            source="semantic_joints", supporting_units=["root_path_curvature"],
            motion_signature=_sig("root_path_shape", "state", "circular_path", "grounded", support_mode="feet", bilateral_symmetry="axial"),
            metadata={"rule": "path_length>=2.0,circle_score>=0.70,curvature>=2.4", **path},
        ))


def _add_climb_proxy_events(out: list[dict[str, Any]], joints: np.ndarray) -> None:
    root_y = _smooth(joints[:, JOINT_INDEX["Pelvis"], 1])
    if len(root_y) < 8:
        return
    cumulative_up = float(root_y[-1] - np.percentile(root_y[: max(4, len(root_y) // 5)], 20))
    scale = _scale(joints)
    path = _root_path_features(joints)
    has_low = any(e.get("cluster_id") in {"WB_SQUAT_HOLD", "TORSO_HUNCHED_FORWARD"} for e in out)
    if cumulative_up < max(0.32, 0.80 * scale) or path["path_length"] < 1.2 or not has_low:
        return
    out.append(_event(
        "whole_body", "WHOLE_BODY_CLIMB", "CLIMB_UP_OVER_PROXY", 0, len(joints) - 1,
        direction="up_over", role="composed", optional_semantic_name="climb_up_over_proxy",
        magnitude=cumulative_up, unit="m", confidence=0.56,
        source="semantic_joints", supporting_units=["root_height_gain", "low_body_or_hunch_state"],
        motion_signature=_sig("height_path_posture", "sequence", "duck_then_climb_up", "grounded_or_object", support_mode="feet_or_hands", bilateral_symmetry="axial"),
        metadata={"rule": "root_height_gain with low/hunched state and nontrivial path", "root_height_gain": cumulative_up, **path},
    ))


def build_semantic_joint_events(joints: np.ndarray | None) -> list[dict[str, Any]]:
    if joints is None:
        return []
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] <= JOINT_INDEX["R_Wrist"]:
        return []
    out: list[dict[str, Any]] = []
    _add_state_events(out, arr)
    _add_climb_proxy_events(out, arr)
    out.sort(key=lambda e: (int(e["start_frame"]), int(e["end_frame"]), str(e["super_family"]), str(e["cluster_id"])))
    return out
