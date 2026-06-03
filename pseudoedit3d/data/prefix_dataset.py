from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pseudoedit3d.constants import BODY_PART_TO_JOINTS, SMPLH_NUM_JOINTS, SMPLH_POSE_DIM
from pseudoedit3d.edit.action_program import build_goal_spec, goal_spec_to_numpy
from pseudoedit3d.edit.attributes import compute_motion_statistics, extract_upper_body_proxy_attributes, resolve_proxy_attribute_key
from pseudoedit3d.edit.schema import EditProgram, MultiEditProgram, LabelSchema, load_label_schema
from pseudoedit3d.edit.skill_context import SKILL_LABEL_TO_PROMPT_PHRASE, infer_skill_context, summarize_skill_attribute
from pseudoedit3d.edit.verbalizer import verbalize_program
from pseudoedit3d.edit.hierarchical_atomic import extract_all_atomic_candidates
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


def _make_prefix_condition_masked(
    poses: np.ndarray,
    trans: np.ndarray,
    prefix_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = poses.shape[0]
    prefix = min(max(1, prefix_frames), num_frames)
    conditioned_pose = np.zeros_like(poses)
    conditioned_trans = np.zeros_like(trans)
    conditioned_pose[:prefix] = poses[:prefix]
    conditioned_trans[:prefix] = trans[:prefix]
    conditioning_mask = np.zeros((num_frames,), dtype=np.float32)
    conditioning_mask[:prefix] = 1.0
    return conditioned_pose, conditioned_trans, conditioning_mask


def _build_future_mask(num_frames: int, prefix_frames: int) -> np.ndarray:
    mask = np.zeros((num_frames,), dtype=np.float32)
    mask[min(prefix_frames, num_frames) :] = 1.0
    return mask


def _infer_valid_end_frame(
    poses: np.ndarray,
    active_joint_ids: list[int],
    start_frame: int,
    peak_frame: int,
    min_quiet_len: int = 4,
) -> int:
    num_frames = poses.shape[0]
    if peak_frame >= num_frames - 1:
        return num_frames - 1
    joint_ids = active_joint_ids if active_joint_ids else list(range(min(22, poses.shape[1])))
    delta = np.diff(poses[:, joint_ids, :], axis=0)
    energy = np.linalg.norm(delta, axis=-1).mean(axis=1)
    active_window = energy[max(0, start_frame - 1) : max(start_frame, peak_frame)]
    if active_window.size == 0:
        active_window = energy[: max(1, peak_frame)]
    ref = float(np.median(active_window)) if active_window.size > 0 else 0.02
    peak_ref = float(np.max(active_window)) if active_window.size > 0 else ref
    quiet_thr = max(0.008, 0.18 * peak_ref, 0.35 * ref)
    quiet_count = 0
    search_start = min(num_frames - 1, max(start_frame + 2, peak_frame + 1))
    for frame_idx in range(search_start, num_frames - 1):
        if float(energy[frame_idx]) <= quiet_thr:
            quiet_count += 1
        else:
            quiet_count = 0
        if quiet_count >= min_quiet_len:
            return frame_idx - quiet_count + 1
    return num_frames - 1


def _sample_delta_bin(schema: LabelSchema, rng) -> str:
    return schema.delta_bin_keys[int(rng.integers(0, len(schema.delta_bin_keys)))]


def _delta_to_bin(schema: LabelSchema, delta_abs: float) -> str:
    best_key = schema.delta_bin_keys[0]
    best_dist = float("inf")
    for key in schema.delta_bin_keys:
        ref = float(schema.delta_bin(key).default_degrees or 0.0)
        dist = abs(ref - delta_abs)
        if dist < best_dist:
            best_key = key
            best_dist = dist
    return best_key


def _attribute_from_proxy_key(proxy_key: str) -> tuple[str, str, str, str]:
    if proxy_key == "left_shoulder_pitch_proxy_deg":
        return "left_arm", "raise", "lower", "deg"
    if proxy_key == "right_shoulder_pitch_proxy_deg":
        return "right_arm", "raise", "lower", "deg"
    if proxy_key == "both_shoulder_pitch_proxy_deg":
        return "both_arms", "raise", "lower", "deg"
    if proxy_key == "left_elbow_flex_proxy_deg":
        return "left_arm", "bend", "extend", "deg"
    if proxy_key == "right_elbow_flex_proxy_deg":
        return "right_arm", "bend", "extend", "deg"
    if proxy_key == "both_elbow_flex_proxy_deg":
        return "both_arms", "bend", "extend", "deg"
    if proxy_key == "torso_pitch_proxy_deg":
        return "torso", "lean_forward", "lean_backward", "deg"
    if proxy_key == "torso_roll_proxy_deg":
        return "torso", "lean_left", "lean_right", "deg"
    if proxy_key == "root_yaw_proxy_deg":
        return "whole_body", "turn_left", "turn_right", "deg"
    if proxy_key == "root_height_proxy":
        return "whole_body", "jump_up", "land", "m"
    raise KeyError(f"Unsupported proxy key: {proxy_key}")


def _build_relative_program(
    schema: LabelSchema,
    skill_context: dict,
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
    rng,
) -> EditProgram:
    skill_label = skill_context["skill_label"]
    if skill_label == "periodic_arm_motion":
        part = skill_context.get("periodic_limb", "both_arms")
        periodic_state = skill_context.get("periodic_states", {}).get(part, {})
        attr_key = periodic_state.get("attr_key", skill_context["dominant_attr_key"])
        part, positive_attr, negative_attr = _attribute_from_proxy_key(attr_key)
        direction = "increase" if int(rng.integers(0, 2)) == 0 else "decrease"
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
        candidates = ["both_shoulder_pitch_proxy_deg", "torso_pitch_proxy_deg"]
        attr_key = candidates[int(rng.integers(0, len(candidates)))]
        part, positive_attr, negative_attr = _attribute_from_proxy_key(attr_key)
        direction = "increase" if int(rng.integers(0, 2)) == 0 else "decrease"
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
        direction = "increase" if int(rng.integers(0, 2)) == 0 else "decrease"
        attribute = positive_attr if direction == "increase" else negative_attr
        preserve_mode = "all_non_target"
        metadata = {
            "source_attr_mean_deg": float(np.mean(proxy_attributes[attr_key])),
            "source_attr_amplitude_deg": float(summarize_skill_attribute(proxy_attributes[attr_key])["amplitude_deg"]),
            "relative_skill_parameter": "attribute_delta_deg",
            "preserve_amplitude": False,
        }

    delta_bin = _sample_delta_bin(schema, rng)
    delta_value_deg = float(schema.delta_bin(delta_bin).default_degrees or 0.0)
    signed_delta = delta_value_deg if direction == "increase" else -delta_value_deg
    if metadata.get("relative_skill_parameter") == "offset_deg":
        metadata["target_offset_deg"] = float(metadata["source_attr_mean_deg"] + signed_delta)
    else:
        metadata["target_offset_deg"] = float("nan")

    start_frame = min(prefix_frames, num_frames - 1)
    end_frame = num_frames - 1
    if num_frames - start_frame > 10:
        start_frame = int(rng.integers(prefix_frames, max(prefix_frames + 1, num_frames - 9)))
        end_frame = int(rng.integers(start_frame + 4, num_frames))

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
        unit=unit,
        preserve_parts=[],
        preserve_mode=preserve_mode,
        skill_label=skill_label,
        skill_phase=float(skill_context["skill_phase"]),
        tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
        metadata=metadata,
    )


def _build_semantic_continue_program(
    schema: LabelSchema,
    skill_context: dict,
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
    rng,
) -> EditProgram:
    candidate_keys = [
        "left_shoulder_pitch_proxy_deg",
        "right_shoulder_pitch_proxy_deg",
        "both_shoulder_pitch_proxy_deg",
        "left_elbow_flex_proxy_deg",
        "right_elbow_flex_proxy_deg",
        "both_elbow_flex_proxy_deg",
        "torso_pitch_proxy_deg",
        "torso_roll_proxy_deg",
        "root_yaw_proxy_deg",
        "root_height_proxy",
    ]
    if skill_context["skill_label"] == "periodic_arm_motion":
        periodic_limb = skill_context.get("periodic_limb", "both_arms")
        periodic_state = skill_context.get("periodic_states", {}).get(periodic_limb, {})
        preferred = periodic_state.get("attr_key")
        if preferred in candidate_keys:
            candidate_keys = [preferred] + [k for k in candidate_keys if k != preferred]

    best = None
    for key in candidate_keys:
        values = np.asarray(proxy_attributes[key], dtype=np.float32)
        source_current = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) == 0:
            continue
        max_idx = int(np.argmax(future))
        min_idx = int(np.argmin(future))
        max_delta = float(future[max_idx] - source_current)
        min_delta = float(future[min_idx] - source_current)
        if abs(max_delta) >= abs(min_delta):
            delta = max_delta
            peak_rel_idx = max_idx
        else:
            delta = min_delta
            peak_rel_idx = min_idx
        score = abs(delta)
        if best is None or score > best["score"]:
            best = {
                "attr_key": key,
                "delta": delta,
                "peak_rel_idx": peak_rel_idx,
                "source_current": source_current,
                "future_peak": float(future[peak_rel_idx]),
                "score": float(score),
            }

    if best is None:
        key = skill_context["dominant_attr_key"]
        values = np.asarray(proxy_attributes[key], dtype=np.float32)
        best = {
            "attr_key": key,
            "delta": float(values[-1] - values[prefix_frames - 1]),
            "peak_rel_idx": len(values[prefix_frames:]) - 1,
            "source_current": float(values[prefix_frames - 1]),
            "future_peak": float(values[-1]),
            "score": float(abs(values[-1] - values[prefix_frames - 1])),
        }

    part, positive_attr, negative_attr, unit = _attribute_from_proxy_key(best["attr_key"])
    direction = "increase" if best["delta"] >= 0.0 else "decrease"
    attribute = positive_attr if direction == "increase" else negative_attr
    delta_value_deg = float(best["delta"])
    delta_bin = _delta_to_bin(schema, abs(delta_value_deg))
    start_frame = prefix_frames
    end_frame = min(num_frames - 1, prefix_frames + max(4, int(best["peak_rel_idx"])))
    preserve_mode = "skill_structure" if skill_context.get("is_relative_friendly", False) else "all_non_target"

    return EditProgram(
        part=part,
        attribute=attribute,
        delta_bin=delta_bin,
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="ignore",
        attribute_key=best["attr_key"],
        direction=direction,
        delta_value_deg=delta_value_deg,
        source_type="same_clip_prefix",
        schema_version=schema.schema_version,
        input_mode="motion_prefix",
        operator="add",
        reference="current_state",
        unit=unit,
        preserve_parts=[],
        preserve_mode=preserve_mode,
        skill_label=skill_context["skill_label"],
        skill_phase=float(skill_context["skill_phase"]),
        tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
        metadata={
            "task_mode": "semantic_continue",
            "source_attr_current_deg": best["source_current"],
            "future_peak_deg": best["future_peak"],
            "source_attr_mean_deg": float(np.mean(proxy_attributes[best["attr_key"]][:prefix_frames])),
            "source_attr_amplitude_deg": float(summarize_skill_attribute(proxy_attributes[best["attr_key"]][:prefix_frames])["amplitude_deg"]),
            "relative_skill_parameter": "attribute_delta_deg",
            "target_offset_deg": float("nan"),
            "preserve_amplitude": False,
        },
    )


def _build_atomic_program(
    schema: LabelSchema,
    proxy_attributes: dict[str, np.ndarray],
    num_frames: int,
    prefix_frames: int,
    rng,
) -> EditProgram:
    candidate_keys = [
        "left_shoulder_pitch_proxy_deg",
        "right_shoulder_pitch_proxy_deg",
        "both_shoulder_pitch_proxy_deg",
        "left_elbow_flex_proxy_deg",
        "right_elbow_flex_proxy_deg",
        "both_elbow_flex_proxy_deg",
        "torso_pitch_proxy_deg",
        "torso_roll_proxy_deg",
        "root_yaw_proxy_deg",
        "root_height_proxy",
    ]
    best = None
    for key in candidate_keys:
        values = np.asarray(proxy_attributes[key], dtype=np.float32)
        source_value = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) == 0:
            continue
        max_idx = int(np.argmax(future))
        min_idx = int(np.argmin(future))
        max_delta = float(future[max_idx] - source_value)
        min_delta = float(future[min_idx] - source_value)
        if abs(max_delta) >= abs(min_delta):
            delta = max_delta
            peak_rel_idx = max_idx
        else:
            delta = min_delta
            peak_rel_idx = min_idx
        score = abs(delta)
        if best is None or score > best["score"]:
            best = {
                "attr_key": key,
                "delta": delta,
                "peak_rel_idx": peak_rel_idx,
                "source_value": source_value,
                "future_peak": float(future[peak_rel_idx]),
                "score": float(score),
            }

    if best is None:
        key = candidate_keys[0]
        values = np.asarray(proxy_attributes[key], dtype=np.float32)
        best = {
            "attr_key": key,
            "delta": float(values[-1] - values[prefix_frames - 1]),
            "peak_rel_idx": len(values[prefix_frames:]) - 1,
            "source_value": float(values[prefix_frames - 1]),
            "future_peak": float(values[-1]),
            "score": float(abs(values[-1] - values[0])),
        }

    part, positive_attr, negative_attr, unit = _attribute_from_proxy_key(best["attr_key"])
    direction = "increase" if best["delta"] >= 0.0 else "decrease"
    attribute = positive_attr if direction == "increase" else negative_attr
    delta_value_deg = float(best["delta"])
    delta_bin = _delta_to_bin(schema, abs(delta_value_deg))
    future_values = np.asarray(proxy_attributes[best["attr_key"]][prefix_frames:], dtype=np.float32)
    threshold = 0.2 * abs(delta_value_deg)
    activation_idx = 0
    for i, value in enumerate(future_values):
        if abs(float(value - best["source_value"])) >= threshold:
            activation_idx = i
            break
    start_frame = min(num_frames - 1, prefix_frames + activation_idx)
    end_frame = min(num_frames - 1, prefix_frames + max(activation_idx + 4, best["peak_rel_idx"]))

    return EditProgram(
        part=part,
        attribute=attribute,
        delta_bin=delta_bin,
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="ignore",
        attribute_key=best["attr_key"],
        direction=direction,
        delta_value_deg=delta_value_deg,
        source_type="same_clip_atomic",
        schema_version=schema.schema_version,
        input_mode="motion_prefix",
        operator="add",
        reference="current_state",
        preserve_parts=[],
        preserve_mode="all_non_target",
        skill_label="static_pose",
        skill_phase=float("nan"),
        tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
        metadata={
            "task_mode": "atomic_realize",
            "source_attr_current_deg": best["source_value"],
            "future_peak_deg": best["future_peak"],
            "source_attr_mean_deg": float(best["source_value"]),
            "source_attr_amplitude_deg": 0.0,
            "relative_skill_parameter": "attribute_delta_deg",
            "target_offset_deg": float("nan"),
            "preserve_amplitude": False,
        },
    )




def _build_multi_atomic_program(
    schema: LabelSchema,
    proxy_attributes: dict[str, np.ndarray],
    num_frames: int,
    prefix_frames: int,
    rng,
    max_events: int = 6,
    overlap_tol: int = 3,
) -> MultiEditProgram:
    candidate_keys = [
        "root_yaw_proxy_deg",
        "root_height_proxy",
        "left_shoulder_pitch_proxy_deg",
        "right_shoulder_pitch_proxy_deg",
        "both_shoulder_pitch_proxy_deg",
        "left_elbow_flex_proxy_deg",
        "right_elbow_flex_proxy_deg",
        "both_elbow_flex_proxy_deg",
        "torso_pitch_proxy_deg",
        "torso_roll_proxy_deg",
    ]
    candidates = []
    for key in candidate_keys:
        values = np.asarray(proxy_attributes[key], dtype=np.float32)
        source_value = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) == 0:
            continue
        max_idx = int(np.argmax(future))
        min_idx = int(np.argmin(future))
        max_delta = float(future[max_idx] - source_value)
        min_delta = float(future[min_idx] - source_value)
        if abs(max_delta) >= abs(min_delta):
            delta = max_delta
            peak_rel_idx = max_idx
        else:
            delta = min_delta
            peak_rel_idx = min_idx
        min_thr = 10.0 if key == "root_yaw_proxy_deg" else (0.08 if key == "root_height_proxy" else 6.0)
        if abs(delta) < min_thr:
            continue
        future_values = np.asarray(proxy_attributes[key][prefix_frames:], dtype=np.float32)
        threshold = 0.2 * abs(delta)
        activation_idx = 0
        for i, value in enumerate(future_values):
            if abs(float(value - source_value)) >= threshold:
                activation_idx = i
                break
        start_frame = min(num_frames - 1, prefix_frames + activation_idx)
        end_frame = min(num_frames - 1, prefix_frames + max(activation_idx + 4, peak_rel_idx))
        part, positive_attr, negative_attr, unit = _attribute_from_proxy_key(key)
        direction = "increase" if delta >= 0.0 else "decrease"
        if key == "root_height_proxy":
            velocity = np.diff(values, prepend=values[:1])
            local_vel = float(np.max(velocity[prefix_frames:prefix_frames + peak_rel_idx + 1])) if peak_rel_idx >= 0 else 0.0
            attribute = "jump_up" if local_vel > 0 else (positive_attr if direction == "increase" else negative_attr)
        else:
            attribute = positive_attr if direction == "increase" else negative_attr
        delta_bin = _delta_to_bin(schema, abs(delta))
        priority = 2.0 if key == "root_yaw_proxy_deg" else (1.5 if key == "root_height_proxy" else 1.0)
        candidates.append({
            "attr_key": key,
            "part": part,
            "attribute": attribute,
            "direction": direction,
            "delta_value_deg": float(delta),
            "delta_bin": delta_bin,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "score": float(abs(delta)) * priority,
            "source_value": source_value,
            "future_peak": float(future[peak_rel_idx]),
            "unit": unit,
        })
    candidates.sort(key=lambda item: (item["start_frame"], -item["score"]))
    global_candidates = [
        {
            "attr_key": cand.attribute_key,
            "part": cand.part,
            "attribute": cand.attribute,
            "direction": cand.direction,
            "delta_value_deg": cand.delta_value,
            "delta_bin": _delta_to_bin(schema, abs(cand.delta_value)),
            "start_frame": int(cand.start_frame),
            "end_frame": int(cand.end_frame),
            "score": float(cand.score),
            "source_value": float(cand.metadata.get("source_attr_current_deg", 0.0)),
            "future_peak": float(cand.metadata.get("future_peak_deg", 0.0)),
            "unit": cand.unit,
            "metadata": cand.metadata,
        }
        for cand in extract_all_atomic_candidates(proxy_attributes, prefix_frames, num_frames)
    ]
    candidates.extend(global_candidates)
    global_first = [c for c in candidates if c.get("part") == "whole_body"]
    local_rest = [c for c in candidates if c.get("part") != "whole_body"]
    global_first.sort(key=lambda item: (-item["score"], item["start_frame"]))
    local_rest.sort(key=lambda item: (item["start_frame"], -item["score"]))
    # Keep the strongest few whole-body events first so turn/jump/land can coexist before local edits fill the budget.
    ordered_candidates = global_first[:3] + local_rest + global_first[3:]
    selected = []
    for cand in ordered_candidates:
        conflict = False
        for prev in selected:
            overlap = not (cand["end_frame"] < prev["start_frame"] - overlap_tol or cand["start_frame"] > prev["end_frame"] + overlap_tol)
            same_attr = cand["attr_key"] == prev["attr_key"]
            same_part = cand["part"] == prev["part"]
            same_semantic = same_attr or (same_part and cand["part"] != "whole_body")
            if overlap and same_semantic:
                conflict = True
                break
        if conflict:
            continue
        selected.append(cand)
        if len(selected) >= max_events:
            break
    if not selected:
        best = _build_atomic_program(schema, proxy_attributes, num_frames, prefix_frames, rng)
        return MultiEditProgram(edits=[best], schema_version=schema.schema_version, metadata={"task_mode": "multi_atomic_realize"})
    edits = []
    for cand in selected:
        edits.append(EditProgram(
            part=cand["part"],
            attribute=cand["attribute"],
            delta_bin=cand["delta_bin"],
            start_frame=cand["start_frame"],
            end_frame=cand["end_frame"],
            contact_policy="ignore",
            attribute_key=cand["attr_key"],
            direction=cand["direction"],
            delta_value_deg=cand["delta_value_deg"],
            source_type="same_clip_multi_atomic",
            schema_version=schema.schema_version,
            input_mode="motion_prefix",
            operator="add",
            reference="current_state",
            unit=cand["unit"],
            preserve_parts=[],
            preserve_mode="all_non_target",
            skill_label="static_pose",
            skill_phase=float("nan"),
            tolerance_deg=float(schema.prompt_defaults.get("default_tolerance_deg", 5.0)),
            metadata=cand.get("metadata", {
                "task_mode": "multi_atomic_realize",
                "source_attr_current_deg": cand["source_value"],
                "future_peak_deg": cand["future_peak"],
                "source_attr_mean_deg": float(cand["source_value"]),
                "source_attr_amplitude_deg": 0.0,
                "relative_skill_parameter": "attribute_delta_deg",
                "target_offset_deg": float("nan"),
                "preserve_amplitude": False,
            }),
        ))
    return MultiEditProgram(edits=edits, schema_version=schema.schema_version, metadata={"task_mode": "multi_atomic_realize"})
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


def _dummy_sequence_spec(edit_dim: int, max_edits: int = 4) -> dict[str, np.ndarray]:
    return {
        "seq_edit_vectors": np.zeros((max_edits, edit_dim), dtype=np.float32),
        "seq_start_frames": np.zeros((max_edits,), dtype=np.int64),
        "seq_end_frames": np.zeros((max_edits,), dtype=np.int64),
        "seq_delta_deg": np.zeros((max_edits,), dtype=np.float32),
        "seq_direction_sign": np.ones((max_edits,), dtype=np.float32),
        "seq_num_edits": np.asarray(0, dtype=np.int64),
    }


def _sequence_spec_from_programs(programs: list[EditProgram], schema: LabelSchema, max_edits: int = 4) -> dict[str, np.ndarray]:
    spec = _dummy_sequence_spec(schema.vector_dim, max_edits=max_edits)
    count = min(len(programs), max_edits)
    spec["seq_num_edits"] = np.asarray(count, dtype=np.int64)
    for idx, program in enumerate(programs[:count]):
        spec["seq_edit_vectors"][idx] = np.asarray(program.to_vector(schema), dtype=np.float32)
        spec["seq_start_frames"][idx] = int(program.start_frame)
        spec["seq_end_frames"][idx] = int(program.end_frame)
        spec["seq_delta_deg"][idx] = float(program.delta_value_deg or 0.0)
        if program.direction == "decrease":
            spec["seq_direction_sign"][idx] = -1.0
        else:
            spec["seq_direction_sign"][idx] = 1.0
    return spec


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
        seed: int = 42,
        input_source_mode: str = "target_prefix",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.schema = load_label_schema(label_schema_path) if label_schema_path else load_label_schema()
        self.prompt_style = prompt_style
        self.tokenizer = CharTokenizer(max_length=prompt_max_length)
        self.prefix_frames = max(1, int(prefix_frames))
        self.task_mode = task_mode
        self.contact_filter = contact_filter
        self.seed = int(seed)
        self.input_source_mode = input_source_mode
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

        proxy_attributes = extract_upper_body_proxy_attributes(poses, trans=trans)
        motion_stats = compute_motion_statistics(poses, trans)
        if self.input_source_mode == "target_prefix_masked":
            prefix_pose, prefix_trans, conditioning_frame_mask = _make_prefix_condition_masked(poses, trans, self.prefix_frames)
        else:
            prefix_pose, prefix_trans, conditioning_frame_mask = _make_prefix_condition(poses, trans, self.prefix_frames)
        future_mask = _build_future_mask(poses.shape[0], self.prefix_frames)
        skill_context = infer_skill_context(
            proxy_attributes,
            motion_stats=motion_stats,
            num_frames=poses.shape[0],
            anchor_frame=max(0, self.prefix_frames - 1),
        )
        rng = np.random.default_rng(self.seed + index * 1000003)
        sequence_spec = _dummy_sequence_spec(self.schema.vector_dim)

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
                "source_prefix_frames": self.prefix_frames,
            }
        elif self.task_mode == "semantic_continue":
            program = _build_semantic_continue_program(
                schema=self.schema,
                skill_context=skill_context,
                proxy_attributes=proxy_attributes,
                prefix_frames=self.prefix_frames,
                num_frames=poses.shape[0],
                rng=rng,
            )
            program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
            target_pose = poses.copy()
            target_trans = trans.copy()
            edit_vector = np.asarray(program.to_vector(self.schema), dtype=np.float32)
            joint_mask = np.ones((poses.shape[0], SMPLH_NUM_JOINTS), dtype=np.float32) * future_mask[:, None]
            time_mask = future_mask
            goal_spec = goal_spec_to_numpy(build_goal_spec(program, schema=self.schema))
            prompt_text = f"continue {SKILL_LABEL_TO_PROMPT_PHRASE.get(skill_context['skill_label'], 'the motion')} and {verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index, total_frames=poses.shape[0]).lower()}"
            program_dict = program.to_dict()
            program_dict["task_mode"] = self.task_mode
            program_dict["source_prefix_frames"] = self.prefix_frames
        elif self.task_mode == "atomic_realize":
            program = _build_atomic_program(
                schema=self.schema,
                proxy_attributes=proxy_attributes,
                num_frames=poses.shape[0],
                prefix_frames=self.prefix_frames,
                rng=rng,
            )
            program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
            target_pose = poses.copy()
            target_trans = trans.copy()
            edit_vector = np.asarray(program.to_vector(self.schema), dtype=np.float32)
            active_joint_ids = BODY_PART_TO_JOINTS[program.part]
            valid_end_frame = _infer_valid_end_frame(
                poses=target_pose,
                active_joint_ids=active_joint_ids,
                start_frame=program.start_frame,
                peak_frame=program.end_frame,
            )
            joint_mask = np.zeros((poses.shape[0], SMPLH_NUM_JOINTS), dtype=np.float32)
            joint_mask[program.start_frame : valid_end_frame + 1, active_joint_ids] = 1.0
            time_mask = np.zeros((poses.shape[0],), dtype=np.float32)
            time_mask[program.start_frame : valid_end_frame + 1] = 1.0
            goal_spec = goal_spec_to_numpy(build_goal_spec(program, schema=self.schema))
            prompt_text = verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index, total_frames=poses.shape[0])
            program_dict = program.to_dict()
            program_dict["task_mode"] = self.task_mode
            program_dict["source_prefix_frames"] = self.prefix_frames
            program_dict["valid_end_frame"] = valid_end_frame
            sequence_spec = _sequence_spec_from_programs([program], self.schema)
        elif self.task_mode == "multi_atomic_realize":
            multi_program = _build_multi_atomic_program(
                schema=self.schema,
                proxy_attributes=proxy_attributes,
                num_frames=poses.shape[0],
                prefix_frames=self.prefix_frames,
                rng=rng,
            )
            target_pose = poses.copy()
            target_trans = trans.copy()
            first_program = multi_program.edits[0]
            first_program.attribute_key = resolve_proxy_attribute_key(first_program.part, first_program.attribute, first_program.attribute_key)
            edit_vector = np.asarray(first_program.to_vector(self.schema), dtype=np.float32)
            joint_mask = np.zeros((poses.shape[0], SMPLH_NUM_JOINTS), dtype=np.float32)
            time_mask = np.zeros((poses.shape[0],), dtype=np.float32)
            prompt_parts = []
            for i, program in enumerate(multi_program.edits):
                program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
                active_joint_ids = BODY_PART_TO_JOINTS[program.part]
                valid_end_frame = _infer_valid_end_frame(
                    poses=target_pose,
                    active_joint_ids=active_joint_ids,
                    start_frame=program.start_frame,
                    peak_frame=program.end_frame,
                )
                joint_mask[program.start_frame : valid_end_frame + 1, active_joint_ids] = 1.0
                time_mask[program.start_frame : valid_end_frame + 1] = 1.0
                program.metadata["valid_end_frame"] = valid_end_frame
                prompt_parts.append(
                    verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index + i, total_frames=poses.shape[0])
                )
            goal_spec = goal_spec_to_numpy(build_goal_spec(first_program, schema=self.schema))
            prompt_text = " ; ".join(prompt_parts)
            program_dict = multi_program.to_dict()
            program_dict["task_mode"] = self.task_mode
            program_dict["source_prefix_frames"] = self.prefix_frames
            sequence_spec = _sequence_spec_from_programs(multi_program.edits, self.schema)
        elif self.task_mode == "relative_edit":
            program = _build_relative_program(
                schema=self.schema,
                skill_context=skill_context,
                proxy_attributes=proxy_attributes,
                prefix_frames=self.prefix_frames,
                num_frames=poses.shape[0],
                rng=rng,
            )
            program.attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
            target_pose, joint_mask, time_mask = _apply_program_to_future(poses, program, self.schema)
            target_trans = trans.copy()
            edit_vector = np.asarray(program.to_vector(self.schema), dtype=np.float32)
            goal_spec = goal_spec_to_numpy(build_goal_spec(program, schema=self.schema))
            prompt_text = verbalize_program(program, schema=self.schema, style=self.prompt_style, variant_index=index)
            program_dict = program.to_dict()
            program_dict["source_prefix_frames"] = self.prefix_frames
            sequence_spec = _sequence_spec_from_programs([program], self.schema)
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
            "source_path": str(record["path"]),
            "betas": torch.from_numpy(betas.astype(np.float32)),
        }
        for key, value in goal_spec.items():
            output[key] = torch.from_numpy(value)
        for key, value in sequence_spec.items():
            output[key] = torch.from_numpy(value)
        return output
