from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pseudoedit3d.edit.attributes import compute_motion_statistics, extract_upper_body_proxy_attributes, summarize_attributes
from pseudoedit3d.edit.schema import EditProgram
from pseudoedit3d.edit.segmentation import detect_active_span
from pseudoedit3d.edit.skill_context import infer_skill_context, summarize_skill_attribute


ATTRIBUTE_TO_PROGRAM = {
    "left_shoulder_pitch_proxy_deg": ("left_arm", "raise", "lower"),
    "right_shoulder_pitch_proxy_deg": ("right_arm", "raise", "lower"),
    "both_shoulder_pitch_proxy_deg": ("both_arms", "raise", "lower"),
    "left_elbow_flex_proxy_deg": ("left_arm", "bend", "extend"),
    "right_elbow_flex_proxy_deg": ("right_arm", "bend", "extend"),
    "both_elbow_flex_proxy_deg": ("both_arms", "bend", "extend"),
    "torso_pitch_proxy_deg": ("torso", "lean_forward", "lean_backward"),
    "torso_roll_proxy_deg": ("torso", "lean_left", "lean_right"),
}

MINEABLE_ATTRIBUTE_KEYS = list(ATTRIBUTE_TO_PROGRAM.keys())


def build_attribute_record(npz_path: str, contact_bucket: str, sequence_group: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    poses = data["poses"].reshape(data["poses"].shape[0], -1, 3).astype(np.float32)
    trans = data["trans"].astype(np.float32)
    attributes = extract_upper_body_proxy_attributes(poses)
    summary = summarize_attributes(attributes)
    summary.update(compute_motion_statistics(poses, trans))
    attribute_segments = {}
    attribute_active_ratios = {}
    serializable_attributes = {}
    for key, values in attributes.items():
        active_mask, segments = detect_active_span(values)
        attribute_segments[key] = [[int(start), int(end)] for start, end in segments]
        attribute_active_ratios[key] = float(active_mask.mean())
        serializable_attributes[key] = [float(v) for v in values]
        summary[f"{key}_active_ratio"] = float(active_mask.mean())
    skill_context = infer_skill_context(attributes, motion_stats=summary, num_frames=int(poses.shape[0]))
    return {
        "path": npz_path,
        "contact_bucket": contact_bucket,
        "sequence_group": sequence_group,
        "dataset_family": sequence_group.rsplit("_", 2)[0] if "_" in sequence_group else sequence_group,
        "summary": summary,
        "attributes": serializable_attributes,
        "segments": attribute_segments,
        "skill_context": skill_context,
        "num_frames": int(poses.shape[0]),
    }


def _compute_feature_scales(records: list[dict], keys: list[str]) -> dict[str, float]:
    scales = {}
    for key in keys:
        values = np.asarray([record["summary"][key] for record in records], dtype=np.float32)
        scales[key] = float(values.std()) if float(values.std()) > 1e-6 else 1.0
    return scales


def _delta_to_bin(delta_abs: float) -> str:
    if delta_abs < 12.0:
        return "small"
    if delta_abs < 24.0:
        return "medium"
    return "large"


def _build_program(attr_key: str, delta_value: float, diff_seq: np.ndarray, contact_bucket: str, source_record: dict) -> EditProgram:
    part, positive_attr, negative_attr = ATTRIBUTE_TO_PROGRAM[attr_key]
    active_mask, segments = detect_active_span(diff_seq, min_len=5)
    start_frame, end_frame = segments[0]
    skill_context = source_record.get("skill_context", {})
    skill_label = skill_context.get("skill_label", "unknown")
    skill_phase = skill_context.get("skill_phase")
    preserve_mode = "skill_structure" if skill_context.get("is_relative_friendly", False) else "all_non_target"
    return EditProgram(
        part=part,
        attribute=positive_attr if delta_value >= 0.0 else negative_attr,
        delta_bin=_delta_to_bin(abs(delta_value)),
        start_frame=start_frame,
        end_frame=end_frame,
        contact_policy="keep" if contact_bucket == "contact" else "ignore",
        attribute_key=attr_key,
        direction="increase" if delta_value >= 0.0 else "decrease",
        delta_value_deg=float(delta_value),
        source_type="mined",
        operator="add",
        reference="current_state",
        skill_label=skill_label,
        skill_phase=skill_phase,
        preserve_mode=preserve_mode,
        preserve_parts=[],
        metadata={
            "source_attr_mean_deg": float(np.mean(source_record["attributes"][attr_key])),
            "source_attr_amplitude_deg": float(summarize_skill_attribute(np.asarray(source_record["attributes"][attr_key], dtype=np.float32))["amplitude_deg"]),
            "relative_skill_parameter": "offset_deg" if skill_label == "periodic_arm_motion" else "attribute_delta_deg",
            "target_offset_deg": float(np.mean(source_record["attributes"][attr_key]) + delta_value) if skill_label == "periodic_arm_motion" else float("nan"),
            "preserve_amplitude": bool(skill_label == "periodic_arm_motion"),
            "periodic_limb": skill_context.get("periodic_limb"),
            "periodic_state": skill_context.get("periodic_states", {}).get(skill_context.get("periodic_limb", ""), {}),
        },
    )


def mine_triplets(
    records: list[dict],
    min_delta_deg: float = 8.0,
    max_delta_deg: float = 45.0,
    candidate_limit: int = 64,
    max_pairs_per_clip: int = 4,
    distance_threshold: float = 2.5,
) -> list[dict]:
    if not records:
        return []
    match_keys = [f"{key}_mean" for key in MINEABLE_ATTRIBUTE_KEYS]
    match_keys += ["pose_velocity_mean", "root_speed_mean", "root_speed_std", "root_displacement"]
    scales = _compute_feature_scales(records, match_keys)

    grouped = {}
    for record in records:
        group_key = (record["contact_bucket"], record["dataset_family"])
        grouped.setdefault(group_key, []).append(record)

    mined_pairs = []
    for group_records in grouped.values():
        by_speed = sorted(group_records, key=lambda item: item["summary"]["root_speed_mean"])
        for source in by_speed:
            source_pairs = 0
            candidates = sorted(
                [record for record in by_speed if record["path"] != source["path"]],
                key=lambda item: abs(item["summary"]["root_speed_mean"] - source["summary"]["root_speed_mean"]),
            )[:candidate_limit]
            for attr_key in MINEABLE_ATTRIBUTE_KEYS:
                if source_pairs >= max_pairs_per_clip:
                    break
                source_mean = source["summary"][f"{attr_key}_mean"]
                best_pair = None
                best_score = None
                for target in candidates:
                    delta_value = target["summary"][f"{attr_key}_mean"] - source_mean
                    delta_abs = abs(delta_value)
                    if delta_abs < min_delta_deg or delta_abs > max_delta_deg:
                        continue
                    other_diffs = []
                    for other_attr in MINEABLE_ATTRIBUTE_KEYS:
                        if other_attr == attr_key:
                            continue
                        key_name = f"{other_attr}_mean"
                        diff = abs(target["summary"][key_name] - source["summary"][key_name]) / scales[key_name]
                        other_diffs.append(diff)
                    motion_dist = 0.0
                    for motion_key in ["pose_velocity_mean", "root_speed_mean", "root_speed_std", "root_displacement"]:
                        motion_dist += abs(target["summary"][motion_key] - source["summary"][motion_key]) / scales[motion_key]
                    score = float(np.mean(other_diffs)) + 0.35 * motion_dist
                    if score > distance_threshold:
                        continue
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = target
                if best_pair is None:
                    continue
                diff_seq = np.asarray(best_pair["attributes"][attr_key], dtype=np.float32) - np.asarray(
                    source["attributes"][attr_key], dtype=np.float32
                )
                delta_value = best_pair["summary"][f"{attr_key}_mean"] - source_mean
                program = _build_program(attr_key, delta_value, diff_seq, source["contact_bucket"], source_record=source)
                mined_pairs.append(
                    {
                        "source_path": source["path"],
                        "target_path": best_pair["path"],
                        "source_contact_bucket": source["contact_bucket"],
                        "target_contact_bucket": best_pair["contact_bucket"],
                        "sequence_group": source["sequence_group"],
                        "program": program.to_dict(),
                        "score": float(best_score),
                    }
                )
                source_pairs += 1
    return mined_pairs


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def dump_jsonl(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
