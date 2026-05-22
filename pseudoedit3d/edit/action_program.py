from __future__ import annotations

import math

import numpy as np

from pseudoedit3d.edit.attributes import PROXY_ATTRIBUTE_INDEX, resolve_proxy_attribute_key
from pseudoedit3d.edit.schema import EditProgram, LabelSchema, get_default_schema


def infer_direction_sign(program: EditProgram) -> float:
    if program.direction == "decrease":
        return -1.0
    if program.direction == "increase":
        return 1.0
    if program.delta_value_deg is not None:
        return 1.0 if float(program.delta_value_deg) >= 0.0 else -1.0
    return 1.0


def infer_delta_value_deg(program: EditProgram, schema: LabelSchema) -> float:
    if program.delta_value_deg is not None:
        return float(abs(program.delta_value_deg))
    entry = schema.delta_bin(program.delta_bin)
    if entry.default_degrees is not None:
        return float(entry.default_degrees)
    return 0.0


def build_goal_spec(program: EditProgram, schema: LabelSchema | None = None) -> dict[str, float | int]:
    schema = schema or get_default_schema()
    proxy_attribute_key = resolve_proxy_attribute_key(program.part, program.attribute, program.attribute_key)
    operator_idx = schema.operator_keys.index(program.operator)
    reference_idx = schema.reference_keys.index(program.reference)
    preserve_mode_idx = schema.preserve_mode_keys.index(program.preserve_mode)
    skill_label_idx = schema.skill_label_keys.index((program.skill_label or "unknown"))
    direction_sign = infer_direction_sign(program)
    magnitude_deg = infer_delta_value_deg(program, schema)
    signed_delta_deg = magnitude_deg * direction_sign
    tolerance_deg = (
        float(program.tolerance_deg)
        if program.tolerance_deg is not None
        else float(schema.prompt_defaults.get("default_tolerance_deg", 5.0))
    )
    target_value_deg = math.nan
    if program.operator == "set":
        target_value_deg = float(program.metadata.get("target_value_deg", signed_delta_deg))
    source_attr_mean_deg = float(program.metadata.get("source_attr_mean_deg", math.nan))
    source_attr_amplitude_deg = float(program.metadata.get("source_attr_amplitude_deg", math.nan))
    target_offset_deg = float(program.metadata.get("target_offset_deg", math.nan))
    preserve_amplitude = float(bool(program.metadata.get("preserve_amplitude", False)))
    return {
        "goal_attr_idx": int(PROXY_ATTRIBUTE_INDEX[proxy_attribute_key]),
        "goal_operator_idx": int(operator_idx),
        "goal_reference_idx": int(reference_idx),
        "goal_preserve_mode_idx": int(preserve_mode_idx),
        "goal_skill_label_idx": int(skill_label_idx),
        "goal_start_frame": int(program.start_frame),
        "goal_end_frame": int(program.end_frame),
        "goal_delta_deg": float(signed_delta_deg),
        "goal_target_value_deg": float(target_value_deg),
        "goal_source_attr_mean_deg": source_attr_mean_deg,
        "goal_source_attr_amplitude_deg": source_attr_amplitude_deg,
        "goal_target_offset_deg": target_offset_deg,
        "goal_preserve_amplitude": preserve_amplitude,
        "goal_direction_sign": float(direction_sign),
        "goal_tolerance_deg": float(tolerance_deg),
        "goal_skill_phase": float(program.skill_phase) if program.skill_phase is not None else math.nan,
    }


def goal_spec_to_numpy(goal_spec: dict[str, float | int]) -> dict[str, np.ndarray]:
    return {
        "goal_attr_idx": np.asarray(goal_spec["goal_attr_idx"], dtype=np.int64),
        "goal_operator_idx": np.asarray(goal_spec["goal_operator_idx"], dtype=np.int64),
        "goal_reference_idx": np.asarray(goal_spec["goal_reference_idx"], dtype=np.int64),
        "goal_preserve_mode_idx": np.asarray(goal_spec["goal_preserve_mode_idx"], dtype=np.int64),
        "goal_skill_label_idx": np.asarray(goal_spec["goal_skill_label_idx"], dtype=np.int64),
        "goal_start_frame": np.asarray(goal_spec["goal_start_frame"], dtype=np.int64),
        "goal_end_frame": np.asarray(goal_spec["goal_end_frame"], dtype=np.int64),
        "goal_delta_deg": np.asarray(goal_spec["goal_delta_deg"], dtype=np.float32),
        "goal_target_value_deg": np.asarray(goal_spec["goal_target_value_deg"], dtype=np.float32),
        "goal_source_attr_mean_deg": np.asarray(goal_spec["goal_source_attr_mean_deg"], dtype=np.float32),
        "goal_source_attr_amplitude_deg": np.asarray(goal_spec["goal_source_attr_amplitude_deg"], dtype=np.float32),
        "goal_target_offset_deg": np.asarray(goal_spec["goal_target_offset_deg"], dtype=np.float32),
        "goal_preserve_amplitude": np.asarray(goal_spec["goal_preserve_amplitude"], dtype=np.float32),
        "goal_direction_sign": np.asarray(goal_spec["goal_direction_sign"], dtype=np.float32),
        "goal_tolerance_deg": np.asarray(goal_spec["goal_tolerance_deg"], dtype=np.float32),
        "goal_skill_phase": np.asarray(goal_spec["goal_skill_phase"], dtype=np.float32),
    }
