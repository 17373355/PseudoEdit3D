from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pseudoedit3d.edit.schema import EditProgram, LabelSchema


@dataclass
class AtomicCandidate:
    layer: str
    part: str
    attribute: str
    direction: str
    delta_value: float
    unit: str
    start_frame: int
    end_frame: int
    score: float
    attribute_key: str
    metadata: dict[str, Any]

    def to_edit_program(self, schema: LabelSchema) -> EditProgram:
        from pseudoedit3d.data.prefix_dataset import _delta_to_bin

        return EditProgram(
            part=self.part,
            attribute=self.attribute,
            delta_bin=_delta_to_bin(schema, abs(self.delta_value)),
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            contact_policy='ignore',
            attribute_key=self.attribute_key,
            direction=self.direction,
            delta_value_deg=self.delta_value,
            source_type='same_clip_hierarchical_atomic',
            schema_version=schema.schema_version,
            input_mode='motion_prefix',
            operator='add',
            reference='current_state',
            unit=self.unit,
            preserve_parts=[],
            preserve_mode='all_non_target',
            skill_label='static_pose',
            skill_phase=float('nan'),
            tolerance_deg=float(schema.prompt_defaults.get('default_tolerance_deg', 5.0)),
            metadata=self.metadata,
        )


DEFAULT_GLOBAL_CFG = {
    'turn_min_deg': 20.0,
    'turn_start_ratio': 0.15,
    'turn_min_duration': 6,
    'turn_priority_small': 2.0,
    'turn_priority_large': 3.0,
    'turn_large_deg': 90.0,
    'jump_min_height_m': 0.08,
    'land_min_drop_m': 0.05,
    'height_start_ratio': 0.20,
    'jump_priority_scale': 100.0,
    'land_priority_scale': 80.0,
    'translate_min_m': 0.15,
    'translate_start_ratio': 0.20,
    'translate_priority_scale': 40.0,
    'lean_min_deg': 8.0,
    'lean_start_ratio': 0.20,
    'lean_priority_scale': 1.0,
}


def _first_threshold_crossing(values: np.ndarray, source_value: float, threshold: float, start_frame: int) -> int:
    future = values[start_frame:]
    for i, value in enumerate(future):
        if abs(float(value - source_value)) >= threshold:
            return start_frame + i
    return start_frame


def _pack_candidate(
    *,
    layer: str,
    part: str,
    attribute: str,
    direction: str,
    delta_value: float,
    unit: str,
    start_frame: int,
    end_frame: int,
    score: float,
    attribute_key: str,
    source_value: float,
    future_peak: float,
    relative_skill_parameter: str,
) -> AtomicCandidate:
    return AtomicCandidate(
        layer=layer,
        part=part,
        attribute=attribute,
        direction=direction,
        delta_value=float(delta_value),
        unit=unit,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        score=float(score),
        attribute_key=attribute_key,
        metadata={
            'task_mode': 'multi_atomic_realize',
            'kinematic_layer': layer,
            'source_attr_current_deg': float(source_value),
            'future_peak_deg': float(future_peak),
            'relative_skill_parameter': relative_skill_parameter,
            'target_offset_deg': float('nan'),
            'preserve_amplitude': False,
        },
    )


def extract_global_atomic_candidates(
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
    cfg: dict[str, float] | None = None,
) -> list[AtomicCandidate]:
    cfg = {**DEFAULT_GLOBAL_CFG, **(cfg or {})}
    candidates: list[AtomicCandidate] = []

    # Turning: positive heading change is treated as turn_left in the current body-heading coordinate.
    if 'root_yaw_proxy_deg' in proxy_attributes:
        values = np.asarray(proxy_attributes['root_yaw_proxy_deg'], dtype=np.float32)
        source_value = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) > 0:
            max_idx = int(np.argmax(future))
            min_idx = int(np.argmin(future))
            max_delta = float(future[max_idx] - source_value)
            min_delta = float(future[min_idx] - source_value)
            if abs(max_delta) >= abs(min_delta):
                delta = max_delta
                peak_idx = max_idx
            else:
                delta = min_delta
                peak_idx = min_idx
            if abs(delta) >= cfg['turn_min_deg']:
                direction = 'increase' if delta >= 0 else 'decrease'
                attr = 'turn_left' if direction == 'increase' else 'turn_right'
                start = _first_threshold_crossing(values, source_value, cfg['turn_start_ratio'] * abs(delta), prefix_frames)
                end = min(num_frames - 1, prefix_frames + max(peak_idx, int(cfg['turn_min_duration'])))
                priority = cfg['turn_priority_large'] if abs(delta) >= cfg['turn_large_deg'] else cfg['turn_priority_small']
                candidates.append(_pack_candidate(
                    layer='global',
                    part='whole_body',
                    attribute=attr,
                    direction=direction,
                    delta_value=delta,
                    unit='deg',
                    start_frame=start,
                    end_frame=end,
                    score=abs(delta) * priority,
                    attribute_key='root_yaw_proxy_deg',
                    source_value=source_value,
                    future_peak=float(future[peak_idx]),
                    relative_skill_parameter='attribute_delta_deg',
                ))

    # Vertical displacement: jump up / land
    if 'root_height_proxy' in proxy_attributes:
        values = np.asarray(proxy_attributes['root_height_proxy'], dtype=np.float32)
        source_value = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) > 0:
            max_idx = int(np.argmax(future))
            min_idx = int(np.argmin(future))
            max_delta = float(future[max_idx] - source_value)
            min_delta = float(future[min_idx] - source_value)
            if max_delta >= cfg['jump_min_height_m']:
                start = _first_threshold_crossing(values, source_value, cfg['height_start_ratio'] * abs(max_delta), prefix_frames)
                end = min(num_frames - 1, prefix_frames + max(max_idx, 4))
                candidates.append(_pack_candidate(
                    layer='global',
                    part='whole_body',
                    attribute='jump_up',
                    direction='increase',
                    delta_value=max_delta,
                    unit='m',
                    start_frame=start,
                    end_frame=end,
                    score=abs(max_delta) * cfg['jump_priority_scale'],
                    attribute_key='root_height_proxy',
                    source_value=source_value,
                    future_peak=float(future[max_idx]),
                    relative_skill_parameter='attribute_delta_m',
                ))
            if min_delta <= -cfg['land_min_drop_m']:
                start = _first_threshold_crossing(values, source_value, cfg['height_start_ratio'] * abs(min_delta), prefix_frames)
                end = min(num_frames - 1, prefix_frames + max(min_idx, 4))
                candidates.append(_pack_candidate(
                    layer='global',
                    part='whole_body',
                    attribute='land',
                    direction='decrease',
                    delta_value=min_delta,
                    unit='m',
                    start_frame=start,
                    end_frame=end,
                    score=abs(min_delta) * cfg['land_priority_scale'],
                    attribute_key='root_height_proxy',
                    source_value=source_value,
                    future_peak=float(future[min_idx]),
                    relative_skill_parameter='attribute_delta_m',
                ))

    # Horizontal displacement / drift.
    if 'root_xz_speed_proxy' in proxy_attributes:
        speed = np.asarray(proxy_attributes['root_xz_speed_proxy'], dtype=np.float32)
        future = speed[prefix_frames:]
        if len(future) > 0:
            peak_idx = int(np.argmax(future))
            peak_val = float(future[peak_idx])
            if peak_val >= cfg['translate_min_m']:
                start = prefix_frames + max(0, peak_idx - 2)
                end = min(num_frames - 1, prefix_frames + peak_idx + 4)
                candidates.append(_pack_candidate(
                    layer='global',
                    part='whole_body',
                    attribute='shift_forward',
                    direction='increase',
                    delta_value=peak_val,
                    unit='m',
                    start_frame=start,
                    end_frame=end,
                    score=peak_val * cfg['translate_priority_scale'],
                    attribute_key='root_xz_speed_proxy',
                    source_value=0.0,
                    future_peak=peak_val,
                    relative_skill_parameter='speed_m',
                ))

    # Global lean placeholders from torso posture proxies.
    if 'torso_pitch_proxy_deg' in proxy_attributes:
        values = np.asarray(proxy_attributes['torso_pitch_proxy_deg'], dtype=np.float32)
        source_value = float(values[prefix_frames - 1])
        future = values[prefix_frames:]
        if len(future) > 0:
            max_idx = int(np.argmax(future))
            min_idx = int(np.argmin(future))
            max_delta = float(future[max_idx] - source_value)
            min_delta = float(future[min_idx] - source_value)
            if max_delta >= cfg['lean_min_deg']:
                start = _first_threshold_crossing(values, source_value, cfg['lean_start_ratio'] * abs(max_delta), prefix_frames)
                end = min(num_frames - 1, prefix_frames + max(max_idx, 4))
                candidates.append(_pack_candidate(
                    layer='global',
                    part='torso',
                    attribute='lean_forward',
                    direction='increase',
                    delta_value=max_delta,
                    unit='deg',
                    start_frame=start,
                    end_frame=end,
                    score=abs(max_delta) * cfg['lean_priority_scale'],
                    attribute_key='torso_pitch_proxy_deg',
                    source_value=source_value,
                    future_peak=float(future[max_idx]),
                    relative_skill_parameter='attribute_delta_deg',
                ))
            if min_delta <= -cfg['lean_min_deg']:
                start = _first_threshold_crossing(values, source_value, cfg['lean_start_ratio'] * abs(min_delta), prefix_frames)
                end = min(num_frames - 1, prefix_frames + max(min_idx, 4))
                candidates.append(_pack_candidate(
                    layer='global',
                    part='torso',
                    attribute='lean_backward',
                    direction='decrease',
                    delta_value=min_delta,
                    unit='deg',
                    start_frame=start,
                    end_frame=end,
                    score=abs(min_delta) * cfg['lean_priority_scale'],
                    attribute_key='torso_pitch_proxy_deg',
                    source_value=source_value,
                    future_peak=float(future[min_idx]),
                    relative_skill_parameter='attribute_delta_deg',
                ))

    return candidates


def extract_posture_atomic_candidates(
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
) -> list[AtomicCandidate]:
    return []


def extract_contact_atomic_candidates(
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
) -> list[AtomicCandidate]:
    return []


def extract_all_atomic_candidates(
    proxy_attributes: dict[str, np.ndarray],
    prefix_frames: int,
    num_frames: int,
    global_cfg: dict[str, float] | None = None,
) -> list[AtomicCandidate]:
    candidates: list[AtomicCandidate] = []
    candidates.extend(extract_global_atomic_candidates(proxy_attributes, prefix_frames, num_frames, cfg=global_cfg))
    candidates.extend(extract_posture_atomic_candidates(proxy_attributes, prefix_frames, num_frames))
    candidates.extend(extract_contact_atomic_candidates(proxy_attributes, prefix_frames, num_frames))
    return candidates
