from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np

from pseudoedit3d.edit.frame_observables import FrameObservables, ObservableSequence


@dataclass
class MicroEvent:
    observable: str
    part: str
    direction: str
    magnitude_bin: str
    duration_bin: str
    start_frame: int
    end_frame: int
    delta_value: float
    unit: str
    confidence: float
    metadata: dict[str, Any]

    def to_symbol(self) -> str:
        part = self.part.upper()
        direction = self.direction.upper()
        mag = self.magnitude_bin.upper()
        return f"{part}_{direction}_{mag}"


STATE_CHANNEL_CONFIG: dict[str, dict[str, Any]] = {
    'root_xz_speed_proxy': {
        'part': 'whole_body',
        'direction': 'locomotion_active',
        'speed_threshold': 0.015,
        'min_duration': 10,
        'merge_gap': 4,
        'min_path_length': 0.20,
        'small': 0.35,
        'medium': 0.90,
    },
}


SUSTAINED_STATE_CHANNEL_CONFIG: dict[str, dict[str, Any]] = {
    'pelvis_to_ankle_height': {
        'part': 'whole_body',
        'direction': 'low_body_hold',
        'max_height': 0.45,
        'min_duration': 12,
        'merge_gap': 6,
        'small': 0.08,
        'medium': 0.18,
    },
}


CHANNEL_CONFIG: dict[str, dict[str, Any]] = {
    'root_yaw_proxy_deg': {
        'part': 'whole_body',
        'pos': 'turn_left',
        'neg': 'turn_right',
        'deadband': 2.0,
        'small': 8.0,
        'medium': 20.0,
        'duration_short': 4,
    },
    'root_height_proxy': {
        'part': 'whole_body',
        'pos': 'root_up',
        'neg': 'root_down',
        'deadband': 0.005,
        'small': 0.02,
        'medium': 0.06,
        'duration_short': 4,
    },
    'root_xz_speed_proxy': {
        'part': 'whole_body',
        'pos': 'root_speed_up',
        'neg': 'root_speed_down',
        'deadband': 0.005,
        'small': 0.03,
        'medium': 0.10,
        'duration_short': 4,
    },
    'pelvis_to_ankle_height': {
        'part': 'whole_body',
        'pos': 'leg_release',
        'neg': 'leg_compress',
        'deadband': 0.005,
        'small': 0.02,
        'medium': 0.06,
        'duration_short': 4,
    },
    'torso_bend_drop_signal': {
        'part': 'torso',
        'pos': 'torso_unbend',
        'neg': 'torso_bend_forward',
        'deadband': 0.005,
        'small': 0.02,
        'medium': 0.06,
        'duration_short': 4,
    },
    'torso_forward_extent': {
        'part': 'torso',
        'pos': 'torso_forward_extend',
        'neg': 'torso_backward_retract',
        'deadband': 0.005,
        'small': 0.01,
        'medium': 0.03,
        'duration_short': 4,
    },
    'left_arm_raise_deg': {
        'part': 'left_arm',
        'pos': 'left_arm_up',
        'neg': 'left_arm_down',
        'deadband': 2.0,
        'small': 8.0,
        'medium': 20.0,
        'duration_short': 4,
    },
    'right_arm_raise_deg': {
        'part': 'right_arm',
        'pos': 'right_arm_up',
        'neg': 'right_arm_down',
        'deadband': 2.0,
        'small': 8.0,
        'medium': 20.0,
        'duration_short': 4,
    },
    'left_elbow_lift_deg': {
        'part': 'left_arm',
        'pos': 'left_elbow_up',
        'neg': 'left_elbow_down',
        'deadband': 2.0,
        'small': 8.0,
        'medium': 20.0,
        'duration_short': 4,
    },
    'right_elbow_lift_deg': {
        'part': 'right_arm',
        'pos': 'right_elbow_up',
        'neg': 'right_elbow_down',
        'deadband': 2.0,
        'small': 8.0,
        'medium': 20.0,
        'duration_short': 4,
    },
    'left_wrist_chest_distance': {
        'part': 'left_arm',
        'pos': 'left_hand_far_from_chest',
        'neg': 'left_hand_near_chest',
        'deadband': 0.005,
        'small': 0.02,
        'medium': 0.05,
        'duration_short': 4,
    },
    'right_wrist_chest_distance': {
        'part': 'right_arm',
        'pos': 'right_hand_far_from_chest',
        'neg': 'right_hand_near_chest',
        'deadband': 0.005,
        'small': 0.02,
        'medium': 0.05,
        'duration_short': 4,
    },
}


def _sign_state(diff: float, deadband: float) -> int:
    if diff > deadband:
        return 1
    if diff < -deadband:
        return -1
    return 0


def _magnitude_bin(abs_delta: float, small: float, medium: float) -> str:
    if abs_delta < small:
        return 's'
    if abs_delta < medium:
        return 'm'
    return 'l'


def _duration_bin(length: int, short_thr: int) -> str:
    if length <= short_thr:
        return 'short'
    if length <= short_thr * 2:
        return 'medium'
    return 'long'


def _confidence(abs_delta: float, small: float, medium: float, duration: int, short_thr: int) -> float:
    mag_score = 0.34 if abs_delta < small else (0.67 if abs_delta < medium else 1.0)
    dur_score = 0.34 if duration <= short_thr else (0.67 if duration <= short_thr * 2 else 1.0)
    return float(round(0.5 * mag_score + 0.5 * dur_score, 3))


def segment_observable(sequence: ObservableSequence, cfg: dict[str, Any]) -> list[MicroEvent]:
    values = np.asarray(sequence.values, dtype=np.float32)
    if len(values) < 2:
        return []
    diffs = np.diff(values, prepend=values[:1])
    states = np.asarray([_sign_state(float(d), float(cfg['deadband'])) for d in diffs], dtype=np.int32)

    events: list[MicroEvent] = []
    start = 1
    current = int(states[1])
    for idx in range(2, len(states) + 1):
        boundary = idx == len(states) or int(states[idx]) != current
        if not boundary:
            continue
        if current != 0:
            seg_start = start
            seg_end = idx - 1
            delta = float(values[seg_end] - values[seg_start - 1])
            abs_delta = abs(delta)
            direction = cfg['pos'] if current > 0 else cfg['neg']
            dur = seg_end - seg_start + 1
            event = MicroEvent(
                observable=sequence.name,
                part=str(cfg['part']),
                direction=str(direction),
                magnitude_bin=_magnitude_bin(abs_delta, float(cfg['small']), float(cfg['medium'])),
                duration_bin=_duration_bin(dur, int(cfg['duration_short'])),
                start_frame=int(seg_start),
                end_frame=int(seg_end),
                delta_value=delta,
                unit=sequence.unit,
                confidence=_confidence(abs_delta, float(cfg['small']), float(cfg['medium']), dur, int(cfg['duration_short'])),
                metadata={'source': sequence.source},
            )
            events.append(event)
        if idx < len(states):
            start = idx
            current = int(states[idx])
    return events


def _boolean_segments(active: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active):
        if bool(flag) and start is None:
            start = idx
        if (not bool(flag) or idx == len(active) - 1) and start is not None:
            end = idx if bool(flag) and idx == len(active) - 1 else idx - 1
            if end >= start:
                segments.append((start, end))
            start = None
    return segments


def _merge_short_gaps(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def segment_locomotion_state(sequence: ObservableSequence, cfg: dict[str, Any]) -> list[MicroEvent]:
    speeds = np.asarray(sequence.values, dtype=np.float32)
    if len(speeds) < int(cfg['min_duration']):
        return []
    active = speeds >= float(cfg['speed_threshold'])
    segments = _merge_short_gaps(_boolean_segments(active), int(cfg['merge_gap']))
    events: list[MicroEvent] = []
    for start, end in segments:
        duration = end - start + 1
        if duration < int(cfg['min_duration']):
            continue
        path_length = float(np.sum(np.maximum(speeds[start:end + 1], 0.0)))
        if path_length < float(cfg['min_path_length']):
            continue
        mean_speed = float(np.mean(speeds[start:end + 1]))
        active_ratio = float(np.mean(active[start:end + 1]))
        forward_values = sequence.metadata.get('root_forward_velocity')
        lateral_values = sequence.metadata.get('root_lateral_velocity')
        forward_displacement = 0.0
        lateral_displacement = 0.0
        abs_forward_displacement = 0.0
        abs_lateral_displacement = 0.0
        trajectory_direction = 'unknown'
        if forward_values is not None and lateral_values is not None:
            forward_arr = np.asarray(forward_values, dtype=np.float32)
            lateral_arr = np.asarray(lateral_values, dtype=np.float32)
            forward_displacement = float(np.sum(forward_arr[start:end + 1]))
            lateral_displacement = float(np.sum(lateral_arr[start:end + 1]))
            abs_forward_displacement = float(np.sum(np.abs(forward_arr[start:end + 1])))
            abs_lateral_displacement = float(np.sum(np.abs(lateral_arr[start:end + 1])))
            net_mag = max(1e-6, float(np.hypot(forward_displacement, lateral_displacement)))
            forward_ratio = abs(forward_displacement) / net_mag
            lateral_ratio = abs(lateral_displacement) / net_mag
            if forward_ratio >= 0.65 and abs(forward_displacement) >= 0.25:
                trajectory_direction = 'forward' if forward_displacement >= 0 else 'backward'
            elif lateral_ratio >= 0.65 and abs(lateral_displacement) >= 0.25:
                trajectory_direction = 'right' if lateral_displacement >= 0 else 'left'
            elif net_mag >= 0.25:
                trajectory_direction = 'mixed'
        confidence = float(round(min(1.0, 0.45 + 0.35 * active_ratio + 0.20 * min(duration / 30.0, 1.0)), 3))
        events.append(MicroEvent(
            observable=sequence.name,
            part=str(cfg['part']),
            direction=str(cfg['direction']),
            magnitude_bin=_magnitude_bin(path_length, float(cfg['small']), float(cfg['medium'])),
            duration_bin=_duration_bin(duration, max(1, int(cfg['min_duration']) // 2)),
            start_frame=int(start),
            end_frame=int(end),
            delta_value=path_length,
            unit='m',
            confidence=confidence,
            metadata={
                'source': sequence.source,
                'state_event': True,
                'mean_speed': mean_speed,
                'path_length': path_length,
                'active_ratio': active_ratio,
                'speed_threshold': float(cfg['speed_threshold']),
                'trajectory_direction': trajectory_direction,
                'forward_displacement': forward_displacement,
                'lateral_displacement': lateral_displacement,
                'abs_forward_displacement': abs_forward_displacement,
                'abs_lateral_displacement': abs_lateral_displacement,
            },
        ))
    return events


def segment_low_body_state(sequence: ObservableSequence, cfg: dict[str, Any]) -> list[MicroEvent]:
    values = np.asarray(sequence.values, dtype=np.float32)
    if len(values) < int(cfg['min_duration']):
        return []
    active = values <= float(cfg['max_height'])
    segments = _merge_short_gaps(_boolean_segments(active), int(cfg['merge_gap']))
    events: list[MicroEvent] = []
    for start, end in segments:
        duration = end - start + 1
        if duration < int(cfg['min_duration']):
            continue
        segment_values = values[start:end + 1]
        active_ratio = float(np.mean(active[start:end + 1]))
        mean_height = float(np.mean(segment_values))
        low_depth = max(0.0, float(cfg['max_height']) - mean_height)
        confidence = float(round(min(1.0, 0.48 + 0.30 * active_ratio + 0.22 * min(duration / 40.0, 1.0)), 3))
        events.append(MicroEvent(
            observable=sequence.name,
            part=str(cfg['part']),
            direction=str(cfg['direction']),
            magnitude_bin=_magnitude_bin(low_depth, float(cfg['small']), float(cfg['medium'])),
            duration_bin=_duration_bin(duration, max(1, int(cfg['min_duration']) // 2)),
            start_frame=int(start),
            end_frame=int(end),
            delta_value=low_depth,
            unit=sequence.unit,
            confidence=confidence,
            metadata={
                'source': sequence.source,
                'state_event': True,
                'state_type': 'sustained_low_body',
                'mean_height': mean_height,
                'min_height': float(np.min(segment_values)),
                'max_height_observed': float(np.max(segment_values)),
                'low_height_threshold': float(cfg['max_height']),
                'active_ratio': active_ratio,
            },
        ))
    return events


def extract_layer1_micro_events(frame_observables: FrameObservables, channels: Iterable[str] | None = None) -> list[MicroEvent]:
    selected = list(channels) if channels is not None else list(CHANNEL_CONFIG.keys())
    events: list[MicroEvent] = []
    for name in selected:
        if name not in frame_observables.sequences or name not in CHANNEL_CONFIG:
            continue
        seq = frame_observables.get(name)
        events.extend(segment_observable(seq, CHANNEL_CONFIG[name]))
        if name in STATE_CHANNEL_CONFIG:
            events.extend(segment_locomotion_state(seq, STATE_CHANNEL_CONFIG[name]))
        if name in SUSTAINED_STATE_CHANNEL_CONFIG:
            events.extend(segment_low_body_state(seq, SUSTAINED_STATE_CHANNEL_CONFIG[name]))
    events.sort(key=lambda e: (e.start_frame, e.end_frame, e.observable, e.direction))
    return events
