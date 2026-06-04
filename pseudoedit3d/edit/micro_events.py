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


def extract_layer1_micro_events(frame_observables: FrameObservables, channels: Iterable[str] | None = None) -> list[MicroEvent]:
    selected = list(channels) if channels is not None else list(CHANNEL_CONFIG.keys())
    events: list[MicroEvent] = []
    for name in selected:
        if name not in frame_observables.sequences or name not in CHANNEL_CONFIG:
            continue
        seq = frame_observables.get(name)
        events.extend(segment_observable(seq, CHANNEL_CONFIG[name]))
    events.sort(key=lambda e: (e.start_frame, e.end_frame, e.observable))
    return events
