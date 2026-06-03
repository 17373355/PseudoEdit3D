from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'
if str(MOMASK_ROOT) not in sys.path:
    sys.path.insert(0, str(MOMASK_ROOT))

from common.skeleton import Skeleton  # type: ignore
from utils import motion_process as mp  # type: ignore
from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets  # type: ignore


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def read_all_prompts(case_id: str) -> list[tuple[str, float, float]]:
    text_path = HML_ROOT / 'texts' / f'{case_id}.txt'
    out = []
    for line in text_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('#')
        caption = parts[0].strip()
        start, end = 0.0, 0.0
        if len(parts) >= 4:
            try:
                start = float(parts[-2])
                end = float(parts[-1])
            except Exception:
                pass
        out.append((caption, start, end))
    return out


def read_first_prompt(case_id: str) -> str:
    prompts = read_all_prompts(case_id)
    if not prompts:
        raise RuntimeError(f'No prompt lines for {case_id}')

    def score(caption: str) -> tuple[float, int]:
        caption_l = caption.lower()
        score = 0.0
        if 'up and down' in caption_l or 'down and up' in caption_l:
            score += 4.0
        if 'jumping up and down' in caption_l or 'bounce' in caption_l or 'hops' in caption_l:
            score += 3.0
        if 'stairs' in caption_l or 'stair' in caption_l or 'downstairs' in caption_l or 'upstairs' in caption_l:
            score += 2.0
        if 'turn' in caption_l or 'spin' in caption_l or 'rotate' in caption_l:
            score += 1.5
        if 'walk' in caption_l or 'run' in caption_l:
            score += 1.0
        if 'sup and down' in caption_l:
            score -= 1.0
        score += min(len(caption.split()) * 0.05, 1.0)
        return score, len(caption)

    return max((caption for caption, _, _ in prompts), key=score)


CAPTION_PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    'stair_descent': (
        'walk down some stairs', 'walks down some stairs', 'walking down the stairs', 'walk down stairs',
        'walks down stairs', 'downstairs', 'down the stairs', 'descends', 'descend', 'steps down stairs',
        'down steps', 'down the steps', 'walk back down', 'climb back down', 'walk back down the stairs',
        'walking downstairs', 'walks downstairs', 'steps downward', 'walking down some stairs',
    ),
    'stair_ascent': (
        'walk up stairs', 'walks up stairs', 'walking up stairs', 'upstairs', 'up the stairs', 'climb up',
        'climbs up', 'climbing up', 'steps up', 'step up', 'goes up steps', 'walking upstairs', 'walks upstairs',
    ),
    'walk_forward': (
        'walk forward', 'walks forward', 'walking forward', 'walk straight', 'walks straight', 'paces',
        'steps forward', 'walks down the road', 'walks casually forward', 'walks straight casually',
    ),
    'walk_backward': (
        'walk backward', 'walks backward', 'walking backward', 'walk back', 'walks back', 'backwards', 'walk back the opposite direction',
    ),
    'turn': (
        'turn', 'turns', 'turn around', 'spins', 'spin', 'rotate', 'rotates', 'rotating', 'turns around',
        'sharp turn', 'turns to the left', 'turns to the right',
    ),
    'jump_up': (
        'jump', 'jumps', 'jump up', 'jumps up', 'hop', 'hops', 'bounce', 'bounces', 'hops up', 'little jump',
    ),
    'crouch_bend': (
        'crouch', 'crouches', 'bend', 'bends', 'bent over', 'squat', 'squats', 'squatting',
    ),
    'arm_support': (
        'railing', 'rail', 'bannister', 'handrail', 'support', 'holding', 'holding on', 'uses their right arm as support',
        'using the handrail', 'holding a railing', 'holding onto',
    ),
    'stop_pause': (
        'stops', 'stop', 'halts', 'halt', 'pause', 'paused',
    ),
}


def build_caption_prior(case_id: str) -> dict[str, float]:
    prompts = read_all_prompts(case_id)
    captions = [caption.lower() for caption, _, _ in prompts]
    total = max(len(captions), 1)

    prior: dict[str, float] = {}
    matched_captions: dict[str, list[str]] = {}
    for key, keywords in CAPTION_PATTERN_GROUPS.items():
        hits = []
        for caption in captions:
            if any(keyword in caption for keyword in keywords):
                hits.append(caption)
        prior[key] = len(hits) / total
        matched_captions[key] = hits
    prior['num_captions'] = float(len(captions))
    prior['_matched_captions'] = matched_captions
    return prior


def load_joints3d(case_id: str) -> np.ndarray:
    pth = HML_ROOT / 'joints3d.pth'
    data = torch.load(pth, map_location='cpu')
    key = f'{case_id}.npy'
    item = data[key]
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    return np.asarray(joints, dtype=np.float32)


def setup_momask_preprocess(example_joints22: np.ndarray) -> None:
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = 22
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, 'cpu')
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_joints22[0]).float())


def heading_deg(joints: np.ndarray) -> np.ndarray:
    l_hip = joints[:, 1][:, [0, 2]]
    r_hip = joints[:, 2][:, [0, 2]]
    l_sh = joints[:, 16][:, [0, 2]]
    r_sh = joints[:, 17][:, [0, 2]]
    across = (r_hip - l_hip) + (r_sh - l_sh)
    forward = np.stack([-across[:, 1], across[:, 0]], axis=-1)
    forward = forward / np.clip(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8, None)
    yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))
    return yaw - yaw[0]


def root_height(joints: np.ndarray) -> np.ndarray:
    return joints[:, 0, 1]


def root_xz_displacement(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    disp = root_xz - root_xz[0:1]
    return np.linalg.norm(disp, axis=-1)


def root_forward_progress(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    delta = root_xz - root_xz[0:1]
    return delta[:, 1]


def root_step_speed(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    step = np.diff(root_xz, axis=0, prepend=root_xz[0:1])
    return np.linalg.norm(step, axis=-1)


def pelvis_to_ankle_height(joints: np.ndarray) -> np.ndarray:
    ankles = 0.5 * (joints[:, 7, 1] + joints[:, 8, 1])
    pelvis = joints[:, 0, 1]
    return pelvis - ankles


def torso_bend_drop_signal(joints: np.ndarray) -> np.ndarray:
    pelvis_y = joints[:, 0, 1]
    neck_rel = joints[:, 12, 1] - pelvis_y
    head_rel = joints[:, 15, 1] - pelvis_y
    shoulder_rel = 0.5 * (joints[:, 16, 1] + joints[:, 17, 1]) - pelvis_y
    return (neck_rel + head_rel + shoulder_rel) / 3.0


def torso_forward_extent(joints: np.ndarray) -> np.ndarray:
    pelvis_xz = joints[:, 0][:, [0, 2]]
    neck_xz = np.linalg.norm(joints[:, 12][:, [0, 2]] - pelvis_xz, axis=-1)
    head_xz = np.linalg.norm(joints[:, 15][:, [0, 2]] - pelvis_xz, axis=-1)
    shoulder_center = 0.5 * (joints[:, 16][:, [0, 2]] + joints[:, 17][:, [0, 2]])
    shoulder_xz = np.linalg.norm(shoulder_center - pelvis_xz, axis=-1)
    return (neck_xz + head_xz + shoulder_xz) / 3.0


def left_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    return (joints[:, 20, 1] - joints[:, 16, 1]) * 180.0


def right_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    return (joints[:, 21, 1] - joints[:, 17, 1]) * 180.0


def left_elbow_lift_deg(joints: np.ndarray) -> np.ndarray:
    return (joints[:, 18, 1] - joints[:, 16, 1]) * 180.0


def right_elbow_lift_deg(joints: np.ndarray) -> np.ndarray:
    return (joints[:, 19, 1] - joints[:, 17, 1]) * 180.0


def both_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    left_raise = left_arm_raise_deg(joints)
    right_raise = right_arm_raise_deg(joints)
    return np.minimum(left_raise, right_raise)


def first_cross(values: np.ndarray, source: float, thr: float, start: int) -> int:
    for i in range(start, len(values)):
        if abs(float(values[i] - source)) >= thr:
            return i
    return start


def find_repeated_height_peaks(values: np.ndarray, source_value: float, start_frame: int, min_peak: float = 0.03, min_gap: int = 3) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    future = values[start_frame:]
    last_peak = -min_gap
    for i in range(1, len(future) - 1):
        delta = float(future[i] - source_value)
        if delta < min_peak:
            continue
        if future[i] > future[i - 1] and future[i] >= future[i + 1] and (i - last_peak) >= min_gap:
            peaks.append((start_frame + i, delta))
            last_peak = i
    return peaks


def find_repeated_height_valleys(values: np.ndarray, source_value: float, start_frame: int, min_drop: float = 0.03, min_gap: int = 3) -> list[tuple[int, float]]:
    valleys: list[tuple[int, float]] = []
    future = values[start_frame:]
    last_valley = -min_gap
    for i in range(1, len(future) - 1):
        delta = float(source_value - future[i])
        if delta < min_drop:
            continue
        if future[i] < future[i - 1] and future[i] <= future[i + 1] and (i - last_valley) >= min_gap:
            valleys.append((start_frame + i, delta))
            last_valley = i
    return valleys


def find_repeated_signal_peaks(values: np.ndarray, source_value: float, start_frame: int, min_peak: float, min_gap: int = 3) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    future = values[start_frame:]
    last_peak = -min_gap
    for i in range(1, len(future) - 1):
        delta = float(future[i] - source_value)
        if delta < min_peak:
            continue
        if future[i] > future[i - 1] and future[i] >= future[i + 1] and (i - last_peak) >= min_gap:
            peaks.append((start_frame + i, delta))
            last_peak = i
    return peaks


def find_repeated_signal_valleys(values: np.ndarray, source_value: float, start_frame: int, min_drop: float, min_gap: int = 3) -> list[tuple[int, float]]:
    valleys: list[tuple[int, float]] = []
    future = values[start_frame:]
    last_valley = -min_gap
    for i in range(1, len(future) - 1):
        delta = float(source_value - future[i])
        if delta < min_drop:
            continue
        if future[i] < future[i - 1] and future[i] <= future[i + 1] and (i - last_valley) >= min_gap:
            valleys.append((start_frame + i, delta))
            last_valley = i
    return valleys


def _format_turn(delta: float) -> str:
    side = 'left' if delta >= 0 else 'right'
    return f'turns {side} by about {abs(delta):.0f} degrees'


def _edit_clause(edit: dict) -> str:
    attr = edit.get('attribute', '')
    delta = float(edit.get('delta_value_deg') or 0.0)
    if attr == 'turn_left' or attr == 'turn_right':
        return _format_turn(delta)
    if attr == 'walk_forward':
        return 'walks forward'
    if attr == 'walk_backward':
        return 'walks backward'
    if attr == 'stair_descent':
        return 'walks down some stairs'
    if attr == 'stair_ascent':
        return 'walks up some stairs'
    if attr == 'jump_up':
        if abs(delta) < 0.15:
            return 'jumps up slightly'
        return 'jumps upward'
    if attr == 'bounce_repeated':
        return 'jumps up and down repeatedly in place'
    if attr == 'hop_repeated':
        return 'hops repeatedly in place'
    if attr == 'crouch_repeated':
        return 'repeatedly squats down and stands back up'
    if attr == 'crouch_bend':
        return 'squats down and rises back up'
    if attr == 'torso_bend_forward':
        return 'bends forward at the waist'
    if attr == 'stop_pause':
        return 'comes to a stop'
    if attr == 'raise':
        return 'raises both arms'
    if attr == 'arm_updown_repeated':
        return 'moves both arms up and down repeatedly'
    if attr == 'elbow_flap_repeated':
        return 'flaps both elbows up and down near the chest'
    if attr == 'left_arm_updown':
        return 'moves the left arm up and down'
    if attr == 'right_arm_updown':
        return 'moves the right arm up and down'
    if attr == 'land':
        return 'lands back down'
    if attr == 'lean_forward':
        return 'leans forward'
    if attr == 'lean_backward':
        return 'leans backward'
    return attr.replace('_', ' ')


def naturalize_auto_prompt(prompts: list[str], edits: list[dict]) -> str:
    del prompts
    attrs = [edit.get('attribute', '') for edit in edits]
    attr_set = set(attrs)

    if 'hop_repeated' in attr_set:
        return 'a person hops repeatedly in place'

    if 'crouch_repeated' in attr_set:
        return 'a person repeatedly squats down and stands back up'

    if 'torso_bend_forward' in attr_set:
        return 'a person bends forward repeatedly at the waist'

    if 'arm_updown_repeated' in attr_set:
        return 'a person moves both arms up and down repeatedly'

    if 'elbow_flap_repeated' in attr_set:
        return 'a person flaps both elbows up and down near the chest'

    if 'left_arm_updown' in attr_set:
        return 'a person moves the left arm up and down'

    if 'right_arm_updown' in attr_set:
        return 'a person moves the right arm up and down'

    if 'bounce_repeated' in attr_set:
        return 'a person is jumping up and down repeatedly in place'

    if 'stair_ascent' in attr_set and 'stair_descent' in attr_set and ('turn_left' in attr_set or 'turn_right' in attr_set):
        return 'a person walks up the stairs, turns around, and walks back down the stairs'

    if 'stair_descent' in attr_set and 'walk_forward' in attr_set and 'stair_ascent' not in attr_set:
        return 'a person walks down some stairs'

    if 'stair_ascent' in attr_set and 'walk_forward' in attr_set and 'stair_descent' not in attr_set:
        return 'a person walks up some stairs'

    if 'walk_backward' in attr_set and ('turn_left' in attr_set or 'turn_right' in attr_set):
        return 'a person walks forward, turns around, and walks back'

    clauses = []
    for edit in edits:
        clause = _edit_clause(edit)
        if clause not in clauses:
            clauses.append(clause)

    if not clauses:
        return 'a person moves naturally'
    if len(clauses) == 1:
        return f'a person {clauses[0]}'
    if len(clauses) == 2:
        return f'a person {clauses[0]} and then {clauses[1]}'
    return 'a person ' + ', then '.join([clauses[0], *clauses[1:]])


def _event_unit(attr: str) -> str:
    if attr in {'turn_left', 'turn_right'}:
        return 'deg'
    if attr in {'walk_forward', 'walk_backward', 'stair_descent', 'stair_ascent', 'jump_up', 'land', 'crouch_bend', 'crouch_repeated', 'bounce_repeated'}:
        return 'm'
    if attr == 'stop_pause':
        return 'frame'
    if attr == 'torso_bend_forward':
        return 'm'
    if attr in {'arm_updown_repeated', 'elbow_flap_repeated', 'left_arm_updown', 'right_arm_updown'}:
        return 'deg'
    return 'unknown'


def _event_direction(attr: str, delta: float) -> str:
    if attr in {'turn_left', 'turn_right'}:
        return 'left' if delta >= 0 else 'right'
    if attr == 'walk_forward':
        return 'forward'
    if attr == 'walk_backward':
        return 'backward'
    if attr == 'stair_descent' or attr == 'land':
        return 'down'
    if attr == 'stair_ascent' or attr == 'jump_up':
        return 'up'
    if attr.startswith('crouch'):
        return 'lower'
    if attr == 'torso_bend_forward':
        return 'forward_down'
    if attr == 'hop_repeated':
        return 'up_down'
    if attr == 'bounce_repeated':
        return 'up_down'
    if attr == 'raise':
        return 'up'
    if attr in {'arm_updown_repeated', 'elbow_flap_repeated', 'left_arm_updown', 'right_arm_updown'}:
        return 'up_down'
    if attr == 'stop_pause':
        return 'stop'
    return 'unknown'


def _event_confidence(edit: dict) -> float:
    attr = edit.get('attribute', '')
    delta = abs(float(edit.get('delta_value_deg') or 0.0))
    if attr in {'turn_left', 'turn_right'}:
        return min(1.0, delta / 180.0)
    if attr in {'walk_forward', 'walk_backward'}:
        return min(1.0, delta / 2.0)
    if attr in {'stair_descent', 'stair_ascent', 'jump_up', 'land', 'crouch_bend', 'crouch_repeated', 'bounce_repeated', 'hop_repeated', 'torso_bend_forward'}:
        return min(1.0, delta / 0.3)
    if attr == 'raise':
        return min(1.0, delta / 90.0)
    if attr == 'stop_pause':
        return 0.75
    return 0.5


def build_auto_program(edits: list[dict]) -> dict:
    events = []
    for edit in edits:
        attr = edit.get('attribute', '')
        delta = float(edit.get('delta_value_deg') or 0.0)
        events.append({
            'type': attr,
            'part': edit.get('part', 'whole_body'),
            'direction': _event_direction(attr, delta),
            'magnitude': abs(delta),
            'unit': _event_unit(attr),
            'count': int(edit.get('count', 1)),
            'start_frame': int(edit.get('start_frame', -1)),
            'end_frame': int(edit.get('end_frame', -1)),
            'confidence': round(_event_confidence(edit), 3),
        })
    return {'events': events}


def build_auto_prompt(case_id: str, prefix_frames: int) -> dict:
    joints = load_joints3d(case_id)
    setup_momask_preprocess(joints)
    motion_263, _, _, _ = mp.process_file(joints.copy(), 0.002)

    yaw = heading_deg(joints)
    yaw_source = float(yaw[min(prefix_frames - 1, len(yaw) - 1)])
    yaw_future = yaw[prefix_frames:]
    height = root_height(joints)
    h_source = float(height[min(prefix_frames - 1, len(height) - 1)])
    h_future = height[prefix_frames:]
    xz_disp = root_xz_displacement(joints)
    xz_source = float(xz_disp[min(prefix_frames - 1, len(xz_disp) - 1)])
    xz_future = xz_disp[prefix_frames:]
    fwd = root_forward_progress(joints)
    fwd_source = float(fwd[min(prefix_frames - 1, len(fwd) - 1)])
    fwd_future = fwd[prefix_frames:]
    step_speed = root_step_speed(joints)
    p2a = pelvis_to_ankle_height(joints)
    p2a_source = float(p2a[min(prefix_frames - 1, len(p2a) - 1)])
    p2a_future = p2a[prefix_frames:]
    torso_bend = torso_bend_drop_signal(joints)
    torso_bend_source = float(torso_bend[min(prefix_frames - 1, len(torso_bend) - 1)])
    torso_bend_future = torso_bend[prefix_frames:]
    torso_xz = torso_forward_extent(joints)
    torso_xz_source = float(torso_xz[min(prefix_frames - 1, len(torso_xz) - 1)])
    torso_xz_future = torso_xz[prefix_frames:]
    left_arm = left_arm_raise_deg(joints)
    right_arm = right_arm_raise_deg(joints)
    left_elbow = left_elbow_lift_deg(joints)
    right_elbow = right_elbow_lift_deg(joints)
    chest = 0.5 * (joints[:, 16] + joints[:, 17])
    left_wrist_chest = np.linalg.norm(joints[:, 20] - chest, axis=-1)
    right_wrist_chest = np.linalg.norm(joints[:, 21] - chest, axis=-1)
    arm = both_arm_raise_deg(joints)
    a_source = float(arm[min(prefix_frames - 1, len(arm) - 1)])
    a_future = arm[prefix_frames:]
    left_source = float(left_arm[min(prefix_frames - 1, len(left_arm) - 1)])
    right_source = float(right_arm[min(prefix_frames - 1, len(right_arm) - 1)])
    left_elbow_source = float(left_elbow[min(prefix_frames - 1, len(left_elbow) - 1)])
    right_elbow_source = float(right_elbow[min(prefix_frames - 1, len(right_elbow) - 1)])
    arm_peaks = find_repeated_signal_peaks(arm, a_source, prefix_frames, min_peak=20.0)
    arm_valleys = find_repeated_signal_valleys(arm, a_source, prefix_frames, min_drop=20.0)
    left_peaks = find_repeated_signal_peaks(left_arm, left_source, prefix_frames, min_peak=20.0)
    left_valleys = find_repeated_signal_valleys(left_arm, left_source, prefix_frames, min_drop=20.0)
    right_peaks = find_repeated_signal_peaks(right_arm, right_source, prefix_frames, min_peak=20.0)
    right_valleys = find_repeated_signal_valleys(right_arm, right_source, prefix_frames, min_drop=20.0)
    left_elbow_peaks = find_repeated_signal_peaks(left_elbow, left_elbow_source, prefix_frames, min_peak=8.0)
    left_elbow_valleys = find_repeated_signal_valleys(left_elbow, left_elbow_source, prefix_frames, min_drop=8.0)
    right_elbow_peaks = find_repeated_signal_peaks(right_elbow, right_elbow_source, prefix_frames, min_peak=8.0)
    right_elbow_valleys = find_repeated_signal_valleys(right_elbow, right_elbow_source, prefix_frames, min_drop=8.0)

    prompts = []
    edits = []
    stair_detected = False
    stair_ascent_detected = False

    if len(yaw_future) > 0:
        max_idx = int(np.argmax(yaw_future))
        min_idx = int(np.argmin(yaw_future))
        max_delta = float(yaw_future[max_idx] - yaw_source)
        min_delta = float(yaw_future[min_idx] - yaw_source)
        if abs(max_delta) >= abs(min_delta):
            delta = max_delta
            peak_idx = max_idx
        else:
            delta = min_delta
            peak_idx = min_idx
        turn_thr = 20.0
        if abs(delta) >= turn_thr:
            attr = 'turn_left' if delta >= 0 else 'turn_right'
            st = first_cross(yaw, yaw_source, 0.15 * abs(delta), prefix_frames)
            ed = min(len(yaw) - 1, prefix_frames + max(peak_idx, 6))
            prompts.append(f'whole_body {attr} by {abs(delta):.1f} degrees from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': attr, 'delta_value_deg': delta, 'start_frame': st, 'end_frame': ed})

    if len(fwd_future) > 0 and len(h_future) > 0:
        forward_delta = float(np.max(fwd_future) - fwd_source)
        backward_delta = float(np.min(fwd_future) - fwd_source)
        total_drop = float(np.min(h_future) - h_source)
        total_rise = float(np.max(h_future) - h_source)
        compression_drop = float(p2a_source - np.min(p2a_future)) if len(p2a_future) > 0 else 0.0
        walk_thr = 0.45
        stair_thr = 0.18
        ascent_thr = 0.18
        if forward_delta >= walk_thr:
            peak_idx = int(np.argmax(fwd_future))
            st = first_cross(fwd, fwd_source, max(0.12, 0.2 * abs(forward_delta)), prefix_frames)
            ed = min(len(fwd) - 1, prefix_frames + max(peak_idx, 8))
            prompts.append(f'whole_body walk_forward by {forward_delta:.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'walk_forward', 'delta_value_deg': forward_delta, 'start_frame': st, 'end_frame': ed})
            if total_drop <= -stair_thr and compression_drop < 0.08:
                stair_detected = True
                stair_idx = int(np.argmin(h_future))
                stair_st = min(st, prefix_frames + max(0, stair_idx - 6))
                stair_ed = min(len(height) - 1, prefix_frames + max(stair_idx, 8))
                prompts.append(f'whole_body stair_descent by {abs(total_drop):.2f} meters from frame {stair_st} to {stair_ed}')
                edits.append({'part': 'whole_body', 'attribute': 'stair_descent', 'delta_value_deg': total_drop, 'start_frame': stair_st, 'end_frame': stair_ed})
            if total_rise >= ascent_thr:
                rise_idx = int(np.argmax(h_future))
                rise_st = min(st, prefix_frames + max(0, rise_idx - 6))
                rise_ed = min(len(height) - 1, prefix_frames + max(rise_idx, 8))
                stair_ascent_detected = True
                prompts.append(f'whole_body stair_ascent by {total_rise:.2f} meters from frame {rise_st} to {rise_ed}')
                edits.append({'part': 'whole_body', 'attribute': 'stair_ascent', 'delta_value_deg': total_rise, 'start_frame': rise_st, 'end_frame': rise_ed})
        elif backward_delta <= -walk_thr:
            back_idx = int(np.argmin(fwd_future))
            st = first_cross(fwd, fwd_source, max(0.12, 0.2 * abs(backward_delta)), prefix_frames)
            ed = min(len(fwd) - 1, prefix_frames + max(back_idx, 8))
            prompts.append(f'whole_body walk_backward by {abs(backward_delta):.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'walk_backward', 'delta_value_deg': backward_delta, 'start_frame': st, 'end_frame': ed})

    if not (stair_detected or stair_ascent_detected) and any(edit.get('attribute') in {'walk_forward', 'walk_backward'} for edit in edits):
        tail_len = min(12, len(step_speed) - prefix_frames)
        if tail_len >= 4:
            tail = step_speed[-tail_len:]
            body = step_speed[prefix_frames:-tail_len] if len(step_speed) > prefix_frames + tail_len else step_speed[prefix_frames:]
            body_peak = float(np.max(body)) if len(body) > 0 else 0.0
            tail_mean = float(np.mean(tail))
            if tail_mean <= 0.02 and body_peak >= 0.03:
                stop_st = len(step_speed) - tail_len
                stop_ed = len(step_speed) - 1
                if not any(edit.get('attribute') == 'stop_pause' for edit in edits):
                    prompts.append(f'whole_body stop_pause from frame {stop_st} to {stop_ed}')
                    edits.append({'part': 'whole_body', 'attribute': 'stop_pause', 'delta_value_deg': 0.0, 'start_frame': stop_st, 'end_frame': stop_ed})

    if len(h_future) > 0:
        max_idx = int(np.argmax(h_future))
        min_idx = int(np.argmin(h_future))
        max_delta = float(h_future[max_idx] - h_source)
        min_delta = float(h_future[min_idx] - h_source)
        repeated_peaks = find_repeated_height_peaks(height, h_source, prefix_frames)
        repeated_valleys = find_repeated_height_valleys(height, h_source, prefix_frames)
        horizontal_range = max(abs(forward_delta), abs(backward_delta)) if len(fwd_future) > 0 else 0.0
        stationary_like = float(np.max(xz_future) - xz_source) <= 0.35
        torso_drop = float(torso_bend_source - np.min(torso_bend_future)) if len(torso_bend_future) > 0 else 0.0
        torso_forward = float(np.max(torso_xz_future) - torso_xz_source) if len(torso_xz_future) > 0 else 0.0
        torso_bend_like = torso_drop >= 0.12 and torso_forward <= 0.05 and horizontal_range <= 0.2 and not (stair_detected or stair_ascent_detected)
        crouch_like = (horizontal_range <= 0.25 or compression_drop >= 0.12) and (max_delta >= 0.12 or min_delta <= -0.12 or compression_drop >= 0.12) and ((max_delta - min_delta) >= 0.18 or compression_drop >= 0.12) and not (stair_detected or stair_ascent_detected) and not torso_bend_like
        if torso_bend_like:
            st = first_cross(torso_bend, torso_bend_source, max(0.04, 0.2 * torso_drop), prefix_frames)
            ed = min(len(torso_bend) - 1, prefix_frames + max(int(np.argmin(torso_bend_future)), 8))
            prompts.append(f'whole_body torso_bend_forward from frame {st} to {ed}')
            edits.append({'part': 'torso', 'attribute': 'torso_bend_forward', 'delta_value_deg': torso_drop, 'start_frame': st, 'end_frame': ed})
        elif len(repeated_peaks) >= 3 and stationary_like and (not stair_detected):
            st = repeated_peaks[0][0]
            ed = repeated_peaks[-1][0]
            peak_height = max(delta for _, delta in repeated_peaks)
            if len(repeated_valleys) >= 1 and abs(min_delta) >= 0.12:
                if compression_drop >= 0.14:
                    prompts.append(f'whole_body crouch_repeated from frame {st} to {ed}')
                    edits.append({'part': 'whole_body', 'attribute': 'crouch_repeated', 'delta_value_deg': max(peak_height, abs(min_delta)), 'start_frame': st, 'end_frame': ed, 'count': len(repeated_peaks)})
                else:
                    prompts.append(f'whole_body hop_repeated from frame {st} to {ed}')
                    edits.append({'part': 'whole_body', 'attribute': 'hop_repeated', 'delta_value_deg': peak_height, 'start_frame': st, 'end_frame': ed, 'count': len(repeated_peaks)})
            else:
                prompts.append(f'whole_body bounce_up_down repeated {len(repeated_peaks)} times from frame {st} to {ed}')
                edits.append({'part': 'whole_body', 'attribute': 'bounce_repeated', 'delta_value_deg': peak_height, 'start_frame': st, 'end_frame': ed, 'count': len(repeated_peaks)})
        elif crouch_like:
            st = first_cross(height, h_source, max(0.05, 0.2 * max(abs(max_delta), abs(min_delta))), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(max(max_idx, min_idx), 8))
            prompts.append(f'whole_body crouch_bend from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'crouch_bend', 'delta_value_deg': max(max_delta, abs(min_delta)), 'start_frame': st, 'end_frame': ed})
        elif max_delta >= 0.08 and not (stair_detected or stair_ascent_detected):
            st = first_cross(height, h_source, 0.2 * abs(max_delta), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(max_idx, 4))
            prompts.append(f'whole_body jump_up by {max_delta:.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'jump_up', 'delta_value_deg': max_delta, 'start_frame': st, 'end_frame': ed})
        if (not stair_detected) and (not crouch_like) and min_delta <= -0.12 and max_delta >= 0.04 and len(repeated_peaks) < 3:
            st = first_cross(height, h_source, 0.2 * abs(min_delta), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(min_idx, 4))
            prompts.append(f'whole_body land by {abs(min_delta):.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'land', 'delta_value_deg': min_delta, 'start_frame': st, 'end_frame': ed})

    if len(a_future) > 0:
        peak_idx = int(np.argmax(a_future))
        delta = float(a_future[peak_idx] - a_source)
        arm_thr = 50.0
        wrist_chest_mean = float(np.mean(np.concatenate([left_wrist_chest[prefix_frames:], right_wrist_chest[prefix_frames:]])))
        elbow_peak_count = min(len(left_elbow_peaks), len(right_elbow_peaks))
        if elbow_peak_count >= 4 and horizontal_range <= 0.2 and abs(delta) >= 40.0 and wrist_chest_mean <= 0.45:
            st = min(left_elbow_peaks[0][0], right_elbow_peaks[0][0])
            ed = max(left_elbow_peaks[-1][0], right_elbow_peaks[-1][0])
            edits.append({'part': 'both_arms', 'attribute': 'elbow_flap_repeated', 'delta_value_deg': max(max(d for _, d in left_elbow_peaks), max(d for _, d in right_elbow_peaks)), 'start_frame': st, 'end_frame': ed, 'count': elbow_peak_count})
            prompts.append(f'both_elbows up_down repeated {elbow_peak_count} times from frame {st} to {ed}')
        elif len(left_peaks) >= 2 and len(right_peaks) >= 2 and len(left_valleys) >= 1 and len(right_valleys) >= 1 and horizontal_range <= 0.5:
            st = min(left_peaks[0][0], right_peaks[0][0])
            ed = max(left_peaks[-1][0], right_peaks[-1][0])
            edits.append({'part': 'both_arms', 'attribute': 'arm_updown_repeated', 'delta_value_deg': max(max(d for _, d in left_peaks), max(d for _, d in right_peaks)), 'start_frame': st, 'end_frame': ed, 'count': min(len(left_peaks), len(right_peaks))})
            prompts.append(f'both_arms up_down repeated {min(len(left_peaks), len(right_peaks))} times from frame {st} to {ed}')
        elif len(left_peaks) >= 1 and len(left_valleys) >= 1 and max((d for _, d in left_peaks), default=0.0) >= 30.0:
            st = min(left_peaks[0][0], left_valleys[0][0])
            ed = max(left_peaks[-1][0], left_valleys[-1][0])
            edits.append({'part': 'left_arm', 'attribute': 'left_arm_updown', 'delta_value_deg': max(d for _, d in left_peaks), 'start_frame': st, 'end_frame': ed, 'count': len(left_peaks)})
            prompts.append(f'left_arm up_down from frame {st} to {ed}')
        elif len(right_peaks) >= 1 and len(right_valleys) >= 1 and max((d for _, d in right_peaks), default=0.0) >= 30.0:
            st = min(right_peaks[0][0], right_valleys[0][0])
            ed = max(right_peaks[-1][0], right_valleys[-1][0])
            edits.append({'part': 'right_arm', 'attribute': 'right_arm_updown', 'delta_value_deg': max(d for _, d in right_peaks), 'start_frame': st, 'end_frame': ed, 'count': len(right_peaks)})
            prompts.append(f'right_arm up_down from frame {st} to {ed}')
        elif abs(delta) >= arm_thr:
            st = first_cross(arm, a_source, 0.2 * abs(delta), prefix_frames)
            ed = min(len(arm) - 1, prefix_frames + max(peak_idx, 4))
            prompts.append(f'both_arms raise by {abs(delta):.1f} degrees from frame {st} to {ed}')
            edits.append({'part': 'both_arms', 'attribute': 'raise', 'delta_value_deg': delta, 'start_frame': st, 'end_frame': ed})

    text_prompt = naturalize_auto_prompt(prompts, edits)
    if edits:
        mask_start = min(int(e['start_frame']) for e in edits)
        mask_end = max(int(e['end_frame']) for e in edits)
    else:
        mask_start = prefix_frames
        mask_end = joints.shape[0] - 1

    auto_program = build_auto_program(edits)
    return {
        'motion_263': motion_263.astype(np.float32),
        'program': {'task_mode': 'multi_atomic_realize', 'edits': edits, 'source_prefix_frames': prefix_frames},
        'auto_program': auto_program,
        'auto_prompt': text_prompt,
        'mask_edit_section': f'{mask_start},{mask_end}',
        'source_num_frames': int(len(joints)),
        'raw_prompt_segments': read_all_prompts(case_id),
    }


def run_gen(prompt: str, ext: str, motion_length: int) -> int:
    cmd = [
        '/mnt/data/home/guoruoxi/miniconda3/envs/momask/bin/python',
        'gen_t2m.py',
        '--gpu_id', '0',
        '--dataset_name', 't2m',
        '--name', 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
        '--res_name', 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
        '--text_prompt', prompt,
        '--motion_length', str(motion_length),
        '--repeat_times', '1',
        '--time_steps', '10',
        '--cond_scale', '4',
        '--ext', ext,
    ]
    subprocess.run(cmd, cwd=MOMASK_ROOT, check=True)
    return int(motion_length // 4) * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--prefix-frames', type=int, default=20)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--skip-generation', action='store_true')
    args = parser.parse_args()

    case_ids = [x.strip() for x in args.case_ids.split(',') if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for case_id in case_ids:
        gt_prompt = read_first_prompt(case_id)
        auto = build_auto_prompt(case_id, args.prefix_frames)
        caption_prior = build_caption_prior(case_id)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / 'source_motion.npy', auto['motion_263'])
        (case_dir / 'auto_meta.json').write_text(json.dumps(_json_safe(auto), ensure_ascii=True, indent=2), encoding='utf-8')
        gt_ext = f'case_study_{case_id}_gtprompt'
        auto_ext = f'case_study_{case_id}_autoprompt'
        source_num_frames = int(auto['source_num_frames'])
        generated_num_frames = int((source_num_frames // 4) * 4)
        if not args.skip_generation:
            generated_num_frames = run_gen(gt_prompt, gt_ext, source_num_frames)
            generated_num_frames = run_gen(auto['auto_prompt'], auto_ext, source_num_frames)
        summary.append({
            'case_id': case_id,
            'gt_prompt': gt_prompt,
            'auto_prompt': auto['auto_prompt'],
            'program': auto['program'],
            'auto_program': auto['auto_program'],
            'gt_ext': gt_ext,
            'auto_ext': auto_ext,
            'source_num_frames': source_num_frames,
            'generated_num_frames': generated_num_frames,
            'raw_prompt_segments': auto['raw_prompt_segments'],
            'caption_prior': caption_prior,
        })

    (out_dir / 'summary.json').write_text(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2))
    print(f'saved_summary={out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
