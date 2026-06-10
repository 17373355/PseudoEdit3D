from __future__ import annotations

import math
from typing import Any

import numpy as np

from pseudoedit3d.constants import JOINT_INDEX

BIMANUAL_FAMILY = 'BIMANUAL_PERIODIC'
BIMANUAL_COARSE_CLUSTERS = {'BI_OUT', 'BI_UP'}


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = int(a.get('start_frame', -1)), int(a.get('end_frame', -1))
    b0, b1 = int(b.get('start_frame', -1)), int(b.get('end_frame', -1))
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    dur = max(1, a1 - a0 + 1)
    return inter / dur


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(value):
        return 0.0
    return value


def _path_length(xz: np.ndarray) -> float:
    if len(xz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(xz[1:] - xz[:-1], axis=-1)))


def _speed(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((len(values),), dtype=np.float32)
    diffs = np.zeros((len(values),), dtype=np.float32)
    diffs[1:] = np.linalg.norm(values[1:] - values[:-1], axis=-1)
    return diffs


def _span(joints: np.ndarray, evt: dict[str, Any]) -> tuple[int, int]:
    start = max(0, int(evt.get('start_frame', 0)))
    end = min(len(joints) - 1, int(evt.get('end_frame', len(joints) - 1)))
    if end < start:
        end = start
    return start, end


def _global_scale(joints: np.ndarray) -> float:
    l_shoulder = joints[:, JOINT_INDEX['L_Shoulder']]
    r_shoulder = joints[:, JOINT_INDEX['R_Shoulder']]
    shoulder_width = float(np.median(np.linalg.norm(l_shoulder - r_shoulder, axis=-1)))
    return max(shoulder_width, 0.20)


def bimanual_span_features(joints: np.ndarray, evt: dict[str, Any], program_events: list[dict[str, Any]]) -> dict[str, float | int]:
    start, end = _span(joints, evt)
    seg = np.asarray(joints[start:end + 1], dtype=np.float32)
    scale = _global_scale(np.asarray(joints, dtype=np.float32))

    lw = seg[:, JOINT_INDEX['L_Wrist']]
    rw = seg[:, JOINT_INDEX['R_Wrist']]
    l_shoulder = seg[:, JOINT_INDEX['L_Shoulder']]
    r_shoulder = seg[:, JOINT_INDEX['R_Shoulder']]
    chest = 0.5 * (l_shoulder + r_shoulder)
    root = seg[:, JOINT_INDEX['Pelvis']]

    hand_dist = np.linalg.norm(lw - rw, axis=-1)
    left_chest = np.linalg.norm(lw - chest, axis=-1)
    right_chest = np.linalg.norm(rw - chest, axis=-1)
    mean_chest = 0.5 * (left_chest + right_chest)
    left_root = np.linalg.norm(lw - root, axis=-1)
    right_root = np.linalg.norm(rw - root, axis=-1)
    mean_root = 0.5 * (left_root + right_root)
    wrist_height_rel = 0.5 * (lw[:, 1] + rw[:, 1]) - chest[:, 1]
    left_speed = _speed(lw)
    right_speed = _speed(rw)
    mean_speed = 0.5 * (left_speed + right_speed)
    root_path = _path_length(root[:, [0, 2]])
    duration = end - start + 1

    locos = [e for e in program_events if e.get('super_family') == 'WHOLE_BODY_LOCOMOTION']
    verticals = [e for e in program_events if e.get('super_family') == 'WHOLE_BODY_VERTICAL']
    rotations = [e for e in program_events if e.get('super_family') == 'WHOLE_BODY_ROTATION']

    return {
        'start_frame': start,
        'end_frame': end,
        'duration': duration,
        'body_scale_m': scale,
        'mean_hand_distance': float(np.mean(hand_dist)),
        'min_hand_distance': float(np.min(hand_dist)),
        'max_hand_distance': float(np.max(hand_dist)),
        'hand_distance_delta': float(hand_dist[-1] - hand_dist[0]) if len(hand_dist) else 0.0,
        'mean_hand_distance_norm': float(np.mean(hand_dist) / scale),
        'min_hand_distance_norm': float(np.min(hand_dist) / scale),
        'max_hand_distance_norm': float(np.max(hand_dist) / scale),
        'hand_distance_delta_norm': float((hand_dist[-1] - hand_dist[0]) / scale) if len(hand_dist) else 0.0,
        'mean_wrist_chest_distance': float(np.mean(mean_chest)),
        'mean_wrist_chest_distance_norm': float(np.mean(mean_chest) / scale),
        'mean_wrist_root_distance': float(np.mean(mean_root)),
        'mean_wrist_root_distance_norm': float(np.mean(mean_root) / scale),
        'mean_wrist_height_rel': float(np.mean(wrist_height_rel)),
        'max_wrist_height_rel': float(np.max(wrist_height_rel)),
        'wrist_height_delta': float(wrist_height_rel[-1] - wrist_height_rel[0]) if len(wrist_height_rel) else 0.0,
        'mean_wrist_height_rel_norm': float(np.mean(wrist_height_rel) / scale),
        'max_wrist_height_rel_norm': float(np.max(wrist_height_rel) / scale),
        'wrist_height_delta_norm': float((wrist_height_rel[-1] - wrist_height_rel[0]) / scale) if len(wrist_height_rel) else 0.0,
        'mean_wrist_speed': float(np.mean(mean_speed)),
        'max_wrist_speed': float(np.max(mean_speed)),
        'mean_wrist_speed_norm': float(np.mean(mean_speed) / scale),
        'max_wrist_speed_norm': float(np.max(mean_speed) / scale),
        'root_path_xz': root_path,
        'root_path_xz_norm': float(root_path / scale),
        'left_right_chest_corr': _safe_corr(left_chest, right_chest),
        'max_loco_overlap': max((_overlap_ratio(evt, e) for e in locos), default=0.0),
        'max_vertical_overlap': max((_overlap_ratio(evt, e) for e in verticals), default=0.0),
        'max_rotation_overlap': max((_overlap_ratio(evt, e) for e in rotations), default=0.0),
    }


def classify_bimanual_event(evt: dict[str, Any], feat: dict[str, float | int]) -> tuple[str, str, float]:
    cluster = str(evt.get('cluster_id', ''))
    duration = int(feat['duration'])
    mean_hand = float(feat['mean_hand_distance_norm'])
    min_hand = float(feat['min_hand_distance_norm'])
    max_hand = float(feat['max_hand_distance_norm'])
    hand_delta = float(feat['hand_distance_delta_norm'])
    mean_chest = float(feat['mean_wrist_chest_distance_norm'])
    mean_root = float(feat['mean_wrist_root_distance_norm'])
    mean_height = float(feat['mean_wrist_height_rel_norm'])
    max_height = float(feat['max_wrist_height_rel_norm'])
    height_delta = float(feat['wrist_height_delta_norm'])
    wrist_speed = float(feat['mean_wrist_speed_norm'])
    root_path = float(feat['root_path_xz_norm'])
    loco_overlap = float(feat['max_loco_overlap'])
    vertical_overlap = float(feat['max_vertical_overlap'])
    lr_corr = float(feat['left_right_chest_corr'])

    hands_close = mean_hand <= 0.90 or min_hand <= 0.55
    hands_far = mean_hand >= 1.70 or max_hand >= 2.35
    hands_spreading = hand_delta >= 0.30
    hands_closing = hand_delta <= -0.20
    high_hands = max_height >= 0.35 or mean_height >= 0.18 or height_delta >= 0.25
    extended = mean_chest >= 1.15 or mean_root >= 2.15
    stable_hands = wrist_speed <= 0.16 and duration >= 8
    moving_body = loco_overlap >= 0.40 or root_path >= 1.80
    synchronized = lr_corr >= 0.35

    if high_hands and (hands_far or hands_spreading):
        return 'BI_RAISE_SPREAD', 'both hands rise while spreading outward', 0.72
    if hands_close or hands_closing:
        if high_hands or height_delta >= 0.20:
            return 'BI_HANDS_CLOSE_RAISE', 'hands stay close while rising', 0.68
        return 'BI_HANDS_CLOSE', 'hands are close together or moving closer', 0.66
    if high_hands or cluster == 'BI_UP':
        conf = 0.70 if high_hands else 0.58
        return 'BI_RAISE', 'both wrists move upward or are above the shoulder line', conf
    if hands_far or hands_spreading or cluster == 'BI_OUT':
        conf = 0.70 if (hands_far or hands_spreading) else 0.58
        return 'BI_SPREAD', 'hands are far apart or moving farther apart', conf
    if moving_body and stable_hands and extended:
        return 'BI_EXTENDED_LOCO_COUPLED', 'extended bilateral hand posture coupled with locomotion', 0.50
    if moving_body and synchronized:
        return 'BI_LOCOMOTION_COUPLED', 'bilateral arm motion coupled with body translation', 0.50
    if vertical_overlap >= 0.40:
        return 'BI_VERTICAL_COUPLED', 'bilateral arm motion coupled with vertical body motion', 0.50
    return 'BI_UNRESOLVED', 'bimanual event lacks enough separating evidence', 0.45


def _split_direction(split_cluster: str, fallback: str) -> str:
    if split_cluster == 'BI_RAISE':
        return 'up'
    if split_cluster == 'BI_RAISE_SPREAD':
        return 'up_outward'
    if split_cluster == 'BI_SPREAD':
        return 'outward'
    if split_cluster in {'BI_HANDS_CLOSE', 'BI_HANDS_CLOSE_RAISE'}:
        return 'inward'
    if split_cluster in {'BI_EXTENDED_LOCO_COUPLED', 'BI_LOCOMOTION_COUPLED'}:
        return 'locomotion_coupled'
    if split_cluster == 'BI_VERTICAL_COUPLED':
        return 'vertical_coupled'
    return fallback


def _split_phase_template(split_cluster: str) -> str:
    return {
        'BI_RAISE': 'bilateral_raise',
        'BI_RAISE_SPREAD': 'bilateral_raise_spread',
        'BI_SPREAD': 'bilateral_spread',
        'BI_HANDS_CLOSE': 'bilateral_hands_close',
        'BI_HANDS_CLOSE_RAISE': 'bilateral_hands_close_raise',
        'BI_EXTENDED_LOCO_COUPLED': 'bilateral_extended_locomotion_coupled',
        'BI_LOCOMOTION_COUPLED': 'bilateral_locomotion_coupled',
        'BI_VERTICAL_COUPLED': 'bilateral_vertical_coupled',
    }.get(split_cluster, 'bilateral_unresolved')


def split_bimanual_events(events: list[dict[str, Any]], joints: np.ndarray | None) -> list[dict[str, Any]]:
    if joints is None:
        return events
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] <= 1 or joints.shape[1] <= JOINT_INDEX['R_Wrist']:
        return events

    out: list[dict[str, Any]] = []
    for evt in events:
        if evt.get('super_family') != BIMANUAL_FAMILY or str(evt.get('cluster_id')) not in BIMANUAL_COARSE_CLUSTERS:
            out.append(evt)
            continue
        features = bimanual_span_features(joints, evt, events)
        split_cluster, reason, split_confidence = classify_bimanual_event(evt, features)
        new_evt = dict(evt)
        original_cluster = str(evt.get('cluster_id'))
        new_evt['cluster_id'] = split_cluster
        new_evt['direction'] = _split_direction(split_cluster, str(evt.get('direction', '')))
        new_evt['optional_semantic_name'] = _split_phase_template(split_cluster)
        new_evt['confidence'] = min(float(evt.get('confidence', 0.0)), float(split_confidence))
        sig = dict(new_evt.get('motion_signature') or {})
        sig['phase_template'] = _split_phase_template(split_cluster)
        sig['bimanual_split_cluster'] = split_cluster
        new_evt['motion_signature'] = sig
        meta = dict(new_evt.get('metadata') or {})
        meta['bimanual_split'] = {
            'original_cluster': original_cluster,
            'split_cluster': split_cluster,
            'reason': reason,
            'confidence': float(split_confidence),
            'features': {
                'mean_hand_distance_norm': features['mean_hand_distance_norm'],
                'min_hand_distance_norm': features['min_hand_distance_norm'],
                'max_hand_distance_norm': features['max_hand_distance_norm'],
                'hand_distance_delta_norm': features['hand_distance_delta_norm'],
                'mean_wrist_chest_distance_norm': features['mean_wrist_chest_distance_norm'],
                'mean_wrist_height_rel_norm': features['mean_wrist_height_rel_norm'],
                'max_wrist_height_rel_norm': features['max_wrist_height_rel_norm'],
                'wrist_height_delta_norm': features['wrist_height_delta_norm'],
                'mean_wrist_speed_norm': features['mean_wrist_speed_norm'],
                'root_path_xz_norm': features['root_path_xz_norm'],
                'max_loco_overlap': features['max_loco_overlap'],
                'max_vertical_overlap': features['max_vertical_overlap'],
                'left_right_chest_corr': features['left_right_chest_corr'],
            },
        }
        new_evt['metadata'] = meta
        out.append(new_evt)
    return out
