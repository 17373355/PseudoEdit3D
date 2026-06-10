from __future__ import annotations

from typing import Any

import numpy as np

from pseudoedit3d.constants import JOINT_INDEX

BIMANUAL_FAMILY = 'BIMANUAL_PERIODIC'
BIMANUAL_ARM_CLUSTERS = {
    'BI_SPREAD',
    'BI_RAISE_SPREAD',
    'BI_RAISE',
    'BI_HANDS_CLOSE',
    'BI_HANDS_CLOSE_RAISE',
}
VERTICAL_JUMP_CLUSTERS = {
    'WB_VERT_UP',
    'WB_VERT_REP',
    'WB_VERT_REP_ALT',
}
LOCOMOTION_FAMILIES = {'WHOLE_BODY_LOCOMOTION'}


def _span(evt: dict[str, Any]) -> tuple[int, int]:
    return int(evt.get('start_frame', -1)), int(evt.get('end_frame', -1))


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    dur = max(1, min(a1 - a0 + 1, b1 - b0 + 1))
    return inter / dur


def _temporal_gap(a: dict[str, Any], b: dict[str, Any]) -> int:
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _near_or_overlap(a: dict[str, Any], b: dict[str, Any], max_gap: int) -> bool:
    return _overlap_ratio(a, b) > 0.0 or _temporal_gap(a, b) <= max_gap


def _root_motion_features(joints: np.ndarray | None, start: int, end: int, pre_frames: int = 20, post_frames: int = 20) -> dict[str, float]:
    if joints is None or len(joints) == 0:
        return {
            'coord_root_xz_displacement': 0.0,
            'coord_root_xz_path': 0.0,
            'coord_pre_root_xz_path': 0.0,
        }
    root = np.asarray(joints[:, JOINT_INDEX['Pelvis'], [0, 2]], dtype=np.float32)
    start = max(0, min(int(start), len(root) - 1))
    end = max(start, min(int(end), len(root) - 1))
    win_start = max(0, start - 8)
    win_end = min(len(root) - 1, end + post_frames)
    pre_start = max(0, start - pre_frames)
    pre_end = max(0, start - 1)

    def path(seg: np.ndarray) -> float:
        if len(seg) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(seg[1:] - seg[:-1], axis=-1)))

    window = root[win_start:win_end + 1]
    pre = root[pre_start:pre_end + 1]
    displacement = float(np.linalg.norm(window[-1] - window[0])) if len(window) >= 2 else 0.0
    return {
        'coord_root_xz_displacement': displacement,
        'coord_root_xz_path': path(window),
        'coord_pre_root_xz_path': path(pre),
    }


def _arm_timing(arm_evt: dict[str, Any], body_evt: dict[str, Any]) -> str:
    arm_start, arm_end = _span(arm_evt)
    body_start, body_end = _span(body_evt)
    if arm_end < body_start:
        return 'preparation'
    if arm_start <= body_end and arm_end >= body_start:
        return 'takeoff_overlap'
    return 'followthrough'


def _confidence(
    vertical_evt: dict[str, Any],
    arm_events: list[dict[str, Any]],
    root_features: dict[str, float],
    *,
    forward_like: bool,
    standing_like: bool,
) -> float:
    score = 0.45
    score += min(0.18, float(vertical_evt.get('magnitude') or 0.0) * 0.9)
    if arm_events:
        score += 0.16
    if forward_like:
        score += 0.12
    if standing_like:
        score += 0.06
    if root_features['coord_root_xz_displacement'] >= 0.55:
        score += 0.05
    return float(round(min(score, 0.95), 3))


def detect_coordination_patterns(events: list[dict[str, Any]], joints: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Detect higher-level body-arm coordination without collapsing arm variants.

    These patterns are intentionally above atomic bimanual clusters. For example,
    forward-jump coordination may use BI_SPREAD or BI_HANDS_CLOSE_RAISE as different
    arm realizations instead of treating either one as the action label.
    """
    vertical_events = [
        e for e in events
        if e.get('super_family') == 'WHOLE_BODY_VERTICAL' and str(e.get('cluster_id')) in VERTICAL_JUMP_CLUSTERS
    ]
    bimanual_events = [
        e for e in events
        if e.get('super_family') == BIMANUAL_FAMILY and str(e.get('cluster_id')) in BIMANUAL_ARM_CLUSTERS
    ]
    locomotion_events = [
        e for e in events
        if e.get('super_family') in LOCOMOTION_FAMILIES and not str(e.get('cluster_id', '')).startswith('LOCO_TURN_')
    ]
    preparation_events = [
        e for e in events
        if (
            (e.get('super_family') == 'WHOLE_BODY_VERTICAL' and e.get('cluster_id') == 'WB_VERT_DOWN')
            or (e.get('super_family') == 'WHOLE_BODY_POSTURE' and e.get('cluster_id') == 'WB_LOW_BODY_HOLD')
        )
    ]

    patterns: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for vertical in vertical_events:
        v_start, v_end = _span(vertical)
        nearby_arms = [arm for arm in bimanual_events if _near_or_overlap(arm, vertical, max_gap=18)]
        if not nearby_arms:
            continue
        nearby_locos = [loco for loco in locomotion_events if _near_or_overlap(loco, vertical, max_gap=12)]
        nearby_preps = [prep for prep in preparation_events if _near_or_overlap(prep, vertical, max_gap=18)]
        root_features = _root_motion_features(joints, v_start, v_end)
        loco_forward = any(str(loco.get('direction')) in {'forward', 'mixed', 'active'} for loco in nearby_locos)
        arm_clusters = sorted({str(arm.get('cluster_id')) for arm in nearby_arms})
        arm_timings = sorted({_arm_timing(arm, vertical) for arm in nearby_arms})
        vertical_magnitude = float(vertical.get('magnitude') or 0.0)
        forward_like = root_features['coord_root_xz_displacement'] >= 0.32 or loco_forward
        standing_like = (
            forward_like
            and bool(nearby_preps)
            and root_features['coord_pre_root_xz_path'] <= 0.12
            and root_features['coord_root_xz_displacement'] >= 0.75
            and vertical_magnitude >= 0.12
            and any(t in {'preparation', 'takeoff_overlap'} for t in arm_timings)
        )
        pattern_id = 'COORD_STANDING_FORWARD_JUMP_CANDIDATE' if standing_like else (
            'COORD_FORWARD_JUMP_ARM_COORDINATION' if forward_like else 'COORD_VERTICAL_JUMP_ARM_COORDINATION'
        )
        start = min([v_start] + [int(arm.get('start_frame', v_start)) for arm in nearby_arms])
        end = max([v_end] + [int(arm.get('end_frame', v_end)) for arm in nearby_arms])
        arm_variant = '+'.join(arm_clusters)
        key = (pattern_id, start, end, arm_variant)
        if key in seen:
            continue
        seen.add(key)
        patterns.append({
            'pattern_id': pattern_id,
            'semantic_family': 'jump_with_arm_coordination',
            'optional_semantic_name': 'standing_forward_jump_candidate' if standing_like else (
                'forward_jump_with_arm_coordination' if forward_like else 'vertical_jump_with_arm_coordination'
            ),
            'start_frame': int(start),
            'end_frame': int(end),
            'confidence': _confidence(vertical, nearby_arms, root_features, forward_like=forward_like, standing_like=standing_like),
            'coordination_slots': {
                'body': {
                    'vertical_cluster': vertical.get('cluster_id'),
                    'vertical_span': [int(v_start), int(v_end)],
                    'forward_like': bool(forward_like),
                    'standing_like': bool(standing_like),
                    'has_low_body_preparation': bool(nearby_preps),
                    'preparation_clusters': [str(p.get('cluster_id')) for p in nearby_preps],
                    **root_features,
                },
                'arms': {
                    'realization_clusters': arm_clusters,
                    'variant_key': arm_variant,
                    'timing': arm_timings,
                    'event_spans': [[int(a.get('start_frame', -1)), int(a.get('end_frame', -1))] for a in nearby_arms],
                },
                'locomotion': {
                    'nearby_locomotion_clusters': [str(l.get('cluster_id')) for l in nearby_locos],
                    'nearby_locomotion_directions': [str(l.get('direction')) for l in nearby_locos],
                },
            },
            'supporting_event_ids': [
                f"{vertical.get('super_family')}/{vertical.get('cluster_id')}[{v_start}-{v_end}]",
                *[
                    f"{arm.get('super_family')}/{arm.get('cluster_id')}[{int(arm.get('start_frame', -1))}-{int(arm.get('end_frame', -1))}]"
                    for arm in nearby_arms
                ],
            ],
        })
    patterns.sort(key=lambda x: (int(x['start_frame']), int(x['end_frame']), str(x['pattern_id'])))
    return patterns
