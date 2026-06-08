from __future__ import annotations

from typing import Any

from pseudoedit3d.edit.phase_patterns import PhasePattern
from pseudoedit3d.edit.submotion_lexicon import SubMotionUnit


def _tempo_bucket(start: int, end: int, count: int | None = None) -> str:
    duration = max(1, int(end) - int(start) + 1)
    span = duration / max(int(count or 1), 1)
    if span <= 6:
        return 'fast'
    if span <= 14:
        return 'medium'
    return 'slow'


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
    source: str = 'submotion',
    supporting_units: list[str] | None = None,
    motion_signature: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        'part': part,
        'super_family': super_family,
        'cluster_id': cluster_id,
        'optional_semantic_name': optional_semantic_name,
        'direction': direction,
        'role': role,
        'start_frame': int(start),
        'end_frame': int(end),
        'confidence': float(confidence),
        'source': source,
        'source_span': [int(start), int(end)],
        'supporting_units': supporting_units or [],
        'motion_signature': motion_signature or {},
    }
    if magnitude is not None:
        out['magnitude'] = float(magnitude)
    if signed_delta is not None:
        out['signed_delta'] = float(signed_delta)
    if unit is not None:
        out['unit'] = unit
    if count is not None:
        out['count'] = int(count)
    if metadata:
        out['metadata'] = metadata
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
        'dominant_axis': dominant_axis,
        'repeat_mode': repeat_mode,
        'phase_template': phase_template,
        'contact_mode': contact_mode,
        'support_mode': support_mode or contact_mode,
        'bilateral_symmetry': bilateral_symmetry or ('bilateral' if 'bi' in dominant_axis else 'unilateral'),
        'alternation': bool(alternation),
        'tempo_bucket': tempo_bucket or 'medium',
    }


def _rotation_magnitude_class(angle_deg: float) -> str:
    angle = abs(float(angle_deg))
    if angle < 75.0:
        return 'small'
    if angle < 135.0:
        return 'quarter'
    if angle < 225.0:
        return 'half'
    if angle < 315.0:
        return 'three_quarter'
    if angle < 450.0:
        return 'full'
    return 'multi'


def _rotation_cluster(direction: str, angle_deg: float) -> tuple[str, str]:
    mag_class = _rotation_magnitude_class(angle_deg)
    side = 'LEFT' if direction == 'left' else 'RIGHT'
    suffix = {
        'small': 'SMALL',
        'quarter': 'QTR',
        'half': 'HALF',
        'three_quarter': 'THREE_QTR',
        'full': 'FULL',
        'multi': 'MULTI',
    }[mag_class]
    return f'WB_ROT_{side}_{suffix}', mag_class


def _submotion_to_event(unit: SubMotionUnit) -> dict[str, Any] | None:
    name = unit.name
    meta = unit.metadata or {}
    magnitude = meta.get('max_abs_delta')
    signed_delta = meta.get('net_delta_sum')
    support = [name]
    start, end = unit.start_frame, unit.end_frame

    if name == 'whole_body_locomotion_active' or meta.get('direction') == 'locomotion_active':
        path_length = float(meta.get('path_length', meta.get('delta_value', signed_delta or 0.0)))
        mean_speed = float(meta.get('mean_speed', 0.0))
        speed_class = 'fast' if mean_speed >= 0.045 else ('medium' if mean_speed >= 0.025 else 'slow')
        trajectory_direction = str(meta.get('trajectory_direction') or 'unknown')
        direction_suffix = {
            'forward': 'FORWARD',
            'backward': 'BACKWARD',
            'left': 'LEFT',
            'right': 'RIGHT',
            'mixed': 'MIXED',
        }.get(trajectory_direction, 'ACTIVE')
        event_direction = trajectory_direction if trajectory_direction != 'unknown' else 'active'
        return _event(
            'whole_body', 'WHOLE_BODY_LOCOMOTION', f'LOCO_{direction_suffix}_{speed_class.upper()}', start, end,
            direction=event_direction, role='state', optional_semantic_name='locomotion_active',
            magnitude=path_length, signed_delta=path_length, unit='m', confidence=0.76,
            source='state_event', supporting_units=support,
            motion_signature=_sig('xz_path', 'state', f'locomotion_{event_direction}_{speed_class}', 'grounded', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={
                'submotion': name,
                'observable': meta.get('observable'),
                'mean_speed': mean_speed,
                'path_length': path_length,
                'active_ratio': meta.get('active_ratio'),
                'speed_threshold': meta.get('speed_threshold'),
                'trajectory_direction': trajectory_direction,
                'forward_displacement': meta.get('forward_displacement'),
                'lateral_displacement': meta.get('lateral_displacement'),
                'abs_forward_displacement': meta.get('abs_forward_displacement'),
                'abs_lateral_displacement': meta.get('abs_lateral_displacement'),
            },
        )

    if name == 'whole_body_low_body_hold' or meta.get('direction') == 'low_body_hold':
        low_depth = float(meta.get('delta_value', magnitude or 0.0))
        return _event(
            'whole_body', 'WHOLE_BODY_POSTURE', 'WB_LOW_BODY_HOLD', start, end,
            direction='low', role='state', optional_semantic_name='low_body_hold',
            magnitude=low_depth, signed_delta=None, unit='m', confidence=0.76,
            source='state_event', supporting_units=support,
            motion_signature=_sig('height', 'state', 'low_body_hold', 'grounded', support_mode='feet_or_body', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={
                'submotion': name,
                'observable': meta.get('observable'),
                'mean_height': meta.get('mean_height'),
                'min_height': meta.get('min_height'),
                'max_height_observed': meta.get('max_height_observed'),
                'low_height_threshold': meta.get('low_height_threshold'),
                'active_ratio': meta.get('active_ratio'),
            },
        )

    if name in {'whole_body_turn_left', 'whole_body_turn_right'} or meta.get('observable') == 'root_yaw_proxy_deg':
        raw_direction = str(meta.get('direction') or ('turn_left' if 'turn_left' in name else 'turn_right'))
        direction = 'left' if raw_direction == 'turn_left' else 'right'
        delta = float(meta.get('delta_value', signed_delta or 0.0))
        duration = int(end) - int(start) + 1
        if abs(delta) < 45.0 or duration < 4:
            return None
        cluster, mag_class = _rotation_cluster(direction, abs(delta))
        return _event(
            'whole_body', 'WHOLE_BODY_ROTATION', cluster, start, end,
            direction=direction, role='primitive', optional_semantic_name='body_turn',
            magnitude=abs(delta), signed_delta=delta, unit='deg', confidence=0.72,
            source='micro_event', supporting_units=support,
            motion_signature=_sig('yaw', 'single', f'turn_{direction}_{mag_class}', 'grounded', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={
                'submotion': name,
                'observable': meta.get('observable'),
                'raw_direction': raw_direction,
                'rotation_magnitude_class': mag_class,
                'layer3_salience_gate': 'abs_angle>=45deg,duration>=4',
            },
        )

    if name in {'crouch_descent', 'crouch_descent_strong', 'root_down_then_leg_compress', 'root_down_compress_release_cycle'}:
        return _event(
            'whole_body', 'WHOLE_BODY_VERTICAL', 'WB_VERT_DOWN', start, end,
            direction='down', role='primitive', optional_semantic_name='crouch_descent',
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.72,
            source='submotion', supporting_units=support,
            motion_signature=_sig('vertical', 'single', 'down_compress', 'grounded', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'hop_ascent', 'hop_ascent_variant', 'hop_unit', 'root_up_then_leg_release', 'root_up_release_compress_cycle'}:
        return _event(
            'whole_body', 'WHOLE_BODY_VERTICAL', 'WB_VERT_UP', start, end,
            direction='up', role='primitive', optional_semantic_name='hop_ascent',
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.74,
            source='submotion', supporting_units=support,
            motion_signature=_sig('vertical', 'single', 'release_up', 'grounded_to_air', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'leg_release_compress_cycle', 'leg_compress_release_cycle', 'leg_bounce_cycle'}:
        return _event(
            'whole_body', 'WHOLE_BODY_VERTICAL', 'WB_VERT_CYCLE', start, end,
            direction='up_down', role='composed', optional_semantic_name='leg_bounce_cycle',
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.7,
            source='submotion', supporting_units=support,
            motion_signature=_sig('vertical', 'cycle', 'compress_release', 'grounded', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'torso_forward_back_forward_cycle', 'torso_back_forward_back_cycle'}:
        return _event(
            'torso', 'TORSO_PERIODIC', 'TORSO_OSC_FB', start, end,
            direction='forward_back', role='composed', optional_semantic_name=None,
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.68,
            source='submotion', supporting_units=support,
            motion_signature=_sig('forward_back', 'cycle', 'forward_back', 'free', support_mode='none', bilateral_symmetry='axial', alternation=True, tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'torso_forward_bend_recover', 'torso_bend_recover', 'torso_rise_back'}:
        return _event(
            'torso', 'TORSO_PERIODIC', 'TORSO_BEND_RECOVER', start, end,
            direction='forward_down', role='composed', optional_semantic_name='torso_bend_recover',
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.72,
            source='submotion', supporting_units=support,
            motion_signature=_sig('forward_down', 'single', 'bend_recover', 'free', support_mode='none', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'left_hand_near_far_cycle', 'left_hand_far_near_cycle'}:
        return _event(
            'left_arm', 'LEFT_ARM_PERIODIC', 'LA_NEAR_FAR', start, end,
            direction='near_far', role='composed', optional_semantic_name=None,
            magnitude=magnitude, signed_delta=signed_delta, unit='deg', confidence=0.7,
            source='submotion', supporting_units=support,
            motion_signature=_sig('near_far', 'single', 'near_far', 'free', support_mode='none', bilateral_symmetry='unilateral', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'left_arm_near_down_far', 'left_arm_far_near_up', 'left_arm_near_up_down', 'left_arm_far_near_far'}:
        return _event(
            'left_arm', 'LEFT_ARM_PERIODIC', 'LA_COMPOSITE', start, end,
            direction='near_far', role='composed', optional_semantic_name=None,
            magnitude=magnitude, signed_delta=signed_delta, unit='deg', confidence=0.7,
            source='submotion', supporting_units=support,
            motion_signature=_sig('near_far', 'single', 'near_down_far', 'free', support_mode='none', bilateral_symmetry='unilateral', alternation=True, tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'right_hand_near_far_cycle', 'right_hand_far_near_cycle'}:
        return _event(
            'right_arm', 'RIGHT_ARM_PERIODIC', 'RA_NEAR_FAR', start, end,
            direction='near_far', role='composed', optional_semantic_name=None,
            magnitude=magnitude, signed_delta=signed_delta, unit='deg', confidence=0.7,
            source='submotion', supporting_units=support,
            motion_signature=_sig('near_far', 'single', 'near_far', 'free', support_mode='none', bilateral_symmetry='unilateral', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name in {'right_arm_near_down_far', 'right_arm_far_near_up', 'right_arm_near_up_down', 'right_arm_far_near_far'}:
        return _event(
            'right_arm', 'RIGHT_ARM_PERIODIC', 'RA_COMPOSITE', start, end,
            direction='near_far', role='composed', optional_semantic_name=None,
            magnitude=magnitude, signed_delta=signed_delta, unit='deg', confidence=0.7,
            source='submotion', supporting_units=support,
            motion_signature=_sig('near_far', 'single', 'near_down_far', 'free', support_mode='none', bilateral_symmetry='unilateral', alternation=True, tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name == 'both_arms_lift':
        return _event(
            'both_arms', 'BIMANUAL_PERIODIC', 'BI_UP', start, end,
            direction='up', role='primitive', optional_semantic_name='both_arms_lift',
            magnitude=magnitude, signed_delta=signed_delta, unit='deg', confidence=0.66,
            source='submotion', supporting_units=support,
            motion_signature=_sig('up', 'single', 'bilateral_raise', 'free', support_mode='none', bilateral_symmetry='bilateral', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    if name == 'hands_move_away_from_chest':
        return _event(
            'both_arms', 'BIMANUAL_PERIODIC', 'BI_OUT', start, end,
            direction='outward', role='primitive', optional_semantic_name='hands_move_away_from_chest',
            magnitude=magnitude, signed_delta=signed_delta, unit='m', confidence=0.66,
            source='submotion', supporting_units=support,
            motion_signature=_sig('outward', 'single', 'bilateral_out', 'free', support_mode='none', bilateral_symmetry='bilateral', tempo_bucket=_tempo_bucket(start, end)),
            metadata={'submotion': name},
        )
    return None


def _phase_to_event(phase: PhasePattern) -> dict[str, Any] | None:
    name = phase.name
    count = int(phase.count)
    support = [name]
    start, end = phase.start_frame, phase.end_frame
    alt = phase.kind == 'alternate'
    if 'leg_release_compress_cycle' in name or 'leg_compress_release_cycle' in name:
        return _event(
            'whole_body', 'WHOLE_BODY_VERTICAL', 'WB_VERT_CYCLE', start, end,
            direction='up_down', role='repeated_phase', optional_semantic_name='repeated_leg_bounce',
            count=count, confidence=0.82, source='phase', supporting_units=support,
            motion_signature=_sig('vertical', 'repeated_cycle', 'compress_release', 'grounded', support_mode='feet', bilateral_symmetry='axial', alternation=alt, tempo_bucket=_tempo_bucket(start, end, count)),
            metadata={'phase_name': name},
        )
    if 'hop_ascent' in name or 'crouch_descent' in name:
        return _event(
            'whole_body', 'WHOLE_BODY_VERTICAL', ('WB_VERT_REP_ALT' if alt else 'WB_VERT_REP'), start, end,
            direction='up_down', role='repeated_phase', optional_semantic_name='repeated_hop_like_cycle',
            count=count, confidence=0.8, source='phase', supporting_units=support,
            motion_signature=_sig('vertical', 'repeated_cycle', 'down_up', 'grounded_to_air', support_mode='feet', bilateral_symmetry='axial', alternation=alt, tempo_bucket=_tempo_bucket(start, end, count)),
            metadata={'phase_name': name},
        )
    if 'torso_torso_forward_extend__torso_torso_backward_retract' in name or 'torso_forward_back_forward_cycle' in name or 'torso_back_forward_back_cycle' in name:
        return _event(
            'torso', 'TORSO_PERIODIC', 'TORSO_OSC_FB', start, end,
            direction='forward_back', role='repeated_phase', optional_semantic_name='repeated_torso_oscillation',
            count=count, confidence=0.78, source='phase', supporting_units=support,
            motion_signature=_sig('forward_back', 'repeated_cycle', 'forward_back', 'free', support_mode='none', bilateral_symmetry='axial', alternation=True, tempo_bucket=_tempo_bucket(start, end, count)),
            metadata={'phase_name': name},
        )
    if 'left_arm' in name and ('near_chest' in name or 'far_from_chest' in name or 'left_arm_up' in name or 'left_arm_down' in name):
        cluster = 'LA_REPEAT_ALT' if alt else 'LA_REPEAT'
        phase_template = 'arm_alternation' if alt else 'arm_cycle'
        return _event(
            'left_arm', 'LEFT_ARM_PERIODIC', cluster, start, end,
            direction='up_down', role='repeated_phase', optional_semantic_name='repeated_left_arm_cycle',
            count=count, confidence=0.8, source='phase', supporting_units=support,
            motion_signature=_sig('up_down', 'repeated_cycle', phase_template, 'free', support_mode='none', bilateral_symmetry='unilateral', alternation=alt, tempo_bucket=_tempo_bucket(start, end, count)),
            metadata={'phase_name': name},
        )
    if 'right_arm' in name and ('near_chest' in name or 'far_from_chest' in name or 'right_arm_up' in name or 'right_arm_down' in name):
        cluster = 'RA_REPEAT_ALT' if alt else 'RA_REPEAT'
        phase_template = 'arm_alternation' if alt else 'arm_cycle'
        return _event(
            'right_arm', 'RIGHT_ARM_PERIODIC', cluster, start, end,
            direction='up_down', role='repeated_phase', optional_semantic_name='repeated_right_arm_cycle',
            count=count, confidence=0.8, source='phase', supporting_units=support,
            motion_signature=_sig('up_down', 'repeated_cycle', phase_template, 'free', support_mode='none', bilateral_symmetry='unilateral', alternation=alt, tempo_bucket=_tempo_bucket(start, end, count)),
            metadata={'phase_name': name},
        )
    return None


def dedupe_phase_patterns(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int]] = set()
    out: list[dict[str, Any]] = []
    for p in phases:
        key = (str(p.get('name')), int(p.get('start_frame', -1)), int(p.get('end_frame', -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    s1, e1 = int(a['start_frame']), int(a['end_frame'])
    s2, e2 = int(b['start_frame']), int(b['end_frame'])
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    dur1 = max(1, e1 - s1 + 1)
    dur2 = max(1, e2 - s2 + 1)
    return inter / max(1, min(dur1, dur2))


def _event_priority(evt: dict[str, Any]) -> tuple[int, float, int]:
    role_rank = {'primitive': 1, 'state': 2, 'composed': 3, 'repeated_phase': 4}.get(str(evt.get('role')), 0)
    confidence = float(evt.get('confidence', 0.0))
    duration = int(evt.get('end_frame', -1)) - int(evt.get('start_frame', -1))
    return (role_rank, confidence, duration)


def _merge_support(prev: dict[str, Any], evt: dict[str, Any]) -> None:
    prev.setdefault('supporting_units', [])
    merged = list(prev['supporting_units']) + list(evt.get('supporting_units', []))
    seen = set()
    uniq = []
    for x in merged:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    prev['supporting_units'] = uniq
    if 'count' in evt:
        prev['count'] = max(int(prev.get('count', 1)), int(evt.get('count', 1)))
    if 'magnitude' in evt:
        prev['magnitude'] = max(float(prev.get('magnitude', 0.0)), float(evt.get('magnitude', 0.0)))
    if 'signed_delta' in evt and 'signed_delta' not in prev:
        prev['signed_delta'] = evt['signed_delta']


def _annotate_context_coupling(events: list[dict[str, Any]]) -> None:
    body_events = [e for e in events if e.get('super_family') in {'WHOLE_BODY_VERTICAL', 'WHOLE_BODY_LOCOMOTION'}]
    split_clusters = {
        'LA_REPEAT',
        'RA_REPEAT',
        'LA_REPEAT_ALT',
        'RA_REPEAT_ALT',
    }
    for evt in events:
        sig = evt.setdefault('motion_signature', {})
        if evt.get('super_family') in {'LEFT_ARM_PERIODIC', 'RIGHT_ARM_PERIODIC', 'BIMANUAL_PERIODIC', 'TORSO_PERIODIC'}:
            max_overlap = 0.0
            for body_evt in body_events:
                max_overlap = max(max_overlap, _overlap_ratio(evt, body_evt))
            coupled = max_overlap >= 0.45
            sig['coupled_with_locomotion'] = bool(coupled)
            sig['context_mode'] = 'locomotion_coupled' if coupled else 'isolated_or_intentional'
            evt.setdefault('metadata', {})['max_whole_body_vertical_overlap'] = float(max_overlap)
            if coupled and evt.get('cluster_id') in split_clusters:
                evt['cluster_id'] = f"{evt['cluster_id']}_LOCO"
        else:
            sig.setdefault('coupled_with_locomotion', False)
            sig.setdefault('context_mode', 'body_driver')


def abstract_atomic_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for evt in sorted(events, key=lambda x: (int(x['start_frame']), int(x['end_frame']), str(x['super_family']), str(x['cluster_id']))):
        merged = False
        for prev in kept:
            if evt['part'] != prev['part']:
                continue
            if evt['super_family'] != prev['super_family']:
                continue
            if evt['cluster_id'] != prev['cluster_id']:
                continue
            if _overlap_ratio(evt, prev) < 0.6:
                continue
            if _event_priority(prev) >= _event_priority(evt):
                _merge_support(prev, evt)
                merged = True
                break
            _merge_support(evt, prev)
            prev.update(evt)
            merged = True
            break
        if not merged:
            kept.append(evt)
    return kept



def _unit_overlap_ratio(a: SubMotionUnit, b: SubMotionUnit) -> float:
    s1, e1 = int(a.start_frame), int(a.end_frame)
    s2, e2 = int(b.start_frame), int(b.end_frame)
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    dur = max(1, min(e1 - s1 + 1, e2 - s2 + 1))
    return inter / dur


def _build_locomotion_turn_events(submotions: list[SubMotionUnit]) -> list[dict[str, Any]]:
    locos = [
        u for u in submotions
        if u.name == 'whole_body_locomotion_active' or (u.metadata or {}).get('direction') == 'locomotion_active'
    ]
    yaw_units = [
        u for u in submotions
        if u.name in {'whole_body_turn_left', 'whole_body_turn_right'} or (u.metadata or {}).get('observable') == 'root_yaw_proxy_deg'
    ]
    out: list[dict[str, Any]] = []
    for loco in locos:
        meta = loco.metadata or {}
        path_length = float(meta.get('path_length', meta.get('delta_value', 0.0)) or 0.0)
        mean_speed = float(meta.get('mean_speed', 0.0) or 0.0)
        if path_length < 1.0:
            continue
        matched: list[SubMotionUnit] = []
        signed_sum = 0.0
        left_sum = 0.0
        right_sum = 0.0
        abs_sum = 0.0
        for yaw in yaw_units:
            if _unit_overlap_ratio(yaw, loco) < 0.10:
                continue
            ymeta = yaw.metadata or {}
            delta = float(ymeta.get('delta_value', 0.0) or 0.0)
            if abs(delta) < 5.0:
                continue
            matched.append(yaw)
            signed_sum += delta
            abs_sum += abs(delta)
            if delta >= 0:
                left_sum += abs(delta)
            else:
                right_sum += abs(delta)
        if not matched:
            continue
        dominance_margin = abs(left_sum - right_sum)
        if abs(signed_sum) < 40.0 and dominance_margin < 45.0:
            continue
        direction = 'left' if signed_sum >= 0 else 'right'
        mag_class = _rotation_magnitude_class(abs(signed_sum))
        side = 'LEFT' if direction == 'left' else 'RIGHT'
        cluster = f'LOCO_TURN_{side}_{mag_class.upper()}'
        start = min(int(u.start_frame) for u in matched)
        end = max(int(u.end_frame) for u in matched)
        out.append(_event(
            'whole_body', 'WHOLE_BODY_LOCOMOTION', cluster, start, end,
            direction=direction, role='state', optional_semantic_name='locomotion_turn',
            magnitude=abs(signed_sum), signed_delta=signed_sum, unit='deg', confidence=0.68,
            source='aggregate_micro_event',
            supporting_units=[f'{u.name}[{int(u.start_frame)}-{int(u.end_frame)}]' for u in matched],
            motion_signature=_sig('yaw_xz_path', 'state', f'locomotion_turn_{direction}_{mag_class}', 'grounded', support_mode='feet', bilateral_symmetry='axial', tempo_bucket=_tempo_bucket(start, end)),
            metadata={
                'submotion': 'whole_body_locomotion_turn',
                'source_locomotion_span': [int(loco.start_frame), int(loco.end_frame)],
                'path_length': path_length,
                'mean_speed': mean_speed,
                'yaw_signed_sum': signed_sum,
                'yaw_abs_sum': abs_sum,
                'yaw_left_sum': left_sum,
                'yaw_right_sum': right_sum,
                'yaw_dominance_margin': dominance_margin,
                'yaw_unit_spans': [[int(u.start_frame), int(u.end_frame)] for u in matched],
                'rule': 'path_length>=1m and cumulative yaw evidence within locomotion state',
            },
        ))
    return out

def build_layer3_atomic_program(submotions: list[SubMotionUnit], phase_patterns: list[PhasePattern]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for unit in submotions:
        evt = _submotion_to_event(unit)
        if evt is not None:
            events.append(evt)
    events.extend(_build_locomotion_turn_events(submotions))
    for phase in phase_patterns:
        evt = _phase_to_event(phase)
        if evt is not None:
            events.append(evt)
    events = abstract_atomic_events(events)
    _annotate_context_coupling(events)
    events.sort(key=lambda x: (int(x['start_frame']), int(x['end_frame']), str(x['super_family']), str(x['cluster_id'])))
    return {'events': events}
