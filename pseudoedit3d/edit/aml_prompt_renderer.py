from __future__ import annotations

from typing import Any


def _event_sort_key(evt: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(evt.get('start_frame', -1)),
        int(evt.get('end_frame', -1)),
        str(evt.get('super_family', '')),
        str(evt.get('cluster_id', '')),
    )


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    s1, e1 = int(a.get('start_frame', -1)), int(a.get('end_frame', -1))
    s2, e2 = int(b.get('start_frame', -1)), int(b.get('end_frame', -1))
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    dur = max(1, min(e1 - s1 + 1, e2 - s2 + 1))
    return inter / dur


def _spin_direction_phrase(direction: str) -> str:
    if direction == 'left':
        return 'counter-clockwise'
    if direction == 'right':
        return 'clockwise'
    return direction


def _max_locomotion_overlap(evt: dict[str, Any], events: list[dict[str, Any]] | None) -> float:
    if not events:
        return 0.0
    overlaps = []
    for other in events:
        if other.get('super_family') != 'WHOLE_BODY_LOCOMOTION':
            continue
        if not str(other.get('cluster_id', '')).startswith('LOCO_'):
            continue
        overlaps.append(_overlap_ratio(evt, other))
    return max(overlaps, default=0.0)


def _has_recent_vertical_down(evt: dict[str, Any], events: list[dict[str, Any]] | None, max_gap: int = 12) -> bool:
    if not events:
        return False
    start = int(evt.get('start_frame', -1))
    for other in events:
        if other.get('super_family') != 'WHOLE_BODY_VERTICAL':
            continue
        if other.get('cluster_id') != 'WB_VERT_DOWN':
            continue
        gap = start - int(other.get('end_frame', -1))
        if 0 <= gap <= max_gap:
            return True
    return False


def _rotation_phrase(evt: dict[str, Any]) -> str:
    cluster = str(evt.get('cluster_id', ''))
    direction = str(evt.get('direction', ''))
    spin_direction = _spin_direction_phrase(direction)
    mag = float(evt.get('magnitude') or abs(float(evt.get('signed_delta') or 0.0)))
    if 'MULTI' in cluster:
        return f'spins {spin_direction} multiple complete rotations'
    if 'FULL' in cluster:
        return f'does one complete 360-degree {spin_direction} spin'
    if 'THREE_QTR' in cluster:
        return f'turns {direction} about three quarters of a turn'
    if 'HALF' in cluster:
        return f'turns {direction} about a half turn'
    if 'QTR' in cluster:
        return f'turns {direction} about a quarter turn'
    return f'turns {direction} by about {mag:.0f} degrees'


def _locomotion_phrase(evt: dict[str, Any]) -> str:
    cluster = str(evt.get('cluster_id', ''))
    direction = str(evt.get('direction', ''))
    mag = float(evt.get('magnitude') or 0.0)
    if cluster.startswith('LOCO_TURN_'):
        return f'turns {direction} while moving'
    speed = 'quickly ' if 'FAST' in cluster else ('slowly ' if 'SLOW' in cluster else '')
    if direction == 'forward':
        return f'walks {speed}forward for about {mag:.1f} meters'
    if direction == 'backward':
        return f'walks {speed}backward for about {mag:.1f} meters'
    if direction in {'left', 'right'}:
        return f'moves {speed}to the {direction} for about {mag:.1f} meters'
    if direction == 'mixed':
        return f'moves {speed}through space with a changing direction for about {mag:.1f} meters'
    if 'FAST' in cluster:
        return f'moves quickly through space for about {mag:.1f} meters'
    if 'SLOW' in cluster:
        return f'moves slowly through space for about {mag:.1f} meters'
    return f'moves through space for about {mag:.1f} meters'


def event_to_prompt_clause(evt: dict[str, Any], context_events: list[dict[str, Any]] | None = None) -> str | None:
    family = str(evt.get('super_family', ''))
    cluster = str(evt.get('cluster_id', ''))
    count = evt.get('count')
    loco_overlap = _max_locomotion_overlap(evt, context_events)
    if family == 'WHOLE_BODY_LOCOMOTION':
        return _locomotion_phrase(evt)
    if family == 'WHOLE_BODY_ROTATION':
        return _rotation_phrase(evt)
    if family == 'WHOLE_BODY_POSTURE':
        if cluster == 'WB_LOW_BODY_HOLD':
            return 'keeps the body low'
    if family == 'WHOLE_BODY_VERTICAL':
        if loco_overlap >= 0.40:
            if cluster in {'WB_VERT_UP', 'WB_VERT_DOWN'}:
                return 'changes body height while moving'
            return None
        if cluster == 'WB_VERT_UP':
            if _has_recent_vertical_down(evt, context_events):
                return 'rises back up'
            return 'jumps upward'
        if cluster == 'WB_VERT_DOWN':
            return 'lowers the body'
        if cluster in {'WB_VERT_REP', 'WB_VERT_REP_ALT'}:
            n = f' {int(count)} times' if count else ''
            return f'repeats a hop-like up-and-down motion{n}'
        return 'makes a small up-and-down body motion'
    if family == 'BIMANUAL_PERIODIC':
        if cluster == 'BI_UP':
            return 'raises both arms'
        if cluster == 'BI_OUT':
            return 'moves both hands outward'
    if family == 'LEFT_ARM_PERIODIC':
        if 'REPEAT' in cluster and ('LOCO' in cluster or loco_overlap >= 0.40):
            return 'swings the left arm while walking'
        if 'REPEAT' in cluster:
            n = f' {int(count)} times' if count else ''
            return f'moves the left arm repeatedly{n}'
        return 'moves the left arm near and far from the body'
    if family == 'RIGHT_ARM_PERIODIC':
        if 'REPEAT' in cluster and ('LOCO' in cluster or loco_overlap >= 0.40):
            return 'swings the right arm while walking'
        if 'REPEAT' in cluster:
            n = f' {int(count)} times' if count else ''
            return f'moves the right arm repeatedly{n}'
        return 'moves the right arm near and far from the body'
    if family == 'TORSO_PERIODIC':
        if cluster == 'TORSO_BEND_RECOVER':
            return 'bends the torso forward and recovers'
        return 'oscillates the torso forward and backward'
    return None


def _has_salient_full_rotation(events: list[dict[str, Any]]) -> bool:
    for evt in events:
        if evt.get('super_family') != 'WHOLE_BODY_ROTATION':
            continue
        if any(k in str(evt.get('cluster_id', '')) for k in ('FULL', 'MULTI')):
            return True
    return False


def _is_low_salience_noise(evt: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    family = str(evt.get('super_family', ''))
    cluster = str(evt.get('cluster_id', ''))
    duration = int(evt.get('end_frame', -1)) - int(evt.get('start_frame', -1)) + 1
    magnitude = float(evt.get('magnitude') or 0.0)

    if family == 'BIMANUAL_PERIODIC' and cluster == 'BI_OUT':
        if duration <= 6 and magnitude < 0.08 and _has_salient_full_rotation(events):
            return True

    if family == 'WHOLE_BODY_LOCOMOTION' and cluster == 'LOCO_ACTIVE_SLOW':
        if magnitude < 0.55 and _has_salient_full_rotation(events):
            return True

    if family == 'WHOLE_BODY_VERTICAL':
        loco_overlap = _max_locomotion_overlap(evt, events)
        if cluster == 'WB_VERT_CYCLE' and magnitude < 0.04:
            return True
        if cluster in {'WB_VERT_CYCLE', 'WB_VERT_REP', 'WB_VERT_REP_ALT'} and loco_overlap >= 0.40:
            return True
        if cluster in {'WB_VERT_UP', 'WB_VERT_DOWN'} and loco_overlap >= 0.40 and magnitude < 0.14:
            return True
        if cluster == 'WB_VERT_CYCLE' and magnitude < 0.06 and _has_salient_full_rotation(events):
            return True

    return False


def _priority(evt: dict[str, Any]) -> tuple[int, int, float]:
    family = str(evt.get('super_family', ''))
    cluster = str(evt.get('cluster_id', ''))
    duration = int(evt.get('end_frame', -1)) - int(evt.get('start_frame', -1)) + 1
    magnitude = float(evt.get('magnitude') or 0.0)
    if family == 'WHOLE_BODY_ROTATION':
        return (0, -duration, -magnitude)
    if family == 'WHOLE_BODY_LOCOMOTION' and cluster.startswith('LOCO_ACTIVE'):
        return (1, -duration, -magnitude)
    if family == 'WHOLE_BODY_LOCOMOTION' and cluster.startswith('LOCO_TURN'):
        return (2, -duration, -magnitude)
    if family == 'WHOLE_BODY_POSTURE':
        return (3, -duration, -magnitude)
    if family == 'WHOLE_BODY_VERTICAL':
        return (4, -duration, -magnitude)
    if family == 'BIMANUAL_PERIODIC':
        return (5, -duration, -magnitude)
    if family in {'LEFT_ARM_PERIODIC', 'RIGHT_ARM_PERIODIC'}:
        return (6, -duration, -magnitude)
    return (9, -duration, -magnitude)


def _merge_arm_swing_clauses(clauses: list[str]) -> list[str]:
    left = 'swings the left arm while walking'
    right = 'swings the right arm while walking'
    if left not in clauses or right not in clauses:
        return clauses
    out: list[str] = []
    inserted = False
    for clause in clauses:
        if clause in {left, right}:
            if not inserted:
                out.append('swings both arms while walking')
                inserted = True
            continue
        out.append(clause)
    return out


def _merge_jump_spin_clause(clauses: list[str], events: list[dict[str, Any]]) -> list[str]:
    full_rot = [e for e in events if e.get('super_family') == 'WHOLE_BODY_ROTATION' and any(k in str(e.get('cluster_id', '')) for k in ('FULL', 'MULTI'))]
    vertical = [e for e in events if e.get('super_family') == 'WHOLE_BODY_VERTICAL']
    if not full_rot or not vertical:
        return clauses
    for rot in full_rot:
        max_overlap = max((_overlap_ratio(rot, v) for v in vertical), default=0.0)
        min_gap = min(
            (
                max(0, int(v.get('start_frame', -1)) - int(rot.get('end_frame', -1)), int(rot.get('start_frame', -1)) - int(v.get('end_frame', -1)))
                for v in vertical
            ),
            default=999,
        )
        if max_overlap < 0.05 and min_gap > 8:
            continue
        rot_clause = event_to_prompt_clause(rot, events)
        if rot_clause not in clauses:
            continue
        direction = _spin_direction_phrase(str(rot.get('direction', '')))
        merged = f'jumps and does one complete 360-degree {direction} spin'
        out = []
        inserted = False
        for clause in clauses:
            if clause == rot_clause:
                out.append(merged)
                inserted = True
                continue
            if clause in {'jumps upward', 'makes a small up-and-down body motion'}:
                continue
            out.append(clause)
        return out if inserted else clauses
    return clauses


def select_prompt_events(events: list[dict[str, Any]], max_events: int = 8) -> list[dict[str, Any]]:
    candidates = [evt for evt in events if not _is_low_salience_noise(evt, events)]
    ranked = sorted(candidates, key=_priority)
    picked: list[dict[str, Any]] = []
    seen_clauses: set[str] = set()
    for evt in ranked:
        if evt.get('super_family') in {'LEFT_ARM_PERIODIC', 'RIGHT_ARM_PERIODIC'}:
            sig = evt.get('motion_signature') or {}
            duration = int(evt.get('end_frame', -1)) - int(evt.get('start_frame', -1)) + 1
            if sig.get('context_mode') == 'locomotion_coupled' and duration <= 10:
                continue
        clause = event_to_prompt_clause(evt, events)
        if not clause or clause in seen_clauses:
            continue
        picked.append(evt)
        seen_clauses.add(clause)
        if len(picked) >= max_events:
            break
    return sorted(picked, key=_event_sort_key)


def render_aml_prompt(program: dict[str, Any], *, max_events: int = 8) -> str:
    events = list((program or {}).get('events') or [])
    picked = select_prompt_events(events, max_events=max_events)
    clauses: list[str] = []
    for evt in picked:
        clause = event_to_prompt_clause(evt, events)
        if clause and clause not in clauses:
            clauses.append(clause)
    clauses = _merge_arm_swing_clauses(clauses)
    clauses = _merge_jump_spin_clause(clauses, events)
    if not clauses:
        return 'a person moves naturally'
    return 'a person ' + ', then '.join(clauses)
