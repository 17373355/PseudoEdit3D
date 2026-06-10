from __future__ import annotations

from typing import Any

from .aml_prompt_renderer import event_to_prompt_clause
from .coarse_signature import build_coarse_action_program


_BANNED_RESIDUAL_PHRASES = {
    'changes body height while moving',
    'repeats a hop-like up-and-down motion',
    'makes a small up-and-down body motion',
    'oscillates the torso forward and backward',
    'bends the torso forward and recovers',
    'moves the left arm repeatedly',
    'moves the right arm repeatedly',
    'moves the left arm near and far from the body',
    'moves the right arm near and far from the body',
    'swings the left arm while walking',
    'swings the right arm while walking',
}

_DEFAULT_MAX_PROBE_CLAUSES = 5
_DEFAULT_MAX_RESIDUAL_CLAUSES = 1
_DEFAULT_MAX_WORDS = 34


_NUMBER_WORD = {
    1: 'once',
    2: 'twice',
    3: '3 times',
    4: '4 times',
    5: '5 times',
}


def _distance_phrase(distance_m: float) -> str:
    if distance_m < 0.7:
        return ''
    if distance_m >= 1.5:
        return f' for about {distance_m:.1f} meters'
    return f' for about {distance_m:.1f} meters'


def _speed_prefix(speed: str) -> str:
    if speed == 'fast':
        return 'quickly '
    if speed == 'slow':
        return 'slowly '
    return ''


def _direction_phrase(direction: str, *, gait: bool = False) -> str:
    if direction == 'forward':
        return 'forward'
    if direction == 'backward':
        return 'backward'
    if direction in {'left', 'right'}:
        return f'to the {direction}' if gait else direction
    if direction == 'mixed':
        return 'through a changing path' if gait else 'with a changing direction'
    if direction == 'in_place':
        return 'in place'
    return ''


def _repeat_phrase(count: int) -> str:
    if count <= 0:
        return ''
    return ' ' + _NUMBER_WORD.get(count, f'{count} times')


def _gait_clause(signature: dict[str, Any], action: dict[str, Any]) -> str:
    locomotion = signature.get('locomotion') or {}
    name = str(action.get('name_hint') or 'walk')
    verb = 'runs' if name == 'run' else 'walks'
    speed_value = str(action.get('speed') or locomotion.get('speed', 'unknown'))
    distance_value = float(action.get('distance_m') if action.get('distance_m') is not None else locomotion.get('distance_m') or 0.0)
    speed = _speed_prefix(speed_value)
    direction = _direction_phrase(str(action.get('primary_direction', locomotion.get('direction', 'none'))), gait=True)
    distance = _distance_phrase(distance_value)
    if direction:
        return f'{verb} {speed}{direction}{distance}'.replace('  ', ' ')
    return f'{verb} {speed}naturally{distance}'.replace('  ', ' ')


def _jump_clause(signature: dict[str, Any], action: dict[str, Any]) -> str:
    vertical = signature.get('vertical') or {}
    direction = str(action.get('primary_direction', 'in_place'))
    repeat = int(action.get('count') or vertical.get('repeat_count') or 0)
    amplitude = float(action.get('vertical_amplitude_m') or vertical.get('max_amplitude_m') or 0.0)
    if direction in {'forward', 'backward'}:
        return f'jumps {direction}'
    if direction in {'left', 'right'}:
        return f'jumps to the {direction}'
    if repeat >= 3:
        return f'jumps up and down{_repeat_phrase(repeat)}'
    if amplitude >= 0.20:
        return 'jumps straight up'
    return 'jumps upward'


def _jumping_jack_clause(signature: dict[str, Any], action: dict[str, Any]) -> str:
    repeat = int(action.get('count') or (signature.get('vertical') or {}).get('repeat_count') or 0)
    if repeat >= 2:
        return f'does jumping jacks{_repeat_phrase(repeat)}'
    return 'does a jumping jack'


def _turn_clause(signature: dict[str, Any], action: dict[str, Any] | None = None) -> str:
    action = action or {}
    rotation = signature.get('rotation') or {}
    direction = str(action.get('primary_direction') or rotation.get('direction', ''))
    angle_bin = str(action.get('angle_bin') or rotation.get('angle_bin', ''))
    angle = float(action.get('angle_deg') if action.get('angle_deg') is not None else rotation.get('angle_deg') or 0.0)
    angle_bin_upper = angle_bin.upper()
    if angle_bin == 'multi' or 'MULTI' in angle_bin_upper:
        return f'spins {direction} multiple times'
    if angle_bin == 'full' or 'FULL' in angle_bin_upper:
        return f'spins {direction} around once'
    if angle_bin == 'three_quarter' or 'THREE_QTR' in angle_bin_upper:
        return f'turns {direction} about three quarters of a turn'
    if angle_bin == 'half' or 'HALF' in angle_bin_upper:
        return f'turns {direction} about a half turn'
    if angle_bin == 'quarter' or 'QTR' in angle_bin_upper:
        return f'turns {direction} about a quarter turn'
    return f'turns {direction} by about {angle:.0f} degrees'


def _bimanual_clause(signature: dict[str, Any]) -> str:
    pattern = str((signature.get('limb_coordination') or {}).get('bimanual_pattern', 'none'))
    if pattern == 'raise_spread_repeated':
        return 'repeatedly raises and spreads both arms'
    if pattern == 'raise_spread':
        return 'raises and spreads both arms'
    if pattern == 'hands_close':
        return 'brings both hands closer together'
    if pattern == 'raise':
        return 'raises both arms'
    if pattern == 'spread':
        return 'spreads both arms outward'
    return 'moves both arms'


def _hands_close_clause(action: dict[str, Any]) -> str:
    del action
    return 'brings both hands together'


def _action_clause(signature: dict[str, Any], action: dict[str, Any]) -> str | None:
    pid = str(action.get('prototype_id', ''))
    if pid in {'TRANSLATING_GAIT', 'TRANSLATING_GAIT_SEGMENT'}:
        return _gait_clause(signature, action)
    if pid == 'IN_PLACE_GAIT':
        name_hint = str(action.get('name_hint'))
        if name_hint == 'run_in_place':
            return 'runs in place'
        if name_hint == 'jog_in_place':
            return 'jogs in place'
        return 'walks in place'
    if pid == 'IN_PLACE_GAIT_PROXY':
        return 'makes a small in-place bouncing motion'
    if pid == 'CELEBRATORY_DANCE_GESTURE':
        count = int(action.get('raise_spread_count') or 0)
        if count >= 3:
            return 'makes a cheer-like dance gesture with repeated arm raises'
        return 'makes a cheer-like dance gesture'
    if pid in {'BALLISTIC_TRANSLATION', 'BALLISTIC_TRANSLATION_SEGMENT', 'VERTICAL_JUMP', 'VERTICAL_JUMP_SEGMENT'}:
        return _jump_clause(signature, action)
    if pid == 'JUMPING_JACK':
        return _jumping_jack_clause(signature, action)
    if pid in {'ROTATION_DOMINANT', 'TURN_SEGMENT'}:
        return _turn_clause(signature, action)
    if pid == 'TERMINAL_STILL':
        return 'comes to a stop and stands still'
    if pid == 'RECOVERY_STEP_SEGMENT':
        direction = str(action.get('primary_direction') or '')
        if direction in {'backward', 'forward'}:
            return f'steps {direction} to regain balance'
        if direction in {'left', 'right'}:
            return f'steps to the {direction} to regain balance'
        return 'takes a recovery step to regain balance'
    if pid == 'BIMANUAL_HANDS_CLOSE':
        return _hands_close_clause(action)
    if pid == 'BIMANUAL_ACTION':
        return _bimanual_clause(signature)
    if pid == 'BIMANUAL_ARM_MIME_CANDIDATE':
        return 'makes a bimanual upper-body gesture'
    if pid == 'UNILATERAL_ARM_MIME_CANDIDATE':
        side = str(action.get('dominant_side') or '')
        if side in {'left', 'right'}:
            return f'makes repeated {side} arm gestures'
        return 'makes repeated one-arm gestures'
    if pid == 'STATIC_OR_SUBTLE_STATE_PROXY':
        return 'holds a mostly still subtle pose'
    if pid == 'TORSO_HUNCHED_FORWARD':
        return 'keeps the torso hunched forward'
    if pid == 'LEFT_HAND_RAISED_HIGH':
        return 'raises the left hand high'
    if pid == 'RIGHT_HAND_RAISED_HIGH':
        return 'raises the right hand high'
    if pid == 'SQUAT_HOLD':
        return 'squats low'
    if pid == 'SQUAT_REPETITION':
        count = int(action.get('count') or 0)
        return f'repeatedly squats low{_repeat_phrase(count)}' if count >= 2 else 'squats low'
    if pid == 'SQUAT_ARM_LIFT':
        count = int(action.get('count') or 0)
        suffix = _repeat_phrase(count) if count >= 2 else ''
        return f'repeatedly squats low while lifting the arms{suffix}'
    if pid == 'LEFT_LEG_KICK_FORWARD':
        return 'kicks the left leg forward'
    if pid == 'RIGHT_LEG_KICK_FORWARD':
        return 'kicks the right leg forward'
    if pid == 'LEG_FORWARD_POSE_CANDIDATE':
        side = str(action.get('dominant_side') or '')
        if side in {'left', 'right'}:
            return f'holds the {side} leg forward'
        return 'holds one leg forward'
    if pid == 'DANCE_LEG_POSE_CANDIDATE':
        side = str(action.get('dominant_side') or '')
        if side in {'left', 'right'}:
            return f'holds a dance-like pose with the {side} leg extended'
        return 'holds a dance-like raised-leg pose'
    if pid == 'CIRCULAR_WALK_PATH':
        return 'walks in a circular path'
    if pid == 'CLIMB_UP_OVER_PROXY':
        return 'climbs upward and over'
    if pid == 'CARTWHEEL_CANDIDATE':
        return 'does a cartwheel-like inverted motion'
    if pid == 'INVERTED_ACROBATICS_CANDIDATE':
        return 'does an inverted acrobatic motion'
    if pid == 'ACROBATIC_SEQUENCE_CANDIDATE':
        count = int(action.get('segment_count') or action.get('count') or 0)
        if count >= 2:
            return f'does repeated inverted acrobatic motions{_repeat_phrase(count)}'
        return 'does an inverted acrobatic motion'
    return None


def _probe_sort_key(action: dict[str, Any]) -> tuple[int, int, int]:
    span = action.get('span') or [0, 0]
    start = int(span[0]) if span else 0
    return (start, int(span[1]) if len(span) > 1 else start, 0)


def _semantic_status(action: dict[str, Any]) -> str:
    family = action.get('semantic_family') or {}
    if isinstance(family, dict):
        status = str(family.get('status') or '')
        if status:
            return status
    slots = action.get('slots') or {}
    return str(slots.get('semantic_family_status') or 'stable')


def _action_salience(action: dict[str, Any]) -> float:
    pid = str(action.get('prototype_id', ''))
    score = {
        'JUMPING_JACK': 1.0,
        'BALLISTIC_TRANSLATION': 0.96,
        'VERTICAL_JUMP': 0.93,
        'CELEBRATORY_DANCE_GESTURE': 0.92,
        'ACROBATIC_SEQUENCE_CANDIDATE': 1.02,
        'CARTWHEEL_CANDIDATE': 0.95,
        'INVERTED_ACROBATICS_CANDIDATE': 0.92,
        'CLIMB_UP_OVER_PROXY': 0.92,
        'SQUAT_ARM_LIFT': 0.96,
        'SQUAT_REPETITION': 0.94,
        'CIRCULAR_WALK_PATH': 0.87,
        'SQUAT_HOLD': 0.84,
        'LEFT_LEG_KICK_FORWARD': 0.82,
        'RIGHT_LEG_KICK_FORWARD': 0.82,
        'LEG_FORWARD_POSE_CANDIDATE': 0.86,
        'DANCE_LEG_POSE_CANDIDATE': 0.97,
        'TORSO_HUNCHED_FORWARD': 0.72,
        'LEFT_HAND_RAISED_HIGH': 0.80,
        'RIGHT_HAND_RAISED_HIGH': 0.80,
        'IN_PLACE_GAIT': 0.90,
        'IN_PLACE_GAIT_PROXY': 0.56,
        'TRANSLATING_GAIT': 0.90,
        'TRANSLATING_GAIT_SEGMENT': 0.74,
        'BALLISTIC_TRANSLATION_SEGMENT': 0.82,
        'RECOVERY_STEP_SEGMENT': 0.70,
        'ROTATION_DOMINANT': 0.76,
        'TURN_SEGMENT': 0.72,
        'TERMINAL_STILL': 0.70,
        'BIMANUAL_HANDS_CLOSE': 0.45,
        'BIMANUAL_ACTION': 0.42,
        'BIMANUAL_ARM_MIME_CANDIDATE': 0.62,
        'UNILATERAL_ARM_MIME_CANDIDATE': 0.58,
        'STATIC_OR_SUBTLE_STATE_PROXY': 0.44,
    }.get(pid, 0.25)
    if action.get('probe_visible') is False:
        score -= 1.0
    status = _semantic_status(action)
    if status == 'unknown':
        score -= 2.0
    elif status == 'candidate':
        score -= 0.08
    elif status == 'proxy':
        score -= 0.12
    try:
        confidence = float(action.get('confidence') or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return score + min(0.1, confidence * 0.1)


def _word_count(clauses: list[str]) -> int:
    return len(('a person ' + ', then '.join(clauses)).split())


def _trim_to_budget(clauses: list[str], *, max_words: int) -> list[str]:
    out: list[str] = []
    for clause in clauses:
        candidate = out + [clause]
        if out and _word_count(candidate) > max_words:
            continue
        out = candidate
    return out or clauses[:1]


def _unique_clause_rows(rows: list[tuple[dict[str, Any], str, float]]) -> list[tuple[dict[str, Any], str, float]]:
    out: list[tuple[dict[str, Any], str, float]] = []
    seen: set[str] = set()
    for row in rows:
        clause = row[1]
        if clause in seen:
            continue
        seen.add(clause)
        out.append(row)
    return out


def _trim_rows_to_budget(
    rows: list[tuple[dict[str, Any], str, float]],
    *,
    max_words: int,
) -> list[tuple[dict[str, Any], str, float]]:
    rows = _unique_clause_rows(rows)
    if not rows:
        return rows
    while len(rows) > 1 and _word_count([row[1] for row in rows]) > max_words:
        remove_idx = min(
            range(len(rows)),
            key=lambda idx: (
                rows[idx][2],
                -len(rows[idx][1].split()),
                _probe_sort_key(rows[idx][0]),
            ),
        )
        rows.pop(remove_idx)
    return rows


def _is_banned_residual(clause: str) -> bool:
    return any(clause.startswith(prefix) for prefix in _BANNED_RESIDUAL_PHRASES)


def _residual_clauses(program: dict[str, Any], coarse: dict[str, Any], max_residual_events: int) -> list[str]:
    events = list((program or {}).get('events') or [])
    covered = set(int(x) for x in coarse.get('covered_event_indices') or [])
    clauses: list[str] = []
    for idx, evt in enumerate(events):
        if idx in covered:
            continue
        clause = event_to_prompt_clause(evt, events)
        if not clause or _is_banned_residual(clause):
            continue
        if clause in clauses:
            continue
        clauses.append(clause)
        if len(clauses) >= max_residual_events:
            break
    return clauses


def render_coarse_aml_prompt(
    program: dict[str, Any],
    *,
    max_residual_events: int = 3,
    max_probe_clauses: int = _DEFAULT_MAX_PROBE_CLAUSES,
    max_residual_clauses: int = _DEFAULT_MAX_RESIDUAL_CLAUSES,
    max_words: int = _DEFAULT_MAX_WORDS,
    return_program: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Render a MoMask-compatible prompt from coarse AML prototypes plus residuals."""
    coarse = build_coarse_action_program(program, max_residual_events=max_residual_events)
    signature = coarse.get('signature') or {}
    action_rows: list[tuple[dict[str, Any], str, float]] = []
    for action in sorted(coarse.get('coarse_actions') or [], key=_probe_sort_key):
        if action.get('probe_visible') is False:
            continue
        if _semantic_status(action) == 'unknown':
            continue
        clause = _action_clause(signature, action)
        if clause:
            action_rows.append((action, clause, _action_salience(action)))

    selected_rows = [
        row for row in action_rows
        if row[2] >= 0.70
    ]
    if not selected_rows and action_rows:
        selected_rows = [max(action_rows, key=lambda row: row[2])]
    if len(selected_rows) < max_probe_clauses:
        for row in sorted(action_rows, key=lambda item: (-item[2], _probe_sort_key(item[0]))):
            if row in selected_rows:
                continue
            if row[2] < 0.40:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= max_probe_clauses:
                break
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (-row[2], _probe_sort_key(row[0])),
    )[:max_probe_clauses]
    selected_rows.sort(key=lambda row: _probe_sort_key(row[0]))
    selected_rows = _trim_rows_to_budget(selected_rows[:max_probe_clauses], max_words=max_words)

    clauses: list[str] = [clause for _action, clause, _score in selected_rows]

    residual_budget = min(max_residual_clauses, max_residual_events)
    if residual_budget and len(clauses) < max_probe_clauses:
        for clause in _residual_clauses(program, coarse, max_residual_events=residual_budget):
            if clause not in clauses:
                clauses.append(clause)
            if len(clauses) >= max_probe_clauses:
                break
    if not clauses:
        clauses = ['moves naturally']
    prompt = 'a person ' + ', then '.join(clauses)
    if return_program:
        return prompt, coarse
    return prompt
