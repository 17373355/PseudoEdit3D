from __future__ import annotations

from typing import Any

from .aml_family_taxonomy import active_family_id
from .aml_prompt_renderer import event_to_prompt_clause
from .aml_proto_registry import registry_map
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


def _low_body_repetition_clause(action: dict[str, Any], *, has_arm_lift: bool = False) -> str:
    count = int(action.get('count') or 0)
    suffix = _repeat_phrase(count) if count >= 2 else ''
    if has_arm_lift:
        return f'repeats a low-body posture while lifting the arms{suffix}'
    return f'repeats a low-body posture{suffix}' if count >= 2 else 'holds a low-body posture'


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


def _renderer_specs() -> dict[str, Any]:
    return registry_map('prompt_renderer')


def _renderer_spec_for(action: dict[str, Any]) -> dict[str, Any]:
    pid = active_family_id(str(action.get('prototype_id', '')))
    specs = _renderer_specs()
    by_prototype = specs.get('by_prototype') or {}
    spec = by_prototype.get(pid) if isinstance(by_prototype, dict) else None
    return dict(spec) if isinstance(spec, dict) else {}


def _field_text(action: dict[str, Any], spec: dict[str, Any]) -> str:
    path = str(spec.get('path', ''))
    value = action.get(path) if path else None
    if value is None:
        value = spec.get('default')
    if value is None:
        return ''
    text = str(value)
    allowed = spec.get('allowed')
    if isinstance(allowed, list) and text not in {str(item) for item in allowed}:
        return ''
    omit_if = {str(item) for item in spec.get('omit_if') or []}
    if text in omit_if:
        return ''
    return text


def _template_fields(action: dict[str, Any], fields: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in fields.items():
        if not isinstance(value, dict):
            continue
        if value.get('kind') == 'repeat_suffix':
            count = int(action.get(str(value.get('path', 'count'))) or 0)
            threshold = int(value.get('min', 1) or 1)
            out[str(key)] = _repeat_phrase(count) if count >= threshold else ''
        else:
            out[str(key)] = _field_text(action, value)
    return out


def _recovery_step_clause(action: dict[str, Any]) -> str:
    direction = str(action.get('primary_direction') or '')
    if direction in {'backward', 'forward'}:
        return f'steps {direction} to regain balance'
    if direction in {'left', 'right'}:
        return f'steps to the {direction} to regain balance'
    return 'takes a recovery step to regain balance'


def _acrobatic_sequence_clause(action: dict[str, Any]) -> str:
    count = int(action.get('segment_count') or action.get('count') or 0)
    if count >= 2:
        return f'does repeated inverted acrobatic motions{_repeat_phrase(count)}'
    return 'does an inverted acrobatic motion'


def _clause_from_spec(signature: dict[str, Any], action: dict[str, Any], spec: dict[str, Any]) -> str | None:
    if not spec:
        return None
    if 'value' in spec:
        return str(spec['value'])
    if 'field_cases' in spec:
        field_cases = spec.get('field_cases') or {}
        field = str(field_cases.get('field', ''))
        value = str(action.get(field) or '')
        cases = field_cases.get('cases') or {}
        if isinstance(cases, dict) and value in cases:
            return str(cases[value])
        if field_cases.get('default') is not None:
            return str(field_cases['default'])
    if spec.get('kind') == 'name_cases':
        cases = spec.get('cases') or {}
        name = str(action.get('name_hint') or '')
        if isinstance(cases, dict) and name in cases:
            return str(cases[name])
        return str(spec.get('default', ''))
    if 'template' in spec:
        fields = _template_fields(action, spec.get('fields') or {})
        if all(value for value in fields.values() if value is not None) or fields:
            try:
                return str(spec['template']).format(**fields)
            except KeyError:
                pass
        if spec.get('fallback') is not None:
            return str(spec['fallback'])
    kind = str(spec.get('kind', ''))
    if kind == 'gait':
        return _gait_clause(signature, action)
    if kind == 'jump':
        return _jump_clause(signature, action)
    if kind == 'turn':
        return _turn_clause(signature, action)
    if kind == 'bimanual_signature':
        return _bimanual_clause(signature)
    if kind == 'low_body_repetition':
        return _low_body_repetition_clause(action, has_arm_lift=bool(spec.get('has_arm_lift')))
    if kind == 'recovery_step':
        return _recovery_step_clause(action)
    if kind == 'acrobatic_sequence':
        return _acrobatic_sequence_clause(action)
    return None


def _action_clause(signature: dict[str, Any], action: dict[str, Any]) -> str | None:
    semantic_alias = action.get('semantic_alias')
    if isinstance(semantic_alias, dict):
        clause = str(semantic_alias.get('clause') or '')
        if clause:
            return clause
    spec = _renderer_spec_for(action).get('clause') or _renderer_specs().get('default_clause') or {}
    return _clause_from_spec(signature, action, dict(spec)) if isinstance(spec, dict) else None


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
    specs = _renderer_specs()
    score = float(_renderer_spec_for(action).get('salience', specs.get('default_salience', 0.25)))
    if isinstance(action.get('semantic_alias'), dict):
        priority = int((action.get('semantic_alias') or {}).get('priority') or 0)
        score = max(score, 0.86)
        score += min(0.10, max(0, priority - 70) * 0.002)
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
    caption_hints: list[str] | str | None = None,
    return_program: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Render a MoMask-compatible prompt from coarse AML prototypes plus residuals."""
    coarse = build_coarse_action_program(
        program,
        max_residual_events=max_residual_events,
        caption_hints=caption_hints,
    )
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
