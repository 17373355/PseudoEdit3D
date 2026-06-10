from __future__ import annotations

from typing import Any


FAMILY_LABELS = {
    'WHOLE_BODY_VERTICAL': 'whole_body.vertical',
    'WHOLE_BODY_ROTATION': 'whole_body.rotation',
    'WHOLE_BODY_LOCOMOTION': 'whole_body.locomotion',
    'TORSO_PERIODIC': 'torso.periodic',
    'LEFT_ARM_PERIODIC': 'left_arm.periodic',
    'RIGHT_ARM_PERIODIC': 'right_arm.periodic',
    'BIMANUAL_PERIODIC': 'both_arms.periodic',
}

CLUSTER_VERBS = {
    'WB_VERT_DOWN': 'move downward / compress',
    'WB_VERT_UP': 'move upward / release',
    'WB_VERT_CYCLE': 'vertical up-down cycle',
    'WB_VERT_REP': 'repeat vertical hop-like cycle',
    'WB_VERT_REP_ALT': 'repeat alternating vertical cycle',
    'WB_ROT_LEFT_SMALL': 'turn left slightly',
    'WB_ROT_LEFT_QTR': 'turn left about a quarter turn',
    'WB_ROT_LEFT_HALF': 'turn left about a half turn',
    'WB_ROT_LEFT_THREE_QTR': 'turn left about a three-quarter turn',
    'WB_ROT_LEFT_FULL': 'turn left about a full turn',
    'WB_ROT_LEFT_MULTI': 'turn left multiple rotations',
    'WB_ROT_RIGHT_SMALL': 'turn right slightly',
    'WB_ROT_RIGHT_QTR': 'turn right about a quarter turn',
    'WB_ROT_RIGHT_HALF': 'turn right about a half turn',
    'WB_ROT_RIGHT_THREE_QTR': 'turn right about a three-quarter turn',
    'WB_ROT_RIGHT_FULL': 'turn right about a full turn',
    'WB_ROT_RIGHT_MULTI': 'turn right multiple rotations',
    'LOCO_ACTIVE_SLOW': 'move through space slowly',
    'LOCO_ACTIVE_MEDIUM': 'move through space',
    'LOCO_ACTIVE_FAST': 'move through space quickly',
    'LOCO_FORWARD_SLOW': 'move forward slowly',
    'LOCO_FORWARD_MEDIUM': 'move forward',
    'LOCO_FORWARD_FAST': 'move forward quickly',
    'LOCO_BACKWARD_SLOW': 'move backward slowly',
    'LOCO_BACKWARD_MEDIUM': 'move backward',
    'LOCO_BACKWARD_FAST': 'move backward quickly',
    'LOCO_LEFT_SLOW': 'move left slowly',
    'LOCO_LEFT_MEDIUM': 'move left',
    'LOCO_LEFT_FAST': 'move left quickly',
    'LOCO_RIGHT_SLOW': 'move right slowly',
    'LOCO_RIGHT_MEDIUM': 'move right',
    'LOCO_RIGHT_FAST': 'move right quickly',
    'LOCO_MIXED_SLOW': 'move through space with mixed direction slowly',
    'LOCO_MIXED_MEDIUM': 'move through space with mixed direction',
    'LOCO_MIXED_FAST': 'move through space with mixed direction quickly',
    'TORSO_OSC_FB': 'oscillate forward-back',
    'TORSO_BEND_RECOVER': 'bend forward and recover',
    'LA_NEAR_FAR': 'left hand near-far cycle',
    'LA_COMPOSITE': 'left arm composite near/far cycle',
    'LA_REPEAT': 'repeat left arm cycle',
    'LA_REPEAT_LOCO': 'repeat left arm cycle coupled with locomotion',
    'LA_REPEAT_ALT': 'repeat alternating left arm cycle',
    'LA_REPEAT_ALT_LOCO': 'repeat alternating left arm cycle coupled with locomotion',
    'RA_NEAR_FAR': 'right hand near-far cycle',
    'RA_COMPOSITE': 'right arm composite near/far cycle',
    'RA_REPEAT': 'repeat right arm cycle',
    'RA_REPEAT_LOCO': 'repeat right arm cycle coupled with locomotion',
    'RA_REPEAT_ALT': 'repeat alternating right arm cycle',
    'RA_REPEAT_ALT_LOCO': 'repeat alternating right arm cycle coupled with locomotion',
    'BI_UP': 'raise both arms',
    'BI_OUT': 'move both hands outward',
    'BI_RAISE': 'raise both arms',
    'BI_RAISE_SPREAD': 'raise and spread both arms',
    'BI_SPREAD': 'move both hands outward',
    'BI_HANDS_CLOSE': 'bring both hands closer together',
    'BI_HANDS_CLOSE_RAISE': 'bring both hands closer while raising them',
    'BI_EXTENDED_LOCO_COUPLED': 'keep both hands extended while moving',
    'BI_LOCOMOTION_COUPLED': 'move both arms while walking',
    'BI_VERTICAL_COUPLED': 'move both arms with vertical body motion',
    'BI_UNRESOLVED': 'bimanual motion unresolved',
}


def _fmt_float(value: Any, digits: int = 1) -> str | None:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    return f'{x:.2f}' if digits > 1 else f'{x:.1f}'


def _measure_phrase(evt: dict[str, Any]) -> str | None:
    unit = evt.get('unit')
    magnitude = evt.get('magnitude')
    signed_delta = evt.get('signed_delta')
    if unit == 'deg':
        angle = _fmt_float(magnitude if magnitude is not None else abs(float(signed_delta or 0.0)))
        signed = _fmt_float(signed_delta)
        if angle is None:
            return None
        if signed is not None:
            return f'angle={angle}deg, signed_delta={signed}deg'
        return f'angle={angle}deg'
    if unit == 'm':
        dist = _fmt_float(magnitude if magnitude is not None else abs(float(signed_delta or 0.0)), digits=2)
        signed = _fmt_float(signed_delta, digits=2)
        if dist is None:
            return None
        if signed is not None:
            return f'amplitude={dist}m, signed_delta={signed}m'
        return f'amplitude={dist}m'
    return None


def aml_event_to_template(evt: dict[str, Any], *, detail: bool = False) -> str:
    family = str(evt.get('super_family', 'UNKNOWN'))
    cluster = str(evt.get('cluster_id', 'UNKNOWN'))
    label = FAMILY_LABELS.get(family, family.lower())
    verb = CLUSTER_VERBS.get(cluster, str(evt.get('direction', 'move')))
    start = int(evt.get('start_frame', -1))
    end = int(evt.get('end_frame', -1))
    sig = evt.get('motion_signature') or {}
    context = sig.get('context_mode', 'unknown')
    count = evt.get('count')
    role = evt.get('role', 'unknown')
    measure = _measure_phrase(evt)

    pieces = [
        f'frames {start}-{end}',
        label,
        verb,
    ]
    if count is not None:
        pieces.append(f'count={int(count)}')
    if measure:
        pieces.append(measure)
    pieces.append(f'context={context}')
    if detail:
        pieces.extend([
            f'family={family}',
            f'cluster={cluster}',
            f'role={role}',
            f'source={evt.get("source", "unknown")}',
        ])
        supporting = evt.get('supporting_units') or []
        if supporting:
            pieces.append('support=' + ','.join(str(x) for x in supporting[:4]))
    return ' | '.join(pieces)


def aml_program_to_templates(program: dict[str, Any], *, detail: bool = False) -> list[str]:
    events = list((program or {}).get('events') or [])
    events.sort(key=lambda e: (int(e.get('start_frame', -1)), int(e.get('end_frame', -1)), str(e.get('super_family', '')), str(e.get('cluster_id', ''))))
    return [aml_event_to_template(evt, detail=detail) for evt in events]


def attach_aml_language(program: dict[str, Any]) -> dict[str, Any]:
    out = dict(program or {})
    out['aml_language_compact'] = aml_program_to_templates(out, detail=False)
    out['aml_language_detailed'] = aml_program_to_templates(out, detail=True)
    return out
