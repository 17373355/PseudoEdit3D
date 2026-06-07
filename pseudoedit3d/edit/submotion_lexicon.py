from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pseudoedit3d.edit.micro_events import MicroEvent


def canonical_symbol(symbol: str) -> str:
    for suffix in ['_S', '_M', '_L', '_SHORT', '_MEDIUM', '_LONG']:
        if symbol.endswith(suffix):
            return symbol[:-len(suffix)]
    return symbol


@dataclass
class SubMotionPattern:
    name: str
    tokens: tuple[str, ...]
    category: str
    description: str
    metadata: dict[str, Any] | None = None


@dataclass
class SubMotionUnit:
    name: str
    category: str
    start_frame: int
    end_frame: int
    support_tokens: list[str]
    metadata: dict[str, Any]


SUBMOTION_LEXICON_V1: list[SubMotionPattern] = [
    SubMotionPattern(
        name='crouch_descent',
        tokens=('WHOLE_BODY_ROOT_DOWN', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='root descends while legs compress',
    ),
    SubMotionPattern(
        name='crouch_descent_strong',
        tokens=('WHOLE_BODY_ROOT_DOWN', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='strong root descent with leg compression',
    ),
    SubMotionPattern(
        name='hop_ascent',
        tokens=('WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_ROOT_UP'),
        category='whole_body',
        description='leg release followed by root ascent',
    ),
    SubMotionPattern(
        name='hop_ascent_variant',
        tokens=('WHOLE_BODY_ROOT_UP', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='root ascent followed by strong leg release',
    ),
    SubMotionPattern(
        name='leg_bounce_cycle',
        tokens=('WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='compress-release leg cycle',
    ),
    SubMotionPattern(
        name='left_arm_lowering',
        tokens=('LEFT_ARM_LEFT_ARM_DOWN', 'LEFT_ARM_LEFT_ELBOW_DOWN'),
        category='left_arm',
        description='left arm and elbow lower together',
    ),
    SubMotionPattern(
        name='right_arm_lowering',
        tokens=('RIGHT_ARM_RIGHT_ARM_DOWN', 'RIGHT_ARM_RIGHT_ELBOW_DOWN'),
        category='right_arm',
        description='right arm and elbow lower together',
    ),
    SubMotionPattern(
        name='both_arms_lift',
        tokens=('LEFT_ARM_LEFT_ARM_UP', 'RIGHT_ARM_RIGHT_ARM_UP'),
        category='both_arms',
        description='both arms lift strongly together',
    ),
    SubMotionPattern(
        name='hands_move_away_from_chest',
        tokens=('LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST', 'RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST'),
        category='both_arms',
        description='both hands move away from the chest',
    ),
    SubMotionPattern(
        name='torso_rise_back',
        tokens=('TORSO_TORSO_UNBEND', 'TORSO_TORSO_BACKWARD_RETRACT'),
        category='torso',
        description='torso rises and retracts backward',
    ),
    SubMotionPattern(
        name='hop_unit',
        tokens=('WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_ROOT_UP', 'TORSO_TORSO_UNBEND'),
        category='whole_body',
        description='release + root ascent + torso rise',
    ),
    SubMotionPattern(
        name='arm_lift_front',
        tokens=('RIGHT_ARM_RIGHT_ARM_UP', 'RIGHT_ARM_RIGHT_ELBOW_UP', 'RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST'),
        category='right_arm',
        description='right arm lifts with elbow and hand extending outward',
    ),

    SubMotionPattern(
        name='leg_release_then_compress',
        tokens=('WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='leg release followed by leg compression',
    ),
    SubMotionPattern(
        name='leg_compress_then_release',
        tokens=('WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='leg compression followed by leg release',
    ),
    SubMotionPattern(
        name='root_down_then_leg_compress',
        tokens=('WHOLE_BODY_ROOT_DOWN', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='root moves down then legs compress',
    ),
    SubMotionPattern(
        name='root_up_then_leg_release',
        tokens=('WHOLE_BODY_ROOT_UP', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='root moves up then legs release',
    ),
    SubMotionPattern(
        name='leg_release_root_down_leg_compress',
        tokens=('WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_ROOT_DOWN', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='release followed by downward compression cycle',
    ),
    SubMotionPattern(
        name='leg_compress_root_up_leg_release',
        tokens=('WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_ROOT_UP', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='compression followed by upward release cycle',
    ),
    SubMotionPattern(
        name='torso_forward_back_cycle',
        tokens=('TORSO_TORSO_FORWARD_EXTEND', 'TORSO_TORSO_BACKWARD_RETRACT'),
        category='torso',
        description='torso forward then back cycle',
    ),
    SubMotionPattern(
        name='torso_back_forward_cycle',
        tokens=('TORSO_TORSO_BACKWARD_RETRACT', 'TORSO_TORSO_FORWARD_EXTEND'),
        category='torso',
        description='torso back then forward cycle',
    ),
    SubMotionPattern(
        name='torso_bend_recover',
        tokens=('TORSO_TORSO_FORWARD_EXTEND', 'TORSO_TORSO_BEND_FORWARD', 'TORSO_TORSO_BACKWARD_RETRACT'),
        category='torso',
        description='torso extends, bends forward, then recovers back',
    ),
    SubMotionPattern(
        name='left_hand_near_far_cycle',
        tokens=('LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST'),
        category='left_arm',
        description='left hand moves from chest outward',
    ),
    SubMotionPattern(
        name='left_hand_far_near_cycle',
        tokens=('LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST', 'LEFT_ARM_LEFT_HAND_NEAR_CHEST'),
        category='left_arm',
        description='left hand returns toward chest',
    ),
    SubMotionPattern(
        name='left_arm_raise_cycle',
        tokens=('LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_ARM_UP', 'LEFT_ARM_LEFT_ARM_DOWN'),
        category='left_arm',
        description='left arm rise-and-fall cycle near chest',
    ),
    SubMotionPattern(
        name='right_hand_near_far_cycle',
        tokens=('RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST'),
        category='right_arm',
        description='right hand moves from chest outward',
    ),
    SubMotionPattern(
        name='right_hand_far_near_cycle',
        tokens=('RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST', 'RIGHT_ARM_RIGHT_HAND_NEAR_CHEST'),
        category='right_arm',
        description='right hand returns toward chest',
    ),
    SubMotionPattern(
        name='right_arm_raise_cycle',
        tokens=('RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_ARM_UP', 'RIGHT_ARM_RIGHT_ARM_DOWN'),
        category='right_arm',
        description='right arm rise-and-fall cycle near chest',
    ),
    SubMotionPattern(
        name='leg_release_compress_cycle',
        tokens=('WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='whole-body release-compress-release cycle',
    ),
    SubMotionPattern(
        name='leg_compress_release_cycle',
        tokens=('WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='whole-body compress-release-compress cycle',
    ),
    SubMotionPattern(
        name='root_down_compress_release_cycle',
        tokens=('WHOLE_BODY_ROOT_DOWN', 'WHOLE_BODY_LEG_COMPRESS', 'WHOLE_BODY_LEG_RELEASE'),
        category='whole_body',
        description='root down then compress-release cycle',
    ),
    SubMotionPattern(
        name='root_up_release_compress_cycle',
        tokens=('WHOLE_BODY_ROOT_UP', 'WHOLE_BODY_LEG_RELEASE', 'WHOLE_BODY_LEG_COMPRESS'),
        category='whole_body',
        description='root up then release-compress cycle',
    ),
    SubMotionPattern(
        name='torso_forward_back_forward_cycle',
        tokens=('TORSO_TORSO_FORWARD_EXTEND', 'TORSO_TORSO_BACKWARD_RETRACT', 'TORSO_TORSO_FORWARD_EXTEND'),
        category='torso',
        description='torso forward-back-forward oscillation',
    ),
    SubMotionPattern(
        name='torso_back_forward_back_cycle',
        tokens=('TORSO_TORSO_BACKWARD_RETRACT', 'TORSO_TORSO_FORWARD_EXTEND', 'TORSO_TORSO_BACKWARD_RETRACT'),
        category='torso',
        description='torso back-forward-back oscillation',
    ),
    SubMotionPattern(
        name='torso_forward_bend_recover',
        tokens=('TORSO_TORSO_FORWARD_EXTEND', 'TORSO_TORSO_BEND_FORWARD', 'TORSO_TORSO_BACKWARD_RETRACT'),
        category='torso',
        description='torso extends, bends, and recovers',
    ),
    SubMotionPattern(
        name='left_arm_near_down_far',
        tokens=('LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_ARM_DOWN', 'LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST'),
        category='left_arm',
        description='left hand near chest then arm lowers outward',
    ),
    SubMotionPattern(
        name='left_arm_far_near_far',
        tokens=('LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST', 'LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST'),
        category='left_arm',
        description='left hand out-in-out cycle',
    ),
    SubMotionPattern(
        name='left_arm_near_up_down',
        tokens=('LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_ARM_UP', 'LEFT_ARM_LEFT_ARM_DOWN'),
        category='left_arm',
        description='left arm near-chest up-down cycle',
    ),
    SubMotionPattern(
        name='left_arm_far_near_up',
        tokens=('LEFT_ARM_LEFT_HAND_FAR_FROM_CHEST', 'LEFT_ARM_LEFT_HAND_NEAR_CHEST', 'LEFT_ARM_LEFT_ARM_UP'),
        category='left_arm',
        description='left hand returns to chest then lifts',
    ),
    SubMotionPattern(
        name='right_arm_near_down_far',
        tokens=('RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_ARM_DOWN', 'RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST'),
        category='right_arm',
        description='right hand near chest then arm lowers outward',
    ),
    SubMotionPattern(
        name='right_arm_far_near_far',
        tokens=('RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST', 'RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST'),
        category='right_arm',
        description='right hand out-in-out cycle',
    ),
    SubMotionPattern(
        name='right_arm_near_up_down',
        tokens=('RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_ARM_UP', 'RIGHT_ARM_RIGHT_ARM_DOWN'),
        category='right_arm',
        description='right arm near-chest up-down cycle',
    ),
    SubMotionPattern(
        name='right_arm_far_near_up',
        tokens=('RIGHT_ARM_RIGHT_HAND_FAR_FROM_CHEST', 'RIGHT_ARM_RIGHT_HAND_NEAR_CHEST', 'RIGHT_ARM_RIGHT_ARM_UP'),
        category='right_arm',
        description='right hand returns to chest then lifts',
    ),
]


def merge_micro_events(events: list[MicroEvent], lexicon: Iterable[SubMotionPattern] | None = None) -> list[SubMotionUnit]:
    lexicon = list(lexicon) if lexicon is not None else SUBMOTION_LEXICON_V1
    symbols = [e.to_symbol() for e in events]
    canon_symbols = [canonical_symbol(s) for s in symbols]
    units: list[SubMotionUnit] = []
    i = 0
    while i < len(events):
        matched = None
        matched_len = 0
        for pattern in lexicon:
            tokens = list(pattern.tokens)
            n = len(tokens)
            if i + n <= len(symbols) and canon_symbols[i:i+n] == tokens:
                if n > matched_len:
                    matched = pattern
                    matched_len = n
        if matched is not None:
            span = events[i:i+matched_len]
            deltas = [float(e.delta_value) for e in span]
            units.append(SubMotionUnit(
                name=matched.name,
                category=matched.category,
                start_frame=span[0].start_frame,
                end_frame=span[-1].end_frame,
                support_tokens=[e.to_symbol() for e in span],
                metadata={
                    'description': matched.description,
                    'canonical_tokens': list(matched.tokens),
                    'component_events': [
                        {
                            'observable': e.observable,
                            'direction': e.direction,
                            'delta_value': float(e.delta_value),
                            'unit': e.unit,
                            'magnitude_bin': e.magnitude_bin,
                            'duration_bin': e.duration_bin,
                            'start_frame': e.start_frame,
                            'end_frame': e.end_frame,
                        } for e in span
                    ],
                    'net_delta_sum': float(sum(deltas)),
                    'max_abs_delta': float(max(abs(d) for d in deltas)) if deltas else 0.0,
                    'component_count': len(span),
                },
            ))
            i += matched_len
        else:
            e = events[i]
            units.append(SubMotionUnit(
                name=canonical_symbol(e.to_symbol()).lower(),
                category='micro_event',
                start_frame=e.start_frame,
                end_frame=e.end_frame,
                support_tokens=[e.to_symbol()],
                metadata={
                    'observable': e.observable,
                    'direction': e.direction,
                    'delta_value': float(e.delta_value),
                    'unit': e.unit,
                    'magnitude_bin': e.magnitude_bin,
                    'duration_bin': e.duration_bin,
                },
            ))
            i += 1
    return units
