from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pseudoedit3d.edit.submotion_lexicon import SubMotionUnit


@dataclass
class PhasePattern:
    name: str
    kind: str
    count: int
    start_frame: int
    end_frame: int
    unit_names: list[str]
    metadata: dict[str, Any]


def canonical_name(name: str) -> str:
    # normalize micro-event suffixes like _s/_m/_l/_short/_medium/_long
    for suffix in ['_s', '_m', '_l', '_short', '_medium', '_long']:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def detect_repeated_phases(units: list[SubMotionUnit], min_repeat: int = 2, max_gap: int = 12) -> list[PhasePattern]:
    patterns: list[PhasePattern] = []
    names = [u.name for u in units]
    canon = [canonical_name(n) for n in names]

    # 1. same-unit repetition: A A A
    i = 0
    n = len(units)
    while i < n:
        j = i + 1
        repeat_units = [units[i]]
        while j < n and canonical_name(units[j].name) == canonical_name(units[i].name) and units[j].start_frame - repeat_units[-1].end_frame <= max_gap:
            repeat_units.append(units[j])
            j += 1
        if len(repeat_units) >= min_repeat:
            base = canonical_name(units[i].name)
            patterns.append(PhasePattern(
                name=f'{base}_repeat_x{len(repeat_units)}',
                kind='repeat',
                count=len(repeat_units),
                start_frame=repeat_units[0].start_frame,
                end_frame=repeat_units[-1].end_frame,
                unit_names=[u.name for u in repeat_units],
                metadata={'base_unit': base},
            ))
            i = j
            continue
        i += 1

    # 2. alternating ABAB
    i = 0
    while i + 3 < n:
        a = canon[i]
        b = canon[i + 1]
        if a != b and canon[i + 2] == a and canon[i + 3] == b:
            seq = units[i:i + 4]
            j = i + 4
            pair_count = 2
            expected = a
            while j < n:
                if canon[j] != expected or units[j].start_frame - seq[-1].end_frame > max_gap:
                    break
                seq.append(units[j])
                expected = b if expected == a else a
                if len(seq) % 2 == 0:
                    pair_count += 1
                j += 1
            if pair_count >= min_repeat:
                patterns.append(PhasePattern(
                    name=f'{a}__{b}_alternate_x{pair_count}',
                    kind='alternate',
                    count=pair_count,
                    start_frame=seq[0].start_frame,
                    end_frame=seq[-1].end_frame,
                    unit_names=[u.name for u in seq],
                    metadata={'unit_a': a, 'unit_b': b},
                ))
                i = j
                continue
        i += 1

    # 3. repeated short subsequences (BPE-like local motifs), lengths 2..4
    seen = set((p.name, p.start_frame, p.end_frame) for p in patterns)
    for width in [4, 3, 2]:
        i = 0
        while i + 2 * width <= n:
            first = tuple(canon[i:i + width])
            second = tuple(canon[i + width:i + 2 * width])
            if first == second:
                reps = 2
                j = i + 2 * width
                while j + width <= n and tuple(canon[j:j + width]) == first:
                    reps += 1
                    j += width
                seq = units[i:j]
                name = '__'.join(first) + f'_loop_x{reps}'
                key = (name, seq[0].start_frame, seq[-1].end_frame)
                if key not in seen:
                    patterns.append(PhasePattern(
                        name=name,
                        kind='subsequence_repeat',
                        count=reps,
                        start_frame=seq[0].start_frame,
                        end_frame=seq[-1].end_frame,
                        unit_names=[u.name for u in seq],
                        metadata={'width': width},
                    ))
                    seen.add(key)
                i = j
                continue
            i += 1

    patterns.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return patterns



def project_units_by_category(units: list[SubMotionUnit], category_prefix: str) -> list[SubMotionUnit]:
    out = []
    for u in units:
        if u.category == category_prefix:
            out.append(u)
        elif u.category == 'micro_event' and u.name.startswith(category_prefix):
            out.append(u)
    return out
