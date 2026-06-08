from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import (
    PhasePattern,
    build_layer3_atomic_program,
    dedupe_phase_patterns,
    detect_repeated_phases,
    extract_layer0_frame_observables,
    extract_layer1_micro_events,
    merge_micro_events,
    project_units_by_category,
)

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')
UPPER_FAMILIES = {'BIMANUAL_PERIODIC', 'LEFT_ARM_PERIODIC', 'RIGHT_ARM_PERIODIC'}
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being', 'person', 'man', 'woman', 'figure',
    'someone', 'their', 'his', 'her', 'he', 'she', 'they', 'then', 'and', 'to', 'of', 'in', 'on',
    'with', 'while', 'for', 'as', 'at', 'from', 'it', 'this', 'that', 'them', 'him', 'herself',
    'himself', 'slowly', 'quickly', 'forward', 'backward', 'left', 'right', 'around', 'before',
    'after', 'starts', 'start', 'begins', 'begin', 'continues', 'continues', 'again', 'normal',
}
UPPER_KEYWORDS = {
    'arm', 'arms', 'hand', 'hands', 'wrist', 'wrists', 'elbow', 'elbows', 'shoulder', 'shoulders',
    'clap', 'claps', 'clapping', 'wave', 'waves', 'waving', 'raise', 'raises', 'raised', 'lift',
    'lifts', 'lifted', 'holding', 'hold', 'holds', 'grab', 'grabs', 'grabbing', 'pick', 'picks',
    'throw', 'throws', 'throwing', 'catch', 'catches', 'push', 'pushes', 'pull', 'pulls', 'press',
    'presses', 'support', 'rail', 'railing', 'wall', 'balance', 'lean', 'leans', 'touch', 'touches',
    'punch', 'punches', 'boxing', 'salute', 'gesture', 'gestures', 'spread', 'extend', 'extends',
    'stretch', 'stretches', 'swing', 'swings', 'swinging', 'out', 'up', 'down', 'wide', 'chest',
    'face', 'head', 'knee', 'waist', 'mouth', 'door', 'object', 'something', 'item', 'ball', 'cup',
}

WORD_FAMILY_PATTERNS: dict[str, list[str]] = {
    'support_contact': [
        r'\bhold(?:s|ing)? on\b', r'\brail(?:ing)?\b', r'\bwall\b', r'\bsupport\b', r'\bbalance\b',
        r'\bpress(?:es|ing)? against\b', r'\blean(?:s|ing)? on\b', r'\busing .* arm.* support\b',
    ],
    'object_hold_or_manipulate': [
        r'\bhold(?:s|ing)?\b', r'\bcarry(?:ing|ies)?\b', r'\bgrab(?:s|bing)?\b', r'\bpick(?:s|ing)? up\b',
        r'\bput(?:s|ting)?\b', r'\bthrow(?:s|ing)?\b', r'\bcatch(?:es|ing)?\b', r'\bobject\b', r'\bsomething\b',
        r'\bitem\b', r'\bbox\b', r'\bball\b', r'\bcup\b', r'\bdoor\b', r'\bknock(?:s|ing)?\b',
    ],
    'arm_raise_lift': [
        r'\braise(?:s|d|ing)? .*arm', r'\blift(?:s|ed|ing)? .*arm', r'\barm[s]? up\b', r'\bhands? up\b',
        r'\boverhead\b', r'\babove (?:the )?head\b', r'\braise(?:s|d|ing)? .*hand', r'\blift(?:s|ed|ing)? .*hand',
    ],
    'arm_extend_spread': [
        r'\bextend(?:s|ed|ing)?\b', r'\bstretch(?:es|ed|ing)?\b', r'\bspread(?:s|ing)?\b',
        r'\bout wide\b', r'\bto (?:the )?side[s]?\b', r'\barms? out\b', r'\bhands? out\b',
    ],
    'arm_swing_walk': [
        r'\bswing(?:s|ing)? (?:his |her |their )?arm', r'\barm[s]? swing(?:s|ing)?\b', r'\bswing(?:s|ing)? .*shoulder',
    ],
    'wave_or_gesture': [
        r'\bwave(?:s|d|ing)?\b', r'\bgesture(?:s|d|ing)?\b', r'\bsalute(?:s|d|ing)?\b', r'\bwaive(?:s|d|ing)?\b',
    ],
    'clap_or_hands_together': [
        r'\bclap(?:s|ped|ping)?\b', r'\bhands? together\b', r'\bpalms?\b', r'\bprayer\b',
    ],
    'touch_body': [
        r'\btouch(?:es|ing)?\b', r'\bhand[s]? .*\b(?:face|head|chest|knee|waist|mouth|hip|shoulder)\b',
        r'\b(?:face|head|chest|knee|waist|mouth|hip|shoulder) .*hand',
    ],
    'punch_boxing': [r'\bpunch(?:es|ing)?\b', r'\bbox(?:es|ing)?\b', r'\bfight(?:s|ing)?\b'],
    'dance_or_circular_gesture': [r'\bdance(?:s|d|ing)?\b', r'\bcircular\b', r'\bcircle(?:s|ing)? .*hand', r'\bflap(?:s|ping)?\b'],
    'generic_upper_body': [r'\barm[s]?\b', r'\bhand[s]?\b', r'\bshoulder[s]?\b', r'\belbow[s]?\b', r'\bwrist[s]?\b'],
}


def load_case_ids(path: Path, max_cases: int | None) -> list[str]:
    out: list[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(str(json.loads(line)['case_id']))
            if max_cases is not None and len(out) >= max_cases:
                break
    return out


def read_captions(case_id: str) -> list[str]:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not path.exists():
        return []
    captions: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        caption = line.split('#')[0].strip()
        if caption:
            captions.append(caption)
    return captions


def normalize(text: str) -> str:
    text = text.lower().replace('-', ' ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def phrase_candidates(captions: Iterable[str], max_ngram: int = 5) -> set[str]:
    phrases: set[str] = set()
    for caption in captions:
        toks = tokens(caption)
        if not toks:
            continue
        upper_positions = [i for i, tok in enumerate(toks) if tok in UPPER_KEYWORDS]
        windows: list[tuple[int, int]] = []
        for pos in upper_positions:
            windows.append((max(0, pos - 3), min(len(toks), pos + 5)))
        if not windows and any(tok in UPPER_KEYWORDS for tok in toks):
            windows.append((0, len(toks)))
        for start, end in windows:
            window = toks[start:end]
            for n in range(1, max_ngram + 1):
                for i in range(0, max(0, len(window) - n + 1)):
                    ng = window[i:i + n]
                    if not any(t in UPPER_KEYWORDS for t in ng):
                        continue
                    if all(t in STOPWORDS for t in ng):
                        continue
                    if len(ng) == 1 and ng[0] in {'arm', 'arms', 'hand', 'hands'}:
                        continue
                    phrase = ' '.join(ng)
                    if len(phrase) >= 3:
                        phrases.add(phrase)
    return phrases


def word_families(captions: Iterable[str]) -> set[str]:
    joined = ' '.join(normalize(c) for c in captions)
    out: set[str] = set()
    for family, patterns in WORD_FAMILY_PATTERNS.items():
        if any(re.search(pattern, joined) for pattern in patterns):
            out.add(family)
    return out


def phase_to_dict(phase: PhasePattern) -> dict[str, Any]:
    return {
        'name': phase.name,
        'kind': phase.kind,
        'count': int(phase.count),
        'start_frame': int(phase.start_frame),
        'end_frame': int(phase.end_frame),
        'unit_names': list(phase.unit_names),
        'metadata': dict(phase.metadata),
    }


def dedupe_phase_objects(phases: list[PhasePattern]) -> list[PhasePattern]:
    deduped = dedupe_phase_patterns([phase_to_dict(p) for p in phases])
    out: list[PhasePattern] = []
    for p in deduped:
        out.append(PhasePattern(
            name=str(p['name']),
            kind=str(p['kind']),
            count=int(p['count']),
            start_frame=int(p['start_frame']),
            end_frame=int(p['end_frame']),
            unit_names=list(p['unit_names']),
            metadata=dict(p.get('metadata', {})),
        ))
    out.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return out


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = int(a.get('start_frame', -1)), int(a.get('end_frame', -1))
    b0, b1 = int(b.get('start_frame', -1)), int(b.get('end_frame', -1))
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    dur = max(1, a1 - a0 + 1)
    return inter / dur


def motion_keys(program: dict[str, Any]) -> set[str]:
    events = list(program.get('events') or [])
    locos = [e for e in events if e.get('super_family') == 'WHOLE_BODY_LOCOMOTION']
    posture = any(e.get('super_family') == 'WHOLE_BODY_POSTURE' for e in events)
    vertical = any(e.get('super_family') == 'WHOLE_BODY_VERTICAL' for e in events)
    rotation = any(e.get('super_family') == 'WHOLE_BODY_ROTATION' for e in events)
    out: set[str] = set()
    for evt in events:
        family = str(evt.get('super_family'))
        if family not in UPPER_FAMILIES:
            continue
        cluster = str(evt.get('cluster_id'))
        loco_overlap = max((overlap_ratio(evt, l) for l in locos), default=0.0)
        context = 'loco' if loco_overlap >= 0.40 else 'nonloco'
        if posture:
            context += '+posture'
        if vertical:
            context += '+vertical'
        if rotation:
            context += '+turn'
        out.add(f'{family}/{cluster}|{context}')
        out.add(f'{family}/{cluster}')
    return out


def extract_program(case_id: str, packed: dict[str, Any]) -> dict[str, Any] | None:
    key = f'{case_id}.npy'
    if key not in packed:
        return None
    joints = packed[key]['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    if len(joints) <= 20:
        return None
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=joints[:, 0, :])
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases: list[PhasePattern] = []
    phases.extend(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
    return build_layer3_atomic_program(layer2, dedupe_phase_objects(phases))


def score_entry(support: int, motion_support: int, phrase_support: int, total_cases: int) -> dict[str, float | int]:
    coverage = support / max(motion_support, 1)
    precision = support / max(phrase_support, 1)
    background = phrase_support / max(total_cases, 1)
    lift = coverage / max(background, 1e-9)
    harmonic = 2 * coverage * precision / max(coverage + precision, 1e-9)
    support_factor = min(1.0, math.log1p(support) / math.log(51))
    score = harmonic * support_factor * math.log1p(lift)
    return {
        'support': support,
        'motion_support': motion_support,
        'global_support': phrase_support,
        'coverage': coverage,
        'precision': precision,
        'lift': lift,
        'score': score,
    }


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for value in counter.values():
        p = value / total
        out -= p * math.log(p + 1e-12, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--min-support', type=int, default=10)
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--progress-every', type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    case_ids = load_case_ids(Path(args.manifest), args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    motion_cases: dict[str, set[str]] = defaultdict(set)
    phrase_cases: dict[str, set[str]] = defaultdict(set)
    family_cases: dict[str, set[str]] = defaultdict(set)
    motion_phrase_cases: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    motion_family_cases: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    motion_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = 0

    for idx, case_id in enumerate(case_ids, start=1):
        captions = read_captions(case_id)
        if not captions:
            continue
        program = extract_program(case_id, packed)
        if program is None:
            continue
        keys = motion_keys(program)
        if not keys:
            continue
        phrases = phrase_candidates(captions)
        families = word_families(captions)
        processed += 1
        for phrase in phrases:
            phrase_cases[phrase].add(case_id)
        for family in families:
            family_cases[family].add(case_id)
        for key in keys:
            motion_cases[key].add(case_id)
            if len(motion_examples[key]) < 8:
                motion_examples[key].append({
                    'case_id': case_id,
                    'captions': captions[:5],
                    'word_families': sorted(families),
                })
            for phrase in phrases:
                motion_phrase_cases[key][phrase].add(case_id)
            for family in families:
                motion_family_cases[key][family].add(case_id)
        if idx % args.progress_every == 0:
            print(f'processed {idx}/{len(case_ids)}, upperbody_valid={processed}, elapsed={time.time()-t0:.1f}s', flush=True)

    total_cases = max(processed, 1)
    motion_reports: dict[str, Any] = {}
    for key, cases in sorted(motion_cases.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        m_support = len(cases)
        phrase_rows = []
        for phrase, pcases in motion_phrase_cases[key].items():
            support = len(pcases)
            if support < args.min_support:
                continue
            metrics = score_entry(support, m_support, len(phrase_cases[phrase]), total_cases)
            if metrics['precision'] < 0.05 and metrics['lift'] < 1.5:
                continue
            phrase_rows.append({'phrase': phrase, **metrics})
        phrase_rows.sort(key=lambda x: (-float(x['score']), -int(x['support']), x['phrase']))

        family_rows = []
        family_counter: Counter[str] = Counter()
        for family, fcases in motion_family_cases[key].items():
            support = len(fcases)
            family_counter[family] = support
            metrics = score_entry(support, m_support, len(family_cases[family]), total_cases)
            family_rows.append({'word_family': family, **metrics})
        family_rows.sort(key=lambda x: (-float(x['score']), -int(x['support']), x['word_family']))

        motion_reports[key] = {
            'motion_support': m_support,
            'word_family_entropy': entropy(family_counter),
            'top_word_families': family_rows[:args.top_k],
            'top_phrases': phrase_rows[:args.top_k],
            'examples': motion_examples.get(key, []),
        }

    out = {
        'run': {
            'manifest': args.manifest,
            'max_cases': args.max_cases,
            'requested_cases': len(case_ids),
            'processed_upperbody_cases': processed,
            'elapsed_sec': time.time() - t0,
            'min_support': args.min_support,
            'top_k': args.top_k,
            'note': 'HML3D captions are used only as a global wording corpus for cluster naming/reference, not as same-case auto-prompt input.',
        },
        'word_family_patterns': WORD_FAMILY_PATTERNS,
        'motion_reports': motion_reports,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')

    report_lines = ['# HML3D Upper-Body Wording Mining v1', '']
    report_lines.append('## Scope')
    report_lines.append('')
    report_lines.append('- captions are used as a global wording corpus only; no same-case caption is used to produce auto-prompt text.')
    report_lines.append('- scoring uses motion-cluster support, phrase coverage, phrase precision, and lift.')
    report_lines.append('')
    report_lines.append('## Run')
    report_lines.append('')
    for k, v in out['run'].items():
        report_lines.append(f'- {k}: `{v}`')
    report_lines.append('')
    report_lines.append('## Top Motion Clusters')
    report_lines.append('')
    for key, rep in list(motion_reports.items())[:40]:
        report_lines.append(f'### {key}')
        report_lines.append('')
        report_lines.append(f"- support: {rep['motion_support']}")
        report_lines.append(f"- word-family entropy: {rep['word_family_entropy']:.3f}")
        report_lines.append('- top word families:')
        for row in rep['top_word_families'][:8]:
            report_lines.append(
                f"  - {row['word_family']}: support={row['support']}, coverage={row['coverage']:.3f}, precision={row['precision']:.3f}, lift={row['lift']:.2f}, score={row['score']:.3f}"
            )
        report_lines.append('- top phrases:')
        for row in rep['top_phrases'][:12]:
            report_lines.append(
                f"  - {row['phrase']}: support={row['support']}, coverage={row['coverage']:.3f}, precision={row['precision']:.3f}, lift={row['lift']:.2f}, score={row['score']:.3f}"
            )
        report_lines.append('')
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'saved={out_path}')
    print(f'report={report_path}')


if __name__ == '__main__':
    main()
