from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
    render_aml_prompt,
)

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')

PHRASE_GROUPS = {
    'problem_vertical': ['jumps upward', 'lowers the body', 'hop-like', 'up-and-down body motion'],
    'neutral_height': ['changes body height while moving', 'rises back up'],
    'arm_cycle_raw': ['repeats a left arm cycle', 'repeats a right arm cycle'],
    'arm_swing_family': ['swings the left arm while walking', 'swings the right arm while walking', 'swings both arms while walking'],
    'bimanual_coarse': ['moves both hands outward', 'raises both arms'],
    'locomotion': ['walks forward', 'walks backward', 'moves to the left', 'moves to the right'],
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


def read_first_prompt(case_id: str) -> str:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            return line.split('#')[0].strip()
    return ''


def extract_prompt(case_id: str, packed: dict[str, Any]) -> str | None:
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
    layer3 = build_layer3_atomic_program(layer2, dedupe_phase_objects(phases))
    return render_aml_prompt(layer3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    case_ids = load_case_ids(Path(args.manifest), args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    processed = 0

    for idx, case_id in enumerate(case_ids, start=1):
        prompt = extract_prompt(case_id, packed)
        if prompt is None:
            continue
        processed += 1
        for group, phrases in PHRASE_GROUPS.items():
            if any(phrase in prompt for phrase in phrases):
                counts[group] += 1
                if len(examples[group]) < 30:
                    examples[group].append({
                        'case_id': case_id,
                        'auto_prompt': prompt,
                        'selected_hml3d_prompt': read_first_prompt(case_id),
                    })
        if idx % args.progress_every == 0:
            print(f'processed {idx}/{len(case_ids)}, valid={processed}, elapsed={time.time()-t0:.1f}s', flush=True)

    out = {
        'run': {
            'manifest': args.manifest,
            'max_cases': args.max_cases,
            'requested_cases': len(case_ids),
            'processed_cases': processed,
            'elapsed_sec': time.time() - t0,
        },
        'phrase_case_counts': counts,
        'phrase_case_share': {k: counts[k] / max(processed, 1) for k in PHRASE_GROUPS},
        'examples': examples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(f'saved={out_path}')


if __name__ == '__main__':
    main()
