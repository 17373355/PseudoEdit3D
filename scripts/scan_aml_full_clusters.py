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


def load_case_ids(manifest: Path, max_cases: int | None = None) -> list[str]:
    out: list[str] = []
    with manifest.open('r', encoding='utf-8') as f:
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


def extract_case(case_id: str, packed: dict[str, Any], *, detect_phase: bool) -> dict[str, Any] | None:
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
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases: list[PhasePattern] = []
    if detect_phase:
        phases.extend(detect_repeated_phases(layer2))
        for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
            phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
        phases = dedupe_phase_objects(phases)
    layer3 = build_layer3_atomic_program(layer2, phases)
    return {
        'case_id': case_id,
        'num_frames': int(len(joints)),
        'layer1_count': int(len(layer1)),
        'layer2_count': int(len(layer2)),
        'layer25_count': int(len(phases)),
        'layer3': layer3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--detect-phase', action='store_true')
    parser.add_argument('--progress-every', type=int, default=500)
    args = parser.parse_args()

    t0 = time.time()
    case_ids = load_case_ids(Path(args.manifest), max_cases=args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    family_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    family_cluster_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    family_support: dict[str, set[str]] = defaultdict(set)
    cluster_support: dict[str, set[str]] = defaultdict(set)
    low_event_cases: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_examples: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    total_layer3 = 0
    layer_count_sums = Counter()

    for idx, case_id in enumerate(case_ids, start=1):
        item = extract_case(case_id, packed, detect_phase=args.detect_phase)
        if item is None:
            skipped += 1
            continue
        processed += 1
        events = item['layer3']['events']
        total_layer3 += len(events)
        layer_count_sums['layer1'] += item['layer1_count']
        layer_count_sums['layer2'] += item['layer2_count']
        layer_count_sums['layer25'] += item['layer25_count']
        if len(events) <= 2:
            low_event_cases.append({
                'case_id': case_id,
                'layer3_count': len(events),
                'selected_hml3d_prompt': read_first_prompt(case_id),
                'auto_prompt': render_aml_prompt(item['layer3']),
            })
        if len(prompt_examples) < 100:
            prompt_examples.append({
                'case_id': case_id,
                'selected_hml3d_prompt': read_first_prompt(case_id),
                'auto_prompt': render_aml_prompt(item['layer3']),
                'num_events': len(events),
            })
        for evt in events:
            family = str(evt.get('super_family', 'UNKNOWN'))
            cluster = str(evt.get('cluster_id', 'UNKNOWN'))
            role = str(evt.get('role', 'unknown'))
            combined = f'{family}/{cluster}'
            family_counts[family] += 1
            cluster_counts[cluster] += 1
            family_cluster_counts[combined] += 1
            role_counts[role] += 1
            family_support[family].add(case_id)
            cluster_support[combined].add(case_id)
            if len(examples[combined]) < 8:
                examples[combined].append({
                    'case_id': case_id,
                    'span': [int(evt.get('start_frame', -1)), int(evt.get('end_frame', -1))],
                    'direction': evt.get('direction'),
                    'role': role,
                    'magnitude': evt.get('magnitude'),
                    'unit': evt.get('unit'),
                    'selected_hml3d_prompt': read_first_prompt(case_id),
                })
        if idx % args.progress_every == 0:
            elapsed = time.time() - t0
            print(f'processed {idx}/{len(case_ids)} manifest rows, valid={processed}, elapsed={elapsed:.1f}s', flush=True)

    out = {
        'run': {
            'manifest': str(Path(args.manifest)),
            'requested_cases': len(case_ids),
            'processed_cases': processed,
            'skipped_cases': skipped,
            'detect_phase': bool(args.detect_phase),
            'elapsed_sec': time.time() - t0,
        },
        'global_stats': {
            'total_layer3_events': total_layer3,
            'avg_layer1_count': layer_count_sums['layer1'] / max(processed, 1),
            'avg_layer2_count': layer_count_sums['layer2'] / max(processed, 1),
            'avg_layer25_count': layer_count_sums['layer25'] / max(processed, 1),
            'avg_layer3_count': total_layer3 / max(processed, 1),
            'low_event_case_count_le2': len(low_event_cases),
        },
        'super_family_counts': family_counts.most_common(),
        'super_family_case_support': {k: len(v) for k, v in sorted(family_support.items())},
        'cluster_counts': cluster_counts.most_common(),
        'family_cluster_counts': family_cluster_counts.most_common(),
        'family_cluster_case_support': {k: len(v) for k, v in sorted(cluster_support.items())},
        'role_counts': role_counts.most_common(),
        'cluster_examples': dict(sorted(examples.items())),
        'low_event_cases_le2': low_event_cases[:200],
        'prompt_examples_first100': prompt_examples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(f'saved={out_path}')


if __name__ == '__main__':
    main()
