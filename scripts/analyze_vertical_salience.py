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


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = int(a.get('start_frame', -1)), int(a.get('end_frame', -1))
    b0, b1 = int(b.get('start_frame', -1)), int(b.get('end_frame', -1))
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    dur = max(1, a1 - a0 + 1)
    return inter / dur


def extract_case(case_id: str, packed: dict[str, Any]) -> dict[str, Any] | None:
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
    phases = dedupe_phase_objects(phases)
    layer3 = build_layer3_atomic_program(layer2, phases, joints=joints)
    return layer3


def bucket(value: float, cuts: list[float]) -> str:
    prev = 0.0
    for cut in cuts:
        if value < cut:
            return f'[{prev:.2f},{cut:.2f})'
        prev = cut
    return f'>={cuts[-1]:.2f}'


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
    mag_by_cluster: dict[str, list[float]] = defaultdict(list)
    dur_by_cluster: dict[str, list[int]] = defaultdict(list)
    overlap_by_cluster: dict[str, list[float]] = defaultdict(list)
    rendered_vertical_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompt_examples: list[dict[str, Any]] = []

    for idx, case_id in enumerate(case_ids, start=1):
        program = extract_case(case_id, packed)
        if program is None:
            continue
        events = list(program.get('events') or [])
        locos = [e for e in events if e.get('super_family') == 'WHOLE_BODY_LOCOMOTION' and str(e.get('cluster_id', '')).startswith('LOCO_')]
        prompt = render_aml_prompt(program)
        if any(x in prompt for x in ('jumps upward', 'lowers the body', 'hop-like', 'up-and-down body motion')):
            rendered_vertical_counts['cases_with_rendered_vertical'] += 1
            if len(prompt_examples) < 80:
                prompt_examples.append({
                    'case_id': case_id,
                    'selected_hml3d_prompt': read_first_prompt(case_id),
                    'auto_prompt': prompt,
                })
        for evt in events:
            if evt.get('super_family') != 'WHOLE_BODY_VERTICAL':
                continue
            cluster = str(evt.get('cluster_id'))
            mag = float(evt.get('magnitude') or 0.0)
            dur = int(evt.get('end_frame', -1)) - int(evt.get('start_frame', -1)) + 1
            max_ov = max((overlap_ratio(evt, l) for l in locos), default=0.0)
            key = f'{cluster}|ov={bucket(max_ov, [0.10, 0.40, 0.70, 0.90])}|mag={bucket(mag, [0.03, 0.06, 0.10, 0.16])}'
            counts[key] += 1
            mag_by_cluster[cluster].append(mag)
            dur_by_cluster[cluster].append(dur)
            overlap_by_cluster[cluster].append(max_ov)
            if len(examples[key]) < 5:
                examples[key].append({
                    'case_id': case_id,
                    'span': [int(evt.get('start_frame', -1)), int(evt.get('end_frame', -1))],
                    'magnitude': mag,
                    'duration': dur,
                    'max_loco_overlap': max_ov,
                    'auto_prompt': prompt,
                    'selected_hml3d_prompt': read_first_prompt(case_id),
                })
        if idx % args.progress_every == 0:
            print(f'processed {idx}/{len(case_ids)}, elapsed={time.time()-t0:.1f}s', flush=True)

    cluster_stats = {}
    for cluster, mags in mag_by_cluster.items():
        arr = np.asarray(mags, dtype=np.float32)
        darr = np.asarray(dur_by_cluster[cluster], dtype=np.float32)
        oarr = np.asarray(overlap_by_cluster[cluster], dtype=np.float32)
        cluster_stats[cluster] = {
            'count': int(len(arr)),
            'mag_p10': float(np.percentile(arr, 10)),
            'mag_p25': float(np.percentile(arr, 25)),
            'mag_median': float(np.median(arr)),
            'mag_p75': float(np.percentile(arr, 75)),
            'mag_p90': float(np.percentile(arr, 90)),
            'dur_median': float(np.median(darr)),
            'overlap_p50': float(np.percentile(oarr, 50)),
            'overlap_p75': float(np.percentile(oarr, 75)),
            'overlap_p90': float(np.percentile(oarr, 90)),
            'overlap_ge_070': int(np.sum(oarr >= 0.70)),
        }

    out = {
        'run': {
            'manifest': args.manifest,
            'max_cases': args.max_cases,
            'processed_cases': len(case_ids),
            'elapsed_sec': time.time() - t0,
        },
        'cluster_stats': cluster_stats,
        'overlap_mag_buckets': counts.most_common(),
        'rendered_vertical_counts': rendered_vertical_counts,
        'bucket_examples': dict(examples),
        'rendered_vertical_prompt_examples': prompt_examples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(f'saved={out_path}')


if __name__ == '__main__':
    main()
