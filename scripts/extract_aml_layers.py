from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import (
    extract_layer0_frame_observables,
    extract_layer1_micro_events,
    merge_micro_events,
    detect_repeated_phases,
    project_units_by_category,
    build_layer3_atomic_program,
    dedupe_phase_patterns,
)

RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'
HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_joints3d_pack():
    return torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')


def serialize_layer0(layer0):
    out = {}
    for name in layer0.names():
        seq = layer0.get(name)
        out[name] = {
            'unit': seq.unit,
            'part': seq.part,
            'source': seq.source,
            'num_frames': len(seq.values),
            'min': float(np.min(seq.values)),
            'max': float(np.max(seq.values)),
            'mean': float(np.mean(seq.values)),
            'std': float(np.std(seq.values)),
            'first_20': [float(x) for x in seq.values[:20]],
        }
    return out


def serialize_micro(events, limit=200):
    return [
        {
            'symbol': e.to_symbol(),
            'observable': e.observable,
            'part': e.part,
            'direction': e.direction,
            'magnitude_bin': e.magnitude_bin,
            'duration_bin': e.duration_bin,
            'start_frame': e.start_frame,
            'end_frame': e.end_frame,
            'delta_value': float(e.delta_value),
            'unit': e.unit,
            'confidence': float(e.confidence),
        }
        for e in events[:limit]
    ]


def serialize_sub(subs, limit=200):
    return [
        {
            'name': s.name,
            'category': s.category,
            'start_frame': s.start_frame,
            'end_frame': s.end_frame,
            'support_tokens': s.support_tokens,
            'metadata': s.metadata,
        }
        for s in subs[:limit]
    ]


def serialize_phase(phases, limit=200):
    return [
        {
            'name': p.name,
            'kind': p.kind,
            'count': p.count,
            'start_frame': p.start_frame,
            'end_frame': p.end_frame,
            'unit_names': p.unit_names,
            'metadata': p.metadata,
        }
        for p in phases[:limit]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()

    mod = load_runner_module()
    packed = load_joints3d_pack()
    result = []
    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        item = packed[f'{case_id}.npy']
        joints = item['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
        trans = joints[:, 0, :]
        layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
        layer1 = extract_layer1_micro_events(layer0)
        layer2 = merge_micro_events(layer1)
        layer25_global = detect_repeated_phases(layer2)
        layer25_projected = {
            'whole_body': detect_repeated_phases(project_units_by_category(layer2, 'whole_body')),
            'torso': detect_repeated_phases(project_units_by_category(layer2, 'torso')),
            'left_arm': detect_repeated_phases(project_units_by_category(layer2, 'left_arm')),
            'right_arm': detect_repeated_phases(project_units_by_category(layer2, 'right_arm')),
        }
        projected_serialized = {k: serialize_phase(v, args.limit) for k, v in layer25_projected.items()}
        layer25_all = list(layer25_global)
        for pats in layer25_projected.values():
            layer25_all.extend(pats)
        layer25_dedup = dedupe_phase_patterns([
            {
                'name': p.name,
                'kind': p.kind,
                'count': p.count,
                'start_frame': p.start_frame,
                'end_frame': p.end_frame,
                'unit_names': p.unit_names,
                'metadata': p.metadata,
            } for p in layer25_all
        ])
        layer25_dedup.sort(key=lambda p: (p['start_frame'], p['end_frame'], p['name']))
        phase_for_layer3 = []
        for p in layer25_dedup:
            from pseudoedit3d.edit.phase_patterns import PhasePattern
            phase_for_layer3.append(PhasePattern(
                name=p['name'],
                kind=p['kind'],
                count=int(p['count']),
                start_frame=int(p['start_frame']),
                end_frame=int(p['end_frame']),
                unit_names=list(p['unit_names']),
                metadata=dict(p.get('metadata', {})),
            ))
        layer3 = build_layer3_atomic_program(layer2, phase_for_layer3)
        result.append({
            'case_id': case_id,
            'selected_hml3d_prompt': mod.read_first_prompt(case_id),
            'num_frames': int(len(joints)),
            'layer0': serialize_layer0(layer0),
            'layer1_micro_events': serialize_micro(layer1, args.limit),
            'layer2_submotions': serialize_sub(layer2, args.limit),
            'layer25_phase_patterns': layer25_dedup[:args.limit],
            'layer25_phase_patterns_global': serialize_phase(layer25_global, args.limit),
            'layer25_phase_patterns_projected': projected_serialized,
            'layer3_atomic_program': layer3,
        })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
