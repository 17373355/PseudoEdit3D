from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events, merge_micro_events
from pseudoedit3d.edit.phase_patterns import detect_repeated_phases, project_units_by_category

RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    mod = load_runner_module()
    out = []
    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        joints = mod.load_joints3d(case_id)
        poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
        trans = joints[:, 0, :]
        layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
        micro = extract_layer1_micro_events(layer0)
        subs = merge_micro_events(micro)
        phases = detect_repeated_phases(subs)
        whole_body_phases = detect_repeated_phases(project_units_by_category(subs, 'whole_body'))
        torso_phases = detect_repeated_phases(project_units_by_category(subs, 'torso'))
        left_arm_phases = detect_repeated_phases(project_units_by_category(subs, 'left_arm'))
        right_arm_phases = detect_repeated_phases(project_units_by_category(subs, 'right_arm'))
        out.append({
            'case_id': case_id,
            'micro_event_count': len(micro),
            'submotion_count': len(subs),
            'phase_patterns': [
                {
                    'name': p.name,
                    'kind': p.kind,
                    'count': p.count,
                    'start_frame': p.start_frame,
                    'end_frame': p.end_frame,
                    'unit_names': p.unit_names,
                } for p in phases
            ],
            'projected_phase_patterns': {
                'whole_body': [
                    {
                        'name': p.name, 'kind': p.kind, 'count': p.count, 'start_frame': p.start_frame, 'end_frame': p.end_frame, 'unit_names': p.unit_names
                    } for p in whole_body_phases
                ],
                'torso': [
                    {
                        'name': p.name, 'kind': p.kind, 'count': p.count, 'start_frame': p.start_frame, 'end_frame': p.end_frame, 'unit_names': p.unit_names
                    } for p in torso_phases
                ],
                'left_arm': [
                    {
                        'name': p.name, 'kind': p.kind, 'count': p.count, 'start_frame': p.start_frame, 'end_frame': p.end_frame, 'unit_names': p.unit_names
                    } for p in left_arm_phases
                ],
                'right_arm': [
                    {
                        'name': p.name, 'kind': p.kind, 'count': p.count, 'start_frame': p.start_frame, 'end_frame': p.end_frame, 'unit_names': p.unit_names
                    } for p in right_arm_phases
                ],
            }
        })
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
