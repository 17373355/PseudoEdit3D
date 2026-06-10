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

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events

RUNNER = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D/scripts/run_momask_case_study.py')


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
        events = extract_layer1_micro_events(layer0)
        out.append({
            'case_id': case_id,
            'num_frames': int(len(joints)),
            'num_observables': len(layer0.names()),
            'num_events': len(events),
            'events': [
                {
                    'symbol': e.to_symbol(),
                    'observable': e.observable,
                    'part': e.part,
                    'direction': e.direction,
                    'magnitude_bin': e.magnitude_bin,
                    'duration_bin': e.duration_bin,
                    'start_frame': e.start_frame,
                    'end_frame': e.end_frame,
                    'delta_value': e.delta_value,
                    'unit': e.unit,
                    'confidence': e.confidence,
                } for e in events[:200]
            ]
        })

    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
