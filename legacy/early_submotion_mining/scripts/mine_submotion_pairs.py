from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_joints3d_pack():
    return torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')


def extract_symbols_for_case(case_id: str, packed):
    item = packed[f'{case_id}.npy']
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    events = extract_layer1_micro_events(layer0)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    pair_counts = Counter()
    triple_counts = Counter()
    case_examples = []
    packed = load_joints3d_pack()

    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        events = extract_symbols_for_case(case_id, packed)
        symbols = [e.to_symbol() for e in events]
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += 1
        for i in range(len(symbols) - 2):
            triple_counts[(symbols[i], symbols[i + 1], symbols[i + 2])] += 1
        case_examples.append({
            'case_id': case_id,
            'num_events': len(events),
            'first_30_symbols': symbols[:30],
        })

    out = {
        'top_pairs': [
            {'pair': list(k), 'count': v} for k, v in pair_counts.most_common(50)
        ],
        'top_triples': [
            {'triple': list(k), 'count': v} for k, v in triple_counts.most_common(50)
        ],
        'case_examples': case_examples,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
