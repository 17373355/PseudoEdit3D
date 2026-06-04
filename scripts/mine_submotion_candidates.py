from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def extract_events_for_case(case_id: str, packed):
    item = packed[f'{case_id}.npy']
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    return extract_layer1_micro_events(layer0)


def part_prefix(symbol: str) -> str:
    return symbol.split('_', 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-support', type=int, default=3)
    args = parser.parse_args()

    packed = load_joints3d_pack()
    pair_stats = defaultdict(lambda: {'count': 0, 'case_ids': set(), 'durations': [], 'parts': defaultdict(int)})
    triple_stats = defaultdict(lambda: {'count': 0, 'case_ids': set(), 'durations': [], 'parts': defaultdict(int)})

    case_ids = [x.strip() for x in args.case_ids.split(',') if x.strip()]
    for case_id in case_ids:
        events = extract_events_for_case(case_id, packed)
        symbols = [e.to_symbol() for e in events]
        for i in range(len(events) - 1):
            key = (symbols[i], symbols[i + 1])
            pair_stats[key]['count'] += 1
            pair_stats[key]['case_ids'].add(case_id)
            pair_stats[key]['durations'].append(events[i + 1].end_frame - events[i].start_frame + 1)
            pair_stats[key]['parts'][events[i].part] += 1
            pair_stats[key]['parts'][events[i + 1].part] += 1
        for i in range(len(events) - 2):
            key = (symbols[i], symbols[i + 1], symbols[i + 2])
            triple_stats[key]['count'] += 1
            triple_stats[key]['case_ids'].add(case_id)
            triple_stats[key]['durations'].append(events[i + 2].end_frame - events[i].start_frame + 1)
            triple_stats[key]['parts'][events[i].part] += 1
            triple_stats[key]['parts'][events[i + 1].part] += 1
            triple_stats[key]['parts'][events[i + 2].part] += 1

    def summarize(stats, width):
        out = []
        for key, val in stats.items():
            support = len(val['case_ids'])
            if support < args.min_support:
                continue
            out.append({
                'tokens': list(key),
                'width': width,
                'count': val['count'],
                'support_cases': support,
                'avg_duration': float(np.mean(val['durations'])) if val['durations'] else 0.0,
                'parts': dict(sorted(val['parts'].items(), key=lambda kv: kv[1], reverse=True)),
                'example_case_ids': sorted(val['case_ids'])[:10],
            })
        out.sort(key=lambda x: (-x['support_cases'], -x['count'], x['tokens']))
        return out

    report = {
        'pairs': summarize(pair_stats, 2),
        'triples': summarize(triple_stats, 3),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
