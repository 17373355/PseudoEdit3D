from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events, merge_micro_events

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_joints3d_pack():
    return torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')


def get_projected_sequences(case_id: str, packed):
    item = packed[f'{case_id}.npy']
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    micro = extract_layer1_micro_events(layer0)
    subs = merge_micro_events(micro)
    streams = {'whole_body': [], 'torso': [], 'left_arm': [], 'right_arm': []}
    for s in subs:
        if s.category in streams:
            streams[s.category].append(s.name)
        elif s.category == 'micro_event':
            if s.name.startswith('whole_body'):
                streams['whole_body'].append(s.name)
            elif s.name.startswith('torso'):
                streams['torso'].append(s.name)
            elif s.name.startswith('left_arm'):
                streams['left_arm'].append(s.name)
            elif s.name.startswith('right_arm'):
                streams['right_arm'].append(s.name)
    return streams


def mine_stream(stream_name: str, sequences: dict[str, list[str]], min_support: int):
    pair_counts = Counter()
    pair_cases = defaultdict(set)
    triple_counts = Counter()
    triple_cases = defaultdict(set)
    for case_id, seq in sequences.items():
        seq = seq.get(stream_name, [])
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
            pair_cases[pair].add(case_id)
        for i in range(len(seq) - 2):
            tri = (seq[i], seq[i + 1], seq[i + 2])
            triple_counts[tri] += 1
            triple_cases[tri].add(case_id)

    def summarize(counts, cases, width):
        out = []
        for key, count in counts.items():
            support = len(cases[key])
            if support < min_support:
                continue
            out.append({
                'tokens': list(key),
                'width': width,
                'count': count,
                'support_cases': support,
                'example_case_ids': sorted(cases[key])[:10],
            })
        out.sort(key=lambda x: (-x['support_cases'], -x['count'], x['tokens']))
        return out

    return {
        'pairs': summarize(pair_counts, pair_cases, 2),
        'triples': summarize(triple_counts, triple_cases, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', default='')
    parser.add_argument('--manifest', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-support', type=int, default=4)
    args = parser.parse_args()

    packed = load_joints3d_pack()
    sequences = {}
    case_ids = [x.strip() for x in args.case_ids.split(',') if x.strip()]
    if args.manifest:
        import json
        manifest_path = Path(args.manifest)
        case_ids = []
        for line in manifest_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            case_ids.append(payload['case_id'])
    for case_id in case_ids:
        sequences[case_id] = get_projected_sequences(case_id, packed)

    report = {
        'whole_body': mine_stream('whole_body', sequences, args.min_support),
        'torso': mine_stream('torso', sequences, args.min_support),
        'left_arm': mine_stream('left_arm', sequences, args.min_support),
        'right_arm': mine_stream('right_arm', sequences, args.min_support),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
