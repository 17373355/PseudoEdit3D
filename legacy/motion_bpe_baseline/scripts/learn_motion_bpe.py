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


def get_submotion_sequence(case_id: str, packed):
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
    return [s.name for s in subs]


def count_pairs(sequences):
    pair_counts = Counter()
    pair_cases = defaultdict(set)
    for case_id, seq in sequences.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
            pair_cases[pair].add(case_id)
    return pair_counts, pair_cases


def apply_merge(seq, pair, merged_token):
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(merged_token)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', default='')
    parser.add_argument('--manifest', default='')
    parser.add_argument('--num-merges', type=int, default=20)
    parser.add_argument('--min-support', type=int, default=4)
    parser.add_argument('--output', required=True)
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
        sequences[case_id] = get_submotion_sequence(case_id, packed)

    merges = []
    for step in range(args.num_merges):
        pair_counts, pair_cases = count_pairs(sequences)
        best_pair = None
        best_count = -1
        best_support = -1
        for pair, count in pair_counts.items():
            support = len(pair_cases[pair])
            if support < args.min_support:
                continue
            if support > best_support or (support == best_support and count > best_count):
                best_pair = pair
                best_count = count
                best_support = support
        if best_pair is None:
            break
        merged_token = best_pair[0] + '++' + best_pair[1]
        merges.append({
            'step': step + 1,
            'pair': list(best_pair),
            'merged_token': merged_token,
            'count': best_count,
            'support_cases': best_support,
            'example_case_ids': sorted(pair_cases[best_pair])[:10],
        })
        for case_id, seq in list(sequences.items()):
            sequences[case_id] = apply_merge(seq, best_pair, merged_token)

    summary = {
        'num_cases': len(sequences),
        'num_merges_applied': len(merges),
        'merges': merges,
        'sample_sequences': {cid: seq[:40] for cid, seq in list(sequences.items())[:10]},
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
