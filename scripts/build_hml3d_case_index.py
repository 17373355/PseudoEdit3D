from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--split', default='all', choices=['all', 'train', 'val', 'test', 'train_val'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    pth = root / 'joints3d.pth'
    texts_dir = root / 'texts'
    split_file = root / f'{args.split}.txt'
    motion = torch.load(pth, map_location='cpu')
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if args.limit > 0:
        ids = ids[:args.limit]

    out = []
    missing_text = 0
    missing_motion = 0
    for case_id in ids:
        key = f'{case_id}.npy'
        if key not in motion:
            missing_motion += 1
            continue
        text_path = texts_dir / f'{case_id}.txt'
        if not text_path.exists():
            missing_text += 1
            continue
        entry = motion[key]
        joints = entry['joints3d']
        name = entry.get('name', '')
        out.append({
            'id': case_id,
            'motion_key': key,
            'name': name,
            'num_frames': int(joints.shape[0]),
            'joints_shape': list(joints.shape),
            'text_path': str(text_path),
            'split': args.split,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')

    print(json.dumps({
        'split': args.split,
        'requested': len(ids),
        'written': len(out),
        'missing_text': missing_text,
        'missing_motion': missing_motion,
        'output': str(output),
    }, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
