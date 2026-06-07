from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'
HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cases', type=int, default=1000)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    mod = load_runner_module()
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')
    text_root = HML_ROOT / 'texts'

    items = []
    for text_path in sorted(text_root.glob('*.txt')):
        case_id = text_path.stem
        key = f'{case_id}.npy'
        if key not in packed:
            continue
        joints = packed[key]['joints3d']
        num_frames = int(joints.shape[0])
        if num_frames <= 20:
            continue
        prompts = mod.read_all_prompts(case_id)
        prior = mod.build_caption_prior(case_id)
        richness = sum(v for k, v in prior.items() if isinstance(v, float) and not k.startswith('_'))
        items.append({
            'case_id': case_id,
            'num_frames': num_frames,
            'selected_hml3d_prompt': mod.read_first_prompt(case_id),
            'raw_prompt_segments': prompts,
            'caption_prior': prior,
            'semantic_richness': float(richness),
        })

    items.sort(key=lambda x: (-x['semantic_richness'], -x['num_frames'], x['case_id']))
    items = items[: args.max_cases]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f'hml3d_mining_{args.max_cases}.jsonl'
    summary = out_dir / f'hml3d_mining_{args.max_cases}_summary.json'
    with manifest.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')
    summary.write_text(json.dumps({
        'num_cases': len(items),
        'mean_num_frames': sum(x['num_frames'] for x in items) / max(len(items), 1),
        'first_20_case_ids': [x['case_id'] for x in items[:20]],
    }, ensure_ascii=True, indent=2), encoding='utf-8')
    print(manifest)
    print(summary)


if __name__ == '__main__':
    main()
