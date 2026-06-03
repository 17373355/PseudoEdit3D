from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D')
RUNNER = ROOT / 'scripts' / 'run_momask_case_study.py'
TEXT_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/texts')
OUT_ROOT = ROOT / 'outputs' / 'hml3d_pattern_batches'

CATEGORY_QUOTAS = {
    'bounce_repeated': 15,
    'stair_descent': 15,
    'stair_ascent': 15,
    'walk_backward': 15,
    'stop_pause': 15,
    'crouch_bend': 10,
    'turn': 15,
}


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def classify_case(prior: dict[str, float], prompts: list[tuple[str, float, float]]) -> list[str]:
    captions = [caption.lower() for caption, _, _ in prompts]
    cats: list[str] = []
    if prior.get('stair_descent', 0.0) > 0:
        cats.append('stair_descent')
    if prior.get('stair_ascent', 0.0) > 0:
        cats.append('stair_ascent')
    if prior.get('walk_backward', 0.0) > 0:
        cats.append('walk_backward')
    if prior.get('stop_pause', 0.0) > 0:
        cats.append('stop_pause')
    if prior.get('crouch_bend', 0.0) > 0:
        cats.append('crouch_bend')
    if prior.get('turn', 0.0) > 0:
        cats.append('turn')
    if any(('up and down' in c) or ('bounce' in c) or ('hops' in c) for c in captions):
        cats.append('bounce_repeated')
    return cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--resolved-cache', default='')
    args = parser.parse_args()

    mod = load_runner_module()
    resolved = set()
    if args.resolved_cache:
        p = Path(args.resolved_cache)
        if p.exists():
            resolved = {line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()}

    candidates = {k: [] for k in CATEGORY_QUOTAS}
    for text_path in sorted(TEXT_ROOT.glob('*.txt')):
        case_id = text_path.stem
        if case_id in resolved:
            continue
        prompts = mod.read_all_prompts(case_id)
        joints = mod.load_joints3d(case_id)
        if len(joints) <= 20:
            continue
        prior = mod.build_caption_prior(case_id)
        cats = classify_case(prior, prompts)
        if not cats:
            continue
        item = {
            'case_id': case_id,
            'categories': cats,
            'selected_hml3d_prompt': mod.read_first_prompt(case_id),
            'raw_prompt_segments': prompts,
            'caption_prior': prior,
        }
        for cat in cats:
            if cat in candidates:
                candidates[cat].append(item)

    selected = []
    selected_ids = set()
    counts = {k: 0 for k in CATEGORY_QUOTAS}
    for cat, quota in CATEGORY_QUOTAS.items():
        pool = sorted(candidates[cat], key=lambda x: (-x['caption_prior'].get(cat, 0.0), x['case_id']))
        for item in pool:
            if item['case_id'] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item['case_id'])
            for c in item['categories']:
                if c in counts:
                    counts[c] += 1
            if counts[cat] >= quota:
                break

    if len(selected) < args.batch_size:
        fallback = []
        for pool in candidates.values():
            fallback.extend(pool)
        fallback = sorted(
            fallback,
            key=lambda x: (-max(x['caption_prior'].get(cat, 0.0) for cat in x['categories']), x['case_id'])
        )
        for item in fallback:
            if item['case_id'] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item['case_id'])
            for c in item['categories']:
                if c in counts:
                    counts[c] += 1
            if len(selected) >= args.batch_size:
                break

    selected = [item for item in selected if item['case_id'] not in resolved]
    selected = selected[:args.batch_size]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / f'batch_{args.batch_id}_manifest.jsonl'
    summary_path = OUT_ROOT / f'batch_{args.batch_id}_manifest_summary.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')
    summary = {
        'batch_id': args.batch_id,
        'num_cases': len(selected),
        'case_ids': [item['case_id'] for item in selected],
        'category_counts': {cat: sum(1 for item in selected if cat in item['categories']) for cat in CATEGORY_QUOTAS},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')
    print(manifest_path)
    print(summary_path)
    print(','.join(summary['case_ids']))


if __name__ == '__main__':
    main()
