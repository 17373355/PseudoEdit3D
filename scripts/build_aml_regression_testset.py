from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import (
    PhasePattern,
    build_layer3_atomic_program,
    dedupe_phase_patterns,
    detect_repeated_phases,
    extract_layer0_frame_observables,
    extract_layer1_micro_events,
    merge_micro_events,
    project_units_by_category,
    render_coarse_aml_prompt,
)
from pseudoedit3d.edit.coordination_patterns import detect_coordination_patterns

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')

IMPORTANT_PREFIXES = [
    'WHOLE_BODY_LOCOMOTION/',
    'WHOLE_BODY_ROTATION/',
    'WHOLE_BODY_VERTICAL/',
    'WHOLE_BODY_POSTURE/',
    'BIMANUAL_PERIODIC/',
    'LEFT_ARM_PERIODIC/',
    'RIGHT_ARM_PERIODIC/',
    'TORSO_PERIODIC/',
    'COORD/',
]

CATEGORY_QUOTAS = [
    ('coordination', 6),
    ('locomotion', 8),
    ('rotation', 7),
    ('vertical', 7),
    ('bimanual', 7),
    ('unilateral_arm', 5),
    ('torso_posture', 5),
    ('simple_other', 5),
]


def load_split(path: Path, max_cases: int | None) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return ids[:max_cases] if max_cases else ids


def read_first_prompt(case_id: str) -> str:
    text_path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not text_path.exists():
        return ''
    for line in text_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            return line.split('#')[0].strip()
    return ''


def phase_to_dict(phase: PhasePattern) -> dict[str, Any]:
    return {
        'name': phase.name,
        'kind': phase.kind,
        'count': int(phase.count),
        'start_frame': int(phase.start_frame),
        'end_frame': int(phase.end_frame),
        'unit_names': list(phase.unit_names),
        'metadata': dict(phase.metadata),
    }


def dedupe_phase_objects(phases: list[PhasePattern]) -> list[PhasePattern]:
    deduped = dedupe_phase_patterns([phase_to_dict(p) for p in phases])
    out: list[PhasePattern] = []
    for p in deduped:
        out.append(PhasePattern(
            name=str(p['name']),
            kind=str(p['kind']),
            count=int(p['count']),
            start_frame=int(p['start_frame']),
            end_frame=int(p['end_frame']),
            unit_names=list(p['unit_names']),
            metadata=dict(p.get('metadata', {})),
        ))
    out.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return out


def extract_case(case_id: str, joints: np.ndarray) -> dict[str, Any]:
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=joints[:, 0, :])
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases = list(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
    phases = dedupe_phase_objects(phases)
    layer3 = build_layer3_atomic_program(layer2, phases, joints=joints)
    events = list(layer3.get('events') or [])
    coordination = detect_coordination_patterns(events, joints=joints)
    cluster_keys = [f"{evt.get('super_family')}/{evt.get('cluster_id')}" for evt in events]
    coord_keys = [f"COORD/{pat.get('pattern_id')}" for pat in coordination]
    bimanual_variants = []
    for pat in coordination:
        arms = (pat.get('coordination_slots') or {}).get('arms') or {}
        variant = arms.get('variant_key')
        if variant:
            bimanual_variants.append(f'COORD_ARM_VARIANT/{variant}')
    feature_keys = sorted(set(cluster_keys + coord_keys + bimanual_variants))
    super_families = sorted(set(str(evt.get('super_family')) for evt in events))
    auto_prompt, coarse_program = render_coarse_aml_prompt(
        layer3,
        max_residual_events=3,
        return_program=True,
    )
    return {
        'case_id': case_id,
        'num_frames': int(len(joints)),
        'layer1_count': int(len(layer1)),
        'layer2_count': int(len(layer2)),
        'layer25_count': int(len(phases)),
        'layer3_count': int(len(events)),
        'coordination_count': int(len(coordination)),
        'feature_keys': feature_keys,
        'super_families': super_families,
        'selected_hml3d_prompt': read_first_prompt(case_id),
        'auto_prompt': auto_prompt,
        'prompt_renderer': 'coarse_v2',
        'coarse_action_program': coarse_program,
        'canonical_actions': coarse_program.get('canonical_actions') or [],
    }


def feature_weight(feature: str, support: int) -> float:
    rarity = 1.0 / max(float(support), 1.0) ** 0.5
    prefix_bonus = 1.0
    if feature.startswith('COORD/'):
        prefix_bonus = 2.2
    elif feature.startswith('COORD_ARM_VARIANT/'):
        prefix_bonus = 1.6
    elif feature.startswith('BIMANUAL_PERIODIC/'):
        prefix_bonus = 1.6
    elif feature.startswith('WHOLE_BODY_ROTATION/'):
        prefix_bonus = 1.4
    elif feature.startswith('WHOLE_BODY_VERTICAL/'):
        prefix_bonus = 1.35
    elif feature.startswith('WHOLE_BODY_POSTURE/'):
        prefix_bonus = 1.35
    elif feature.startswith('TORSO_PERIODIC/'):
        prefix_bonus = 1.25
    return prefix_bonus * rarity


def select_diverse_cases(items: list[dict[str, Any]], total_cases: int) -> list[dict[str, Any]]:
    feature_support = Counter()
    for item in items:
        feature_support.update(set(item['feature_keys']))
    remaining = list(items)
    selected: list[dict[str, Any]] = []
    covered: Counter[str] = Counter()
    selected_ids: set[str] = set()

    while remaining and len(selected) < total_cases:
        best_idx = -1
        best_score = -1e9
        for idx, item in enumerate(remaining):
            features = set(item['feature_keys'])
            novelty = sum(feature_weight(f, feature_support[f]) / (1.0 + covered[f]) for f in features)
            density = min(float(item['layer3_count']), 14.0) * 0.02
            coord_bonus = float(item['coordination_count']) * 0.18
            length_bonus = min(float(item['num_frames']) / 200.0, 1.0) * 0.03
            duplicate_penalty = 1.0 if item['case_id'] in selected_ids else 0.0
            score = novelty + density + coord_bonus + length_bonus - duplicate_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_ids.add(str(chosen['case_id']))
        covered.update(set(chosen['feature_keys']))
    return selected


def canonical_case_id(case_id: str) -> str:
    if case_id.startswith('M') and case_id[1:].isdigit():
        return case_id[1:]
    return case_id


def category_tags(item: dict[str, Any]) -> set[str]:
    features = set(str(f) for f in item.get('feature_keys', []))
    families = set(str(f) for f in item.get('super_families', []))
    tags: set[str] = set()
    if item.get('coordination_count', 0) > 0 or any(f.startswith('COORD/') for f in features):
        tags.add('coordination')
    if 'whole_body_locomotion' in families or any(f.startswith('WHOLE_BODY_LOCOMOTION/') for f in features):
        tags.add('locomotion')
    if any(f.startswith('WHOLE_BODY_ROTATION/') for f in features):
        tags.add('rotation')
    if any(f.startswith('WHOLE_BODY_VERTICAL/') for f in features):
        tags.add('vertical')
    if any(f.startswith('BIMANUAL_PERIODIC/') for f in features):
        tags.add('bimanual')
    has_left = any(f.startswith('LEFT_ARM_PERIODIC/') for f in features)
    has_right = any(f.startswith('RIGHT_ARM_PERIODIC/') for f in features)
    if (has_left or has_right) and not any(f.startswith('BIMANUAL_PERIODIC/') for f in features):
        tags.add('unilateral_arm')
    if 'torso' in families or 'whole_body_posture' in families:
        tags.add('torso_posture')
    if any(f.startswith('TORSO_PERIODIC/') or f.startswith('WHOLE_BODY_POSTURE/') for f in features):
        tags.add('torso_posture')
    if not tags or item.get('layer3_count', 0) <= 3:
        tags.add('simple_other')
    return tags


def _category_specific_score(item: dict[str, Any], category: str) -> float:
    features = set(str(f) for f in item.get('feature_keys', []))
    score = min(float(item.get('layer3_count', 0)), 16.0) * 0.03
    score += min(float(item.get('num_frames', 0)) / 220.0, 1.0) * 0.03
    if category == 'coordination':
        score += float(item.get('coordination_count', 0)) * 0.7
        score += sum(0.15 for f in features if f.startswith('COORD_ARM_VARIANT/'))
    elif category == 'locomotion':
        score += sum(0.35 for f in features if f.startswith('WHOLE_BODY_LOCOMOTION/'))
        if any('FAST' in f or 'MEDIUM' in f for f in features):
            score += 0.1
    elif category == 'rotation':
        score += sum(0.42 for f in features if f.startswith('WHOLE_BODY_ROTATION/'))
        if any('HALF' in f or 'FULL' in f for f in features):
            score += 0.12
    elif category == 'vertical':
        score += sum(0.36 for f in features if f.startswith('WHOLE_BODY_VERTICAL/'))
        if any('REP' in f or 'CYCLE' in f for f in features):
            score += 0.1
    elif category == 'bimanual':
        score += sum(0.34 for f in features if f.startswith('BIMANUAL_PERIODIC/'))
        if any('HAND' in f or 'SPREAD' in f for f in features):
            score += 0.1
    elif category == 'unilateral_arm':
        score += sum(0.3 for f in features if f.startswith('LEFT_ARM_PERIODIC/') or f.startswith('RIGHT_ARM_PERIODIC/'))
    elif category == 'torso_posture':
        score += sum(0.32 for f in features if f.startswith('TORSO_PERIODIC/') or f.startswith('WHOLE_BODY_POSTURE/'))
    elif category == 'simple_other':
        score += max(0.0, 5.0 - float(item.get('layer3_count', 0))) * 0.15
    return score


def select_stratified_cases(items: list[dict[str, Any]], total_cases: int, group_size: int) -> list[dict[str, Any]]:
    feature_support = Counter()
    for item in items:
        item['category_tags'] = sorted(category_tags(item))
        feature_support.update(set(item['feature_keys']))

    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in CATEGORY_QUOTAS}
    for item in items:
        tags = set(item['category_tags'])
        for category in by_category:
            if category in tags:
                by_category[category].append(item)

    for category, category_items in by_category.items():
        category_items.sort(
            key=lambda item: (
                -_category_specific_score(item, category),
                -sum(feature_weight(f, feature_support[f]) for f in set(item['feature_keys'])),
                canonical_case_id(str(item['case_id'])),
                str(item['case_id']),
            )
        )

    num_groups = max(1, (total_cases + group_size - 1) // group_size)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_bases: set[str] = set()
    covered_features: Counter[str] = Counter()

    def pick(category: str, group: list[dict[str, Any]], allow_mirror: bool = False) -> bool:
        best_item: dict[str, Any] | None = None
        best_score = -1e9
        group_ids = {str(item['case_id']) for item in group}
        group_bases = {canonical_case_id(str(item['case_id'])) for item in group}
        for item in by_category.get(category, []):
            case_id = str(item['case_id'])
            base_id = canonical_case_id(case_id)
            if case_id in selected_ids or case_id in group_ids:
                continue
            if not allow_mirror and (base_id in selected_bases or base_id in group_bases):
                continue
            features = set(item['feature_keys'])
            novelty = sum(feature_weight(f, feature_support[f]) / (1.0 + covered_features[f]) for f in features)
            score = _category_specific_score(item, category) + novelty
            score += min(float(item.get('num_frames', 0)) / 220.0, 1.0) * 0.02
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None:
            return False
        best_item['selection_bucket'] = category
        selected.append(best_item)
        group.append(best_item)
        selected_ids.add(str(best_item['case_id']))
        selected_bases.add(canonical_case_id(str(best_item['case_id'])))
        covered_features.update(set(best_item['feature_keys']))
        return True

    for _group_idx in range(num_groups):
        if len(selected) >= total_cases:
            break
        group: list[dict[str, Any]] = []
        for category, quota in CATEGORY_QUOTAS:
            for _ in range(quota):
                if len(group) >= group_size or len(selected) >= total_cases:
                    break
                if not pick(category, group, allow_mirror=False):
                    pick(category, group, allow_mirror=True)
        while len(group) < group_size and len(selected) < total_cases:
            remaining_categories = sorted(
                by_category,
                key=lambda cat: len(by_category.get(cat, [])),
                reverse=True,
            )
            if not any(pick(cat, group, allow_mirror=False) for cat in remaining_categories):
                if not any(pick(cat, group, allow_mirror=True) for cat in remaining_categories):
                    break

    return selected[:total_cases]


def group_items(items: list[dict[str, Any]], group_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-file', default=str(HML_ROOT / 'test.txt'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--total-cases', type=int, default=250)
    parser.add_argument('--group-size', type=int, default=50)
    parser.add_argument('--max-split-cases', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--strategy', choices=['stratified', 'diverse'], default='stratified')
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = out_dir.name if out_dir.name.startswith('aml_regression_testset_') else 'aml_regression_testset'
    case_ids = load_split(Path(args.split_file), args.max_split_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    items: list[dict[str, Any]] = []
    skipped: list[str] = []
    for idx, case_id in enumerate(case_ids, start=1):
        key = f'{case_id}.npy'
        if key not in packed:
            skipped.append(case_id)
            continue
        joints = packed[key]['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        if len(joints) <= 20:
            skipped.append(case_id)
            continue
        items.append(extract_case(case_id, joints))
        if idx % args.progress_every == 0:
            print(f'processed {idx}/{len(case_ids)}, valid={len(items)}, elapsed={time.time()-t0:.1f}s', flush=True)

    if args.strategy == 'diverse':
        selected = select_diverse_cases(items, total_cases=args.total_cases)
    else:
        selected = select_stratified_cases(items, total_cases=args.total_cases, group_size=args.group_size)
    groups = group_items(selected, args.group_size)
    feature_support = Counter()
    selected_feature_support = Counter()
    for item in items:
        feature_support.update(set(item['feature_keys']))
    for item in selected:
        selected_feature_support.update(set(item['feature_keys']))

    rows = []
    for rank, item in enumerate(selected, start=1):
        group_idx = (rank - 1) // args.group_size + 1
        rows.append({
            'rank': rank,
            'group': group_idx,
            'case_id': item['case_id'],
            'num_frames': item['num_frames'],
            'layer3_count': item['layer3_count'],
            'coordination_count': item['coordination_count'],
            'super_families': '|'.join(item['super_families']),
            'feature_keys': '|'.join(item['feature_keys']),
            'category_tags': '|'.join(item.get('category_tags', sorted(category_tags(item)))),
            'selection_bucket': item.get('selection_bucket', ''),
            'selected_hml3d_prompt': item['selected_hml3d_prompt'],
            'auto_prompt': item['auto_prompt'],
        })
    write_csv(out_dir / f'{artifact_name}.csv', rows)
    (out_dir / f'{artifact_name}.json').write_text(json.dumps({
        'run': {
            'split_file': str(Path(args.split_file)),
            'split_cases': len(case_ids),
            'valid_cases': len(items),
            'skipped_cases': len(skipped),
            'selected_cases': len(selected),
            'group_size': args.group_size,
            'strategy': args.strategy,
            'category_quotas': CATEGORY_QUOTAS,
            'num_groups': len(groups),
            'elapsed_sec': time.time() - t0,
        },
        'selected': selected,
        'feature_support_top_full': feature_support.most_common(80),
        'feature_support_top_selected': selected_feature_support.most_common(80),
        'skipped': skipped,
    }, ensure_ascii=True, indent=2), encoding='utf-8')

    for group_idx, group in enumerate(groups, start=1):
        group_path = out_dir / f'group_{group_idx:02d}_case_ids.txt'
        group_path.write_text('\n'.join(str(item['case_id']) for item in group) + '\n', encoding='utf-8')
        write_csv(out_dir / f'group_{group_idx:02d}.csv', [
            {
                'rank_in_group': i,
                'case_id': item['case_id'],
                'num_frames': item['num_frames'],
                'layer3_count': item['layer3_count'],
                'coordination_count': item['coordination_count'],
                'selected_hml3d_prompt': item['selected_hml3d_prompt'],
                'auto_prompt': item['auto_prompt'],
                'feature_keys': '|'.join(item['feature_keys']),
                'category_tags': '|'.join(item.get('category_tags', sorted(category_tags(item)))),
                'selection_bucket': item.get('selection_bucket', ''),
            }
            for i, item in enumerate(group, start=1)
        ])

    report = [
        f'# {artifact_name}',
        '',
        '## Purpose',
        '',
        '- Fixed HumanML3D test split subset for repeated AutoPrompt/AML visual regression.',
        '- Selected by AML feature diversity, not by same-case caption content.',
        '- Cases are grouped by 50 for iterative visual review.',
        '- Default selection uses category quotas so each group is not dominated by one frequent feature family.',
        '',
        '## Run',
        '',
        f'- split file: `{args.split_file}`',
        f'- valid cases scanned: `{len(items)}`',
        f'- selected cases: `{len(selected)}`',
        f'- group size: `{args.group_size}`',
        f'- strategy: `{args.strategy}`',
        f'- groups: `{len(groups)}`',
        f'- elapsed sec: `{time.time()-t0:.1f}`',
        '',
        '## Files',
        '',
        f'- `{artifact_name}.json`',
        f'- `{artifact_name}.csv`',
        '- `group_01_case_ids.txt`, `group_02_case_ids.txt`, ...',
        '- `group_01.csv`, `group_02.csv`, ...',
        '',
        '## Top Selected Feature Support',
        '',
    ]
    for feature, count in selected_feature_support.most_common(30):
        report.append(f'- {feature}: {count}')
    (out_dir / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'saved={out_dir}')
    print(f'groups={len(groups)} selected={len(selected)} valid={len(items)}')


if __name__ == '__main__':
    main()
