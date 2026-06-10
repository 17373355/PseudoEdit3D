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
)
from pseudoedit3d.edit.coordination_patterns import detect_coordination_patterns

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_case_ids(manifest: Path, max_cases: int | None) -> list[str]:
    out: list[str] = []
    with manifest.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            out.append(str(json.loads(line)['case_id']))
            if max_cases is not None and len(out) >= max_cases:
                break
    return out


def read_first_prompt(case_id: str) -> str:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not path.exists():
        return ''
    for line in path.read_text(encoding='utf-8').splitlines():
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


def extract_layer3(joints: np.ndarray) -> dict[str, Any]:
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=joints[:, 0, :])
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases = list(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
    phases = dedupe_phase_objects(phases)
    return build_layer3_atomic_program(layer2, phases, joints=joints)


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
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--csv-output', default=None)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--max-examples-per-pattern', type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    case_ids = load_case_ids(Path(args.manifest), args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    pattern_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    pattern_variant_counts: Counter[str] = Counter()
    timing_counts: Counter[str] = Counter()
    pattern_case_support: dict[str, set[str]] = defaultdict(set)
    variant_case_support: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0

    for idx, case_id in enumerate(case_ids, start=1):
        key = f'{case_id}.npy'
        if key not in packed:
            skipped += 1
            continue
        joints = packed[key]['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        if len(joints) <= 20:
            skipped += 1
            continue
        program = extract_layer3(joints)
        patterns = detect_coordination_patterns(list(program.get('events') or []), joints=joints)
        processed += 1
        prompt = read_first_prompt(case_id)
        for pat in patterns:
            pattern_id = str(pat['pattern_id'])
            arms = pat.get('coordination_slots', {}).get('arms', {})
            body = pat.get('coordination_slots', {}).get('body', {})
            variant = str(arms.get('variant_key') or 'none')
            timing = '+'.join(arms.get('timing') or [])
            pattern_counts[pattern_id] += 1
            variant_counts[variant] += 1
            pattern_variant_counts[f'{pattern_id}|{variant}'] += 1
            timing_counts[timing] += 1
            pattern_case_support[pattern_id].add(case_id)
            variant_case_support[variant].add(case_id)
            row = {
                'case_id': case_id,
                'pattern_id': pattern_id,
                'semantic_name': pat.get('optional_semantic_name'),
                'start_frame': pat.get('start_frame'),
                'end_frame': pat.get('end_frame'),
                'confidence': pat.get('confidence'),
                'arm_variant': variant,
                'arm_timing': timing,
                'root_xz_displacement': body.get('coord_root_xz_displacement'),
                'root_xz_path': body.get('coord_root_xz_path'),
                'pre_root_xz_path': body.get('coord_pre_root_xz_path'),
                'forward_like': body.get('forward_like'),
                'standing_like': body.get('standing_like'),
                'selected_hml3d_prompt': prompt,
            }
            flat_rows.append(row)
            if len(examples[pattern_id]) < args.max_examples_per_pattern:
                examples[pattern_id].append({
                    **row,
                    'supporting_event_ids': pat.get('supporting_event_ids', []),
                })
        if idx % args.progress_every == 0:
            print(f'processed {idx}/{len(case_ids)}, valid={processed}, patterns={len(flat_rows)}, elapsed={time.time()-t0:.1f}s', flush=True)

    out = {
        'run': {
            'manifest': args.manifest,
            'requested_cases': len(case_ids),
            'processed_cases': processed,
            'skipped_cases': skipped,
            'elapsed_sec': time.time() - t0,
            'note': 'Layer4 diagnostic: body-arm coordination patterns preserve bimanual realization variants instead of replacing them with action labels.',
        },
        'pattern_counts': pattern_counts.most_common(),
        'pattern_case_support': {k: len(v) for k, v in sorted(pattern_case_support.items())},
        'arm_variant_counts': variant_counts.most_common(),
        'arm_variant_case_support': {k: len(v) for k, v in sorted(variant_case_support.items())},
        'pattern_variant_counts': pattern_variant_counts.most_common(),
        'arm_timing_counts': timing_counts.most_common(),
        'examples_by_pattern': dict(sorted(examples.items())),
        'sample_rows': flat_rows[:300],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.csv_output:
        write_csv(Path(args.csv_output), flat_rows)

    lines = ['# Coordination Pattern Diagnostic v1', '']
    lines.append('## Scope')
    lines.append('')
    lines.append('- Detects higher-level body-arm coordination above atomic bimanual clusters.')
    lines.append('- Keeps bimanual variants such as spread / raise-spread / hands-close as realization slots.')
    lines.append('- Does not use HumanML3D captions to assign pattern labels; captions are shown only for inspection.')
    lines.append('')
    lines.append('## Run')
    lines.append('')
    for k, v in out['run'].items():
        lines.append(f'- {k}: `{v}`')
    lines.append('')
    lines.append('## Pattern Counts')
    lines.append('')
    for pattern, count in out['pattern_counts']:
        lines.append(f'- {pattern}: events={count}, cases={out["pattern_case_support"].get(pattern, 0)}')
    lines.append('')
    lines.append('## Arm Realization Variants')
    lines.append('')
    for variant, count in out['arm_variant_counts'][:20]:
        lines.append(f'- {variant}: events={count}, cases={out["arm_variant_case_support"].get(variant, 0)}')
    lines.append('')
    lines.append('## Examples')
    lines.append('')
    for pattern, exs in out['examples_by_pattern'].items():
        lines.append(f'### {pattern}')
        for ex in exs[:8]:
            lines.append(
                f"- {ex['case_id']} [{ex['start_frame']}-{ex['end_frame']}] variant={ex['arm_variant']} "
                f"timing={ex['arm_timing']} conf={float(ex['confidence']):.2f} "
                f"root_disp={float(ex['root_xz_displacement']):.2f} pre_path={float(ex['pre_root_xz_path']):.2f} | "
                f"{ex['selected_hml3d_prompt']}"
            )
        lines.append('')
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\\n'.join(lines), encoding='utf-8')
    print(f'saved={out_path}')
    print(f'report={report_path}')
    if args.csv_output:
        print(f'csv={args.csv_output}')


if __name__ == '__main__':
    main()
