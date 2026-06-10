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

from pseudoedit3d.edit.bimanual_split import (
    bimanual_span_features as core_bimanual_span_features,
    classify_bimanual_event,
)
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

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')
BIMANUAL_FAMILY = 'BIMANUAL_PERIODIC'
BIMANUAL_CLUSTERS = {'BI_OUT', 'BI_UP'}


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


def read_captions(case_id: str) -> list[str]:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not path.exists():
        return []
    captions: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            caption = line.split('#')[0].strip()
            if caption:
                captions.append(caption)
    return captions


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


def extract_program(case_id: str, joints: np.ndarray, *, detect_phase: bool) -> dict[str, Any] | None:
    if len(joints) <= 20:
        return None
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=joints[:, 0, :])
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases: list[PhasePattern] = []
    if detect_phase:
        phases.extend(detect_repeated_phases(layer2))
        for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
            phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
        phases = dedupe_phase_objects(phases)
    return build_layer3_atomic_program(layer2, phases)


def _feature_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        'mean_hand_distance_norm',
        'min_hand_distance_norm',
        'max_hand_distance_norm',
        'hand_distance_delta_norm',
        'mean_wrist_chest_distance_norm',
        'mean_wrist_root_distance_norm',
        'mean_wrist_height_rel_norm',
        'max_wrist_height_rel_norm',
        'wrist_height_delta_norm',
        'mean_wrist_speed_norm',
        'root_path_xz_norm',
        'max_loco_overlap',
        'max_vertical_overlap',
        'left_right_chest_corr',
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(r['features'][key]) for r in rows], dtype=np.float32)
        if len(values) == 0:
            continue
        out[f'{key}.mean'] = float(np.mean(values))
        out[f'{key}.p50'] = float(np.percentile(values, 50))
        out[f'{key}.p90'] = float(np.percentile(values, 90))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(str(r['split_cluster']) for r in rows)
    labels = [k for k, _ in split_counts.most_common()]
    values = [split_counts[k] for k in labels]

    fig_h = max(4.5, 0.42 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, values, color='#F58518')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('event count')
    ax.set_title('Bimanual Candidate Split Counts', fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value, idx, f' {value}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / '01_bimanual_split_counts.png', dpi=180)
    plt.close(fig)

    originals = sorted(set(str(r['original_cluster']) for r in rows))
    splits = labels
    mat = np.zeros((len(originals), len(splits)), dtype=np.float32)
    for r in rows:
        mat[originals.index(str(r['original_cluster'])), splits.index(str(r['split_cluster']))] += 1
    fig_w = max(10, 0.7 * len(splits) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    bottom = np.zeros((len(originals),), dtype=np.float32)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(splits), 1)))
    for idx, split in enumerate(splits):
        ax.bar(originals, mat[:, idx], bottom=bottom, label=split, color=colors[idx])
        bottom += mat[:, idx]
    ax.set_ylabel('event count')
    ax.set_title('Original BI_OUT/BI_UP -> Candidate Splits', fontsize=13, pad=12)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / '02_original_to_split_stacked.png', dpi=180)
    plt.close(fig)

    if not rows:
        return
    max_points = min(len(rows), 8000)
    sampled = rows[:max_points]
    x = np.asarray([float(r['features']['mean_hand_distance_norm']) for r in sampled])
    yv = np.asarray([float(r['features']['mean_wrist_height_rel_norm']) for r in sampled])
    split_to_idx = {s: i for i, s in enumerate(splits)}
    c = np.asarray([split_to_idx[str(r['split_cluster'])] for r in sampled])
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    scatter = ax.scatter(x, yv, c=c, cmap='tab20', s=12, alpha=0.58, edgecolors='none')
    ax.set_xlabel('mean hand-hand distance / shoulder width')
    ax.set_ylabel('mean wrist height above shoulders / shoulder width')
    ax.set_title('Bimanual Split Feature Space', fontsize=13, pad=12)
    ax.grid(alpha=0.22)
    handles, _ = scatter.legend_elements(num=len(splits))
    ax.legend(handles, splits, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / '03_feature_scatter_hand_distance_vs_height.png', dpi=180)
    plt.close(fig)

    norm = mat / np.clip(mat.sum(axis=1, keepdims=True), 1.0, None)
    fig_w = max(9, 0.75 * len(splits) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    im = ax.imshow(norm, cmap='YlGnBu', aspect='auto', vmin=0.0, vmax=max(0.01, float(norm.max())))
    ax.set_xticks(np.arange(len(splits)))
    ax.set_yticks(np.arange(len(originals)))
    ax.set_xticklabels(splits, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(originals, fontsize=9)
    ax.set_title('Normalized Bimanual Split Matrix', fontsize=13, pad=12)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] > 0:
                ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('share within original cluster')
    fig.tight_layout()
    fig.savefig(fig_dir / '04_original_to_split_heatmap.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--csv-output', default=None)
    parser.add_argument('--figure-dir', default=None)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--detect-phase', action='store_true')
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--max-examples-per-split', type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    manifest = Path(args.manifest)
    case_ids = load_case_ids(manifest, args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    original_to_split: dict[str, Counter[str]] = defaultdict(Counter)
    split_case_support: dict[str, set[str]] = defaultdict(set)
    original_case_support: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    processed = 0
    skipped = 0
    bimanual_cases: set[str] = set()

    for idx, case_id in enumerate(case_ids, start=1):
        key = f'{case_id}.npy'
        if key not in packed:
            skipped += 1
            continue
        joints = packed[key]['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        program = extract_program(case_id, joints, detect_phase=bool(args.detect_phase))
        if program is None:
            skipped += 1
            continue
        processed += 1
        events = list(program.get('events') or [])
        bimanual_events = [
            e for e in events
            if e.get('super_family') == BIMANUAL_FAMILY and str(e.get('cluster_id')) in BIMANUAL_CLUSTERS
        ]
        if bimanual_events:
            bimanual_cases.add(case_id)
        captions = read_captions(case_id) if bimanual_events else []
        for event_idx, evt in enumerate(bimanual_events):
            feat = core_bimanual_span_features(joints, evt, events)
            split, reason, confidence = classify_bimanual_event(evt, feat)
            original = str(evt.get('cluster_id'))
            row = {
                'case_id': case_id,
                'event_idx': event_idx,
                'original_cluster': original,
                'split_cluster': split,
                'split_reason': reason,
                'split_confidence': confidence,
                'start_frame': int(feat['start_frame']),
                'end_frame': int(feat['end_frame']),
                'duration': int(feat['duration']),
                'source': evt.get('source'),
                'role': evt.get('role'),
                'magnitude': evt.get('magnitude'),
                'unit': evt.get('unit'),
                'caption_0_for_analysis_only': captions[0] if captions else '',
                'features': feat,
            }
            rows.append(row)
            split_counts[split] += 1
            original_to_split[original][split] += 1
            split_case_support[split].add(case_id)
            original_case_support[original].add(case_id)
            if len(examples[split]) < args.max_examples_per_split:
                examples[split].append({
                    'case_id': case_id,
                    'original_cluster': original,
                    'span': [int(feat['start_frame']), int(feat['end_frame'])],
                    'confidence': confidence,
                    'reason': reason,
                    'caption_0_for_analysis_only': captions[0] if captions else '',
                    'features': {
                        'mean_hand_distance_norm': feat['mean_hand_distance_norm'],
                        'mean_wrist_height_rel_norm': feat['mean_wrist_height_rel_norm'],
                        'mean_wrist_chest_distance_norm': feat['mean_wrist_chest_distance_norm'],
                        'mean_wrist_speed_norm': feat['mean_wrist_speed_norm'],
                        'max_loco_overlap': feat['max_loco_overlap'],
                    },
                })
        if idx % args.progress_every == 0:
            print(
                f'processed {idx}/{len(case_ids)}, valid={processed}, bimanual_events={len(rows)}, elapsed={time.time()-t0:.1f}s',
                flush=True,
            )

    by_split_rows = {
        split: [r for r in rows if r['split_cluster'] == split]
        for split in split_counts
    }
    out = {
        'run': {
            'manifest': str(manifest),
            'requested_cases': len(case_ids),
            'processed_cases': processed,
            'skipped_cases': skipped,
            'detect_phase': bool(args.detect_phase),
            'elapsed_sec': time.time() - t0,
            'note': 'This is a diagnostic split of motion-derived bimanual events. HML3D captions are read only for analysis examples, not for auto-prompt generation.',
        },
        'thresholds': {
            'hands_close': 'min_hand_distance_norm<=0.85 or mean_hand_distance_norm<=1.05',
            'hands_far': 'mean_hand_distance_norm>=1.75 or max_hand_distance_norm>=2.35',
            'hands_spreading': 'hand_distance_delta_norm>=0.30',
            'high_hands': 'max_wrist_height_rel_norm>=0.35 or mean_wrist_height_rel_norm>=0.18 or wrist_height_delta_norm>=0.25',
            'extended_locomotion_proxy': 'moving_body and stable_hands and extended; diagnostic only, not object/contact proof',
            'stable_hands': 'mean_wrist_speed_norm<=0.16 and duration>=8',
            'moving_body': 'max_loco_overlap>=0.40 or root_path_xz_norm>=1.80',
        },
        'event_count': len(rows),
        'case_support': len(bimanual_cases),
        'split_counts': split_counts.most_common(),
        'split_case_support': {k: len(v) for k, v in sorted(split_case_support.items())},
        'original_counts': Counter(str(r['original_cluster']) for r in rows).most_common(),
        'original_case_support': {k: len(v) for k, v in sorted(original_case_support.items())},
        'original_to_split_counts': {k: v.most_common() for k, v in sorted(original_to_split.items())},
        'feature_summary_by_split': {
            split: _feature_summary(split_rows)
            for split, split_rows in sorted(by_split_rows.items())
        },
        'examples_by_split': dict(sorted(examples.items())),
        'sample_rows': rows[:300],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')

    if args.csv_output:
        flat_rows = []
        for r in rows:
            flat = {k: v for k, v in r.items() if k != 'features'}
            for fk, fv in r['features'].items():
                flat[f'feature.{fk}'] = fv
            flat_rows.append(flat)
        write_csv(Path(args.csv_output), flat_rows)

    if args.figure_dir:
        save_figures(rows, Path(args.figure_dir))

    report_lines = ['# Bimanual Candidate Split Analysis v1', '']
    report_lines.append('## Scope')
    report_lines.append('')
    report_lines.append('- The split is motion-derived and diagnostic; it does not use same-case HumanML3D text to generate auto-prompts.')
    report_lines.append('- HML3D captions in examples are only for human inspection of whether the split matches common wording.')
    report_lines.append('- Extended-locomotion subclasses are only motion proxies because HumanML3D joints do not provide wall/object geometry.')
    report_lines.append('')
    report_lines.append('## Run')
    report_lines.append('')
    for k, v in out['run'].items():
        report_lines.append(f'- {k}: `{v}`')
    report_lines.append('')
    report_lines.append('## Split Counts')
    report_lines.append('')
    for split, count in out['split_counts']:
        report_lines.append(f'- {split}: events={count}, cases={out["split_case_support"].get(split, 0)}')
    report_lines.append('')
    report_lines.append('## Original Cluster -> Split')
    report_lines.append('')
    for original, pairs in out['original_to_split_counts'].items():
        report_lines.append(f'### {original}')
        for split, count in pairs:
            report_lines.append(f'- {split}: {count}')
        report_lines.append('')
    report_lines.append('## Examples')
    report_lines.append('')
    for split, exs in out['examples_by_split'].items():
        report_lines.append(f'### {split}')
        for ex in exs[:8]:
            feats = ex['features']
            report_lines.append(
                f"- {ex['case_id']} {ex['span']} from={ex['original_cluster']} conf={ex['confidence']:.2f} "
                f"hand_dist={float(feats['mean_hand_distance_norm']):.2f} "
                f"height={float(feats['mean_wrist_height_rel_norm']):.2f} "
                f"chest={float(feats['mean_wrist_chest_distance_norm']):.2f} "
                f"speed={float(feats['mean_wrist_speed_norm']):.2f} "
                f"loco_overlap={float(feats['max_loco_overlap']):.2f} | {ex['caption_0_for_analysis_only']}"
            )
        report_lines.append('')
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'saved={out_path}')
    print(f'report={report_path}')
    if args.csv_output:
        print(f'csv={args.csv_output}')
    if args.figure_dir:
        print(f'figures={args.figure_dir}')


if __name__ == '__main__':
    main()
