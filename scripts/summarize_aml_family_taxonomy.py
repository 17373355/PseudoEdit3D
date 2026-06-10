from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
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

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')


def load_manifest_case_ids(path: Path, max_cases: int | None = None) -> list[str]:
    case_ids: list[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            case_ids.append(str(item['case_id']))
            if max_cases is not None and len(case_ids) >= max_cases:
                break
    return case_ids


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
        out.append(
            PhasePattern(
                name=str(p['name']),
                kind=str(p['kind']),
                count=int(p['count']),
                start_frame=int(p['start_frame']),
                end_frame=int(p['end_frame']),
                unit_names=list(p['unit_names']),
                metadata=dict(p.get('metadata', {})),
            )
        )
    out.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return out


def extract_case_layer3(case_id: str, packed: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    key = f'{case_id}.npy'
    if key not in packed:
        return None, 'missing_joints'
    item = packed[key]
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    joints = np.asarray(joints, dtype=np.float32)
    if len(joints) <= 20:
        return None, 'too_short'

    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)

    phases = list(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        projected = project_units_by_category(layer2, category)
        phases.extend(detect_repeated_phases(projected))
    phases = dedupe_phase_objects(phases)
    layer3 = build_layer3_atomic_program(layer2, phases, joints=joints)
    return {
        'case_id': case_id,
        'num_frames': int(len(joints)),
        'layer1_count': len(layer1),
        'layer2_count': len(layer2),
        'layer25_count': len(phases),
        'layer3': layer3,
    }, None


def signature_key(evt: dict[str, Any]) -> str:
    sig = evt.get('motion_signature') or {}
    fields = [
        sig.get('dominant_axis', 'unknown'),
        sig.get('repeat_mode', 'unknown'),
        sig.get('phase_template', 'unknown'),
        sig.get('contact_mode', 'unknown'),
        sig.get('support_mode', 'unknown'),
        sig.get('bilateral_symmetry', 'unknown'),
        str(bool(sig.get('alternation', False))).lower(),
        sig.get('tempo_bucket', 'unknown'),
        sig.get('context_mode', 'unknown'),
    ]
    return '|'.join(str(x) for x in fields)


def core_signature_key(evt: dict[str, Any]) -> str:
    sig = evt.get('motion_signature') or {}
    fields = [
        sig.get('dominant_axis', 'unknown'),
        sig.get('repeat_mode', 'unknown'),
        sig.get('phase_template', 'unknown'),
        sig.get('contact_mode', 'unknown'),
        sig.get('support_mode', 'unknown'),
        sig.get('bilateral_symmetry', 'unknown'),
        str(bool(sig.get('alternation', False))).lower(),
        sig.get('context_mode', 'unknown'),
    ]
    return '|'.join(str(x) for x in fields)


def summarize_cases(case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    family_cluster_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    cluster_signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_core_signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_tempo_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_case_support: dict[str, set[str]] = defaultdict(set)
    cluster_case_support: dict[str, set[str]] = defaultdict(set)
    case_rows: list[dict[str, Any]] = []

    total_events = 0
    for item in case_summaries:
        case_id = str(item['case_id'])
        events = item['layer3']['events']
        total_events += len(events)
        case_families: Counter[str] = Counter()
        case_clusters: Counter[str] = Counter()
        case_cluster_core_signatures: dict[str, Counter[str]] = defaultdict(Counter)
        for evt in events:
            family = str(evt.get('super_family', 'UNKNOWN'))
            cluster = str(evt.get('cluster_id', 'UNKNOWN'))
            role = str(evt.get('role', 'unknown'))
            combined = f'{family}/{cluster}'
            sig_key = signature_key(evt)
            core_sig_key = core_signature_key(evt)
            tempo = str((evt.get('motion_signature') or {}).get('tempo_bucket', 'unknown'))

            family_counts[family] += 1
            cluster_counts[cluster] += 1
            family_cluster_counts[combined] += 1
            role_counts[role] += 1
            signature_counts[sig_key] += 1
            cluster_signature_counts[combined][sig_key] += 1
            cluster_core_signature_counts[combined][core_sig_key] += 1
            cluster_tempo_counts[combined][tempo] += 1
            family_case_support[family].add(case_id)
            cluster_case_support[combined].add(case_id)
            case_families[family] += 1
            case_clusters[combined] += 1
            case_cluster_core_signatures[combined][core_sig_key] += 1

        case_rows.append({
            'case_id': case_id,
            'num_frames': int(item['num_frames']),
            'layer1_count': int(item['layer1_count']),
            'layer2_count': int(item['layer2_count']),
            'layer25_count': int(item['layer25_count']),
            'layer3_count': len(events),
            'families': dict(case_families),
            'clusters': dict(case_clusters),
            'cluster_core_signatures': {
                cluster: dict(counter)
                for cluster, counter in sorted(case_cluster_core_signatures.items())
            },
        })

    family_support = {
        family: len(cases)
        for family, cases in sorted(family_case_support.items())
    }
    cluster_support = {
        cluster: len(cases)
        for cluster, cases in sorted(cluster_case_support.items())
    }
    cluster_signature_summary = {}
    for cluster, counter in cluster_signature_counts.items():
        core_counter = cluster_core_signature_counts[cluster]
        tempo_counter = cluster_tempo_counts[cluster]
        total = sum(counter.values())
        dominant_core_count = core_counter.most_common(1)[0][1] if core_counter else 0
        cluster_signature_summary[cluster] = {
            'total_events': total,
            'num_signature_variants': len(counter),
            'num_core_signature_variants': len(core_counter),
            'core_purity': float(dominant_core_count / max(total, 1)),
            'top_signatures': counter.most_common(5),
            'top_core_signatures': core_counter.most_common(5),
            'tempo_distribution': tempo_counter.most_common(),
        }

    split_candidates = []
    stable_clusters = []
    for cluster, summary in cluster_signature_summary.items():
        support = cluster_support.get(cluster, 0)
        row = {
            'cluster': cluster,
            'case_support': support,
            'total_events': summary['total_events'],
            'num_core_signature_variants': summary['num_core_signature_variants'],
            'core_purity': summary['core_purity'],
            'top_core_signatures': summary['top_core_signatures'],
            'tempo_distribution': summary['tempo_distribution'],
        }
        if support >= 20 and summary['num_core_signature_variants'] > 1 and summary['core_purity'] < 0.9:
            split_candidates.append(row)
        if support >= 20 and summary['num_core_signature_variants'] == 1:
            stable_clusters.append(row)
    split_candidates.sort(key=lambda x: (x['core_purity'], -x['case_support'], x['cluster']))
    stable_clusters.sort(key=lambda x: (-x['case_support'], x['cluster']))

    return {
        'global_stats': {
            'num_cases': len(case_summaries),
            'total_layer3_events': total_events,
            'avg_layer1_count': sum(x['layer1_count'] for x in case_summaries) / max(len(case_summaries), 1),
            'avg_layer2_count': sum(x['layer2_count'] for x in case_summaries) / max(len(case_summaries), 1),
            'avg_layer25_count': sum(x['layer25_count'] for x in case_summaries) / max(len(case_summaries), 1),
            'avg_layer3_count': total_events / max(len(case_summaries), 1),
        },
        'super_family_counts': family_counts.most_common(),
        'super_family_case_support': family_support,
        'cluster_counts': cluster_counts.most_common(),
        'family_cluster_counts': family_cluster_counts.most_common(),
        'family_cluster_case_support': cluster_support,
        'role_counts': role_counts.most_common(),
        'signature_counts': signature_counts.most_common(50),
        'cluster_signature_summary': cluster_signature_summary,
        'split_candidates': split_candidates,
        'stable_clusters': stable_clusters,
        'cases': case_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-cases', type=int, default=None)
    args = parser.parse_args()

    case_ids = load_manifest_case_ids(Path(args.manifest), max_cases=args.max_cases)
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    extracted: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for idx, case_id in enumerate(case_ids, start=1):
        item, reason = extract_case_layer3(case_id, packed)
        if item is None:
            skipped[str(reason or 'unknown')] += 1
            continue
        extracted.append(item)
        if idx % 100 == 0:
            print(f'processed {idx}/{len(case_ids)} cases')

    out = summarize_cases(extracted)
    out['run'] = {
        'manifest': str(Path(args.manifest)),
        'requested_cases': len(case_ids),
        'processed_cases': len(extracted),
        'skipped': dict(skipped),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
