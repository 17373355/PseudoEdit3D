from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYWORD_GROUPS = {
    'vertical': ['jump', 'hop', 'bounce', 'squat', 'crouch', 'leap', 'stairs'],
    'locomotion': ['walk', 'run', 'step', 'turn', 'back', 'forward'],
    'arm': ['arm', 'hand', 'clap', 'wave', 'knock', 'tap', 'rope'],
    'support': ['rail', 'wall', 'beam', 'hold', 'support'],
}


def caption_tags(prompt: str | None) -> list[str]:
    text = (prompt or '').lower()
    tags = []
    for tag, words in KEYWORD_GROUPS.items():
        if any(word in text for word in words):
            tags.append(tag)
    return tags or ['other']


def canonical_case_id(case_id: str) -> str:
    return case_id[1:] if case_id.startswith('M') else case_id


def sort_examples(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == 'top_count':
        return sorted(rows, key=lambda x: (-x['cluster_event_count'], x['layer3_count'], x['case_id']))
    # Prefer cases where this cluster is dominant but the whole Layer 3 is not too dense.
    return sorted(rows, key=lambda x: (-x['dominance_score'], x['layer3_count'], -x['cluster_event_count'], x['case_id']))


def pick_diverse(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    picked = []
    seen_base = set()
    seen_tags = set()
    for row in rows:
        base = canonical_case_id(row['case_id'])
        tags = tuple(row.get('caption_tags') or ['other'])
        if base in seen_base:
            continue
        if tags in seen_tags and len(picked) < max(1, top_k // 2):
            continue
        picked.append(row)
        seen_base.add(base)
        seen_tags.add(tags)
        if len(picked) >= top_k:
            return picked
    for row in rows:
        base = canonical_case_id(row['case_id'])
        if base in seen_base:
            continue
        picked.append(row)
        seen_base.add(base)
        if len(picked) >= top_k:
            break
    return picked


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            out[str(item['case_id'])] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--min-support', type=int, default=20)
    parser.add_argument('--mode', choices=['top_count', 'diverse'], default='diverse')
    args = parser.parse_args()

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding='utf-8'))
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)

    cases = taxonomy.get('cases', [])
    stable_clusters = [
        row for row in taxonomy.get('stable_clusters', [])
        if int(row.get('case_support', 0)) >= args.min_support
    ]

    clusters = []
    all_case_ids: list[str] = []
    for cluster_row in stable_clusters:
        cluster = str(cluster_row['cluster'])
        matched = []
        for case in cases:
            count = int((case.get('clusters') or {}).get(cluster, 0))
            if count <= 0:
                continue
            case_id = str(case['case_id'])
            meta = manifest.get(case_id, {})
            layer3_count = int(case.get('layer3_count', 0))
            prompt = meta.get('selected_hml3d_prompt')
            matched.append({
                'case_id': case_id,
                'cluster_event_count': count,
                'dominance_score': float(count / max(layer3_count, 1)),
                'num_frames': int(case.get('num_frames', 0)),
                'layer3_count': layer3_count,
                'caption_tags': caption_tags(prompt),
                'selected_hml3d_prompt': prompt,
            })
        matched = sort_examples(matched, args.mode)
        examples = pick_diverse(matched, args.top_k) if args.mode == 'diverse' else matched[: args.top_k]
        all_case_ids.extend(x['case_id'] for x in examples)
        clusters.append({
            'cluster': cluster,
            'case_support': int(cluster_row.get('case_support', 0)),
            'total_events': int(cluster_row.get('total_events', 0)),
            'core_purity': float(cluster_row.get('core_purity', 0.0)),
            'top_core_signatures': cluster_row.get('top_core_signatures', []),
            'tempo_distribution': cluster_row.get('tempo_distribution', []),
            'examples': examples,
        })

    unique_case_ids = []
    seen = set()
    for case_id in all_case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        unique_case_ids.append(case_id)

    out = {
        'source_taxonomy': str(Path(args.taxonomy)),
        'source_manifest': str(Path(args.manifest)) if args.manifest else None,
        'top_k': int(args.top_k),
        'min_support': int(args.min_support),
        'mode': args.mode,
        'generalization_notes': [
            'representatives are for human inspection, not for defining the taxonomy',
            'diverse mode ranks by cluster dominance and removes mirror duplicates before filling examples',
            'tempo remains a control field and should not dominate representative selection',
        ],
        'num_clusters': len(clusters),
        'num_unique_cases': len(unique_case_ids),
        'unique_case_ids': unique_case_ids,
        'clusters': clusters,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
