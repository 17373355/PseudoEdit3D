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


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (
            -float(x['signature_dominance']),
            int(x['layer3_count']),
            -int(x['signature_event_count']),
            str(x['case_id']),
        ),
    )


def pick_diverse(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen_base: set[str] = set()
    seen_tags: set[tuple[str, ...]] = set()
    for row in rows:
        base = canonical_case_id(str(row['case_id']))
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
        base = canonical_case_id(str(row['case_id']))
        if base in seen_base:
            continue
        picked.append(row)
        seen_base.add(base)
        if len(picked) >= top_k:
            break
    return picked


def select_examples(
    taxonomy: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    cluster: str,
    signature: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for case in taxonomy.get('cases', []):
        sig_counts = ((case.get('cluster_core_signatures') or {}).get(cluster) or {})
        sig_count = int(sig_counts.get(signature, 0))
        if sig_count <= 0:
            continue
        cluster_count = int((case.get('clusters') or {}).get(cluster, 0))
        layer3_count = int(case.get('layer3_count', 0))
        case_id = str(case['case_id'])
        meta = manifest.get(case_id, {})
        prompt = meta.get('selected_hml3d_prompt')
        rows.append({
            'case_id': case_id,
            'signature_event_count': sig_count,
            'cluster_event_count': cluster_count,
            'signature_dominance': float(sig_count / max(cluster_count, 1)),
            'cluster_dominance': float(cluster_count / max(layer3_count, 1)),
            'num_frames': int(case.get('num_frames', 0)),
            'layer3_count': layer3_count,
            'caption_tags': caption_tags(prompt),
            'selected_hml3d_prompt': prompt,
        })
    return pick_diverse(sort_rows(rows), top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--min-support', type=int, default=20)
    args = parser.parse_args()

    taxonomy_path = Path(args.taxonomy)
    taxonomy = json.loads(taxonomy_path.read_text(encoding='utf-8'))
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)

    clusters = []
    all_case_ids: list[str] = []
    for cand in taxonomy.get('split_candidates', []):
        if int(cand.get('case_support', 0)) < args.min_support:
            continue
        cluster = str(cand['cluster'])
        signatures = []
        for sig, count in cand.get('top_core_signatures', []):
            examples = select_examples(taxonomy, manifest, cluster, str(sig), args.top_k)
            all_case_ids.extend(x['case_id'] for x in examples)
            signatures.append({
                'core_signature': str(sig),
                'event_count': int(count),
                'examples': examples,
            })
        clusters.append({
            'cluster': cluster,
            'case_support': int(cand.get('case_support', 0)),
            'total_events': int(cand.get('total_events', 0)),
            'core_purity': float(cand.get('core_purity', 0.0)),
            'num_core_signature_variants': int(cand.get('num_core_signature_variants', 0)),
            'tempo_distribution': cand.get('tempo_distribution', []),
            'signatures': signatures,
        })

    unique_case_ids = []
    seen = set()
    for case_id in all_case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        unique_case_ids.append(case_id)

    out = {
        'source_taxonomy': str(taxonomy_path),
        'source_manifest': str(Path(args.manifest)) if args.manifest else None,
        'top_k': int(args.top_k),
        'min_support': int(args.min_support),
        'generalization_notes': [
            'split-candidate representatives are selected from core signatures, not caption names',
            'cases are for diagnosis and visualization; taxonomy decisions still require full-corpus support and purity',
            'signature dominance favors cases where the candidate signature is easy to inspect',
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
