from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def top_missing(items):
    counts = {}
    for item in items:
        for cat in item.get('missing_categories', []):
            counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triage-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    triage = Path(args.triage_dir)
    good = load_jsonl(triage / 'good.jsonl')
    soft_bad = load_jsonl(triage / 'soft_bad.jsonl')
    hard_bad = load_jsonl(triage / 'hard_bad.jsonl')

    report = {
        'counts': {
            'good': len(good),
            'soft_bad': len(soft_bad),
            'hard_bad': len(hard_bad),
            'total': len(good) + len(soft_bad) + len(hard_bad),
        },
        'soft_bad_top_missing': top_missing(soft_bad),
        'hard_bad_top_missing': top_missing(hard_bad),
        'examples': {
            'soft_bad': soft_bad[:10],
            'hard_bad': hard_bad[:10],
        },
    }
    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out)
    print(json.dumps(report['counts'], ensure_ascii=True))


if __name__ == '__main__':
    main()
