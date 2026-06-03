from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LEXICON = {
    'turn': [r'\bturn\b', r'\bspin\b', r'\brotate\b', r'\btwist\b', r'\blook around\b'],
    'jump': [r'\bjump\b', r'\bleap\b', r'\bhop\b', r'\bbounce\b'],
    'land': [r'\bland\b', r'\bcome down\b', r'\btouch down\b'],
    'crouch': [r'\bcrouch\b', r'\bsquat\b', r'\bkneel\b', r'\bduck\b'],
    'bend': [r'\bbend\b', r'\bcurl\b', r'\bflex\b', r'\blean\b', r'\bstoop\b'],
    'reach': [r'\breach\b', r'\bgrab\b', r'\bpick up\b', r'\blift\b', r'\braise\b'],
    'walk_run': [r'\bwalk\b', r'\brun\b', r'\bstep\b', r'\bstride\b', r'\bshuffle\b'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--texts-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts_dir = Path(args.texts_dir)
    files = sorted(texts_dir.glob('*.txt'))
    if args.limit > 0:
        files = files[:args.limit]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        for line in lines:
            caption = line.split('#')[0].strip()
            lower = caption.lower()
            for key, patterns in LEXICON.items():
                if any(re.search(p, lower) for p in patterns):
                    buckets[key].append({
                        'id': path.stem,
                        'caption': caption,
                    })

    summary = {
        'texts_dir': str(texts_dir),
        'num_files': len(files),
        'lexicon': LEXICON,
        'counts': {k: len(v) for k, v in buckets.items()},
        'examples': {k: v[:20] for k, v in buckets.items()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(summary['counts'], ensure_ascii=True, indent=2))
    print(f'saved_lexicon={out}')


if __name__ == '__main__':
    main()
