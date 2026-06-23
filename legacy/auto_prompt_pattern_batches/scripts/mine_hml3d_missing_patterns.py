from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D')
RUNNER = ROOT / 'scripts' / 'run_momask_case_study.py'

STOPWORDS = {
    'a','an','the','and','or','to','of','in','on','at','with','their','his','her','its','is','are','was','were',
    'then','while','from','into','out','up','down','left','right','forward','backward','backwards','person','man',
    'someone','figure','character','slowly','slightly','quickly','very','some','few'
}


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        chunk = tokens[i:i+n]
        if any(tok in STOPWORDS for tok in chunk):
            continue
        yield ' '.join(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bad-jsonl', required=True)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    mod = load_runner_module()
    covered_phrases = set()
    for vals in mod.CAPTION_PATTERN_GROUPS.values():
        for phrase in vals:
            covered_phrases.add(normalize(phrase))

    items = [json.loads(line) for line in Path(args.bad_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    phrase_counts = Counter()
    phrase_to_cases: dict[str, set[str]] = defaultdict(set)
    phrase_to_missing: dict[str, Counter] = defaultdict(Counter)

    for item in items:
        case_id = item['case_id']
        missing = item['missing_categories']
        raw_captions = [item['selected_hml3d_prompt']]
        if item.get('raw_prompt_segments'):
            raw_captions = [seg[0] for seg in item['raw_prompt_segments']]
        for caption in raw_captions:
            tokens = normalize(caption).split()
            for n in (2, 3):
                for phrase in ngrams(tokens, n):
                    if phrase in covered_phrases:
                        continue
                    phrase_counts[phrase] += 1
                    phrase_to_cases[phrase].add(case_id)
                    for cat in missing:
                        phrase_to_missing[phrase][cat] += 1

    out = []
    for phrase, count in phrase_counts.most_common(args.top_k):
        out.append({
            'phrase': phrase,
            'count': count,
            'num_cases': len(phrase_to_cases[phrase]),
            'case_ids': sorted(phrase_to_cases[phrase])[:20],
            'missing_categories': dict(phrase_to_missing[phrase].most_common()),
            'source': 'raw_hml3d_captions_only',
        })

    Path(args.output).write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
