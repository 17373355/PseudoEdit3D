from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D')
OUT_ROOT = ROOT / 'outputs' / 'hml3d_pattern_batches'

SUPPORTED = {
    'bounce_repeated': ('bounce_up_down', 'up and down', 'hops', 'bounce', 'jumping up and down'),
    'stair_descent': ('stair_descent', 'down some stairs', 'walks down stairs', 'walk down stairs', 'walk down the stairs', 'walks down the stairs', 'down the stairs', 'walk down the steps', 'walks down the steps', 'walk back down', 'back down the stairs', 'go down the stairs', 'downstairs', 'go down the steps'),
    'stair_ascent': ('stair_ascent', 'up some stairs', 'walks up stairs', 'walk up stairs', 'walk up the stairs', 'walks up the stairs', 'up the stairs', 'upstairs', 'climbs up a set of stairs', 'climb up the stairs', 'go up the stairs', 'steps up'),
    'walk_backward': ('walk_backward', 'walks backward', 'walk backward', 'walking backwards', 'walk back', 'walks back', 'backwards', 'opposite direction', 'returns to his original location', 'returns to their starting position', 'walk back to where', 'walks back to where', 'walk back the opposite direction'),
    'stop_pause': ('stop_pause', 'stops', 'stop', 'halts', 'halt', 'come to a stop', 'comes to a stop', 'stops dramatically', 'stop after', 'then stops'),
    'turn': ('turn_left', 'turn_right', 'turns', 'turn ', 'turn around', 'turns around', 'turns to the left', 'turns to the right', 'spin', 'spins', 'counter-clockwise', 'clockwise', 'rotates'),
    'crouch_bend': ('crouch', 'bend', 'bends', 'bending', 'squat', 'squats', 'squatting', 'stoops', 'stoop')
}


def has_semantic(text: str, cat: str) -> bool:
    text = text.lower()
    return any(token in text for token in SUPPORTED[cat])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--batch-id', required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    meta = {item['case_id']: item for item in manifest}

    good = []
    bad = []
    for item in summary:
        case_id = item['case_id']
        cats = meta[case_id]['categories']
        auto_prompt = item['auto_prompt'].lower()
        matched = [cat for cat in cats if has_semantic(auto_prompt, cat)]
        record = {
            'case_id': case_id,
            'categories': cats,
            'matched_categories': matched,
            'missing_categories': [cat for cat in cats if cat not in matched],
            'selected_hml3d_prompt': item['gt_prompt'],
            'auto_prompt': item['auto_prompt'],
        }
        if len(record['missing_categories']) == 0:
            good.append(record)
        else:
            bad.append(record)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    good_path = OUT_ROOT / f'batch_{args.batch_id}_good.jsonl'
    bad_path = OUT_ROOT / f'batch_{args.batch_id}_bad.jsonl'
    with good_path.open('w', encoding='utf-8') as f:
        for item in good:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')
    with bad_path.open('w', encoding='utf-8') as f:
        for item in bad:
            f.write(json.dumps(item, ensure_ascii=True) + '\n')
    print(good_path)
    print(bad_path)
    print(json.dumps({'good': len(good), 'bad': len(bad)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
