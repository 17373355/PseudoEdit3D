from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    arr = json.loads(Path(args.input).read_text(encoding='utf-8'))
    summary = []
    for item in arr:
        layer0_count = len(item['layer0'])
        layer1_count = len(item['layer1_micro_events'])
        layer2_count = len(item['layer2_submotions'])
        layer25_count = len(item['layer25_phase_patterns'])
        summary.append({
            'case_id': item['case_id'],
            'num_frames': item['num_frames'],
            'layer0_count': layer0_count,
            'layer1_count': layer1_count,
            'layer2_count': layer2_count,
            'layer25_count': layer25_count,
            'layer2_over_layer1': float(layer2_count / max(layer1_count, 1)),
            'layer25_over_layer2': float(layer25_count / max(layer2_count, 1)),
            'phase_names': [p['name'] for p in item['layer25_phase_patterns'][:20]],
        })

    global_stats = {
        'num_cases': len(summary),
        'avg_layer1_count': sum(x['layer1_count'] for x in summary) / max(len(summary), 1),
        'avg_layer2_count': sum(x['layer2_count'] for x in summary) / max(len(summary), 1),
        'avg_layer25_count': sum(x['layer25_count'] for x in summary) / max(len(summary), 1),
        'avg_layer2_over_layer1': sum(x['layer2_over_layer1'] for x in summary) / max(len(summary), 1),
        'avg_layer25_over_layer2': sum(x['layer25_over_layer2'] for x in summary) / max(len(summary), 1),
    }

    out = {
        'global_stats': global_stats,
        'cases': summary,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
