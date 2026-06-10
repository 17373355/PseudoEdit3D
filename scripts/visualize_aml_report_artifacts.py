from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TARGET_MOTION_KEYS = [
    'BIMANUAL_PERIODIC/BI_OUT',
    'BIMANUAL_PERIODIC/BI_UP',
    'LEFT_ARM_PERIODIC/LA_REPEAT',
    'RIGHT_ARM_PERIODIC/RA_REPEAT',
    'LEFT_ARM_PERIODIC/LA_REPEAT_LOCO',
    'RIGHT_ARM_PERIODIC/RA_REPEAT_LOCO',
    'LEFT_ARM_PERIODIC/LA_NEAR_FAR',
    'RIGHT_ARM_PERIODIC/RA_NEAR_FAR',
]
TARGET_WORD_FAMILIES = [
    'support_contact',
    'object_hold_or_manipulate',
    'arm_raise_lift',
    'arm_extend_spread',
    'arm_swing_walk',
    'wave_or_gesture',
    'clap_or_hands_together',
    'touch_body',
    'punch_boxing',
    'dance_or_circular_gesture',
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def save_barh(labels: list[str], values: list[float], path: Path, title: str, xlabel: str, color: str = '#4C78A8') -> None:
    fig_h = max(4.5, 0.38 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value, idx, f' {value:.0f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grouped_bars(labels: list[str], series: dict[str, list[float]], path: Path, title: str, ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, (name, values) in enumerate(series.items()):
        ax.bar(x + (idx - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(matrix: np.ndarray, rows: list[str], cols: list[str], path: Path, title: str, cbar_label: str) -> None:
    fig_w = max(9, 0.85 * len(cols) + 3)
    fig_h = max(5, 0.48 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=13, pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value > 0:
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7, color='black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    parser.add_argument('--cluster-scan', required=True)
    parser.add_argument('--cluster-scan-prev', default=None)
    parser.add_argument('--phrase-counts', required=True)
    parser.add_argument('--upperbody-mining', required=True)
    parser.add_argument('--vertical-report-json', default=None)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'figures'
    table_dir = out_dir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    cluster = load_json(Path(args.cluster_scan))
    cluster_prev = load_json(Path(args.cluster_scan_prev)) if args.cluster_scan_prev else None
    phrase_counts = load_json(Path(args.phrase_counts))
    mining = load_json(Path(args.upperbody_mining))
    vertical = load_json(Path(args.vertical_report_json)) if args.vertical_report_json else None

    # 1. Extraction layer capability summary.
    stats = cluster['global_stats']
    layer_labels = ['Layer1\nmicro events', 'Layer2\nsubmotion units', 'Layer2.5\nphase patterns', 'Layer3\nAML events']
    layer_values = [stats['avg_layer1_count'], stats['avg_layer2_count'], stats['avg_layer25_count'], stats['avg_layer3_count']]
    save_barh(layer_labels, layer_values, fig_dir / '01_layer_average_counts.png', 'AML Extraction Depth Per Motion', 'Average count per case', '#4C78A8')

    rows = [{
        'metric': 'processed_cases',
        'value': cluster['run']['processed_cases'],
    }, {
        'metric': 'avg_layer1_count',
        'value': stats['avg_layer1_count'],
    }, {
        'metric': 'avg_layer2_count',
        'value': stats['avg_layer2_count'],
    }, {
        'metric': 'avg_layer25_count',
        'value': stats['avg_layer25_count'],
    }, {
        'metric': 'avg_layer3_count',
        'value': stats['avg_layer3_count'],
    }, {
        'metric': 'low_event_case_count_le2',
        'value': stats['low_event_case_count_le2'],
    }]
    write_csv(table_dir / '01_layer_capability_summary.csv', rows)

    # 2. Family event counts and case support.
    family_counts = dict(cluster['super_family_counts'])
    support = cluster.get('super_family_case_support', {})
    fams = [name for name, _ in cluster['super_family_counts']]
    event_vals = [family_counts[f] for f in fams]
    support_vals = [support.get(f, 0) for f in fams]
    save_grouped_bars(fams, {'event count': event_vals, 'case support': support_vals}, fig_dir / '02_super_family_event_vs_support.png', 'AML Super-Family Distribution', 'Count')
    write_csv(table_dir / '02_super_family_distribution.csv', [
        {'super_family': f, 'event_count': family_counts[f], 'case_support': support.get(f, 0), 'case_share': support.get(f, 0) / max(cluster['run']['processed_cases'], 1)}
        for f in fams
    ])

    # 3. Top clusters.
    top_clusters = cluster['family_cluster_counts'][:25]
    save_barh([k for k, _ in top_clusters], [v for _, v in top_clusters], fig_dir / '03_top_aml_clusters.png', 'Top AML Family/Cluster Counts', 'Event count', '#F58518')
    write_csv(table_dir / '03_top_aml_clusters.csv', [
        {'family_cluster': k, 'event_count': v, 'case_support': cluster.get('family_cluster_case_support', {}).get(k, 0)}
        for k, v in top_clusters
    ])

    # 4. Mechanism progress deltas.
    progress_rows: list[dict[str, Any]] = []
    if cluster_prev is not None:
        cur = cluster['global_stats']
        prev = cluster_prev['global_stats']
        progress_rows.extend([
            {'metric': 'avg_layer3_count', 'before': prev['avg_layer3_count'], 'after': cur['avg_layer3_count'], 'delta': cur['avg_layer3_count'] - prev['avg_layer3_count']},
            {'metric': 'low_event_case_count_le2', 'before': prev['low_event_case_count_le2'], 'after': cur['low_event_case_count_le2'], 'delta': cur['low_event_case_count_le2'] - prev['low_event_case_count_le2']},
        ])
    if vertical is not None:
        v_count = vertical['rendered_vertical_counts'].get('cases_with_rendered_vertical', 0)
        progress_rows.append({'metric': 'full_problem_vertical_prompts_after_gate', 'before': '', 'after': v_count, 'delta': ''})
    pc = phrase_counts['phrase_case_counts']
    progress_rows.extend([
        {'metric': 'raw_arm_cycle_prompt_cases', 'before': '', 'after': pc.get('arm_cycle_raw', 0), 'delta': ''},
        {'metric': 'arm_swing_family_prompt_cases', 'before': '', 'after': pc.get('arm_swing_family', 0), 'delta': ''},
        {'metric': 'bimanual_coarse_prompt_cases', 'before': '', 'after': pc.get('bimanual_coarse', 0), 'delta': ''},
    ])
    write_csv(table_dir / '04_mechanism_progress_summary.csv', progress_rows)

    # 5. Prompt phrase distribution.
    phrase_labels = list(phrase_counts['phrase_case_share'].keys())
    phrase_values = [100.0 * phrase_counts['phrase_case_share'][k] for k in phrase_labels]
    save_barh(phrase_labels, phrase_values, fig_dir / '05_prompt_phrase_distribution.png', 'Prompt Phrase Groups After Current Renderer', 'Case share (%)', '#54A24B')
    write_csv(table_dir / '05_prompt_phrase_distribution.csv', [
        {'phrase_group': k, 'case_count': phrase_counts['phrase_case_counts'].get(k, 0), 'case_share': phrase_counts['phrase_case_share'][k]}
        for k in phrase_labels
    ])

    # 6. Upper-body word-family heatmap.
    reports = mining['motion_reports']
    available_keys = [k for k in TARGET_MOTION_KEYS if k in reports]
    matrix = np.zeros((len(available_keys), len(TARGET_WORD_FAMILIES)), dtype=np.float32)
    word_rows: list[dict[str, Any]] = []
    for i, key in enumerate(available_keys):
        family_map = {row['word_family']: row for row in reports[key]['top_word_families']}
        for j, fam in enumerate(TARGET_WORD_FAMILIES):
            row = family_map.get(fam)
            if row:
                matrix[i, j] = float(row['coverage'])
                word_rows.append({
                    'motion_key': key,
                    'word_family': fam,
                    'support': row['support'],
                    'coverage': row['coverage'],
                    'precision': row['precision'],
                    'lift': row['lift'],
                    'score': row['score'],
                })
    save_heatmap(matrix, available_keys, TARGET_WORD_FAMILIES, fig_dir / '06_upperbody_motion_word_family_heatmap.png', 'Motion Cluster -> HML3D Word-Family Coverage', 'Coverage within motion cluster')
    write_csv(table_dir / '06_upperbody_motion_word_family.csv', word_rows)

    # 7. Top phrase table for target motion clusters.
    phrase_rows: list[dict[str, Any]] = []
    for key in available_keys:
        for row in reports[key].get('top_phrases', [])[:12]:
            phrase_rows.append({
                'motion_key': key,
                'phrase': row['phrase'],
                'support': row['support'],
                'coverage': row['coverage'],
                'precision': row['precision'],
                'lift': row['lift'],
                'score': row['score'],
            })
    write_csv(table_dir / '07_upperbody_top_phrases.csv', phrase_rows)

    # 8. Markdown report index.
    md = []
    md.append('# AML Report Artifacts')
    md.append('')
    md.append('## Figures')
    md.append('')
    for fig in sorted(fig_dir.glob('*.png')):
        md.append(f'- `{fig.relative_to(out_dir)}`')
    md.append('')
    md.append('## Tables')
    md.append('')
    for table in sorted(table_dir.glob('*.csv')):
        md.append(f'- `{table.relative_to(out_dir)}`')
    md.append('')
    md.append('## Key Readout')
    md.append('')
    md.append(f"- processed cases: `{cluster['run']['processed_cases']}`")
    md.append(f"- avg Layer3 events: `{stats['avg_layer3_count']:.3f}`")
    md.append(f"- low-event cases <=2: `{stats['low_event_case_count_le2']}`")
    md.append(f"- raw arm-cycle prompt cases: `{pc.get('arm_cycle_raw', 0)}`")
    md.append(f"- bimanual coarse prompt cases: `{pc.get('bimanual_coarse', 0)}`")
    if vertical is not None:
        md.append(f"- problematic vertical prompt cases after gate: `{vertical['rendered_vertical_counts'].get('cases_with_rendered_vertical', 0)}`")
    md.append('')
    md.append('## Upper-Body Mining Note')
    md.append('')
    md.append('- HML3D captions are used as a global wording inventory for cluster naming/reference only, not as per-case auto-prompt input.')
    md.append('- The heatmap should be read as candidate semantic family evidence, not direct label assignment.')
    (out_dir / 'README.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'saved_report_artifacts={out_dir}')


if __name__ == '__main__':
    main()
