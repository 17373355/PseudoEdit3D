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

BIMANUAL_PREFIX = 'BIMANUAL_PERIODIC/'
BIMANUAL_ORDER = [
    'BIMANUAL_PERIODIC/BI_SPREAD',
    'BIMANUAL_PERIODIC/BI_RAISE_SPREAD',
    'BIMANUAL_PERIODIC/BI_RAISE',
    'BIMANUAL_PERIODIC/BI_HANDS_CLOSE',
    'BIMANUAL_PERIODIC/BI_HANDS_CLOSE_RAISE',
    'BIMANUAL_PERIODIC/BI_EXTENDED_LOCO_COUPLED',
    'BIMANUAL_PERIODIC/BI_LOCOMOTION_COUPLED',
    'BIMANUAL_PERIODIC/BI_VERTICAL_COUPLED',
    'BIMANUAL_PERIODIC/BI_UNRESOLVED',
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_barh(labels: list[str], values: list[float], path: Path, title: str, xlabel: str, color: str) -> None:
    fig_h = max(4.8, 0.42 * len(labels) + 1.4)
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


def save_heatmap(matrix: np.ndarray, rows: list[str], cols: list[str], path: Path, title: str) -> None:
    fig_w = max(8.5, 0.82 * len(cols) + 3)
    fig_h = max(4.8, 0.48 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title(title, fontsize=13, pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('share')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-scan', required=True)
    parser.add_argument('--diagnostic-split', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'figures'
    table_dir = out_dir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    cluster = load_json(Path(args.cluster_scan))
    diagnostic = load_json(Path(args.diagnostic_split))
    event_counts = dict(cluster.get('family_cluster_counts') or [])
    support_counts = cluster.get('family_cluster_case_support') or {}
    processed = int(cluster['run']['processed_cases'])

    present = [
        key for key in BIMANUAL_ORDER
        if key in event_counts or key in support_counts
    ]
    extra = sorted(
        key for key in event_counts
        if key.startswith(BIMANUAL_PREFIX) and key not in present
    )
    keys = present + extra
    labels = [k.replace(BIMANUAL_PREFIX, '') for k in keys]
    event_values = [int(event_counts.get(k, 0)) for k in keys]
    support_values = [int(support_counts.get(k, 0)) for k in keys]

    save_barh(labels, event_values, fig_dir / '01_full_bimanual_event_counts.png', 'Full HumanML3D Bimanual Split Event Counts', 'event count', '#F58518')
    save_barh(labels, support_values, fig_dir / '02_full_bimanual_case_support.png', 'Full HumanML3D Bimanual Split Case Support', 'case support', '#4C78A8')

    rows = []
    total_bi_events = max(sum(event_values), 1)
    for key, label, event_count, case_count in zip(keys, labels, event_values, support_values):
        rows.append({
            'family_cluster': key,
            'cluster': label,
            'event_count': event_count,
            'event_share_within_bimanual': event_count / total_bi_events,
            'case_support': case_count,
            'case_share': case_count / max(processed, 1),
        })
    write_csv(table_dir / '01_full_bimanual_split_counts.csv', rows)

    diag_pairs = diagnostic.get('original_to_split_counts') or {}
    original_labels = sorted(diag_pairs)
    split_labels = [row[0] for row in diagnostic.get('split_counts') or []]
    mat = np.zeros((len(original_labels), len(split_labels)), dtype=np.float32)
    for i, original in enumerate(original_labels):
        for split, count in diag_pairs[original]:
            if split in split_labels:
                mat[i, split_labels.index(split)] = float(count)
    norm = mat / np.clip(mat.sum(axis=1, keepdims=True), 1.0, None)
    save_heatmap(norm, original_labels, split_labels, fig_dir / '03_diagnostic_original_to_split_heatmap.png', 'Diagnostic Coarse BI_OUT/BI_UP -> Split Matrix')
    write_csv(table_dir / '02_diagnostic_original_to_split.csv', [
        {
            'original_cluster': original,
            'split_cluster': split,
            'event_count': int(mat[i, j]),
            'share_within_original': float(norm[i, j]),
        }
        for i, original in enumerate(original_labels)
        for j, split in enumerate(split_labels)
        if mat[i, j] > 0
    ])

    feature_rows = []
    for split, summary in (diagnostic.get('feature_summary_by_split') or {}).items():
        feature_rows.append({
            'split_cluster': split,
            'mean_hand_distance_norm_mean': summary.get('mean_hand_distance_norm.mean'),
            'mean_wrist_height_rel_norm_mean': summary.get('mean_wrist_height_rel_norm.mean'),
            'mean_wrist_chest_distance_norm_mean': summary.get('mean_wrist_chest_distance_norm.mean'),
            'mean_wrist_speed_norm_mean': summary.get('mean_wrist_speed_norm.mean'),
            'max_loco_overlap_mean': summary.get('max_loco_overlap.mean'),
            'left_right_chest_corr_mean': summary.get('left_right_chest_corr.mean'),
        })
    write_csv(table_dir / '03_diagnostic_feature_summary.csv', feature_rows)

    readme = [
        '# Bimanual Split Report v1',
        '',
        '## Inputs',
        '',
        f'- full cluster scan: `{args.cluster_scan}`',
        f'- diagnostic split: `{args.diagnostic_split}`',
        '',
        '## Main Figures',
        '',
        '- `figures/01_full_bimanual_event_counts.png`: full HumanML3D event count per bimanual split cluster.',
        '- `figures/02_full_bimanual_case_support.png`: full HumanML3D case support per bimanual split cluster.',
        '- `figures/03_diagnostic_original_to_split_heatmap.png`: how old coarse BI_OUT/BI_UP events split into the new subclasses.',
        '',
        '## Interpretation',
        '',
        '- The current split separates coarse bimanual events into spread, raise+spread, raise, hands-close, and hands-close-raise groups.',
        '- Object/support/clap names are intentionally not used yet because these require stronger evidence than joints-only motion proxies.',
        '- The next acceptance check should verify whether each split cluster has visually coherent examples and whether any split remains semantically overloaded.',
    ]
    (out_dir / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print(f'saved={out_dir}')


if __name__ == '__main__':
    main()
