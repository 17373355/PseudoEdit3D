from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.constants import BODY_PART_TO_JOINTS
from pseudoedit3d.edit.bimanual_split import bimanual_span_features
from pseudoedit3d.visualization.skeleton_gif import (
    _draw_skeleton,
    _load_font,
    _normalize_points,
    _part_edge_indices,
    _project_points,
    _wrap_text,
)

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')
BIMANUAL_CLUSTERS = [
    'BIMANUAL_PERIODIC/BI_SPREAD',
    'BIMANUAL_PERIODIC/BI_RAISE_SPREAD',
    'BIMANUAL_PERIODIC/BI_RAISE',
    'BIMANUAL_PERIODIC/BI_HANDS_CLOSE',
    'BIMANUAL_PERIODIC/BI_HANDS_CLOSE_RAISE',
]
BIMANUAL_COLOR = (175, 95, 210)
CLUSTER_TEXT = {
    'BI_SPREAD': 'moves both hands outward',
    'BI_RAISE_SPREAD': 'raises and spreads both arms',
    'BI_RAISE': 'raises both arms',
    'BI_HANDS_CLOSE': 'brings both hands closer together',
    'BI_HANDS_CLOSE_RAISE': 'brings both hands closer while raising them',
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def read_first_prompt(case_id: str) -> str:
    text_path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not text_path.exists():
        return ''
    for line in text_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            return line.split('#')[0].strip()
    return ''


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, max_width: int, fill: tuple[int, int, int], max_lines: int) -> int:
    x, y = xy
    for line in _wrap_text(draw, text, font, max_width)[:max_lines]:
        draw.text((x, y), line, fill=fill, font=font)
        y += 16
    return y


def draw_pose_panel(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    frame_idx: int,
    box: tuple[int, int, int, int],
    label: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, outline=(218, 218, 226), width=1, fill=(255, 255, 255))
    pts = projected[frame_idx].copy()
    panel_w = x1 - x0
    panel_h = y1 - y0
    pts[:, 0] = pts[:, 0] - panel_w / 2.0 + (x0 + x1) / 2.0
    pts[:, 1] = pts[:, 1] - panel_h / 2.0 + (y0 + y1) / 2.0 + 8
    highlight_joints = set(BODY_PART_TO_JOINTS['both_arms'])
    highlight_edges = _part_edge_indices('both_arms')
    _draw_skeleton(
        draw,
        pts,
        base_color=(176, 181, 193),
        highlight_color=BIMANUAL_COLOR,
        highlight_joints=highlight_joints,
        highlight_edges=highlight_edges,
        radius=3,
        width=3,
    )
    font = _load_font(12)
    draw.text((x0 + 8, y0 + 6), f'{label}: frame {frame_idx + 1}', fill=(70, 70, 82), font=font)


def pseudo_event(cluster_key: str, span: list[int]) -> dict[str, Any]:
    family, cluster = cluster_key.split('/', 1)
    return {
        'super_family': family,
        'cluster_id': cluster,
        'part': 'both_arms',
        'start_frame': int(span[0]),
        'end_frame': int(span[1]),
        'direction': cluster.lower(),
    }


def feature_text(cluster_key: str, feat: dict[str, Any]) -> list[tuple[str, str]]:
    keys = [
        ('hand_dist', 'mean_hand_distance_norm'),
        ('delta_hand', 'hand_distance_delta_norm'),
        ('height', 'mean_wrist_height_rel_norm'),
        ('max_height', 'max_wrist_height_rel_norm'),
        ('chest_dist', 'mean_wrist_chest_distance_norm'),
        ('speed', 'mean_wrist_speed_norm'),
        ('loco_overlap', 'max_loco_overlap'),
    ]
    rows = []
    for label, key in keys:
        value = feat.get(key)
        rows.append((label, f'{float(value):.2f}' if value is not None else 'n/a'))
    rows.append(('cluster', cluster_key.split('/', 1)[1]))
    return rows


def render_cluster_sheet(
    cluster_key: str,
    examples: list[dict[str, Any]],
    packed: dict[str, Any],
    output_path: Path,
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = examples[:max_examples]
    row_h = 250
    header_h = 78
    canvas_w = 1680
    canvas_h = header_h + row_h * max(1, len(samples))
    img = Image.new('RGB', (canvas_w, canvas_h), color=(247, 247, 250))
    draw = ImageDraw.Draw(img)
    font_title = _load_font(24)
    font_body = _load_font(15)
    font_small = _load_font(12)

    cluster = cluster_key.split('/', 1)[1]
    draw.text((28, 20), f'Bimanual Cluster Contact Sheet: {cluster_key}', fill=(18, 18, 22), font=font_title)
    draw.text((28, 50), 'Rows show before / middle / after frames around the event span; features are computed directly from joints.', fill=(74, 74, 86), font=font_body)

    for row_idx, example in enumerate(samples):
        y0 = header_h + row_idx * row_h
        draw.rounded_rectangle((18, y0 + 8, canvas_w - 18, y0 + row_h - 8), radius=14, outline=(222, 222, 230), width=1, fill=(255, 255, 255))
        case_id = str(example['case_id'])
        span = [int(example['span'][0]), int(example['span'][1])]
        key = f'{case_id}.npy'
        if key not in packed:
            continue
        joints = packed[key]['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        event = pseudo_event(cluster_key, span)
        feat = bimanual_span_features(joints, event, [event])
        start, end = span
        mid = (start + end) // 2
        frame_ids = [max(0, start - 5), mid, min(len(joints) - 1, end + 5)]
        projected = _normalize_points(_project_points(joints), width=205, height=175, margin=18)

        text_x = 32
        text_y = y0 + 22
        draw.text((text_x, text_y), f'{row_idx + 1}. case {case_id}  span {start}-{end}', fill=(24, 24, 30), font=font_body)
        text_y += 22
        label = CLUSTER_TEXT.get(cluster, cluster)
        text_y = draw_wrapped(draw, (text_x, text_y), f'AML split label: {label}', font_small, 390, (42, 42, 52), 2)
        text_y += 4
        prompt = read_first_prompt(case_id)
        draw_wrapped(draw, (text_x, text_y), f'HML3D: {prompt}', font_small, 390, (70, 105, 190), 5)

        panel_x = 450
        panel_y = y0 + 38
        panel_w = 220
        panel_h = 176
        for label_name, frame_idx in zip(['before', 'middle', 'after'], frame_ids):
            draw_pose_panel(draw, projected, frame_idx, (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), label_name)
            panel_x += panel_w + 18

        feat_x = 1170
        feat_y = y0 + 28
        draw.text((feat_x, feat_y), 'Joint-Derived Split Features', fill=(24, 24, 30), font=font_body)
        feat_y += 24
        for name, value in feature_text(cluster_key, feat):
            draw.text((feat_x, feat_y), f'{name}: {value}', fill=(58, 58, 70), font=font_small)
            feat_y += 17

        rows.append({
            'cluster_key': cluster_key,
            'case_id': case_id,
            'span_start': start,
            'span_end': end,
            'event_label': label,
            'prompt': prompt,
            'mean_hand_distance_norm': feat.get('mean_hand_distance_norm'),
            'hand_distance_delta_norm': feat.get('hand_distance_delta_norm'),
            'mean_wrist_height_rel_norm': feat.get('mean_wrist_height_rel_norm'),
            'max_wrist_height_rel_norm': feat.get('max_wrist_height_rel_norm'),
            'mean_wrist_chest_distance_norm': feat.get('mean_wrist_chest_distance_norm'),
            'mean_wrist_speed_norm': feat.get('mean_wrist_speed_norm'),
            'max_loco_overlap': feat.get('max_loco_overlap'),
            'image_path': str(output_path),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return rows


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
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples-per-cluster', type=int, default=6)
    parser.add_argument('--clusters', default=','.join(BIMANUAL_CLUSTERS))
    args = parser.parse_args()

    cluster_scan = load_json(Path(args.cluster_scan))
    cluster_examples = cluster_scan.get('cluster_examples') or {}
    clusters = [x.strip() for x in args.clusters.split(',') if x.strip()]
    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'contact_sheets'
    all_rows: list[dict[str, Any]] = []
    for cluster_key in clusters:
        examples = list(cluster_examples.get(cluster_key) or [])
        if not examples:
            print(f'skip_no_examples={cluster_key}')
            continue
        short_name = cluster_key.replace('/', '__')
        output_path = fig_dir / f'{short_name}.png'
        rows = render_cluster_sheet(cluster_key, examples, packed, output_path, max_examples=args.samples_per_cluster)
        all_rows.extend(rows)
        print(f'saved_contact_sheet={output_path}')

    write_csv(out_dir / 'bimanual_contact_sheet_index.csv', all_rows)
    readme = [
        '# Bimanual Cluster Contact Sheets',
        '',
        f'- input cluster scan: `{args.cluster_scan}`',
        f'- samples per cluster: `{args.samples_per_cluster}`',
        '- each row shows event-span before / middle / after frames and joints-derived split features.',
        '- this is a lightweight visual check; it does not rerun the full AML extractor for each row.',
        '',
        '## Files',
        '',
    ]
    for cluster_key in clusters:
        short_name = cluster_key.replace('/', '__')
        readme.append(f'- `{fig_dir.name}/{short_name}.png`')
    readme.extend([
        '',
        '## Acceptance Use',
        '',
        '- Check whether samples inside each cluster share the same visible bimanual pattern.',
        '- Check whether different clusters are visually separable in hand distance, wrist height, and timing.',
        '- Mark clusters for another split if their rows show multiple incompatible action families.',
    ])
    (out_dir / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print(f'saved_index={out_dir / "bimanual_contact_sheet_index.csv"}')
    print(f'saved_readme={out_dir / "README.md"}')


if __name__ == '__main__':
    main()
