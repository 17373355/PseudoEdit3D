from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.constants import JOINT_INDEX

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'


def _load_summary(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def _find_joint_file(ext: str) -> Path:
    joint_dir = MOMASK_ROOT / 'generation' / ext / 'joints' / '0'
    matches = sorted(p for p in joint_dir.glob('sample0_repeat0_len*.npy') if '_ik' not in p.name)
    if not matches:
        raise FileNotFoundError(f'No generated joint file found for ext={ext}')
    return matches[0]


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) == 0:
        raise ValueError('Cannot pad empty motion')
    pad = np.repeat(arr[-1:], target_len - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def _path_length(xz: np.ndarray) -> float:
    if len(xz) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(xz[1:] - xz[:-1], axis=-1)))


def _metrics(joints: np.ndarray) -> dict[str, float]:
    root = joints[:, JOINT_INDEX['Pelvis']]
    l_wrist = joints[:, JOINT_INDEX['L_Wrist']]
    r_wrist = joints[:, JOINT_INDEX['R_Wrist']]
    hands_dist = np.linalg.norm(l_wrist - r_wrist, axis=-1)
    xz = root[:, [0, 2]]
    net = float(np.linalg.norm(xz[-1] - xz[0])) if len(xz) else 0.0
    path = _path_length(xz)
    vertical_amp = float(np.max(root[:, 1]) - np.min(root[:, 1])) if len(root) else 0.0
    return {
        'root_path_xz_m': path,
        'root_net_xz_m': net,
        'root_vertical_amp_m': vertical_amp,
        'mean_speed_xz_mpf': path / max(1, len(joints) - 1),
        'hand_distance_mean_m': float(np.mean(hands_dist)) if len(hands_dist) else 0.0,
        'hand_distance_amp_m': float(np.max(hands_dist) - np.min(hands_dist)) if len(hands_dist) else 0.0,
    }


def _ratio(pred: float, gt: float) -> float | None:
    if abs(gt) < 1e-6:
        return None
    return pred / gt


def _flag(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    gt = row['gt_metrics']
    pred = row['pred_metrics']
    path_ratio = _ratio(pred['root_path_xz_m'], gt['root_path_xz_m'])
    vert_ratio = _ratio(pred['root_vertical_amp_m'], gt['root_vertical_amp_m'])
    if path_ratio is not None and (path_ratio < 0.35 or path_ratio > 2.8):
        flags.append('root_path_mismatch')
    elif path_ratio is not None and (path_ratio < 0.60 or path_ratio > 1.50):
        flags.append('root_path_scale_review')
    if vert_ratio is not None and (vert_ratio < 0.35 or vert_ratio > 2.8):
        flags.append('vertical_amp_mismatch')
    elif vert_ratio is not None and (vert_ratio < 0.55 or vert_ratio > 2.20):
        flags.append('vertical_amp_scale_review')
    if gt['root_path_xz_m'] < 0.35 and pred['root_path_xz_m'] > 1.2:
        flags.append('unexpected_translation')
    if gt['root_vertical_amp_m'] < 0.12 and pred['root_vertical_amp_m'] > 0.35:
        flags.append('unexpected_jumpiness')
    if row['length_delta_frames'] > 8:
        flags.append('length_delta')
    return flags


def analyze(summary_path: Path) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    pack = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')
    rows: list[dict[str, Any]] = []
    for case in summary:
        case_id = str(case['case_id'])
        source_len = int(case.get('source_num_frames') or 0)
        gt = _as_numpy(pack[f'{case_id}.npy']['joints3d'])
        if source_len > 0:
            gt = gt[:source_len]
        pred = _as_numpy(np.load(_find_joint_file(str(case['auto_ext']))))
        compare_len = min(len(gt), len(pred))
        pred_for_compare = _pad_or_trim(pred, len(gt))
        row = {
            'case_id': case_id,
            'auto_prompt': case.get('auto_prompt', ''),
            'canonical_ids': [a.get('canonical_id') for a in case.get('canonical_actions') or []],
            'gt_len': int(len(gt)),
            'pred_len': int(len(pred)),
            'compare_len': int(compare_len),
            'length_delta_frames': int(abs(len(gt) - len(pred))),
            'gt_metrics': _metrics(gt),
            'pred_metrics': _metrics(pred_for_compare),
        }
        row['ratios'] = {
            'root_path_xz': _ratio(row['pred_metrics']['root_path_xz_m'], row['gt_metrics']['root_path_xz_m']),
            'root_net_xz': _ratio(row['pred_metrics']['root_net_xz_m'], row['gt_metrics']['root_net_xz_m']),
            'root_vertical_amp': _ratio(row['pred_metrics']['root_vertical_amp_m'], row['gt_metrics']['root_vertical_amp_m']),
            'hand_distance_amp': _ratio(row['pred_metrics']['hand_distance_amp_m'], row['gt_metrics']['hand_distance_amp_m']),
        }
        row['flags'] = _flag(row)
        rows.append(row)
    return {
        'run': {
            'summary': str(summary_path),
            'num_cases': len(rows),
            'note': 'Coarse kinematic sanity check only; not a perceptual or FID metric.',
        },
        'rows': rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_report(result: dict[str, Any], report_path: Path) -> None:
    lines = ['# MoMask Probe Kinematic Sanity Report', '']
    lines.append('This is a rough regression check, not a final evaluation metric.')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f"- cases: `{result['run']['num_cases']}`")
    lines.append(f"- source summary: `{result['run']['summary']}`")
    lines.append('')
    lines.append('## Cases')
    lines.append('')
    lines.append('| case | words | GT path | Pred path | path ratio | GT vert | Pred vert | vert ratio | flags |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in result['rows']:
        words = len(str(row.get('auto_prompt', '')).split())
        flags = ','.join(row['flags']) if row['flags'] else 'ok'
        lines.append(
            f"| `{row['case_id']}` | {words} | "
            f"{_fmt(row['gt_metrics']['root_path_xz_m'])} | {_fmt(row['pred_metrics']['root_path_xz_m'])} | {_fmt(row['ratios']['root_path_xz'])} | "
            f"{_fmt(row['gt_metrics']['root_vertical_amp_m'])} | {_fmt(row['pred_metrics']['root_vertical_amp_m'])} | {_fmt(row['ratios']['root_vertical_amp'])} | {flags} |"
        )
    lines.append('')
    lines.append('## Prompts')
    lines.append('')
    for row in result['rows']:
        lines.append(f"### {row['case_id']}")
        lines.append('')
        lines.append(f"- canonical ids: `{', '.join(str(x) for x in row['canonical_ids'])}`")
        lines.append(f"- prompt: {row['auto_prompt']}")
        lines.append('')
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    result = analyze(Path(args.summary))
    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding='utf-8')
    write_report(result, report_path)
    print(f'saved={out_path}')
    print(f'report={report_path}')


if __name__ == '__main__':
    main()
