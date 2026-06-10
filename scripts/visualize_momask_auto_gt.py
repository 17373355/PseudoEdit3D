from __future__ import annotations

import argparse
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

from pseudoedit3d.visualization.skeleton_gif import (
    _draw_skeleton,
    _load_font,
    _normalize_points,
    _project_points,
    _wrap_text,
)

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'


def _load_summary(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_np(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def _find_joint_file(base_dir: Path) -> Path:
    matches = sorted([p for p in base_dir.glob('sample0_repeat0_len*.npy') if '_ik' not in p.name])
    if not matches:
        raise FileNotFoundError(f'No joint file found in {base_dir}')
    return matches[0]


def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len]
    if len(arr) == 0:
        raise ValueError('Cannot pad empty array')
    pad = np.repeat(arr[-1:], target_len - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def _load_gt_pack() -> dict[str, Any]:
    return torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')


def _gt_joints_from_pack(pack: dict[str, Any], case_id: str) -> np.ndarray:
    item = pack[f'{case_id}.npy']
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    return np.asarray(joints, dtype=np.float32)


def _normalize_panel(points_list: list[np.ndarray], width: int = 390, height: int = 390) -> list[np.ndarray]:
    stacked = np.concatenate(points_list, axis=0)
    proj = _project_points(stacked)
    norm = _normalize_points(proj, width=width, height=height)
    out = []
    offset = 0
    for points in points_list:
        n = points.shape[0]
        out.append(norm[offset: offset + n])
        offset += n
    return out


def frame_indices_for_render(num_frames: int, frame_stride: int = 1, max_render_frames: int | None = None) -> list[int]:
    if num_frames <= 0:
        return []
    if max_render_frames is not None and max_render_frames > 0 and max_render_frames < num_frames:
        indices = np.linspace(0, num_frames - 1, max_render_frames, dtype=np.int32).tolist()
    else:
        stride = max(1, int(frame_stride))
        indices = list(range(0, num_frames, stride))
        if indices[-1] != num_frames - 1:
            indices.append(num_frames - 1)
    out: list[int] = []
    seen = set()
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def render_case(
    case: dict[str, Any],
    gt_pack: dict[str, Any],
    output_path: Path,
    fps: int = 10,
    frame_stride: int = 1,
    max_render_frames: int | None = None,
    show_hml3d_reference: bool = True,
) -> dict[str, Any]:
    case_id = case['case_id']
    source_num_frames = int(case.get('source_num_frames', 60))
    gt_joints = _gt_joints_from_pack(gt_pack, case_id)[:source_num_frames]
    auto_ext = case['auto_ext']
    auto_pred = _load_np(_find_joint_file(MOMASK_ROOT / 'generation' / auto_ext / 'joints' / '0'))
    auto_pred = _pad_to_length(auto_pred, source_num_frames)
    gt_proj, auto_proj = _normalize_panel([gt_joints, auto_pred])

    render_indices = frame_indices_for_render(
        len(gt_joints),
        frame_stride=frame_stride,
        max_render_frames=max_render_frames,
    )
    canvas_w = 1520
    canvas_h = 650
    gt_box = (20, 56, 430, 466)
    prompt_box = (455, 56, 1065, 620)
    auto_box = (1090, 56, 1500, 466)
    font_title = _load_font(23)
    font_body = _load_font(13)
    font_prompt = _load_font(12)
    font_small = _load_font(10)

    def place(draw: ImageDraw.ImageDraw, panel: np.ndarray, frame_idx: int, box, base_color, hi_color):
        p = panel[frame_idx].copy()
        p[:, 0] = p[:, 0] - 195 + (box[0] + box[2]) / 2.0
        p[:, 1] = p[:, 1] - 195 + (box[1] + box[3]) / 2.0
        _draw_skeleton(
            draw,
            p,
            base_color=base_color,
            highlight_color=hi_color,
            highlight_joints=set(),
            highlight_edges=set(),
            radius=4,
            width=3,
        )

    auto_prompt = str(case.get('auto_prompt') or '')
    gt_prompt = str(case.get('gt_prompt') or '')
    raw_prompt_segments = case.get('raw_prompt_segments') or []
    hml3d_prompts: list[str] = []
    for segment in raw_prompt_segments:
        if isinstance(segment, (list, tuple)) and segment:
            hml3d_prompts.append(str(segment[0]))
        elif isinstance(segment, str):
            hml3d_prompts.append(segment)
    if not hml3d_prompts and gt_prompt:
        hml3d_prompts.append(gt_prompt)
    frames = []
    for frame_idx in render_indices:
        img = Image.new('RGB', (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        for box in [gt_box, prompt_box, auto_box]:
            draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((42, 18), 'GT Motion', fill=(20, 20, 20), font=font_title)
        draw.text((485, 18), 'HML3D Captions + AutoPrompt', fill=(20, 20, 20), font=font_title)
        draw.text((1125, 18), 'MoMask from AutoPrompt', fill=(20, 20, 20), font=font_title)

        place(draw, gt_proj, frame_idx, gt_box, (176, 181, 193), (60, 150, 85))
        place(draw, auto_proj, frame_idx, auto_box, (176, 181, 193), (205, 115, 25))

        draw.text((gt_box[0] + 20, gt_box[3] - 50), f'frame {frame_idx + 1}/{len(gt_joints)}', fill=(70, 70, 80), font=font_body)
        draw.text((gt_box[0] + 20, gt_box[3] - 26), f'case {case_id}', fill=(70, 70, 80), font=font_body)
        draw.text((auto_box[0] + 20, auto_box[3] - 50), f'generated len {int(case.get("generated_num_frames", len(auto_pred)))}', fill=(70, 70, 80), font=font_body)
        draw.text((auto_box[0] + 20, auto_box[3] - 26), 'auto prompt conditioned', fill=(70, 70, 80), font=font_body)

        y = prompt_box[1] + 16
        meta_lines = [
            f'case: {case_id}',
            f'GT frames: {len(gt_joints)}   rendered frames: {len(render_indices)}',
            f'L3 events: {len((case.get("auto_program") or {}).get("events") or [])}',
        ]
        for line in meta_lines:
            draw.text((prompt_box[0] + 18, y), line, fill=(55, 55, 65), font=font_body)
            y += 16
        y += 6
        max_text_w = prompt_box[2] - prompt_box[0] - 36
        if show_hml3d_reference and hml3d_prompts:
            draw.text((prompt_box[0] + 18, y), 'HML3D captions (reference only)', fill=(55, 95, 205), font=font_body)
            y += 17
            for i, prompt in enumerate(hml3d_prompts[:4], start=1):
                for line in _wrap_text(draw, f'{i}. {prompt}', font_small, max_text_w):
                    if y > prompt_box[3] - 210:
                        break
                    draw.text((prompt_box[0] + 18, y), line, fill=(55, 95, 205), font=font_small)
                    y += 12
                y += 2
                if y > prompt_box[3] - 210:
                    break
            y += 6
        draw.text((prompt_box[0] + 18, y), 'motion-only AutoPrompt', fill=(205, 115, 25), font=font_body)
        y += 17
        for line in _wrap_text(draw, auto_prompt, font_prompt, max_text_w):
            if y > prompt_box[3] - 16:
                break
            draw.text((prompt_box[0] + 18, y), line, fill=(205, 115, 25), font=font_prompt)
            y += 14
        frames.append(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(str(output_path), save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return {
        'case_id': case_id,
        'gif_path': str(output_path),
        'num_frames': int(len(gt_joints)),
        'rendered_frames': int(len(render_indices)),
        'frame_stride': int(frame_stride),
        'auto_ext': auto_ext,
        'auto_prompt': auto_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--case-ids', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--max-render-frames', type=int, default=None)
    parser.add_argument('--show-hml3d-reference', action='store_true')
    parser.add_argument('--hide-hml3d-reference', action='store_true')
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary))
    selected = {x.strip() for x in args.case_ids.split(',') if x.strip()}
    if selected:
        summary = [case for case in summary if case['case_id'] in selected]
    if not summary:
        raise SystemExit('No cases to visualize')

    gt_pack = _load_gt_pack()
    output_dir = Path(args.output_dir)
    outputs = []
    for case in summary:
        out_path = output_dir / f"case_{case['case_id']}.gif"
        outputs.append(render_case(
            case,
            gt_pack,
            out_path,
            fps=args.fps,
            frame_stride=args.frame_stride,
            max_render_frames=args.max_render_frames,
            show_hml3d_reference=(args.show_hml3d_reference or not args.hide_hml3d_reference),
        ))
        print(f'saved_auto_gt_vis={out_path}', flush=True)
    (output_dir / 'summary.json').write_text(json.dumps(outputs, ensure_ascii=True, indent=2), encoding='utf-8')
    print(output_dir / 'summary.json')


if __name__ == '__main__':
    main()
