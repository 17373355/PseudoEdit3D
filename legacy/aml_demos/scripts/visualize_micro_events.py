from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events
from pseudoedit3d.visualization.skeleton_gif import _draw_skeleton, _load_font, _normalize_points, _project_points, _wrap_text

RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _normalize_panel(points: np.ndarray, width: int = 420, height: int = 420) -> np.ndarray:
    proj = _project_points(points)
    return _normalize_points(proj, width=width, height=height)


def render_case(case_id: str, output_path: Path, fps: int = 12) -> None:
    mod = load_runner_module()
    joints = mod.load_joints3d(case_id)
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    events = extract_layer1_micro_events(layer0)
    gt_prompt = mod.read_first_prompt(case_id)

    gt_joints = joints
    gt_proj = _normalize_panel(gt_joints)

    canvas_w = 1600
    canvas_h = 500
    left_box = (20, 50, 500, 470)
    right_box = (540, 50, 1580, 470)
    font_title = _load_font(24)
    font_body = _load_font(16)
    font_small = _load_font(13)

    frames = []
    total = len(gt_joints)
    for frame_idx in range(total):
        img = Image.new('RGB', (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        for box in [left_box, right_box]:
            draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((40, 12), 'GT Motion', fill=(20, 20, 20), font=font_title)
        draw.text((560, 12), 'Layer 1 Micro-Events', fill=(20, 20, 20), font=font_title)

        panel = gt_proj[frame_idx].copy()
        panel[:, 0] = panel[:, 0] - 210 + (left_box[0] + left_box[2]) / 2.0
        panel[:, 1] = panel[:, 1] - 210 + (left_box[1] + left_box[3]) / 2.0
        _draw_skeleton(draw, panel, base_color=(180,184,195), highlight_color=(80,170,80))

        y = right_box[1] + 20
        draw.text((right_box[0] + 20, y), f'case: {case_id}', fill=(60, 60, 70), font=font_body)
        y += 22
        draw.text((right_box[0] + 20, y), f'frame: {frame_idx + 1}/{total}', fill=(60, 60, 70), font=font_body)
        y += 22
        draw.text((right_box[0] + 20, y), f'num_events: {len(events)}', fill=(60, 60, 70), font=font_body)
        y += 28

        prompt_lines = _wrap_text(draw, f'selected_hml3d_prompt: {gt_prompt}', font_body, max_width=(right_box[2] - right_box[0] - 40))
        for line in prompt_lines[:4]:
            draw.text((right_box[0] + 20, y), line, fill=(70, 120, 220), font=font_body)
            y += 18
        y += 12

        active = [e for e in events if e.start_frame <= frame_idx <= e.end_frame]
        draw.text((right_box[0] + 20, y), 'active_events:', fill=(20, 20, 20), font=font_body)
        y += 22
        if not active:
            draw.text((right_box[0] + 28, y), '(none)', fill=(120, 120, 130), font=font_body)
            y += 18
        else:
            for e in active[:14]:
                line = f'{e.to_symbol()}  [{e.start_frame}-{e.end_frame}]  d={e.delta_value:.3f} {e.unit}'
                draw.text((right_box[0] + 28, y), line, fill=(190, 120, 40), font=font_small)
                y += 16
        y += 10
        upcoming = [e for e in events if e.start_frame > frame_idx]
        draw.text((right_box[0] + 20, y), 'next_events:', fill=(20, 20, 20), font=font_body)
        y += 22
        for e in upcoming[:8]:
            line = f'{e.to_symbol()}  starts@{e.start_frame}'
            draw.text((right_box[0] + 28, y), line, fill=(95, 95, 105), font=font_small)
            y += 16

        frames.append(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(str(output_path), save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        out_path = out_dir / f'case_{case_id}.gif'
        render_case(case_id, out_path, fps=args.fps)
        print(f'saved_micro_event_vis={out_path}')


if __name__ == '__main__':
    main()
