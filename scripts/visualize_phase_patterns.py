from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import extract_layer0_frame_observables, extract_layer1_micro_events, merge_micro_events, detect_repeated_phases
from pseudoedit3d.edit.phase_patterns import project_units_by_category
from pseudoedit3d.visualization.skeleton_gif import _draw_skeleton, _load_font, _normalize_points, _project_points, _wrap_text

RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def render_case(case_id: str, output_path: Path, fps: int = 12):
    mod = load_runner_module()
    joints = mod.load_joints3d(case_id)
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    micro = extract_layer1_micro_events(layer0)
    subs = merge_micro_events(micro)
    phases_all = detect_repeated_phases(subs)
    proj = _normalize_points(_project_points(joints), width=420, height=420)
    prompt = mod.read_first_prompt(case_id)

    projected = {
        'whole_body': detect_repeated_phases(project_units_by_category(subs, 'whole_body')),
        'torso': detect_repeated_phases(project_units_by_category(subs, 'torso')),
        'left_arm': detect_repeated_phases(project_units_by_category(subs, 'left_arm')),
        'right_arm': detect_repeated_phases(project_units_by_category(subs, 'right_arm')),
    }

    canvas_w = 1600
    canvas_h = 500
    left_box = (20, 50, 500, 470)
    right_box = (540, 50, 1580, 470)
    font_title = _load_font(24)
    font_body = _load_font(16)
    font_small = _load_font(13)

    frames = []
    for frame_idx in range(len(joints)):
        img = Image.new('RGB', (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        for box in [left_box, right_box]:
            draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((40, 12), 'GT Motion', fill=(20, 20, 20), font=font_title)
        draw.text((560, 12), 'Phase Patterns', fill=(20, 20, 20), font=font_title)

        panel = proj[frame_idx].copy()
        panel[:, 0] = panel[:, 0] - 210 + (left_box[0] + left_box[2]) / 2.0
        panel[:, 1] = panel[:, 1] - 210 + (left_box[1] + left_box[3]) / 2.0
        _draw_skeleton(draw, panel, base_color=(180,184,195), highlight_color=(80,170,80))

        y = right_box[1] + 20
        draw.text((right_box[0] + 20, y), f'case: {case_id}', fill=(60,60,70), font=font_body); y += 22
        draw.text((right_box[0] + 20, y), f'frame: {frame_idx+1}/{len(joints)}', fill=(60,60,70), font=font_body); y += 22
        prompt_lines = _wrap_text(draw, f'selected_hml3d_prompt: {prompt}', font_body, max_width=(right_box[2]-right_box[0]-40))
        for line in prompt_lines[:4]:
            draw.text((right_box[0]+20, y), line, fill=(70,120,220), font=font_body)
            y += 18
        y += 8

        draw.text((right_box[0]+20, y), f'all_phase_patterns: {len(phases_all)}', fill=(20,20,20), font=font_body); y += 22
        for p in phases_all[:6]:
            draw.text((right_box[0]+28, y), f'{p.name} [{p.start_frame}-{p.end_frame}]', fill=(190,120,40), font=font_small)
            y += 16
        y += 10
        for key, pats in projected.items():
            if not pats:
                continue
            draw.text((right_box[0]+20, y), f'{key}:', fill=(20,20,20), font=font_body); y += 20
            for p in pats[:4]:
                draw.text((right_box[0]+28, y), f'{p.name} [{p.start_frame}-{p.end_frame}]', fill=(95,95,105), font=font_small)
                y += 16
            y += 6
        frames.append(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(str(output_path), save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        out_path = out_dir / f'case_{case_id}.gif'
        render_case(case_id, out_path, fps=args.fps)
        print(f'saved_phase_vis={out_path}')


if __name__ == '__main__':
    main()
