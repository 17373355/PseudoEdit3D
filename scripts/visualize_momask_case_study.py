from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.visualization.skeleton_gif import (
    _compute_joints,
    _draw_skeleton,
    _load_font,
    _normalize_points,
    _part_edge_indices,
    _project_points,
    _wrap_text,
)
from pseudoedit3d.constants import BODY_PART_TO_JOINTS


MOMASK_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes")
HML_ROOT = MOMASK_ROOT / "dataset" / "HumanML3D"


def _load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _load_gt_joints(case_id: str) -> np.ndarray:
    data = torch.load(HML_ROOT / "joints3d.pth", map_location="cpu")
    item = data[f"{case_id}.npy"]
    joints = item["joints3d"]
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    return np.asarray(joints, dtype=np.float32)


def _normalize_panel(points_list: list[np.ndarray], width: int = 360, height: int = 360) -> list[np.ndarray]:
    stacked = np.concatenate(points_list, axis=0)
    proj = _project_points(stacked)
    norm = _normalize_points(proj, width=width, height=height)
    out = []
    offset = 0
    for points in points_list:
        n = points.shape[0]
        out.append(norm[offset : offset + n])
        offset += n
    return out


def _program_lines(program: dict) -> list[str]:
    lines = [
        f"task: {program.get('task_mode', 'multi_atomic_realize')}",
        f"num_edits: {len(program.get('edits', []))}",
        f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
    ]
    for i, edit in enumerate(program.get("edits", [])[:4], start=1):
        lines.extend(
            [
                f"edit{i}.part: {edit.get('part', '-')}",
                f"edit{i}.attr: {edit.get('attribute', '-')}",
                f"edit{i}.delta: {float(edit.get('delta_value_deg') or 0.0):+.1f}",
                f"edit{i}.span: {int(edit.get('start_frame', -1))}-{int(edit.get('end_frame', -1))}",
            ]
        )
    return lines


def render_case(case: dict, output_path: Path, fps: int = 12) -> None:
    case_id = case["case_id"]
    gt_prompt = case["gt_prompt"]
    auto_prompt = case["auto_prompt"]
    program = case["program"]
    raw_prompt_segments = case.get("raw_prompt_segments", [])

    source_num_frames = int(case.get('source_num_frames', 60))
    gt_joints = _load_gt_joints(case_id)[:source_num_frames]
    gt_pred = _load_np(_find_joint_file(MOMASK_ROOT / "generation" / case["gt_ext"] / "joints" / "0"))
    auto_pred = _load_np(_find_joint_file(MOMASK_ROOT / "generation" / case["auto_ext"] / "joints" / "0"))
    gt_pred = _pad_to_length(gt_pred, source_num_frames)
    auto_pred = _pad_to_length(auto_pred, source_num_frames)

    gt_proj, gtpred_proj, autopred_proj = _normalize_panel([gt_joints, gt_pred, auto_pred])

    canvas_w = 1900
    canvas_h = 560
    left_box = (20, 50, 380, 410)
    mid_box = (410, 50, 930, 520)
    gtpred_box = (960, 50, 1320, 410)
    auto_box = (1350, 50, 1710, 410)
    font_title = _load_font(23)
    font_body = _load_font(13)
    font_prompt = _load_font(14)
    font_small = _load_font(11)

    highlight_joints = set()
    highlight_edges = set()
    if program.get("edits"):
        focus_part = program["edits"][0].get("part", "")
        highlight_joints = set(BODY_PART_TO_JOINTS.get(focus_part, []))
        highlight_edges = _part_edge_indices(focus_part)

    def place(draw: ImageDraw.ImageDraw, panel: np.ndarray, frame_idx: int, box, base_color, hi_color):
        p = panel[frame_idx].copy()
        p[:, 0] = p[:, 0] - 180 + (box[0] + box[2]) / 2.0
        p[:, 1] = p[:, 1] - 180 + (box[1] + box[3]) / 2.0
        _draw_skeleton(
            draw,
            p,
            base_color=base_color,
            highlight_color=hi_color,
            highlight_joints=highlight_joints,
            highlight_edges=highlight_edges,
        )

    frames = []
    program_lines = _program_lines(program)
    for frame_idx in range(len(gt_joints)):
        img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        for box in [left_box, mid_box, gtpred_box, auto_box]:
            draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((40, 16), "GT Motion", fill=(20, 20, 20), font=font_title)
        draw.text((450, 16), "Prompts / Program", fill=(20, 20, 20), font=font_title)
        draw.text((990, 16), "MoMask from Selected HML3D", fill=(20, 20, 20), font=font_title)
        draw.text((1380, 16), "MoMask from Auto", fill=(20, 20, 20), font=font_title)

        place(draw, gt_proj, frame_idx, left_box, (180, 184, 195), (80, 170, 80))
        place(draw, gtpred_proj, frame_idx, gtpred_box, (180, 184, 195), (70, 120, 220))
        place(draw, autopred_proj, frame_idx, auto_box, (180, 184, 195), (190, 120, 40))

        y = mid_box[1] + 14
        for line in program_lines[:8]:
            draw.text((mid_box[0] + 18, y), line, fill=(55, 55, 65), font=font_body)
            y += 15
        y += 6
        max_text_w = mid_box[2] - mid_box[0] - 36
        bottom = mid_box[3] - 12
        sections = [
            ("selected_hml3d_prompt: " + gt_prompt, (70, 120, 220), font_prompt, 14),
            ("auto_prompt: " + auto_prompt, (190, 120, 40), font_prompt, 14),
        ]
        if raw_prompt_segments:
            raw_text = "all_hml3d_prompts: " + " | ".join(seg[0] for seg in raw_prompt_segments)
            sections.append((raw_text, (120, 120, 130), font_small, 12))
        for text, color, font, line_h in sections:
            lines = _wrap_text(draw, text, font, max_width=max_text_w)
            for line in lines:
                if y + line_h > bottom:
                    break
                draw.text((mid_box[0] + 18, y), line, fill=color, font=font)
                y += line_h
            y += 5
            if y >= bottom:
                break
        draw.text((left_box[0] + 20, left_box[3] - 48), f"frame {frame_idx + 1}/ {len(gt_joints)}".replace(" ", ""), fill=(80, 80, 90), font=font_body)
        draw.text((left_box[0] + 20, left_box[3] - 24), f"case {case_id}", fill=(80, 80, 90), font=font_body)
        frames.append(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--case-ids", required=False, default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary))
    selected = {x.strip() for x in args.case_ids.split(",") if x.strip()}
    if selected:
        summary = [case for case in summary if case["case_id"] in selected]

    output_dir = Path(args.output_dir)
    for case in summary:
        out_path = output_dir / f"case_{case['case_id']}.gif"
        render_case(case, out_path, fps=args.fps)
        print(f"saved_case_study_vis={out_path}")


if __name__ == "__main__":
    main()
