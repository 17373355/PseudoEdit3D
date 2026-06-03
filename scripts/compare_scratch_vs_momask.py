from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.inference.predict import run_prefix_case_inference
from pseudoedit3d.visualization.skeleton_gif import (
    _build_prefix_locked_display_motion,
    _compute_joints,
    _draw_skeleton,
    _normalize_points,
    _part_edge_indices,
    _project_points,
    _wrap_text,
    _load_font,
)
from pseudoedit3d.constants import BODY_PART_TO_JOINTS


def _load_bridge_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_momask_joints(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


def _normalize_panel(points_list: list[np.ndarray], width: int = 360, height: int = 360) -> list[np.ndarray]:
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


def _program_lines(program: dict) -> list[str]:
    if "edits" in program:
        lines = [
            f"task: {program.get('task_mode', 'multi_atomic_realize')}",
            f"num_edits: {len(program.get('edits', []))}",
            f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
        ]
        for i, edit in enumerate(program.get("edits", [])[:3], start=1):
            lines.extend(
                [
                    f"edit{i}.part: {edit.get('part', '-')}",
                    f"edit{i}.attr: {edit.get('attribute', '-')}",
                    f"edit{i}.dir: {edit.get('direction', '-')}",
                    f"edit{i}.delta_deg: {float(edit.get('delta_value_deg') or 0.0):+.1f}",
                    f"edit{i}.span: {int(edit.get('start_frame', -1))}-{int(edit.get('end_frame', -1))}",
                ]
            )
        return lines
    return [
        f"task: {program.get('task_mode', 'atomic_realize')}",
        f"part: {program.get('part', '-')}",
        f"attr: {program.get('attribute', '-')}",
        f"dir: {program.get('direction', '-')}",
        f"delta_deg: {float(program.get('delta_value_deg') or 0.0):+.1f}",
        f"start: {int(program.get('start_frame', -1))}",
        f"end: {int(program.get('end_frame', -1))}",
        f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
    ]


def render_case(case_result: dict, momask_joints: np.ndarray, momask_meta: dict, output_path: Path, fps: int = 12) -> None:
    source_pose = case_result["source_pose"]
    target_pose = case_result["target_pose"]
    pred_pose = case_result["pred_pose"]
    source_trans = case_result["source_trans"]
    target_trans = case_result["target_trans"]
    program = case_result["program"]
    prompt_text = case_result["prompt_text"]
    momask_text_prompt = momask_meta.get("text_prompt", "")
    momask_mask_section = momask_meta.get("mask_edit_section", "")

    total_frames = pred_pose.shape[0]
    prefix_frames = int(program.get("source_prefix_frames", 1))
    valid_end_frame = int(program.get("valid_end_frame", total_frames - 1))
    display_total_frames = min(total_frames, valid_end_frame + 1)
    frame_indices = np.arange(display_total_frames)

    source_display_pose, source_display_trans = _build_prefix_locked_display_motion(
        prefix_pose=source_pose,
        future_pose=np.repeat(source_pose[prefix_frames - 1:prefix_frames], total_frames, axis=0),
        reference_trans=target_trans,
        prefix_frames=prefix_frames,
    )
    source_joints = _compute_joints(source_display_pose[frame_indices], source_display_trans[frame_indices], None, None)
    gt_joints = _compute_joints(target_pose[frame_indices], target_trans[frame_indices], None, None)
    scratch_display_pose, scratch_display_trans = _build_prefix_locked_display_motion(
        prefix_pose=source_pose,
        future_pose=pred_pose,
        reference_trans=target_trans,
        prefix_frames=prefix_frames,
    )
    scratch_joints = _compute_joints(scratch_display_pose[frame_indices], scratch_display_trans[frame_indices], None, None)
    momask_joints = momask_joints[:display_total_frames]

    source_proj, gt_proj, scratch_proj, momask_proj = _normalize_panel(
        [source_joints, gt_joints, scratch_joints, momask_joints]
    )

    canvas_w = 1980
    canvas_h = 440
    left_box = (20, 40, 380, 400)
    mid_box = (410, 40, 790, 400)
    gt_box = (820, 40, 1180, 400)
    scratch_box = (1210, 40, 1570, 400)
    momask_box = (1600, 40, 1960, 400)
    font_title = _load_font(24)
    font_body = _load_font(18)
    font_small = _load_font(13)

    highlight_joints = set()
    highlight_edges = set()
    if "edits" in program and program["edits"]:
        focus_part = program["edits"][0]["part"]
        highlight_joints = set(BODY_PART_TO_JOINTS.get(focus_part, []))
        highlight_edges = _part_edge_indices(focus_part)
    elif "part" in program:
        focus_part = program["part"]
        highlight_joints = set(BODY_PART_TO_JOINTS.get(focus_part, []))
        highlight_edges = _part_edge_indices(focus_part)

    frames = []
    program_lines = _program_lines(program)
    prompt_lines = []
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    prompt_lines = _wrap_text(dummy_draw, f"prompt: {prompt_text}", font_small, max_width=(mid_box[2] - mid_box[0] - 40))

    for frame_idx in range(display_total_frames):
        img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        for box in [left_box, mid_box, gt_box, scratch_box, momask_box]:
            draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((40, 10), "Source Prefix Motion", fill=(20, 20, 20), font=font_title)
        draw.text((450, 10), "EditProgram", fill=(20, 20, 20), font=font_title)
        draw.text((835, 10), "GT Full Motion", fill=(20, 20, 20), font=font_title)
        draw.text((1220, 10), "Scratch Prediction", fill=(20, 20, 20), font=font_title)
        draw.text((1610, 10), "MoMask Prediction", fill=(20, 20, 20), font=font_title)

        def place(panel: np.ndarray, box, base_color, hi_color):
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

        place(source_proj, left_box, (180, 184, 195), (55, 100, 230))
        place(gt_proj, gt_box, (180, 184, 195), (80, 170, 80))
        place(scratch_proj, scratch_box, (180, 184, 195), (220, 80, 60))
        place(momask_proj, momask_box, (180, 184, 195), (190, 120, 40))

        y = mid_box[1] + 30
        for line in program_lines[:12]:
            draw.text((mid_box[0] + 20, y), line, fill=(55, 55, 65), font=font_body)
            y += 22
        y += 8
        scratch_prompt_lines = _wrap_text(draw, f"scratch_prompt: {prompt_text}", font_small, max_width=(mid_box[2] - mid_box[0] - 40))
        momask_prompt_lines = _wrap_text(draw, f"momask_prompt: {momask_text_prompt}", font_small, max_width=(mid_box[2] - mid_box[0] - 40))
        mask_lines = _wrap_text(draw, f"momask_mask: {momask_mask_section}", font_small, max_width=(mid_box[2] - mid_box[0] - 40))
        for line in scratch_prompt_lines[:2]:
            draw.text((mid_box[0] + 20, y), line, fill=(95, 95, 105), font=font_small)
            y += 16
        for line in momask_prompt_lines[:2]:
            draw.text((mid_box[0] + 20, y), line, fill=(95, 95, 105), font=font_small)
            y += 16
        for line in mask_lines[:1]:
            draw.text((mid_box[0] + 20, y), line, fill=(95, 95, 105), font=font_small)
            y += 16

        draw.text((mid_box[0] + 20, mid_box[3] - 50), f"frame {frame_idx + 1}/{display_total_frames}", fill=(80, 80, 90), font=font_body)
        draw.text((mid_box[0] + 20, mid_box[3] - 25), f"case {case_result['case_idx']}", fill=(80, 80, 90), font=font_body)
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-indices", required=True)
    parser.add_argument("--momask-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    case_indices = [int(x.strip()) for x in args.case_indices.split(",") if x.strip()]
    results = run_prefix_case_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        case_indices=case_indices,
        device="cpu",
    )
    momask_root = Path(args.momask_root)
    output_dir = Path(args.output_dir)
    for result in results:
        out_path = output_dir / f"case_{result['case_idx']:04d}.gif"
        if result['case_idx'] == 54:
            case_root = momask_root / 'pseudoedit_heldout_case54_rich'
            meta_path = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs/momask_bridge_heldout_case54_rich/meta.json')
        else:
            case_root = momask_root / f"pseudoedit_heldout_case{result['case_idx']}"
            meta_path = Path('/mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs') / f"momask_bridge_heldout_case{result['case_idx']}" / 'meta.json'
        joints = _load_momask_joints(case_root / 'joints' / '0' / 'sample0_repeat0_len60.npy')
        momask_meta = _load_bridge_meta(meta_path)
        render_case(result, joints, momask_meta, out_path, fps=args.fps)
        print(f"saved_compare={out_path}")


if __name__ == "__main__":
    main()
