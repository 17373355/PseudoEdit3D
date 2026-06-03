from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pseudoedit3d.constants import BODY_PART_TO_JOINTS


def _load_font(size: int = 20):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except Exception:
        return ImageFont.load_default()


BODY_EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (12, 15),
]

SMPLH_STICK_PARENTS = np.asarray([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 12, 12, 12, 13, 14, 16, 17, 18, 19,
], dtype=np.int64)


SMPLH_STICK_OFFSETS = np.asarray([
    [0.0, 0.0, 0.0],
    [0.09, -0.10, 0.0],
    [-0.09, -0.10, 0.0],
    [0.0, 0.12, 0.0],
    [0.0, -0.42, 0.02],
    [0.0, -0.42, 0.02],
    [0.0, 0.14, 0.0],
    [0.0, -0.43, 0.02],
    [0.0, -0.43, 0.02],
    [0.0, 0.15, 0.0],
    [0.0, -0.05, 0.10],
    [0.0, -0.05, -0.10],
    [0.0, 0.18, 0.0],
    [0.08, 0.12, 0.0],
    [-0.08, 0.12, 0.0],
    [0.0, 0.13, 0.0],
    [0.28, 0.0, 0.0],
    [-0.28, 0.0, 0.0],
    [0.27, 0.0, 0.0],
    [-0.27, 0.0, 0.0],
    [0.20, 0.0, 0.0],
    [-0.20, 0.0, 0.0],
], dtype=np.float32)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.clip(angle, 1e-8, None)
    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    zeros = np.zeros_like(x)
    k = np.concatenate([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), axis.shape[:-1] + (3, 3))
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    return eye + sin * k + (1.0 - cos) * np.matmul(k, k)


def _compute_joints(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray | None = None, smplh_model_path: str | None = None) -> np.ndarray:
    del betas, smplh_model_path
    pose = poses[:, :22].astype(np.float32)
    rot_mats = _axis_angle_to_matrix(pose)
    num_frames = pose.shape[0]
    joints = np.zeros((num_frames, 22, 3), dtype=np.float32)
    global_rot = np.zeros((num_frames, 22, 3, 3), dtype=np.float32)
    for j in range(22):
        parent = int(SMPLH_STICK_PARENTS[j])
        if parent < 0:
            global_rot[:, j] = rot_mats[:, j]
            joints[:, j] = trans + SMPLH_STICK_OFFSETS[j]
        else:
            global_rot[:, j] = np.matmul(global_rot[:, parent], rot_mats[:, j])
            joints[:, j] = joints[:, parent] + np.einsum('fij,j->fi', global_rot[:, parent], SMPLH_STICK_OFFSETS[j])
    return joints


def _project_points(points3d: np.ndarray) -> np.ndarray:
    x = points3d[..., 0]
    y = points3d[..., 1]
    z = points3d[..., 2]
    u = 0.72 * x + 0.45 * z
    v = y
    return np.stack([u, v], axis=-1)


def _normalize_points(points2d: np.ndarray, width: int, height: int, margin: int = 30) -> np.ndarray:
    mins = points2d.reshape(-1, 2).min(axis=0)
    maxs = points2d.reshape(-1, 2).max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-4)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    norm = (points2d - center[None, None, :]) * scale
    norm[..., 0] += width / 2.0
    norm[..., 1] = height / 2.0 - norm[..., 1]
    return norm


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _part_edge_indices(part: str) -> set[int]:
    part_joints = set(BODY_PART_TO_JOINTS.get(part, []))
    highlighted = set()
    for idx, (parent, child) in enumerate(BODY_EDGES):
        if parent in part_joints or child in part_joints:
            highlighted.add(idx)
    return highlighted


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    joints2d: np.ndarray,
    base_color=(150, 155, 170),
    highlight_color=(44, 97, 255),
    highlight_joints: set[int] | None = None,
    highlight_edges: set[int] | None = None,
    radius: int = 4,
    width: int = 3,
):
    highlight_joints = highlight_joints or set()
    highlight_edges = highlight_edges or set()
    for parent, child in BODY_EDGES:
        pass
    for edge_idx, (parent, child) in enumerate(BODY_EDGES):
        p = tuple(joints2d[parent].tolist())
        c = tuple(joints2d[child].tolist())
        color = highlight_color if edge_idx in highlight_edges else base_color
        draw.line([p, c], fill=color, width=width)
    for joint_idx, (x, y) in enumerate(joints2d):
        color = highlight_color if joint_idx in highlight_joints else base_color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)




def _build_prefix_locked_display_motion(
    prefix_pose: np.ndarray,
    future_pose: np.ndarray,
    reference_trans: np.ndarray,
    prefix_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = future_pose.shape[0]
    prefix = min(max(1, prefix_frames), total_frames)
    display_pose = future_pose.copy()
    display_trans = np.repeat(reference_trans[prefix - 1 : prefix], total_frames, axis=0)
    display_pose[:prefix] = prefix_pose[:prefix]
    display_trans[:prefix] = reference_trans[:prefix]
    return display_pose, display_trans


def export_case_gif(
    case_result: dict,
    output_path: str,
    smplh_model_path: str | None = None,
    fps: int = 12,
    frame_limit: int | None = None,
) -> str:
    source_pose = case_result["source_pose"]
    target_pose = case_result["target_pose"]
    pred_pose = case_result["pred_pose"]
    source_trans = case_result["source_trans"]
    target_trans = case_result["target_trans"]
    betas = case_result["betas"]
    prompt_text = case_result["prompt_text"]
    program = case_result["program"]
    task_mode = program.get("task_mode", "")
    is_edit_task = (
        "part" in program and "attribute" in program and "start_frame" in program and "end_frame" in program
        and task_mode not in {"semantic_continue", "atomic_realize"}
    )

    total_frames = pred_pose.shape[0]
    prefix_frames = int(program.get("source_prefix_frames", 1))
    valid_end_frame = int(program.get("valid_end_frame", total_frames - 1))
    display_total_frames = min(total_frames, valid_end_frame + 1)
    if frame_limit is None or frame_limit >= total_frames:
        frame_indices = np.arange(display_total_frames)
    else:
        frame_indices = np.arange(min(frame_limit, display_total_frames))
    num_frames = len(frame_indices)

    source_display_pose, source_display_trans = _build_prefix_locked_display_motion(
        prefix_pose=source_pose,
        future_pose=np.repeat(source_pose[prefix_frames - 1 : prefix_frames], total_frames, axis=0),
        reference_trans=target_trans,
        prefix_frames=prefix_frames,
    )
    source_joints = _compute_joints(source_display_pose[frame_indices], source_display_trans[frame_indices], betas[:1], None)
    if is_edit_task:
        target_masked_pose = source_pose.copy()
        target_joint_ids = BODY_PART_TO_JOINTS.get(program["part"], [])
        span_start = int(program["start_frame"])
        span_end = int(program["end_frame"])
        target_masked_pose[span_start:span_end + 1, target_joint_ids] = target_pose[span_start:span_end + 1, target_joint_ids]
        target_masked_trans = target_trans.copy()
        gt_title = "GT Prompt-Scoped Motion"
    else:
        target_masked_pose = target_pose.copy()
        target_masked_trans = target_trans.copy()
        gt_title = "GT Full Motion"

    target_joints = _compute_joints(target_masked_pose[frame_indices], target_masked_trans[frame_indices], betas[:1], None)
    pred_display_pose, pred_display_trans = _build_prefix_locked_display_motion(
        prefix_pose=source_pose,
        future_pose=pred_pose,
        reference_trans=target_trans,
        prefix_frames=prefix_frames,
    )
    pred_joints = _compute_joints(pred_display_pose[frame_indices], pred_display_trans[frame_indices], betas[:1], None)

    all_points = np.concatenate([source_joints, target_joints, pred_joints], axis=0)
    all_proj = _project_points(all_points)
    norm_proj = _normalize_points(all_proj, width=360, height=360)
    source_proj = norm_proj[:num_frames]
    target_proj = norm_proj[num_frames : 2 * num_frames]
    pred_proj = norm_proj[2 * num_frames :]

    canvas_w = 1600
    canvas_h = 420
    left_box = (20, 40, 380, 400)
    mid_box = (410, 40, 790, 400)
    gt_box = (820, 40, 1180, 400)
    right_box = (1210, 40, 1570, 400)
    font_title = _load_font(24)
    font_body = _load_font(18)
    font_small = _load_font(13)

    frames = []
    for frame_idx in range(num_frames):
        img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(left_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(mid_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(gt_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(right_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        source_title = "Source Prefix Motion" if prefix_frames > 1 else "Source Pose"
        draw.text((40, 10), source_title, fill=(20, 20, 20), font=font_title)
        draw.text((450, 10), "EditProgram", fill=(20, 20, 20), font=font_title)
        draw.text((835, 10), gt_title, fill=(20, 20, 20), font=font_title)
        draw.text((1230, 10), "Predicted Target Motion", fill=(20, 20, 20), font=font_title)

        has_part = "part" in program
        highlight_joints = set(BODY_PART_TO_JOINTS.get(program.get("part", ""), [])) if has_part else set()
        highlight_edges = _part_edge_indices(program["part"]) if has_part else set()

        source_panel = source_proj[frame_idx].copy()
        source_panel[:, 0] = source_panel[:, 0] - 180 + (left_box[0] + left_box[2]) / 2.0
        source_panel[:, 1] = source_panel[:, 1] - 180 + (left_box[1] + left_box[3]) / 2.0
        in_prefix = frame_indices[frame_idx] < prefix_frames
        source_base = (180, 184, 195) if in_prefix else (220, 223, 230)
        source_highlight = (55, 100, 230) if in_prefix else (150, 168, 215)
        _draw_skeleton(
            draw,
            source_panel,
            base_color=source_base,
            highlight_color=source_highlight,
            highlight_joints=highlight_joints,
            highlight_edges=highlight_edges,
        )

        gt_panel = target_proj[frame_idx].copy()
        gt_panel[:, 0] = gt_panel[:, 0] - 180 + (gt_box[0] + gt_box[2]) / 2.0
        gt_panel[:, 1] = gt_panel[:, 1] - 180 + (gt_box[1] + gt_box[3]) / 2.0
        _draw_skeleton(
            draw,
            gt_panel,
            base_color=(180, 184, 195),
            highlight_color=(80, 170, 80),
            highlight_joints=highlight_joints,
            highlight_edges=highlight_edges,
        )

        pred_panel = pred_proj[frame_idx].copy()
        pred_panel[:, 0] = pred_panel[:, 0] - 180 + (right_box[0] + right_box[2]) / 2.0
        pred_panel[:, 1] = pred_panel[:, 1] - 180 + (right_box[1] + right_box[3]) / 2.0
        _draw_skeleton(
            draw,
            pred_panel,
            base_color=(180, 184, 195),
            highlight_color=(220, 80, 60),
            highlight_joints=highlight_joints,
            highlight_edges=highlight_edges,
        )

        if is_edit_task:
            info_lines = [
                f"task: edit_task",
                f"part: {program['part']}",
                f"attribute: {program['attribute']}",
                f"direction: {program.get('direction', '-')}",
                f"delta_deg: {float(program.get('delta_value_deg') or 0.0):+.1f}",
                f"start: {int(program.get('start_frame', -1))}",
                f"end: {int(program.get('end_frame', -1))}",
                f"shown_frames: {int(frame_indices[0])}-{int(frame_indices[-1])}",
            ]
        elif task_mode == "semantic_continue":
            info_lines = [
                f"task: {task_mode}",
                f"part: {program['part']}",
                f"attribute: {program['attribute']}",
                f"direction: {program.get('direction', '-')}",
                f"delta_deg: {float(program.get('delta_value_deg') or 0.0):+.1f}",
                f"start: {int(program.get('start_frame', -1))}",
                f"end: {int(program.get('end_frame', -1))}",
                f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
                f"shown_frames: {int(frame_indices[0])}-{int(frame_indices[-1])}",
            ]
        else:
            if 'edits' in program:
                info_lines = [
                    f"task: {program.get('task_mode', 'multi_atomic_realize')}",
                    f"num_edits: {len(program.get('edits', []))}",
                    f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
                ]
                for i, edit in enumerate(program.get('edits', [])[:3], start=1):
                    info_lines.extend([
                        f"edit{i}.part: {edit.get('part', '-')}",
                        f"edit{i}.attribute: {edit.get('attribute', '-')}",
                        f"edit{i}.direction: {edit.get('direction', '-')}",
                        f"edit{i}.delta_deg: {float(edit.get('delta_value_deg') or 0.0):+.1f}",
                        f"edit{i}.span: {int(edit.get('start_frame', -1))}-{int(edit.get('end_frame', -1))}",
                    ])
            else:
                info_lines = [
                    f"task: {program.get('task_mode', 'continue')}",
                    f"part: {program.get('part', '-')}",
                    f"attribute: {program.get('attribute', '-')}",
                    f"direction: {program.get('direction', '-')}",
                    f"delta_deg: {float(program.get('delta_value_deg') or 0.0):+.1f}",
                    f"start: {int(program.get('start_frame', -1))}",
                    f"end: {int(program.get('end_frame', -1))}",
                    f"prefix_frames: {int(program.get('source_prefix_frames', 0))}",
                ]
        y_info = mid_box[1] + 30
        for line in info_lines[:12]:
            draw.text((mid_box[0] + 20, y_info), line, fill=(55, 55, 65), font=font_body)
            y_info += 22
        prompt_lines = _wrap_text(draw, f"prompt: {prompt_text}", font_small, max_width=(mid_box[2] - mid_box[0] - 40))
        y_prog = min(y_info + 8, mid_box[3] - 92)
        for line in prompt_lines[:3]:
            draw.text((mid_box[0] + 20, y_prog), line, fill=(95, 95, 105), font=font_small)
            y_prog += 16
        draw.text((mid_box[0] + 20, mid_box[3] - 50), f"frame {int(frame_indices[frame_idx]) + 1}/{display_total_frames}", fill=(80, 80, 90), font=font_body)
        draw.text((mid_box[0] + 20, mid_box[3] - 25), f"case {case_result['case_idx']}", fill=(80, 80, 90), font=font_body)

        frames.append(img)

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return output_path


def export_case_summary(results: list[dict], output_path: str) -> str:
    output = []
    for result in results:
        output.append(
            {
                "case_idx": result["case_idx"],
                "prompt_text": result["prompt_text"],
                "source_path": result["source_path"],
                "target_path": result["target_path"],
                "program": result["program"],
                "gif_path": result.get("gif_path"),
            }
        )
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    return output_path
