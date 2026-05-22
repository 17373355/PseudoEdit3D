from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHARRET_MULTI_ROOT = Path("/mnt/data/home/guoruoxi/code/CharRet_multi")
if str(CHARRET_MULTI_ROOT) not in sys.path:
    sys.path.append(str(CHARRET_MULTI_ROOT))

import body_models.smpl_skeleton_simple as smpl_skeleton_simple  # type: ignore

SMPLSkeleton = smpl_skeleton_simple.SMPLSkeleton

from pseudoedit3d.constants import BODY_PART_TO_JOINTS


BODY_EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (12, 13), (12, 14), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (12, 15),
]

DEFAULT_SMPLH_CANDIDATES = [
    "/mnt/data/home/guoruoxi/code/CharRet_multi/body_models/smplh/SMPLH_NEUTRAL.npz",
    "/mnt/data/home/guoruoxi/code/LoopReg/body_models/smplh/SMPLH_NEUTRAL.npz",
    "/mnt/data/home/guoruoxi/code/MyGVHMR/inputs/checkpoints/body_models/smplh/SMPLH_NEUTRAL.npz",
]


def _load_font(size: int = 20):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def resolve_smplh_model_path(user_path: str | None = None) -> str:
    if user_path and os.path.exists(user_path):
        return user_path
    for path in DEFAULT_SMPLH_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find a usable SMPL-H model. Checked: " + ", ".join(DEFAULT_SMPLH_CANDIDATES)
    )


def _load_model_data_allow_pickle(model_path: str):
    model_path = os.path.abspath(model_path)
    assert os.path.exists(model_path), f"Path {model_path} does not exist!"
    if model_path.endswith(".npz"):
        data = np.load(model_path, allow_pickle=True)
        return dict(data)
    if model_path.endswith(".pkl"):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    raise ValueError(f"Unsupported model file: {model_path}")


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


def _compute_joints(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray, smplh_model_path: str) -> np.ndarray:
    smpl_skeleton_simple.load_model_data = _load_model_data_allow_pickle
    model = SMPLSkeleton(model_path=smplh_model_path)
    device = model.device
    poses_t = poses.reshape(poses.shape[0], -1)
    if betas.shape[0] == 1:
        betas = np.repeat(betas, poses.shape[0], axis=0)
    params = {
        "poses": __import__("torch").tensor(poses_t, dtype=__import__("torch").float32, device=device),
        "trans": __import__("torch").tensor(trans, dtype=__import__("torch").float32, device=device),
        "shapes": __import__("torch").tensor(betas, dtype=__import__("torch").float32, device=device),
    }
    with __import__("torch").no_grad():
        joints = model(params)["keypoints3d"].cpu().numpy()
    return joints[:, :22]


def export_case_gif(
    case_result: dict,
    output_path: str,
    smplh_model_path: str | None = None,
    fps: int = 12,
    frame_limit: int | None = None,
) -> str:
    smplh_model_path = resolve_smplh_model_path(smplh_model_path)
    source_pose = case_result["source_pose"]
    target_pose = case_result["target_pose"]
    pred_pose = case_result["pred_pose"]
    source_trans = case_result["source_trans"]
    target_trans = case_result["target_trans"]
    betas = case_result["betas"]
    prompt_text = case_result["prompt_text"]
    program = case_result["program"]

    total_frames = pred_pose.shape[0]
    if frame_limit is None or frame_limit >= total_frames:
        frame_indices = np.arange(total_frames)
    else:
        span_start = int(program["start_frame"])
        span_end = int(program["end_frame"])
        span_center = 0.5 * (span_start + span_end)
        half = frame_limit // 2
        window_start = int(round(span_center)) - half
        window_start = max(0, min(window_start, total_frames - frame_limit))
        frame_indices = np.arange(window_start, window_start + frame_limit)
    num_frames = len(frame_indices)

    source_joints = _compute_joints(source_pose[frame_indices], source_trans[frame_indices], betas[:1], smplh_model_path)
    target_masked_pose = np.repeat(source_pose[:1], total_frames, axis=0)
    target_masked_pose[:] = source_pose[:1]
    target_joint_ids = BODY_PART_TO_JOINTS.get(program["part"], [])
    span_start = int(program["start_frame"])
    span_end = int(program["end_frame"])
    target_masked_pose[span_start:span_end + 1, target_joint_ids] = target_pose[span_start:span_end + 1, target_joint_ids]
    target_masked_trans = np.repeat(source_trans[:1], total_frames, axis=0)

    target_joints = _compute_joints(target_masked_pose[frame_indices], target_masked_trans[frame_indices], betas[:1], smplh_model_path)
    pred_joints = _compute_joints(pred_pose[frame_indices], target_masked_trans[frame_indices], betas[:1], smplh_model_path)

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

    frames = []
    for frame_idx in range(num_frames):
        img = Image.new("RGB", (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(left_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(mid_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(gt_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
        draw.rounded_rectangle(right_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        draw.text((40, 10), "Source Pose", fill=(20, 20, 20), font=font_title)
        draw.text((450, 10), "Prompt", fill=(20, 20, 20), font=font_title)
        draw.text((835, 10), "GT Prompt-Scoped Motion", fill=(20, 20, 20), font=font_title)
        draw.text((1230, 10), "Predicted Target Motion", fill=(20, 20, 20), font=font_title)

        highlight_joints = set(BODY_PART_TO_JOINTS.get(program["part"], []))
        highlight_edges = _part_edge_indices(program["part"])

        source_panel = source_proj[frame_idx].copy()
        source_panel[:, 0] = source_panel[:, 0] - 180 + (left_box[0] + left_box[2]) / 2.0
        source_panel[:, 1] = source_panel[:, 1] - 180 + (left_box[1] + left_box[3]) / 2.0
        _draw_skeleton(
            draw,
            source_panel,
            base_color=(180, 184, 195),
            highlight_color=(55, 100, 230),
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

        text_lines = _wrap_text(draw, prompt_text, font_body, max_width=(mid_box[2] - mid_box[0] - 40))
        y = mid_box[1] + 30
        for line in text_lines:
            draw.text((mid_box[0] + 20, y), line, fill=(30, 30, 35), font=font_body)
            y += 28
        info_lines = [
            f"part={program['part']}",
            f"attr={program['attribute']}",
            f"delta={float(program.get('delta_value_deg') or 0.0):+.1f} deg",
            f"span={program['start_frame']}-{program['end_frame']}",
            f"skill={program.get('skill_label', 'unknown')}",
            f"shown_frames={int(frame_indices[0])}-{int(frame_indices[-1])}",
        ]
        y_info = mid_box[1] + 170
        for line in info_lines:
            draw.text((mid_box[0] + 20, y_info), line, fill=(80, 80, 90), font=font_body)
            y_info += 24
        draw.text((mid_box[0] + 20, mid_box[3] - 50), f"frame {int(frame_indices[frame_idx]) + 1}/{total_frames}", fill=(80, 80, 90), font=font_body)
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
