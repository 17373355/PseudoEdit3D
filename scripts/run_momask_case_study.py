from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'
if str(MOMASK_ROOT) not in sys.path:
    sys.path.insert(0, str(MOMASK_ROOT))

from common.skeleton import Skeleton  # type: ignore
from utils import motion_process as mp  # type: ignore
from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets  # type: ignore


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def read_all_prompts(case_id: str) -> list[tuple[str, float, float]]:
    text_path = HML_ROOT / 'texts' / f'{case_id}.txt'
    out = []
    for line in text_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('#')
        caption = parts[0].strip()
        start, end = 0.0, 0.0
        if len(parts) >= 4:
            try:
                start = float(parts[-2])
                end = float(parts[-1])
            except Exception:
                pass
        out.append((caption, start, end))
    return out


def read_first_prompt(case_id: str) -> str:
    prompts = read_all_prompts(case_id)
    if not prompts:
        raise RuntimeError(f'No prompt lines for {case_id}')

    def score(caption: str) -> tuple[float, int]:
        caption_l = caption.lower()
        score = 0.0
        if 'up and down' in caption_l or 'down and up' in caption_l:
            score += 4.0
        if 'jumping up and down' in caption_l or 'bounce' in caption_l or 'hops' in caption_l:
            score += 3.0
        if 'stairs' in caption_l or 'stair' in caption_l or 'downstairs' in caption_l or 'upstairs' in caption_l:
            score += 2.0
        if 'turn' in caption_l or 'spin' in caption_l or 'rotate' in caption_l:
            score += 1.5
        if 'walk' in caption_l or 'run' in caption_l:
            score += 1.0
        if 'sup and down' in caption_l:
            score -= 1.0
        score += min(len(caption.split()) * 0.05, 1.0)
        return score, len(caption)

    return max((caption for caption, _, _ in prompts), key=score)


CAPTION_PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    'stair_descent': (
        'walk down some stairs', 'walks down some stairs', 'walking down the stairs', 'walk down stairs',
        'walks down stairs', 'downstairs', 'down the stairs', 'descends', 'descend', 'steps down stairs',
        'down steps', 'down the steps', 'walk back down', 'climb back down', 'walk back down the stairs',
        'walking downstairs', 'walks downstairs', 'steps downward', 'walking down some stairs',
    ),
    'stair_ascent': (
        'walk up stairs', 'walks up stairs', 'walking up stairs', 'upstairs', 'up the stairs', 'climb up',
        'climbs up', 'climbing up', 'steps up', 'step up', 'goes up steps', 'walking upstairs', 'walks upstairs',
    ),
    'walk_forward': (
        'walk forward', 'walks forward', 'walking forward', 'walk straight', 'walks straight', 'paces',
        'steps forward', 'walks down the road', 'walks casually forward', 'walks straight casually',
    ),
    'walk_backward': (
        'walk backward', 'walks backward', 'walking backward', 'walk back', 'walks back', 'backwards', 'walk back the opposite direction',
    ),
    'turn': (
        'turn', 'turns', 'turn around', 'spins', 'spin', 'rotate', 'rotates', 'rotating', 'turns around',
        'sharp turn', 'turns to the left', 'turns to the right',
    ),
    'jump_up': (
        'jump', 'jumps', 'jump up', 'jumps up', 'hop', 'hops', 'bounce', 'bounces', 'hops up', 'little jump',
    ),
    'crouch_bend': (
        'crouch', 'crouches', 'bend', 'bends', 'bent over', 'squat', 'squats', 'squatting',
    ),
    'arm_support': (
        'railing', 'rail', 'bannister', 'handrail', 'support', 'holding', 'holding on', 'uses their right arm as support',
        'using the handrail', 'holding a railing', 'holding onto',
    ),
    'stop_pause': (
        'stops', 'stop', 'halts', 'halt', 'pause', 'paused',
    ),
}


def build_caption_prior(case_id: str) -> dict[str, float]:
    prompts = read_all_prompts(case_id)
    captions = [caption.lower() for caption, _, _ in prompts]
    total = max(len(captions), 1)

    prior: dict[str, float] = {}
    matched_captions: dict[str, list[str]] = {}
    for key, keywords in CAPTION_PATTERN_GROUPS.items():
        hits = []
        for caption in captions:
            if any(keyword in caption for keyword in keywords):
                hits.append(caption)
        prior[key] = len(hits) / total
        matched_captions[key] = hits
    prior['num_captions'] = float(len(captions))
    prior['_matched_captions'] = matched_captions
    return prior


def load_joints3d(case_id: str) -> np.ndarray:
    pth = HML_ROOT / 'joints3d.pth'
    data = torch.load(pth, map_location='cpu')
    key = f'{case_id}.npy'
    item = data[key]
    joints = item['joints3d']
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    return np.asarray(joints, dtype=np.float32)


def setup_momask_preprocess(example_joints22: np.ndarray) -> None:
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = 22
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, 'cpu')
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_joints22[0]).float())


def heading_deg(joints: np.ndarray) -> np.ndarray:
    l_hip = joints[:, 1][:, [0, 2]]
    r_hip = joints[:, 2][:, [0, 2]]
    l_sh = joints[:, 16][:, [0, 2]]
    r_sh = joints[:, 17][:, [0, 2]]
    across = (r_hip - l_hip) + (r_sh - l_sh)
    forward = np.stack([-across[:, 1], across[:, 0]], axis=-1)
    forward = forward / np.clip(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8, None)
    yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))
    return yaw - yaw[0]


def root_height(joints: np.ndarray) -> np.ndarray:
    return joints[:, 0, 1]


def root_xz_displacement(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    disp = root_xz - root_xz[0:1]
    return np.linalg.norm(disp, axis=-1)


def root_forward_progress(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    delta = root_xz - root_xz[0:1]
    return delta[:, 1]


def root_step_speed(joints: np.ndarray) -> np.ndarray:
    root_xz = joints[:, 0][:, [0, 2]]
    step = np.diff(root_xz, axis=0, prepend=root_xz[0:1])
    return np.linalg.norm(step, axis=-1)


def both_arm_raise_deg(joints: np.ndarray) -> np.ndarray:
    left_raise = (joints[:, 20, 1] - joints[:, 16, 1]) * 180.0
    right_raise = (joints[:, 21, 1] - joints[:, 17, 1]) * 180.0
    return np.minimum(left_raise, right_raise)


def first_cross(values: np.ndarray, source: float, thr: float, start: int) -> int:
    for i in range(start, len(values)):
        if abs(float(values[i] - source)) >= thr:
            return i
    return start


def find_repeated_height_peaks(values: np.ndarray, source_value: float, start_frame: int, min_peak: float = 0.03, min_gap: int = 3) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    future = values[start_frame:]
    last_peak = -min_gap
    for i in range(1, len(future) - 1):
        delta = float(future[i] - source_value)
        if delta < min_peak:
            continue
        if future[i] > future[i - 1] and future[i] >= future[i + 1] and (i - last_peak) >= min_gap:
            peaks.append((start_frame + i, delta))
            last_peak = i
    return peaks


def _format_turn(delta: float) -> str:
    side = 'left' if delta >= 0 else 'right'
    return f'turns {side} by about {abs(delta):.0f} degrees'


def _edit_clause(edit: dict) -> str:
    attr = edit.get('attribute', '')
    delta = float(edit.get('delta_value_deg') or 0.0)
    if attr == 'turn_left' or attr == 'turn_right':
        return _format_turn(delta)
    if attr == 'walk_forward':
        return 'walks forward'
    if attr == 'walk_backward':
        return 'walks backward'
    if attr == 'stair_descent':
        return 'walks down some stairs'
    if attr == 'stair_ascent':
        return 'walks up some stairs'
    if attr == 'jump_up':
        if abs(delta) < 0.15:
            return 'jumps up slightly'
        return 'jumps upward'
    if attr == 'bounce_repeated':
        return 'jumps up and down repeatedly in place'
    if attr == 'crouch_bend':
        return 'squats down and rises back up'
    if attr == 'stop_pause':
        return 'comes to a stop'
    if attr == 'raise':
        return 'raises both arms'
    if attr == 'land':
        return 'lands back down'
    if attr == 'lean_forward':
        return 'leans forward'
    if attr == 'lean_backward':
        return 'leans backward'
    return attr.replace('_', ' ')


def naturalize_auto_prompt(prompts: list[str], edits: list[dict]) -> str:
    del prompts
    attrs = [edit.get('attribute', '') for edit in edits]
    attr_set = set(attrs)

    if 'bounce_repeated' in attr_set:
        return 'a person is jumping up and down repeatedly in place'

    if 'stair_ascent' in attr_set and 'stair_descent' in attr_set and ('turn_left' in attr_set or 'turn_right' in attr_set):
        return 'a person walks up the stairs, turns around, and walks back down the stairs'

    if 'stair_descent' in attr_set and 'walk_forward' in attr_set and 'stair_ascent' not in attr_set:
        return 'a person walks down some stairs'

    if 'stair_ascent' in attr_set and 'walk_forward' in attr_set and 'stair_descent' not in attr_set:
        return 'a person walks up some stairs'

    if 'walk_backward' in attr_set and ('turn_left' in attr_set or 'turn_right' in attr_set):
        return 'a person walks forward, turns around, and walks back'

    clauses = []
    for edit in edits:
        clause = _edit_clause(edit)
        if clause not in clauses:
            clauses.append(clause)

    if not clauses:
        return 'a person moves naturally'
    if len(clauses) == 1:
        return f'a person {clauses[0]}'
    if len(clauses) == 2:
        return f'a person {clauses[0]} and then {clauses[1]}'
    return 'a person ' + ', then '.join([clauses[0], *clauses[1:]])


def build_auto_prompt(case_id: str, prefix_frames: int) -> dict:
    joints = load_joints3d(case_id)
    setup_momask_preprocess(joints)
    motion_263, _, _, _ = mp.process_file(joints.copy(), 0.002)

    yaw = heading_deg(joints)
    yaw_source = float(yaw[min(prefix_frames - 1, len(yaw) - 1)])
    yaw_future = yaw[prefix_frames:]
    height = root_height(joints)
    h_source = float(height[min(prefix_frames - 1, len(height) - 1)])
    h_future = height[prefix_frames:]
    xz_disp = root_xz_displacement(joints)
    xz_source = float(xz_disp[min(prefix_frames - 1, len(xz_disp) - 1)])
    xz_future = xz_disp[prefix_frames:]
    fwd = root_forward_progress(joints)
    fwd_source = float(fwd[min(prefix_frames - 1, len(fwd) - 1)])
    fwd_future = fwd[prefix_frames:]
    step_speed = root_step_speed(joints)
    arm = both_arm_raise_deg(joints)
    a_source = float(arm[min(prefix_frames - 1, len(arm) - 1)])
    a_future = arm[prefix_frames:]

    prompts = []
    edits = []
    stair_detected = False
    stair_ascent_detected = False

    if len(yaw_future) > 0:
        max_idx = int(np.argmax(yaw_future))
        min_idx = int(np.argmin(yaw_future))
        max_delta = float(yaw_future[max_idx] - yaw_source)
        min_delta = float(yaw_future[min_idx] - yaw_source)
        if abs(max_delta) >= abs(min_delta):
            delta = max_delta
            peak_idx = max_idx
        else:
            delta = min_delta
            peak_idx = min_idx
        turn_thr = 20.0
        if abs(delta) >= turn_thr:
            attr = 'turn_left' if delta >= 0 else 'turn_right'
            st = first_cross(yaw, yaw_source, 0.15 * abs(delta), prefix_frames)
            ed = min(len(yaw) - 1, prefix_frames + max(peak_idx, 6))
            prompts.append(f'whole_body {attr} by {abs(delta):.1f} degrees from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': attr, 'delta_value_deg': delta, 'start_frame': st, 'end_frame': ed})

    if len(fwd_future) > 0 and len(h_future) > 0:
        forward_delta = float(np.max(fwd_future) - fwd_source)
        backward_delta = float(np.min(fwd_future) - fwd_source)
        total_drop = float(np.min(h_future) - h_source)
        total_rise = float(np.max(h_future) - h_source)
        walk_thr = 0.45
        stair_thr = 0.18
        ascent_thr = 0.18
        if forward_delta >= walk_thr:
            peak_idx = int(np.argmax(fwd_future))
            st = first_cross(fwd, fwd_source, max(0.12, 0.2 * abs(forward_delta)), prefix_frames)
            ed = min(len(fwd) - 1, prefix_frames + max(peak_idx, 8))
            prompts.append(f'whole_body walk_forward by {forward_delta:.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'walk_forward', 'delta_value_deg': forward_delta, 'start_frame': st, 'end_frame': ed})
            tail = step_speed[min(len(step_speed) - 1, prefix_frames + peak_idx + 1):]
            peak_speed = float(np.max(step_speed[prefix_frames:])) if len(step_speed) > prefix_frames else 0.0
            if len(tail) >= 4 and float(np.mean(tail[-min(8, len(tail)):])) <= 0.02 and peak_speed >= 0.04:
                stop_st = min(len(step_speed) - 1, prefix_frames + peak_idx + 1)
                stop_ed = min(len(step_speed) - 1, stop_st + max(4, min(12, len(tail))))
                prompts.append(f'whole_body stop_pause from frame {stop_st} to {stop_ed}')
                edits.append({'part': 'whole_body', 'attribute': 'stop_pause', 'delta_value_deg': 0.0, 'start_frame': stop_st, 'end_frame': stop_ed})
            if total_drop <= -stair_thr:
                stair_detected = True
                stair_idx = int(np.argmin(h_future))
                stair_st = min(st, prefix_frames + max(0, stair_idx - 6))
                stair_ed = min(len(height) - 1, prefix_frames + max(stair_idx, 8))
                prompts.append(f'whole_body stair_descent by {abs(total_drop):.2f} meters from frame {stair_st} to {stair_ed}')
                edits.append({'part': 'whole_body', 'attribute': 'stair_descent', 'delta_value_deg': total_drop, 'start_frame': stair_st, 'end_frame': stair_ed})
            if total_rise >= ascent_thr:
                rise_idx = int(np.argmax(h_future))
                rise_st = min(st, prefix_frames + max(0, rise_idx - 6))
                rise_ed = min(len(height) - 1, prefix_frames + max(rise_idx, 8))
                stair_ascent_detected = True
                prompts.append(f'whole_body stair_ascent by {total_rise:.2f} meters from frame {rise_st} to {rise_ed}')
                edits.append({'part': 'whole_body', 'attribute': 'stair_ascent', 'delta_value_deg': total_rise, 'start_frame': rise_st, 'end_frame': rise_ed})
        elif backward_delta <= -walk_thr:
            back_idx = int(np.argmin(fwd_future))
            st = first_cross(fwd, fwd_source, max(0.12, 0.2 * abs(backward_delta)), prefix_frames)
            ed = min(len(fwd) - 1, prefix_frames + max(back_idx, 8))
            prompts.append(f'whole_body walk_backward by {abs(backward_delta):.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'walk_backward', 'delta_value_deg': backward_delta, 'start_frame': st, 'end_frame': ed})

    if len(h_future) > 0:
        max_idx = int(np.argmax(h_future))
        min_idx = int(np.argmin(h_future))
        max_delta = float(h_future[max_idx] - h_source)
        min_delta = float(h_future[min_idx] - h_source)
        repeated_peaks = find_repeated_height_peaks(height, h_source, prefix_frames)
        horizontal_range = max(abs(forward_delta), abs(backward_delta)) if len(fwd_future) > 0 else 0.0
        stationary_like = float(np.max(xz_future) - xz_source) <= 0.35
        crouch_like = (horizontal_range <= 0.25) and (max_delta >= 0.12 or min_delta <= -0.12) and ((max_delta - min_delta) >= 0.18) and not (stair_detected or stair_ascent_detected)
        if len(repeated_peaks) >= 3 and stationary_like and (not stair_detected):
            st = repeated_peaks[0][0]
            ed = repeated_peaks[-1][0]
            peak_height = max(delta for _, delta in repeated_peaks)
            prompts.append(f'whole_body bounce_up_down repeated {len(repeated_peaks)} times from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'bounce_repeated', 'delta_value_deg': peak_height, 'start_frame': st, 'end_frame': ed})
        elif crouch_like:
            st = first_cross(height, h_source, max(0.05, 0.2 * max(abs(max_delta), abs(min_delta))), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(max(max_idx, min_idx), 8))
            prompts.append(f'whole_body crouch_bend from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'crouch_bend', 'delta_value_deg': max(max_delta, abs(min_delta)), 'start_frame': st, 'end_frame': ed})
        elif max_delta >= 0.08 and not (stair_detected or stair_ascent_detected):
            st = first_cross(height, h_source, 0.2 * abs(max_delta), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(max_idx, 4))
            prompts.append(f'whole_body jump_up by {max_delta:.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'jump_up', 'delta_value_deg': max_delta, 'start_frame': st, 'end_frame': ed})
        if (not stair_detected) and (not crouch_like) and min_delta <= -0.12 and max_delta >= 0.04 and len(repeated_peaks) < 3:
            st = first_cross(height, h_source, 0.2 * abs(min_delta), prefix_frames)
            ed = min(len(height) - 1, prefix_frames + max(min_idx, 4))
            prompts.append(f'whole_body land by {abs(min_delta):.2f} meters from frame {st} to {ed}')
            edits.append({'part': 'whole_body', 'attribute': 'land', 'delta_value_deg': min_delta, 'start_frame': st, 'end_frame': ed})

    if len(a_future) > 0:
        peak_idx = int(np.argmax(a_future))
        delta = float(a_future[peak_idx] - a_source)
        arm_thr = 50.0
        if abs(delta) >= arm_thr:
            st = first_cross(arm, a_source, 0.2 * abs(delta), prefix_frames)
            ed = min(len(arm) - 1, prefix_frames + max(peak_idx, 4))
            prompts.append(f'both_arms raise by {abs(delta):.1f} degrees from frame {st} to {ed}')
            edits.append({'part': 'both_arms', 'attribute': 'raise', 'delta_value_deg': delta, 'start_frame': st, 'end_frame': ed})

    text_prompt = naturalize_auto_prompt(prompts, edits)
    if edits:
        mask_start = min(int(e['start_frame']) for e in edits)
        mask_end = max(int(e['end_frame']) for e in edits)
    else:
        mask_start = prefix_frames
        mask_end = joints.shape[0] - 1

    return {
        'motion_263': motion_263.astype(np.float32),
        'program': {'task_mode': 'multi_atomic_realize', 'edits': edits, 'source_prefix_frames': prefix_frames},
        'auto_prompt': text_prompt,
        'mask_edit_section': f'{mask_start},{mask_end}',
        'source_num_frames': int(len(joints)),
        'raw_prompt_segments': read_all_prompts(case_id),
    }


def run_gen(prompt: str, ext: str) -> None:
    cmd = [
        '/mnt/data/home/guoruoxi/miniconda3/envs/momask/bin/python',
        'gen_t2m.py',
        '--gpu_id', '0',
        '--dataset_name', 't2m',
        '--name', 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
        '--res_name', 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
        '--text_prompt', prompt,
        '--motion_length', '60',
        '--repeat_times', '1',
        '--time_steps', '10',
        '--cond_scale', '4',
        '--ext', ext,
    ]
    subprocess.run(cmd, cwd=MOMASK_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--prefix-frames', type=int, default=20)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    case_ids = [x.strip() for x in args.case_ids.split(',') if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for case_id in case_ids:
        gt_prompt = read_first_prompt(case_id)
        auto = build_auto_prompt(case_id, args.prefix_frames)
        caption_prior = build_caption_prior(case_id)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / 'source_motion.npy', auto['motion_263'])
        (case_dir / 'auto_meta.json').write_text(json.dumps(_json_safe(auto), ensure_ascii=True, indent=2), encoding='utf-8')
        gt_ext = f'case_study_{case_id}_gtprompt'
        auto_ext = f'case_study_{case_id}_autoprompt'
        run_gen(gt_prompt, gt_ext)
        run_gen(auto['auto_prompt'], auto_ext)
        summary.append({
            'case_id': case_id,
            'gt_prompt': gt_prompt,
            'auto_prompt': auto['auto_prompt'],
            'program': auto['program'],
            'gt_ext': gt_ext,
            'auto_ext': auto_ext,
            'source_num_frames': auto['source_num_frames'],
            'raw_prompt_segments': auto['raw_prompt_segments'],
            'caption_prior': caption_prior,
        })

    (out_dir / 'summary.json').write_text(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2))
    print(f'saved_summary={out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
