from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.constants import BODY_PART_TO_JOINTS
from pseudoedit3d.edit import (
    aml_event_to_template,
    aml_program_to_templates,
    attach_aml_language,
    PhasePattern,
    build_layer3_atomic_program,
    dedupe_phase_patterns,
    detect_repeated_phases,
    extract_layer0_frame_observables,
    extract_layer1_micro_events,
    merge_micro_events,
    project_units_by_category,
)
from pseudoedit3d.visualization.skeleton_gif import (
    _draw_skeleton,
    _load_font,
    _normalize_points,
    _part_edge_indices,
    _project_points,
    _wrap_text,
)

HML_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D')

PART_COLORS = {
    'whole_body': (60, 150, 85),
    'torso': (220, 140, 45),
    'left_arm': (52, 113, 235),
    'right_arm': (205, 80, 85),
    'both_arms': (175, 95, 210),
}


def load_case_ids(args: argparse.Namespace) -> list[str]:
    case_ids: list[str] = []
    if args.case_ids:
        case_ids.extend(x.strip() for x in args.case_ids.split(',') if x.strip())
    if args.case_list:
        for line in Path(args.case_list).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                case_ids.append(line)
    seen = set()
    out = []
    for case_id in case_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        out.append(case_id)
    return out


def read_first_prompt(case_id: str) -> str:
    text_path = HML_ROOT / 'texts' / f'{case_id}.txt'
    if not text_path.exists():
        return ''
    prompts = []
    for line in text_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(line.split('#')[0].strip())
    return prompts[0] if prompts else ''


def phase_to_dict(phase: PhasePattern) -> dict[str, Any]:
    return {
        'name': phase.name,
        'kind': phase.kind,
        'count': int(phase.count),
        'start_frame': int(phase.start_frame),
        'end_frame': int(phase.end_frame),
        'unit_names': list(phase.unit_names),
        'metadata': dict(phase.metadata),
    }


def dedupe_phase_objects(phases: list[PhasePattern]) -> list[PhasePattern]:
    deduped = dedupe_phase_patterns([phase_to_dict(p) for p in phases])
    out: list[PhasePattern] = []
    for p in deduped:
        out.append(
            PhasePattern(
                name=str(p['name']),
                kind=str(p['kind']),
                count=int(p['count']),
                start_frame=int(p['start_frame']),
                end_frame=int(p['end_frame']),
                unit_names=list(p['unit_names']),
                metadata=dict(p.get('metadata', {})),
            )
        )
    out.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return out


def extract_layer3(joints: np.ndarray) -> dict[str, Any]:
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases = list(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
    phases = dedupe_phase_objects(phases)
    layer3 = attach_aml_language(build_layer3_atomic_program(layer2, phases))
    return {
        'layer1_count': len(layer1),
        'layer2_count': len(layer2),
        'layer25_count': len(phases),
        'layer3': layer3,
    }


def active_events(events: list[dict[str, Any]], frame_idx: int) -> list[dict[str, Any]]:
    return [e for e in events if int(e['start_frame']) <= frame_idx <= int(e['end_frame'])]


def event_label(evt: dict[str, Any]) -> str:
    return aml_event_to_template(evt, detail=False)


def event_sort_key(evt: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(evt.get('start_frame', -1)),
        int(evt.get('end_frame', -1)),
        str(evt.get('super_family', '')),
        str(evt.get('cluster_id', '')),
    )


def draw_timeline(
    draw: ImageDraw.ImageDraw,
    events: list[dict[str, Any]],
    frame_idx: int,
    num_frames: int,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, outline=(220, 220, 228), width=1, fill=(250, 250, 252))
    width = max(1, x1 - x0 - 20)
    base_y = y0 + 20
    for idx, evt in enumerate(events[:16]):
        y = base_y + idx * 8
        if y > y1 - 12:
            break
        s = int(evt.get('start_frame', 0))
        e = int(evt.get('end_frame', s))
        xs = x0 + 10 + int(width * s / max(num_frames - 1, 1))
        xe = x0 + 10 + int(width * e / max(num_frames - 1, 1))
        color = PART_COLORS.get(str(evt.get('part')), (120, 120, 130))
        draw.line((xs, y, max(xs + 2, xe), y), fill=color, width=3)
    xf = x0 + 10 + int(width * frame_idx / max(num_frames - 1, 1))
    draw.line((xf, y0 + 6, xf, y1 - 6), fill=(20, 20, 25), width=2)


def draw_program_panel(
    draw: ImageDraw.ImageDraw,
    case_id: str,
    prompt: str,
    events: list[dict[str, Any]],
    current: list[dict[str, Any]],
    frame_idx: int,
    num_frames: int,
    counts: dict[str, int],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    font_title = _load_font(22)
    font_body = _load_font(15)
    font_small = _load_font(12)
    draw.rounded_rectangle(box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))
    y = y0 + 14
    draw.text((x0 + 18, y), 'AML Atomic Program', fill=(20, 20, 20), font=font_title)
    y += 28
    draw.text((x0 + 18, y), f'case: {case_id}    frame: {frame_idx + 1}/{num_frames}', fill=(70, 70, 80), font=font_body)
    y += 22
    draw.text((x0 + 18, y), f"L1={counts['layer1_count']}  L2={counts['layer2_count']}  Phase={counts['layer25_count']}  L3={len(events)}", fill=(70, 70, 80), font=font_body)
    y += 26
    for line in _wrap_text(draw, f'HML3D prompt: {prompt}', font_small, x1 - x0 - 36)[:4]:
        draw.text((x0 + 18, y), line, fill=(70, 105, 190), font=font_small)
        y += 15
    y += 8
    draw.text((x0 + 18, y), 'Active events', fill=(20, 20, 25), font=font_body)
    y += 20
    if current:
        for evt in sorted(current, key=event_sort_key)[:6]:
            color = PART_COLORS.get(str(evt.get('part')), (120, 120, 130))
            draw.rounded_rectangle((x0 + 18, y + 3, x0 + 28, y + 13), radius=3, fill=color)
            for line in _wrap_text(draw, event_label(evt), font_small, x1 - x0 - 56)[:2]:
                draw.text((x0 + 34, y), line, fill=(40, 40, 50), font=font_small)
                y += 15
            y += 3
    else:
        draw.text((x0 + 24, y), '(none)', fill=(130, 130, 140), font=font_small)
        y += 20
    y += 4
    timeline_h = 150
    draw_timeline(draw, sorted(events, key=event_sort_key), frame_idx, num_frames, (x0 + 18, y, x1 - 18, y + timeline_h))
    y += timeline_h + 16
    draw.text((x0 + 18, y), 'Event list', fill=(20, 20, 25), font=font_body)
    y += 20
    for evt in sorted(events, key=event_sort_key)[:10]:
        is_active = evt in current
        fill = (15, 15, 20) if is_active else (85, 85, 96)
        prefix = '* ' if is_active else '  '
        for line in _wrap_text(draw, prefix + event_label(evt), font_small, x1 - x0 - 36)[:2]:
            if y > y1 - 18:
                return
            draw.text((x0 + 18, y), line, fill=fill, font=font_small)
            y += 14


def render_case(case_id: str, joints: np.ndarray, output_path: Path, fps: int, max_events: int | None = None) -> dict[str, Any]:
    extracted = extract_layer3(joints)
    events = sorted(extracted['layer3']['events'], key=event_sort_key)
    if max_events is not None:
        events = events[:max_events]
    prompt = read_first_prompt(case_id)

    projected = _normalize_points(_project_points(joints), width=520, height=520)
    num_frames = len(joints)
    canvas_w = 1280
    canvas_h = 620
    motion_box = (20, 56, 570, 596)
    program_box = (600, 56, 1260, 596)
    font_title = _load_font(24)
    font_body = _load_font(16)

    frames = []
    for frame_idx in range(num_frames):
        img = Image.new('RGB', (canvas_w, canvas_h), color=(247, 247, 250))
        draw = ImageDraw.Draw(img)
        draw.text((40, 16), 'GT Full Motion', fill=(20, 20, 20), font=font_title)
        draw.text((610, 16), 'Motion-derived AML', fill=(20, 20, 20), font=font_title)
        draw.rounded_rectangle(motion_box, radius=16, outline=(210, 210, 220), width=2, fill=(255, 255, 255))

        current = active_events(events, frame_idx)
        highlight_joints: set[int] = set()
        highlight_edges: set[int] = set()
        active_parts = {str(evt.get('part')) for evt in current}
        for part in active_parts:
            highlight_joints.update(BODY_PART_TO_JOINTS.get(part, []))
            highlight_edges.update(_part_edge_indices(part))
        highlight_color = PART_COLORS.get(next(iter(active_parts), ''), (80, 170, 80))

        panel = projected[frame_idx].copy()
        panel[:, 0] = panel[:, 0] - 260 + (motion_box[0] + motion_box[2]) / 2.0
        panel[:, 1] = panel[:, 1] - 260 + (motion_box[1] + motion_box[3]) / 2.0
        _draw_skeleton(
            draw,
            panel,
            base_color=(176, 181, 193),
            highlight_color=highlight_color,
            highlight_joints=highlight_joints,
            highlight_edges=highlight_edges,
            radius=4,
            width=3,
        )
        draw.text((motion_box[0] + 20, motion_box[3] - 56), f'frame {frame_idx + 1}/{num_frames}', fill=(70, 70, 80), font=font_body)
        draw.text((motion_box[0] + 20, motion_box[3] - 30), f'case {case_id}', fill=(70, 70, 80), font=font_body)

        draw_program_panel(
            draw,
            case_id=case_id,
            prompt=prompt,
            events=events,
            current=current,
            frame_idx=frame_idx,
            num_frames=num_frames,
            counts={
                'layer1_count': int(extracted['layer1_count']),
                'layer2_count': int(extracted['layer2_count']),
                'layer25_count': int(extracted['layer25_count']),
            },
            box=program_box,
        )
        frames.append(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / max(fps, 1)))
    frames[0].save(str(output_path), save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    return {
        'case_id': case_id,
        'num_frames': int(num_frames),
        'gif_path': str(output_path),
        'selected_hml3d_prompt': prompt,
        'layer1_count': int(extracted['layer1_count']),
        'layer2_count': int(extracted['layer2_count']),
        'layer25_count': int(extracted['layer25_count']),
        'layer3_count': len(events),
        'events': events,
        'aml_language_compact': aml_program_to_templates({'events': events}, detail=False),
        'aml_language_detailed': aml_program_to_templates({'events': events}, detail=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', default=None)
    parser.add_argument('--case-list', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--max-events', type=int, default=None)
    args = parser.parse_args()

    case_ids = load_case_ids(args)
    if not case_ids:
        raise SystemExit('No case ids provided')

    packed = torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')
    out_dir = Path(args.output_dir)
    summaries = []
    for case_id in case_ids:
        key = f'{case_id}.npy'
        if key not in packed:
            print(f'skip_missing={case_id}')
            continue
        joints = packed[key]['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        out_path = out_dir / f'case_{case_id}.gif'
        item = render_case(case_id, joints, output_path=out_path, fps=args.fps, max_events=args.max_events)
        summaries.append(item)
        print(f'saved_aml_vis={out_path}')

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summaries, ensure_ascii=True, indent=2), encoding='utf-8')
    print(summary_path)


if __name__ == '__main__':
    main()
