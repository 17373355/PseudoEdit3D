"""Legacy selected-HML3D-vs-event-stream MoMask probe.

Current motion-only AutoPrompt probing uses
``scripts/run_momask_aml_autoprompt_probe.py --prompt-mode coarse``.
This script is kept to reproduce earlier selected prompt vs old event-stream
comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.edit import (
    PhasePattern,
    attach_aml_language,
    build_layer3_atomic_program,
    dedupe_phase_patterns,
    detect_repeated_phases,
    extract_layer0_frame_observables,
    extract_layer1_micro_events,
    merge_micro_events,
    project_units_by_category,
    render_aml_prompt,
)

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'
RUNNER = ROOT_DIR / 'scripts' / 'run_momask_case_study.py'


def _json_safe(obj: Any) -> Any:
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


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_momask_case_study', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_prompts(case_id: str) -> list[str]:
    path = HML_ROOT / 'texts' / f'{case_id}.txt'
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line.split('#')[0].strip())
    return out


def read_first_prompt(case_id: str) -> str:
    prompts = read_prompts(case_id)
    return prompts[0] if prompts else ''


def load_joints3d_pack() -> dict[str, Any]:
    return torch.load(HML_ROOT / 'joints3d.pth', map_location='cpu')


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
        out.append(PhasePattern(
            name=str(p['name']),
            kind=str(p['kind']),
            count=int(p['count']),
            start_frame=int(p['start_frame']),
            end_frame=int(p['end_frame']),
            unit_names=list(p['unit_names']),
            metadata=dict(p.get('metadata', {})),
        ))
    out.sort(key=lambda p: (p.start_frame, p.end_frame, p.name))
    return out


def extract_aml_program(joints: np.ndarray) -> dict[str, Any]:
    poses = np.zeros((len(joints), 52, 3), dtype=np.float32)
    trans = joints[:, 0, :]
    layer0 = extract_layer0_frame_observables(poses=poses, joints=joints, trans=trans)
    layer1 = extract_layer1_micro_events(layer0)
    layer2 = merge_micro_events(layer1)
    phases = list(detect_repeated_phases(layer2))
    for category in ('whole_body', 'torso', 'left_arm', 'right_arm'):
        phases.extend(detect_repeated_phases(project_units_by_category(layer2, category)))
    phases = dedupe_phase_objects(phases)
    layer3 = attach_aml_language(build_layer3_atomic_program(layer2, phases, joints=joints))
    return {
        'layer1_count': len(layer1),
        'layer2_count': len(layer2),
        'layer25_count': len(phases),
        'layer3': layer3,
    }



def run_gen(prompt: str, ext: str, motion_length: int) -> int:
    cmd = [
        '/mnt/data/home/guoruoxi/miniconda3/envs/momask/bin/python',
        'gen_t2m.py',
        '--gpu_id', '0',
        '--dataset_name', 't2m',
        '--name', 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
        '--res_name', 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
        '--text_prompt', prompt,
        '--motion_length', str(motion_length),
        '--repeat_times', '1',
        '--time_steps', '10',
        '--cond_scale', '4',
        '--ext', ext,
    ]
    subprocess.run(cmd, cwd=MOMASK_ROOT, check=True)
    return int(motion_length // 4) * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--max-events', type=int, default=8)
    parser.add_argument('--ext-prefix', default='aml_probe')
    args = parser.parse_args()

    packed = load_joints3d_pack()
    mod = load_runner_module()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for case_id in [x.strip() for x in args.case_ids.split(',') if x.strip()]:
        key = f'{case_id}.npy'
        item = packed[key]
        joints = item['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        aml = extract_aml_program(joints)
        program = aml['layer3']
        auto_prompt = render_aml_prompt(program, max_events=args.max_events)
        gt_prompt = read_first_prompt(case_id)
        source_num_frames = int(len(joints))
        generated_num_frames = int((source_num_frames // 4) * 4)
        gt_ext = f'{args.ext_prefix}_{case_id}_gtprompt'
        auto_ext = f'{args.ext_prefix}_{case_id}_autoprompt'
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'aml_meta.json').write_text(json.dumps(_json_safe({
            'case_id': case_id,
            'gt_prompt': gt_prompt,
            'auto_prompt': auto_prompt,
            'aml': aml,
        }), ensure_ascii=True, indent=2), encoding='utf-8')
        if not args.skip_generation:
            generated_num_frames = run_gen(gt_prompt, gt_ext, source_num_frames)
            generated_num_frames = run_gen(auto_prompt, auto_ext, source_num_frames)
        summary.append({
            'case_id': case_id,
            'gt_prompt': gt_prompt,
            'auto_prompt': auto_prompt,
            'program': {'task_mode': 'aml_prompt_probe', 'source_prefix_frames': 0, 'edits': []},
            'auto_program': program,
            'gt_ext': gt_ext,
            'auto_ext': auto_ext,
            'source_num_frames': source_num_frames,
            'generated_num_frames': generated_num_frames,
            'raw_prompt_segments': [(x, 0.0, 0.0) for x in read_prompts(case_id)],
            'caption_prior': mod.build_caption_prior(case_id),
        })
        print(json.dumps(_json_safe(summary[-1]), ensure_ascii=True, indent=2))
    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2), encoding='utf-8')
    print(f'saved_summary={summary_path}')


if __name__ == '__main__':
    main()
