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

from pseudoedit3d.edit import build_coarse_action_program, render_aml_prompt, render_coarse_aml_prompt

SOURCE = ROOT_DIR / 'scripts' / 'run_momask_aml_prompt_probe.py'
MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
HML_ROOT = MOMASK_ROOT / 'dataset' / 'HumanML3D'


def _load_source_module():
    spec = importlib.util.spec_from_file_location('run_momask_aml_prompt_probe', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


src = _load_source_module()


def _json_safe(obj: Any) -> Any:
    return src._json_safe(obj)


def load_case_ids(case_ids: str | None, case_list: str | None) -> list[str]:
    out: list[str] = []
    if case_ids:
        out.extend(x.strip() for x in case_ids.split(',') if x.strip())
    if case_list:
        for line in Path(case_list).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                out.append(line)
    seen = set()
    deduped = []
    for case_id in out:
        if case_id in seen:
            continue
        seen.add(case_id)
        deduped.append(case_id)
    return deduped


def generated_joint_exists(ext: str) -> bool:
    joint_dir = MOMASK_ROOT / 'generation' / ext / 'joints' / '0'
    return any(joint_dir.glob('sample0_repeat0_len*.npy'))


def run_gen(prompt: str, ext: str, motion_length: int, gpu_id: str, time_steps: int, cond_scale: int) -> int:
    cmd = [
        '/mnt/data/home/guoruoxi/miniconda3/envs/momask/bin/python',
        'gen_t2m.py',
        '--gpu_id', str(gpu_id),
        '--dataset_name', 't2m',
        '--name', 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
        '--res_name', 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
        '--text_prompt', prompt,
        '--motion_length', str(motion_length),
        '--repeat_times', '1',
        '--time_steps', str(time_steps),
        '--cond_scale', str(cond_scale),
        '--ext', ext,
    ]
    subprocess.run(cmd, cwd=MOMASK_ROOT, check=True)
    return int(motion_length // 4) * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-ids', default=None)
    parser.add_argument('--case-list', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--max-events', type=int, default=8)
    parser.add_argument('--prompt-mode', choices=['coarse', 'event_stream'], default='coarse')
    parser.add_argument('--ext-prefix', default='aml_auto_probe')
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--time-steps', type=int, default=10)
    parser.add_argument('--cond-scale', type=int, default=4)
    parser.add_argument('--reuse-existing', action='store_true')
    parser.add_argument('--verbose-json', action='store_true')
    parser.add_argument(
        '--caption-semantic-aliases',
        action='store_true',
        help='Use HML3D captions only to name compatible geometry patterns; motion evidence still comes from AML.',
    )
    parser.add_argument(
        '--caption-alias-source',
        choices=['first', 'all'],
        default='first',
        help='Caption source for --caption-semantic-aliases. first matches the displayed reference prompt; all uses every HML3D caption.',
    )
    args = parser.parse_args()

    case_ids = load_case_ids(args.case_ids, args.case_list)
    if not case_ids:
        raise SystemExit('No case ids provided')

    packed = src.load_joints3d_pack()
    mod = src.load_runner_module()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx, case_id in enumerate(case_ids, start=1):
        key = f'{case_id}.npy'
        if key not in packed:
            print(f'skip_missing={case_id}', flush=True)
            continue
        item = packed[key]
        joints = item['joints3d']
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()
        joints = np.asarray(joints, dtype=np.float32)
        aml = src.extract_aml_program(joints)
        program = aml['layer3']
        coarse_program = None
        caption_hints = None
        if args.caption_semantic_aliases:
            caption_hints = src.read_prompts(case_id) if args.caption_alias_source == 'all' else src.read_first_prompt(case_id)
        if args.prompt_mode == 'coarse':
            auto_prompt, coarse_program = render_coarse_aml_prompt(
                program,
                max_residual_events=args.max_events,
                caption_hints=caption_hints,
                return_program=True,
            )
        else:
            auto_prompt = render_aml_prompt(program, max_events=args.max_events)
            coarse_program = build_coarse_action_program(
                program,
                max_residual_events=args.max_events,
                caption_hints=caption_hints,
            )
        gt_prompt = src.read_first_prompt(case_id)
        source_num_frames = int(len(joints))
        generated_num_frames = int((source_num_frames // 4) * 4)
        auto_ext = f'{args.ext_prefix}_{case_id}_{args.prompt_mode}_autoprompt'
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'aml_meta.json').write_text(json.dumps(_json_safe({
            'case_id': case_id,
            'selected_hml3d_prompt_for_reference_only': gt_prompt,
            'auto_prompt': auto_prompt,
            'prompt_mode': args.prompt_mode,
            'caption_semantic_aliases_enabled': bool(args.caption_semantic_aliases),
            'caption_alias_source': args.caption_alias_source if args.caption_semantic_aliases else None,
            'canonical_actions': (coarse_program or {}).get('canonical_actions') or [],
            'coarse_action_program': coarse_program,
            'aml': aml,
        }), ensure_ascii=True, indent=2), encoding='utf-8')
        if not args.skip_generation:
            if args.reuse_existing and generated_joint_exists(auto_ext):
                print(f'[{idx}/{len(case_ids)}] reuse_existing case={case_id} ext={auto_ext}', flush=True)
            else:
                print(f'[{idx}/{len(case_ids)}] gen_auto case={case_id} frames={source_num_frames}', flush=True)
                generated_num_frames = run_gen(
                    auto_prompt,
                    auto_ext,
                    source_num_frames,
                    gpu_id=args.gpu_id,
                    time_steps=args.time_steps,
                    cond_scale=args.cond_scale,
                )
        row = {
            'case_id': case_id,
            'gt_prompt': gt_prompt,
            'auto_prompt': auto_prompt,
            'program': {'task_mode': 'aml_autoprompt_probe', 'source_prefix_frames': 0, 'edits': []},
            'auto_program': program,
            'canonical_actions': (coarse_program or {}).get('canonical_actions') or [],
            'coarse_action_program': coarse_program,
            'gt_ext': None,
            'auto_ext': auto_ext,
            'source_num_frames': source_num_frames,
            'generated_num_frames': generated_num_frames,
            'raw_prompt_segments': [(x, 0.0, 0.0) for x in src.read_prompts(case_id)],
            'caption_prior': mod.build_caption_prior(case_id),
            'generation': {
                'mode': 'auto_only',
                'prompt_mode': args.prompt_mode,
                'gpu_id': str(args.gpu_id),
                'time_steps': int(args.time_steps),
                'cond_scale': int(args.cond_scale),
                'caption_semantic_aliases_enabled': bool(args.caption_semantic_aliases),
                'caption_alias_source': args.caption_alias_source if args.caption_semantic_aliases else None,
            },
        }
        summary.append(row)
        if args.verbose_json:
            print(json.dumps(_json_safe(row), ensure_ascii=True, indent=2), flush=True)
        else:
            short_prompt = auto_prompt[:140] + ('...' if len(auto_prompt) > 140 else '')
            print(f'[{idx}/{len(case_ids)}] saved_case={case_id} generated_frames={generated_num_frames} prompt={short_prompt}', flush=True)

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2), encoding='utf-8')
    print(f'saved_summary={summary_path}', flush=True)


if __name__ == '__main__':
    main()
