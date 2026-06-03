from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pseudoedit3d.data.prefix_dataset import PrefixMotionDataset

MOMASK_ROOT = Path('/mnt/data/home/guoruoxi/code/momask-codes')
if str(MOMASK_ROOT) not in sys.path:
    sys.path.append(str(MOMASK_ROOT))

CHARRET_ROOT = Path('/mnt/data/home/guoruoxi/code/CharRet_multi')
if str(CHARRET_ROOT) not in sys.path:
    sys.path.append(str(CHARRET_ROOT))

from utils import motion_process as mp  # type: ignore
from body_models.smpl_skeleton_simple import SMPLSkeleton  # type: ignore
from common.skeleton import Skeleton  # type: ignore
from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets  # type: ignore


def _setup_momask_preprocess() -> None:
    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_r, mp.fid_l = [8, 11], [7, 10]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = 22
    mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    mp.kinematic_chain = t2m_kinematic_chain


def _init_target_offsets(example_joints22: np.ndarray) -> None:
    tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, 'cpu')
    mp.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_joints22[0]).float())


def _recover_smpl22(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray) -> np.ndarray:
    model_path = '/mnt/data/home/guoruoxi/code/CharRet_multi/body_models/smplh/SMPLH_NEUTRAL.npz'
    model = SMPLSkeleton(model_path=model_path)
    num_frames = poses.shape[0]
    if betas.ndim == 2 and betas.shape[0] == 1:
        betas = np.repeat(betas, num_frames, axis=0)
    elif betas.ndim == 1:
        betas = np.repeat(betas[None], num_frames, axis=0)
    params = {
        'poses': torch.from_numpy(poses.reshape(num_frames, -1)).float(),
        'trans': torch.from_numpy(trans).float(),
        'shapes': torch.from_numpy(betas).float(),
    }
    with torch.no_grad():
        joints = model(params)['keypoints3d'].cpu().numpy()
    return joints[:, :22].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--case-idx', type=int, required=True)
    parser.add_argument('--prefix-frames', type=int, default=20)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    ds = PrefixMotionDataset(
        manifest_path=args.manifest,
        prefix_frames=args.prefix_frames,
        task_mode='multi_atomic_realize',
        input_source_mode='target_prefix_masked',
    )
    sample = ds[args.case_idx]
    program = json.loads(sample['program_json'])
    source_path = Path(sample['source_path'])
    data = np.load(source_path, allow_pickle=True)
    poses = data['poses'].reshape(-1, 52, 3).astype(np.float32)
    trans = data['trans'].astype(np.float32)
    betas = np.asarray(data.get('betas', np.zeros((1, 16), dtype=np.float32)), dtype=np.float32)

    _setup_momask_preprocess()
    joints22 = _recover_smpl22(poses, trans, betas)
    _init_target_offsets(joints22)
    motion_263, _, _, _ = mp.process_file(joints22.copy(), 0.002)

    edits = program.get('edits', [])
    text_prompt = ' then '.join([
        f"{e['part']} {e['attribute']} {e['direction']} by {float(e.get('delta_value_deg') or 0.0):.1f} degrees from frame {int(e['start_frame'])} to {int(e['end_frame'])}"
        for e in edits
    ])
    if not text_prompt:
        text_prompt = sample['prompt_text']

    if edits:
        mask_start = min(int(e['start_frame']) for e in edits)
        mask_end = max(int(e['end_frame']) for e in edits)
    else:
        mask_start = args.prefix_frames
        mask_end = motion_263.shape[0] - 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'source_motion.npy', motion_263.astype(np.float32))
    meta = {
        'case_idx': args.case_idx,
        'source_path': str(source_path),
        'text_prompt': text_prompt,
        'mask_edit_section': f'{mask_start},{mask_end}',
        'program': program,
        'prompt_text': sample['prompt_text'],
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=True, indent=2))
    print(f'saved_source_motion={out_dir / "source_motion.npy"}')


if __name__ == '__main__':
    main()
