from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


DEFAULT_MOMASK_ROOT = Path("/mnt/data/home/guoruoxi/code/momask-codes")
DEFAULT_PYTHON = Path("/mnt/data/home/guoruoxi/miniconda3/envs/momask/bin/python")


def _load_cases(summary_paths: list[Path], prompt_key: str, ext_key: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in summary_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            ext = str(row.get(ext_key) or "")
            if not ext or ext in seen:
                continue
            seen.add(ext)
            cases.append(
                {
                    "case_id": str(row["case_id"]),
                    "ext": ext,
                    "prompt": str(row.get(prompt_key) or ""),
                    "prompt_key": prompt_key,
                    "ext_key": ext_key,
                    "source_num_frames": int(row.get("source_num_frames") or row.get("generated_num_frames") or 0),
                    "summary": str(path),
                }
            )
    return cases


def _joint_exists(momask_root: Path, ext: str) -> bool:
    joint_dir = momask_root / "generation" / ext / "joints" / "0"
    return any(path for path in joint_dir.glob("sample0_repeat0_len*.npy") if "_ik" not in path.name)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_momask_imports(momask_root: Path) -> None:
    os.chdir(momask_root)
    if str(momask_root) not in sys.path:
        sys.path.insert(0, str(momask_root))


def _load_models(momask_root: Path, gpu_id: str, time_steps: int, cond_scale: float) -> dict[str, Any]:
    _prepare_momask_imports(momask_root)

    from gen_t2m import load_res_model, load_trans_model, load_vq_model
    from utils.get_opt import get_opt

    opt = SimpleNamespace(
        checkpoints_dir="./checkpoints",
        dataset_name="t2m",
        name="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns",
        res_name="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw",
        gpu_id=str(gpu_id),
        time_steps=int(time_steps),
        cond_scale=float(cond_scale),
        temperature=1.0,
        topkr=0.9,
        gumbel_sample=False,
    )
    opt.device = torch.device("cpu" if str(gpu_id) == "-1" else "cuda:" + str(gpu_id))

    dim_pose = 263
    root_dir = Path(opt.checkpoints_dir) / opt.dataset_name / opt.name
    model_opt = get_opt(str(root_dir / "opt.txt"), device=opt.device)

    vq_opt = get_opt(
        str(Path(opt.checkpoints_dir) / opt.dataset_name / model_opt.vq_name / "opt.txt"),
        device=opt.device,
    )
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt = get_opt(
        str(Path(opt.checkpoints_dir) / opt.dataset_name / opt.res_name / "opt.txt"),
        device=opt.device,
    )
    res_model = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    mean = np.load(Path(opt.checkpoints_dir) / opt.dataset_name / model_opt.vq_name / "meta" / "mean.npy")
    std = np.load(Path(opt.checkpoints_dir) / opt.dataset_name / model_opt.vq_name / "meta" / "std.npy")

    return {
        "opt": opt,
        "model_opt": model_opt,
        "vq_model": vq_model,
        "res_model": res_model,
        "t2m_transformer": t2m_transformer,
        "mean": mean,
        "std": std,
    }


def _save_motion(
    momask_root: Path,
    ext: str,
    length: int,
    joint_data: np.ndarray,
    save_bvh: bool,
    converter: Any | None,
) -> dict[str, Any]:
    from utils.motion_process import recover_from_ric

    joint_path = momask_root / "generation" / ext / "joints" / "0"
    animation_path = momask_root / "generation" / ext / "animations" / "0"
    joint_path.mkdir(parents=True, exist_ok=True)
    if save_bvh:
        animation_path.mkdir(parents=True, exist_ok=True)

    joint_data = joint_data[:length]
    recovered = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    joint = recovered
    ik_joint = recovered
    if save_bvh:
        if converter is None:
            raise ValueError("converter is required when save_bvh=True")
        _, ik_joint = converter.convert(
            recovered,
            filename=str(animation_path / f"sample0_repeat0_len{length}_ik.bvh"),
            iterations=100,
        )
        _, joint = converter.convert(
            recovered,
            filename=str(animation_path / f"sample0_repeat0_len{length}.bvh"),
            iterations=100,
            foot_ik=False,
        )

    np.save(joint_path / f"sample0_repeat0_len{length}.npy", joint)
    np.save(joint_path / f"sample0_repeat0_len{length}_ik.npy", ik_joint)
    return {
        "joint_file": str(joint_path / f"sample0_repeat0_len{length}.npy"),
        "ik_joint_file": str(joint_path / f"sample0_repeat0_len{length}_ik.npy"),
        "generated_len": int(length),
        "saved_bvh": bool(save_bvh),
    }


def _generate_batch(
    cases: list[dict[str, Any]],
    models: dict[str, Any],
    momask_root: Path,
    chunk_size: int,
    save_bvh: bool,
) -> list[dict[str, Any]]:
    opt = models["opt"]
    t2m_transformer = models["t2m_transformer"]
    res_model = models["res_model"]
    vq_model = models["vq_model"]
    mean = models["mean"]
    std = models["std"]
    converter = None
    if save_bvh:
        from visualization.joints2bvh import Joint2BVHConvertor

        converter = Joint2BVHConvertor()

    rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), chunk_size):
        chunk = cases[start : start + chunk_size]
        captions = [case["prompt"] for case in chunk]
        token_lens = torch.LongTensor([max(1, int(case["source_num_frames"]) // 4) for case in chunk])
        token_lens = token_lens.to(opt.device).long()
        motion_lens = (token_lens * 4).detach().cpu().numpy().astype(np.int32).tolist()
        print(f"batch {start + 1}-{start + len(chunk)} / {len(cases)}", flush=True)
        with torch.no_grad():
            mids = t2m_transformer.generate(
                captions,
                token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = pred_motions * std + mean

        for case, length, joint_data in zip(chunk, motion_lens, data):
            saved = _save_motion(
                momask_root=momask_root,
                ext=str(case["ext"]),
                length=int(length),
                joint_data=np.asarray(joint_data, dtype=np.float32),
                save_bvh=save_bvh,
                converter=converter,
            )
            row = dict(case)
            row.update(saved)
            rows.append(row)
            print(f"saved case={case['case_id']} ext={case['ext']} len={length}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prompt-key", default="auto_prompt")
    parser.add_argument("--ext-key", default="auto_ext")
    parser.add_argument("--momask-root", default=str(DEFAULT_MOMASK_ROOT))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--cond-scale", type=float, default=4.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--save-bvh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    momask_root = Path(args.momask_root)
    cases = _load_cases([Path(path) for path in args.summary], prompt_key=str(args.prompt_key), ext_key=str(args.ext_key))
    empty_prompt_cases = [case for case in cases if not str(case.get("prompt") or "").strip()]
    if empty_prompt_cases:
        raise SystemExit(f"{len(empty_prompt_cases)} cases have empty prompt for key={args.prompt_key}: {empty_prompt_cases[:5]}")
    skipped = []
    missing = []
    for case in cases:
        if args.reuse_existing and _joint_exists(momask_root, str(case["ext"])):
            skipped.append(case)
        else:
            missing.append(case)

    report_path = Path(args.report).resolve()
    print(f"cases={len(cases)} missing_or_requested={len(missing)} skipped_existing={len(skipped)}", flush=True)
    if args.dry_run:
        _write_json(report_path, {"cases": cases, "missing": missing, "skipped_existing": skipped, "generated": []})
        print(f"dry_run_report={report_path}", flush=True)
        return

    generated: list[dict[str, Any]] = []
    if missing:
        models = _load_models(momask_root, gpu_id=str(args.gpu_id), time_steps=args.time_steps, cond_scale=args.cond_scale)
        generated = _generate_batch(
            missing,
            models=models,
            momask_root=momask_root,
            chunk_size=max(1, int(args.chunk_size)),
            save_bvh=bool(args.save_bvh),
        )

    remaining_missing = [case for case in cases if not _joint_exists(momask_root, str(case["ext"]))]
    report = {
        "run": {
            "summaries": [str(path) for path in args.summary],
            "prompt_key": str(args.prompt_key),
            "ext_key": str(args.ext_key),
            "momask_root": str(momask_root),
            "gpu_id": str(args.gpu_id),
            "time_steps": int(args.time_steps),
            "cond_scale": float(args.cond_scale),
            "chunk_size": int(args.chunk_size),
            "save_bvh": bool(args.save_bvh),
        },
        "num_cases": len(cases),
        "num_skipped_existing": len(skipped),
        "num_generated": len(generated),
        "num_remaining_missing": len(remaining_missing),
        "skipped_existing": skipped,
        "generated": generated,
        "remaining_missing": remaining_missing,
    }
    _write_json(report_path, report)
    print(f"batch_report={report_path}", flush=True)
    if remaining_missing:
        raise SystemExit(f"Still missing {len(remaining_missing)} generated joint files")


if __name__ == "__main__":
    main()
