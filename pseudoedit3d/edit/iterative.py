from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pseudoedit3d.data import load_mined_pair_arrays
from pseudoedit3d.edit.attributes import extract_upper_body_proxy_attributes
from pseudoedit3d.edit.mining import dump_jsonl, load_jsonl, mine_triplets
from pseudoedit3d.edit.schema import EditProgram
from pseudoedit3d.training.train_stage1 import build_model, train_from_config


def dedupe_pair_records(records: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for record in records:
        program = record["program"]
        key = (
            record["source_path"],
            record["target_path"],
            program["part"],
            program["attribute"],
            int(program["start_frame"]),
            int(program["end_frame"]),
            program.get("attribute_key"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def select_seed_pairs(
    records: list[dict],
    keep_ratio: float = 0.5,
    max_pairs: int = 0,
    max_pairs_per_source: int = 2,
) -> list[dict]:
    records = dedupe_pair_records(records)
    target_count = max(1, int(len(records) * keep_ratio))
    if max_pairs > 0:
        target_count = min(target_count, max_pairs)
    selected = []
    per_source = defaultdict(int)
    for record in sorted(records, key=lambda item: (float(item.get("score", 0.0)), -abs(item["program"].get("delta_value_deg") or 0.0))):
        source_path = record["source_path"]
        if per_source[source_path] >= max_pairs_per_source:
            continue
        selected.append(record)
        per_source[source_path] += 1
        if len(selected) >= target_count:
            break
    return selected


def _masked_l1_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    masked_error = np.abs(pred - target) * mask
    denom = max(float(mask.sum()), 1.0)
    return float(masked_error.sum() / denom)


def score_pair_record(record: dict, model: torch.nn.Module, device: torch.device) -> dict:
    arrays = load_mined_pair_arrays(record)
    source_pose = torch.from_numpy(arrays["source_pose"]).reshape(1, arrays["source_pose"].shape[0], -1).to(device)
    target_pose = torch.from_numpy(arrays["target_pose"]).reshape(1, arrays["target_pose"].shape[0], -1).to(device)
    edit_vector = torch.from_numpy(arrays["edit_vector"]).reshape(1, -1).to(device)
    joint_mask = arrays["joint_mask"][..., None].repeat(3, axis=-1).reshape(arrays["joint_mask"].shape[0], -1)
    keep_mask = 1.0 - joint_mask

    with torch.no_grad():
        pred_pose = model(source_pose, edit_vector).cpu().numpy()[0]
    target_pose_np = target_pose.cpu().numpy()[0]
    source_pose_np = source_pose.cpu().numpy()[0]

    edit_l1 = _masked_l1_np(pred_pose, target_pose_np, joint_mask)
    keep_l1 = _masked_l1_np(pred_pose, source_pose_np, keep_mask)
    smooth_l1 = float(np.abs(pred_pose[1:] - pred_pose[:-1]).mean())

    program = EditProgram.from_dict(record["program"])
    attr_key = program.attribute_key
    attr_metrics = {}
    if attr_key:
        pred_pose_j = pred_pose.reshape(-1, 52, 3)
        target_pose_j = target_pose_np.reshape(-1, 52, 3)
        source_pose_j = source_pose_np.reshape(-1, 52, 3)
        pred_attr = extract_upper_body_proxy_attributes(pred_pose_j)[attr_key]
        target_attr = extract_upper_body_proxy_attributes(target_pose_j)[attr_key]
        source_attr = extract_upper_body_proxy_attributes(source_pose_j)[attr_key]
        sl = slice(program.start_frame, program.end_frame + 1)
        pred_mean = float(pred_attr[sl].mean())
        target_mean = float(target_attr[sl].mean())
        source_mean = float(source_attr[sl].mean())
        attr_metrics = {
            "attribute_key": attr_key,
            "source_attr_mean": source_mean,
            "target_attr_mean": target_mean,
            "pred_attr_mean": pred_mean,
            "target_delta_abs": abs(target_mean - source_mean),
            "pred_delta_abs": abs(pred_mean - source_mean),
            "delta_error_abs": abs(pred_mean - target_mean),
        }

    heuristic_score = float(record.get("heuristic_score", record.get("score", 0.0)))
    combined_score = edit_l1 + 0.5 * keep_l1 + 0.1 * heuristic_score
    if attr_metrics:
        combined_score += 0.02 * float(attr_metrics["delta_error_abs"])

    scored = dict(record)
    scored["heuristic_score"] = heuristic_score
    scored["model_metrics"] = {
        "edit_l1": edit_l1,
        "keep_l1": keep_l1,
        "smooth_l1": smooth_l1,
        **attr_metrics,
    }
    scored["combined_score"] = float(combined_score)
    return scored


def score_pair_manifest(records: list[dict], model_checkpoint: str, cfg) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state_dict = torch.load(model_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return [score_pair_record(record, model, device) for record in records]


def select_refined_pairs(
    scored_records: list[dict],
    keep_ratio: float = 0.5,
    max_pairs: int = 0,
    max_pairs_per_source: int = 2,
    edit_l1_quantile: float = 0.75,
    keep_l1_quantile: float = 0.75,
) -> list[dict]:
    scored_records = dedupe_pair_records(scored_records)
    if not scored_records:
        return []

    edit_values = np.asarray([record["model_metrics"]["edit_l1"] for record in scored_records], dtype=np.float32)
    keep_values = np.asarray([record["model_metrics"]["keep_l1"] for record in scored_records], dtype=np.float32)
    edit_thr = float(np.quantile(edit_values, edit_l1_quantile))
    keep_thr = float(np.quantile(keep_values, keep_l1_quantile))

    filtered = [
        record for record in scored_records
        if record["model_metrics"]["edit_l1"] <= edit_thr and record["model_metrics"]["keep_l1"] <= keep_thr
    ]
    if not filtered:
        filtered = sorted(scored_records, key=lambda item: item["combined_score"])[: max(1, len(scored_records) // 4)]

    target_count = max(1, int(len(filtered) * keep_ratio))
    if max_pairs > 0:
        target_count = min(target_count, max_pairs)

    selected = []
    per_source = defaultdict(int)
    for record in sorted(filtered, key=lambda item: (float(item["combined_score"]), float(item["heuristic_score"]))):
        source_path = record["source_path"]
        if per_source[source_path] >= max_pairs_per_source:
            continue
        selected.append(record)
        per_source[source_path] += 1
        if len(selected) >= target_count:
            break
    return selected


def run_iterative_refinement(
    attribute_cache_path: str,
    output_dir: str,
    base_cfg,
    num_rounds: int = 2,
    initial_keep_ratio: float = 0.6,
    refine_keep_ratio: float = 0.6,
    max_pairs_per_source: int = 2,
    max_train_pairs: int = 0,
    mine_min_delta_deg: float = 8.0,
    mine_max_delta_deg: float = 45.0,
    mine_candidate_limit: int = 64,
    mine_max_pairs_per_clip: int = 4,
    mine_distance_threshold: float = 2.5,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    attribute_records = load_jsonl(attribute_cache_path)
    raw_pairs = mine_triplets(
        records=attribute_records,
        min_delta_deg=mine_min_delta_deg,
        max_delta_deg=mine_max_delta_deg,
        candidate_limit=mine_candidate_limit,
        max_pairs_per_clip=mine_max_pairs_per_clip,
        distance_threshold=mine_distance_threshold,
    )
    raw_pairs_path = output_root / "iter_raw_pairs.jsonl"
    dump_jsonl(raw_pairs_path, raw_pairs)

    current_train_records = select_seed_pairs(
        raw_pairs,
        keep_ratio=initial_keep_ratio,
        max_pairs=max_train_pairs,
        max_pairs_per_source=max_pairs_per_source,
    )
    train_manifest_path = output_root / "iter_seed_pairs.jsonl"
    dump_jsonl(train_manifest_path, current_train_records)

    summary = {
        "attribute_cache_path": str(attribute_cache_path),
        "raw_pairs_path": str(raw_pairs_path),
        "num_raw_pairs": len(raw_pairs),
        "rounds": [],
    }

    for round_idx in range(num_rounds):
        round_dir = output_root / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        current_manifest_path = round_dir / "train_pairs.jsonl"
        dump_jsonl(current_manifest_path, current_train_records)

        round_cfg = replace(
            base_cfg,
            data_mode="mined",
            pair_manifest_path=str(current_manifest_path),
            max_clips=max_train_pairs,
            save_dir=str(round_dir / "train_outputs"),
        )
        train_result = train_from_config(round_cfg, checkpoint_name=f"round_{round_idx:02d}_last.pt")

        scored_records = score_pair_manifest(raw_pairs, train_result["checkpoint_path"], round_cfg)
        scored_path = round_dir / "scored_pairs.jsonl"
        dump_jsonl(scored_path, scored_records)

        next_train_records = select_refined_pairs(
            scored_records,
            keep_ratio=refine_keep_ratio,
            max_pairs=max_train_pairs,
            max_pairs_per_source=max_pairs_per_source,
        )
        next_manifest_path = round_dir / "refined_pairs.jsonl"
        dump_jsonl(next_manifest_path, next_train_records)

        summary["rounds"].append(
            {
                "round_idx": round_idx,
                "input_pairs": len(current_train_records),
                "scored_pairs": len(scored_records),
                "refined_pairs": len(next_train_records),
                "checkpoint_path": train_result["checkpoint_path"],
                "scored_pairs_path": str(scored_path),
                "refined_pairs_path": str(next_manifest_path),
                "last_loss": train_result["last_loss"],
            }
        )
        current_train_records = next_train_records

    summary_path = output_root / "iteration_summary.jsonl"
    dump_jsonl(summary_path, [summary])
    summary["summary_path"] = str(summary_path)
    return summary
