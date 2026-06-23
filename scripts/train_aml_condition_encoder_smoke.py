"""Smoke-train an AML condition encoder on span-level motion geometry.

This is the first G9 smoke. It is deliberately small: it does not generate
motion and does not change the AML tree. It checks whether the clean condition
batch can be aligned with motion spans and whether condition tokens/slots carry
enough signal to overfit simple span geometry targets.

Quick run:
    /mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
      scripts/train_aml_condition_encoder_smoke.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pseudoedit3d.data import AMLConditionMotionDataset


DEFAULT_CONDITION_BATCH_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean"
)
DEFAULT_MOTION_BATCH_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_program_condition_motion_batch_v0_train_clean"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/overfit"
)
DEFAULT_SPLIT_OUTPUT_DIR = Path(
    "outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/split_smoke"
)

TARGET_NAMES = [
    "root_delta_x",
    "root_delta_y",
    "root_delta_z",
    "root_path_length",
    "root_vertical_range",
    "mean_joint_disp",
    "max_joint_disp",
    "upper_joint_disp",
    "lower_joint_disp",
    "torso_joint_disp",
    "mean_joint_speed",
    "max_joint_speed",
]

LOWER_JOINTS = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.long)
TORSO_JOINTS = torch.tensor([0, 3, 6, 9, 12, 15], dtype=torch.long)
UPPER_JOINTS = torch.tensor([13, 14, 16, 17, 18, 19, 20, 21], dtype=torch.long)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in vocab.items()}


def _token_field_and_vocab(schema: dict[str, Any], token_source: str) -> tuple[str, str, str]:
    if token_source == "condition":
        return "condition_condition_id", "condition_vocab", "condition_name"
    if token_source == "structure":
        return "condition_motion_structure_id", "motion_structure_vocab", "motion_structure_name"
    return "condition_family_id", "family_vocab", "family_name"


def _span_target(joints: torch.Tensor, start: int, end: int, n_frames: int) -> torch.Tensor:
    n_frames = max(int(n_frames), 1)
    start = max(0, min(int(start), n_frames - 1))
    end = max(start + 1, min(int(end), n_frames))
    seg = joints[start:end]
    if seg.shape[0] == 0:
        seg = joints[start : start + 1]
    root = seg[:, 0, :]
    root_delta = root[-1] - root[0]
    if root.shape[0] > 1:
        root_vel = root[1:] - root[:-1]
        root_path = root_vel.norm(dim=-1).sum()
        joint_vel = seg[1:] - seg[:-1]
        joint_speed = joint_vel.norm(dim=-1)
        mean_joint_speed = joint_speed.mean()
        max_joint_speed = joint_speed.max()
    else:
        root_path = torch.zeros((), dtype=joints.dtype)
        mean_joint_speed = torch.zeros((), dtype=joints.dtype)
        max_joint_speed = torch.zeros((), dtype=joints.dtype)
    root_vertical_range = root[:, 1].max() - root[:, 1].min()
    joint_disp = (seg[-1] - seg[0]).norm(dim=-1)
    target = torch.stack(
        [
            root_delta[0],
            root_delta[1],
            root_delta[2],
            root_path,
            root_vertical_range,
            joint_disp.mean(),
            joint_disp.max(),
            joint_disp.index_select(0, UPPER_JOINTS).mean(),
            joint_disp.index_select(0, LOWER_JOINTS).mean(),
            joint_disp.index_select(0, TORSO_JOINTS).mean(),
            mean_joint_speed,
            max_joint_speed,
        ]
    )
    return target.float()


def build_condition_rows(dataset: AMLConditionMotionDataset, schema: dict[str, Any], token_source: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    token_field, vocab_key, meta_name_key = _token_field_and_vocab(schema, token_source)
    tensors: dict[str, list[torch.Tensor]] = {
        "token_id": [],
        "status_id": [],
        "score_weight": [],
        "span_norm": [],
        "numeric_slots": [],
        "numeric_slot_mask": [],
        "categorical_slots": [],
        "categorical_slot_mask": [],
        "target": [],
    }
    inv_token = _inverse_vocab(schema[vocab_key])
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        num_frames = int(sample["source_num_frames"].item())
        for cond_idx in range(int(sample["condition_mask"].shape[0])):
            if float(sample["condition_mask"][cond_idx].item()) <= 0.0:
                continue
            span = sample["condition_span"][cond_idx]
            target = _span_target(
                sample["joints"],
                int(span[0].item()),
                int(span[1].item()),
                num_frames,
            )
            token_id = int(sample[token_field][cond_idx].item())
            family_id = int(sample["condition_family_id"][cond_idx].item())
            condition_id = int(sample["condition_condition_id"][cond_idx].item())
            structure_id = int(sample["condition_motion_structure_id"][cond_idx].item())
            status_id = int(sample["condition_status_id"][cond_idx].item())
            tensors["token_id"].append(sample[token_field][cond_idx].long())
            tensors["status_id"].append(sample["condition_status_id"][cond_idx].long())
            tensors["score_weight"].append(
                torch.stack(
                    [
                        sample["condition_score"][cond_idx].float(),
                        sample["condition_weight"][cond_idx].float(),
                    ]
                )
            )
            tensors["span_norm"].append(sample["condition_span_norm"][cond_idx].float())
            tensors["numeric_slots"].append(sample["condition_numeric_slots"][cond_idx].float())
            tensors["numeric_slot_mask"].append(sample["condition_numeric_slot_mask"][cond_idx].float())
            tensors["categorical_slots"].append(sample["condition_categorical_slots"][cond_idx].long())
            tensors["categorical_slot_mask"].append(sample["condition_categorical_slot_mask"][cond_idx].float())
            tensors["target"].append(target)
            rows.append(
                {
                    "row_index": len(rows),
                    "case_id": sample["case_id"],
                    "condition_index": cond_idx,
                    "token_source": token_source,
                    "token_id": token_id,
                    "token_name": inv_token.get(token_id, "<unk>"),
                    meta_name_key: inv_token.get(token_id, "<unk>"),
                    "family_id": family_id,
                    "condition_id": condition_id,
                    "motion_structure_id": structure_id,
                    "status_id": status_id,
                    "span": [int(span[0].item()), int(span[1].item())],
                    "target": {name: float(value) for name, value in zip(TARGET_NAMES, target.tolist())},
                }
            )
    if not rows:
        raise ValueError("No real conditions found in dataset")
    stacked = {key: torch.stack(value, dim=0) for key, value in tensors.items()}
    return stacked, rows


class AMLConditionGeometryEncoder(nn.Module):
    def __init__(
        self,
        *,
        family_vocab_size: int,
        status_vocab_size: int,
        categorical_vocab_sizes: list[int],
        numeric_dim: int,
        target_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.family_embed = nn.Embedding(family_vocab_size, embed_dim)
        self.status_embed = nn.Embedding(status_vocab_size, embed_dim // 2)
        cat_dim = max(4, embed_dim // 4)
        self.cat_embeds = nn.ModuleList(nn.Embedding(size, cat_dim) for size in categorical_vocab_sizes)
        input_dim = (
            embed_dim
            + embed_dim // 2
            + len(categorical_vocab_sizes) * cat_dim
            + 2
            + 4
            + numeric_dim
            + numeric_dim
            + len(categorical_vocab_sizes)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        family = self.family_embed(batch["token_id"])
        status = self.status_embed(batch["status_id"])
        cat_parts = []
        for idx, embed in enumerate(self.cat_embeds):
            cat = embed(batch["categorical_slots"][:, idx])
            cat = cat * batch["categorical_slot_mask"][:, idx : idx + 1]
            cat_parts.append(cat)
        x = torch.cat(
            [
                family,
                status,
                *cat_parts,
                batch["score_weight"],
                batch["span_norm"],
                batch["numeric_slots"] * batch["numeric_slot_mask"],
                batch["numeric_slot_mask"],
                batch["categorical_slot_mask"],
            ],
            dim=-1,
        )
        return self.net(x)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _batch_indices(num_rows: int, batch_size: int, shuffle: bool, device: torch.device) -> list[torch.Tensor]:
    order = torch.randperm(num_rows) if shuffle else torch.arange(num_rows)
    return [order[start : start + batch_size].to(device) for start in range(0, num_rows, batch_size)]


def _select(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def _loss_metrics(pred: torch.Tensor, target_norm: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> dict[str, Any]:
    loss = torch.mean((pred - target_norm) ** 2)
    denorm = pred * target_std + target_mean
    target = target_norm * target_std + target_mean
    mae = (denorm - target).abs().mean(dim=0)
    return {
        "mse_norm": float(loss.item()),
        "mae_mean": float(mae.mean().item()),
        "mae_by_target": {name: float(value) for name, value in zip(TARGET_NAMES, mae.tolist())},
    }


def _make_case_split(row_meta: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[int], list[int], dict[str, Any]]:
    case_ids = sorted({str(row["case_id"]) for row in row_meta})
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    val_count = max(1, int(round(len(case_ids) * float(val_fraction))))
    val_cases = set(case_ids[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx, row in enumerate(row_meta):
        if str(row["case_id"]) in val_cases:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    if not train_indices or not val_indices:
        raise ValueError("train/val split produced an empty side")
    return train_indices, val_indices, {
        "split_policy": "case_id_random",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_cases": len({str(row_meta[idx]["case_id"]) for idx in train_indices}),
        "val_cases": len({str(row_meta[idx]["case_id"]) for idx in val_indices}),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "val_case_ids": sorted(val_cases),
    }


def _make_row_stratified_split(row_meta: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[int], list[int], dict[str, Any]]:
    by_family: dict[str, list[int]] = {}
    for idx, row in enumerate(row_meta):
        by_family.setdefault(str(row["token_name"]), []).append(idx)
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    singleton_families: list[str] = []
    for family, indices in sorted(by_family.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_indices.extend(shuffled)
            singleton_families.append(family)
            continue
        val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])
    if not val_indices:
        raise ValueError("row-stratified split has no validation rows; need repeated families")
    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices, {
        "split_policy": "row_family_stratified",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_cases": len({str(row_meta[idx]["case_id"]) for idx in train_indices}),
        "val_cases": len({str(row_meta[idx]["case_id"]) for idx in val_indices}),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "val_case_ids": sorted({str(row_meta[idx]["case_id"]) for idx in val_indices}),
        "singleton_family_count": len(singleton_families),
        "singleton_families_train_only": singleton_families,
    }


def _family_counts(row_meta: list[dict[str, Any]], indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        family = str(row_meta[idx]["token_name"])
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _evaluate_subset(
    model: nn.Module,
    rows: dict[str, torch.Tensor],
    target_norm: torch.Tensor,
    indices: list[int],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(_select(rows, idx))
        target = target_norm.index_select(0, idx)
        return _loss_metrics(pred, target, target_mean, target_std)


def _baseline_metrics(
    *,
    target_norm: torch.Tensor,
    row_meta: list[dict[str, Any]],
    train_indices: list[int],
    eval_indices: list[int],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    eval_idx = torch.tensor(eval_indices, dtype=torch.long, device=device)
    eval_target = target_norm.index_select(0, eval_idx)
    global_pred = torch.zeros_like(eval_target)
    global_metrics = _loss_metrics(global_pred, eval_target, target_mean, target_std)

    family_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for idx in train_indices:
        family_values[str(row_meta[idx]["token_name"])].append(target_norm[idx].detach())
    family_means = {
        family: torch.stack(values, dim=0).mean(dim=0)
        for family, values in family_values.items()
    }
    family_preds: list[torch.Tensor] = []
    unseen_families: list[str] = []
    for idx in eval_indices:
        family = str(row_meta[idx]["token_name"])
        if family in family_means:
            family_preds.append(family_means[family])
        else:
            family_preds.append(torch.zeros((target_norm.shape[1],), dtype=target_norm.dtype, device=device))
            unseen_families.append(family)
    family_pred = torch.stack(family_preds, dim=0)
    family_metrics = _loss_metrics(family_pred, eval_target, target_mean, target_std)
    return {
        "global_train_mean": global_metrics,
        "family_train_mean": family_metrics,
        "unseen_eval_family_count": len(set(unseen_families)),
        "unseen_eval_families": sorted(set(unseen_families)),
    }


def train_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _set_seed(int(args.seed))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dataset = AMLConditionMotionDataset(
        condition_batch_dir=args.condition_batch_dir,
        motion_batch_dir=args.motion_batch_dir,
        max_cases=args.max_cases,
    )
    schema = _load_json(Path(args.condition_batch_dir) / "condition_batch_schema.json")
    token_field, vocab_key, _meta_name_key = _token_field_and_vocab(schema, args.token_source)
    rows, row_meta = build_condition_rows(dataset, schema, args.token_source)
    target = rows.pop("target")
    all_indices = list(range(int(target.shape[0])))
    if args.mode == "case_split":
        train_indices, val_indices, split_info = _make_case_split(row_meta, args.val_fraction, args.seed)
        stat_indices = torch.tensor(train_indices, dtype=torch.long)
        train_loop_indices = train_indices
        eval_indices = val_indices
    elif args.mode == "row_split":
        train_indices, val_indices, split_info = _make_row_stratified_split(row_meta, args.val_fraction, args.seed)
        stat_indices = torch.tensor(train_indices, dtype=torch.long)
        train_loop_indices = train_indices
        eval_indices = val_indices
    else:
        train_indices = all_indices
        val_indices = []
        split_info = {
            "split_policy": "overfit_all_rows",
            "train_cases": len({str(row["case_id"]) for row in row_meta}),
            "val_cases": 0,
            "train_rows": len(all_indices),
            "val_rows": 0,
            "val_case_ids": [],
        }
        stat_indices = torch.tensor(all_indices, dtype=torch.long)
        train_loop_indices = all_indices
        eval_indices = all_indices
    target_mean = target.index_select(0, stat_indices).mean(dim=0)
    target_std = target.index_select(0, stat_indices).std(dim=0).clamp_min(1e-5)
    target_norm = (target - target_mean) / target_std
    rows = _to_device(rows, device)
    target_norm = target_norm.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    num_rows = int(target_norm.shape[0])

    cat_vocab_sizes = [
        len(schema["categorical_slot_vocabs"][name])
        for name in schema["categorical_slot_names"]
    ]
    model = AMLConditionGeometryEncoder(
        family_vocab_size=len(schema[vocab_key]),
        status_vocab_size=len(schema["status_vocab"]),
        categorical_vocab_sizes=cat_vocab_sizes,
        numeric_dim=len(schema["numeric_slot_names"]),
        target_dim=len(TARGET_NAMES),
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    with torch.no_grad():
        initial_train = _evaluate_subset(model, rows, target_norm, train_indices, target_mean, target_std, device)
        initial_eval = _evaluate_subset(model, rows, target_norm, eval_indices, target_mean, target_std, device)

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for indices in _batch_indices(len(train_loop_indices), int(args.batch_size), shuffle=True, device=device):
            real_indices = torch.tensor(train_loop_indices, dtype=torch.long, device=device).index_select(0, indices)
            batch = _select(rows, real_indices)
            pred = model(batch)
            loss = torch.mean((pred - target_norm.index_select(0, real_indices)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = int(real_indices.numel())
            epoch_loss += float(loss.item()) * n
            seen += n
        if epoch == 1 or epoch == int(args.epochs) or epoch % int(args.log_every) == 0:
            model.eval()
            train_metrics = _evaluate_subset(model, rows, target_norm, train_indices, target_mean, target_std, device)
            eval_metrics = _evaluate_subset(model, rows, target_norm, eval_indices, target_mean, target_std, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loop_mse_norm": epoch_loss / max(1, seen),
                    "train_mse_norm": train_metrics["mse_norm"],
                    "eval_mse_norm": eval_metrics["mse_norm"],
                    "eval_mae_mean": eval_metrics["mae_mean"],
                }
            )

    model.eval()
    with torch.no_grad():
        pred_norm = model(rows)
        final_train = _evaluate_subset(model, rows, target_norm, train_indices, target_mean, target_std, device)
        final_eval = _evaluate_subset(model, rows, target_norm, eval_indices, target_mean, target_std, device)
        pred = (pred_norm * target_std + target_mean).cpu()
        target_denorm = (target_norm * target_std + target_mean).cpu()
        baselines = _baseline_metrics(
            target_norm=target_norm,
            row_meta=row_meta,
            train_indices=train_indices,
            eval_indices=eval_indices,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )

    examples = []
    example_indices = eval_indices[: int(args.example_rows)] if args.mode == "split" else all_indices[: int(args.example_rows)]
    for idx in example_indices:
        meta = row_meta[idx]
        examples.append(
            {
                **meta,
                "split": "val" if idx in set(val_indices) else "train",
                "pred": {name: float(value) for name, value in zip(TARGET_NAMES, pred[idx].tolist())},
                "abs_error": {
                    name: float(value)
                    for name, value in zip(TARGET_NAMES, (pred[idx] - target_denorm[idx]).abs().tolist())
                },
            }
        )

    train_loss_reduction = 1.0 - final_train["mse_norm"] / max(initial_train["mse_norm"], 1e-8)
    eval_loss_reduction = 1.0 - final_eval["mse_norm"] / max(initial_eval["mse_norm"], 1e-8)
    if args.mode in {"case_split", "row_split"}:
        status = "pass" if final_eval["mse_norm"] <= float(args.pass_mse) or eval_loss_reduction >= float(args.pass_reduction) else "warn"
    else:
        status = "pass" if final_train["mse_norm"] <= float(args.pass_mse) or train_loss_reduction >= float(args.pass_reduction) else "warn"
    best_eval = min(history, key=lambda row: row["eval_mse_norm"]) if history else {}
    summary = {
        "schema_version": "aml_condition_encoder_geometry_smoke_v1",
        "mode": str(args.mode),
        "status": status,
        "condition_batch_dir": str(args.condition_batch_dir),
        "motion_batch_dir": str(args.motion_batch_dir),
        "device": str(device),
        "dataset_cases": len(dataset),
        "condition_rows": num_rows,
        "split": {
            **split_info,
            "train_family_counts": _family_counts(row_meta, train_indices),
            "val_family_counts": _family_counts(row_meta, val_indices),
        },
        "token_source": str(args.token_source),
        "token_field": token_field,
        "token_vocab_key": vocab_key,
        "target_names": TARGET_NAMES,
        "target_mean": {name: float(value) for name, value in zip(TARGET_NAMES, target_mean.detach().cpu().tolist())},
        "target_std": {name: float(value) for name, value in zip(TARGET_NAMES, target_std.detach().cpu().tolist())},
        "initial_train_metrics": initial_train,
        "initial_eval_metrics": initial_eval,
        "final_train_metrics": final_train,
        "final_eval_metrics": final_eval,
        "initial_metrics": initial_train if args.mode == "overfit" else initial_eval,
        "final_metrics": final_train if args.mode == "overfit" else final_eval,
        "train_loss_reduction": float(train_loss_reduction),
        "eval_loss_reduction": float(eval_loss_reduction),
        "loss_reduction": float(train_loss_reduction if args.mode == "overfit" else eval_loss_reduction),
        "baseline_metrics": baselines,
        "best_eval_history": best_eval,
        "history": history,
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "embed_dim": int(args.embed_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "summary": summary,
        "schema": schema,
    }
    return summary, examples, checkpoint


def write_report(path: Path, summary: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    lines = [
        "# AML Condition Encoder Smoke v0",
        "",
        "## Inputs",
        "",
        f"- condition batch dir: `{summary['condition_batch_dir']}`",
        f"- motion batch dir: `{summary['motion_batch_dir']}`",
        "",
        "## Status",
        "",
        f"- status: `{summary['status']}`",
        f"- mode: `{summary['mode']}`",
        f"- token source: `{summary['token_source']}`",
        f"- device: `{summary['device']}`",
        f"- cases: `{summary['dataset_cases']}`",
        f"- condition rows: `{summary['condition_rows']}`",
        f"- train rows: `{summary['split']['train_rows']}`",
        f"- val rows: `{summary['split']['val_rows']}`",
        f"- initial eval normalized MSE: `{summary['initial_eval_metrics']['mse_norm']:.6f}`",
        f"- final eval normalized MSE: `{summary['final_eval_metrics']['mse_norm']:.6f}`",
        f"- final train normalized MSE: `{summary['final_train_metrics']['mse_norm']:.6f}`",
        f"- best eval normalized MSE: `{summary['best_eval_history'].get('eval_mse_norm', 0.0):.6f}` at epoch `{summary['best_eval_history'].get('epoch', 'n/a')}`",
        f"- loss reduction: `{summary['loss_reduction']:.4f}`",
        f"- final mean absolute error: `{summary['final_metrics']['mae_mean']:.6f}`",
        f"- global-mean eval MSE: `{summary['baseline_metrics']['global_train_mean']['mse_norm']:.6f}`",
        f"- family-mean eval MSE: `{summary['baseline_metrics']['family_train_mean']['mse_norm']:.6f}`",
        "",
        "## Target MAE",
        "",
        "| target | MAE |",
        "| --- | --- |",
    ]
    for name, value in summary["final_metrics"]["mae_by_target"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(["", "## Baselines", ""])
    lines.append("| baseline | eval_mse_norm | eval_mae_mean |")
    lines.append("| --- | --- | --- |")
    for baseline_name, metrics in [
        ("global_train_mean", summary["baseline_metrics"]["global_train_mean"]),
        ("family_train_mean", summary["baseline_metrics"]["family_train_mean"]),
        ("model_final", summary["final_eval_metrics"]),
    ]:
        lines.append(f"| {baseline_name} | {metrics['mse_norm']:.6f} | {metrics['mae_mean']:.6f} |")
    if summary["baseline_metrics"]["unseen_eval_family_count"]:
        lines.append("")
        lines.append(
            f"- unseen validation families: `{summary['baseline_metrics']['unseen_eval_family_count']}`"
        )
    lines.extend(["", "## Training History", "", "| epoch | train_loop_mse_norm | train_mse_norm | eval_mse_norm |", "| --- | --- | --- | --- |"])
    for row in summary["history"]:
        lines.append(f"| {row['epoch']} | {row['train_loop_mse_norm']:.6f} | {row['train_mse_norm']:.6f} | {row['eval_mse_norm']:.6f} |")
    if summary["mode"] in {"case_split", "row_split"}:
        lines.extend(["", "## Split", ""])
        lines.append(f"- train cases: `{summary['split']['train_cases']}`")
        lines.append(f"- val cases: `{summary['split']['val_cases']}`")
        lines.append(f"- val case ids: `{', '.join(summary['split']['val_case_ids'])}`")
        lines.append("")
        if summary["mode"] == "case_split":
            lines.append("Validation is intentionally harsh for this tiny 40-case batch: many validation families may be unseen or one-shot.")
        else:
            lines.append("Row-stratified validation is a same-distribution sanity check; singleton families remain train-only.")
    lines.extend(["", "## Example Rows", ""])
    for item in examples:
        lines.append(f"### row {item['row_index']} | {item['case_id']} | cond {item['condition_index']} | {item.get('split', '')}")
        lines.append(f"- token: `{item['token_name']}`")
        lines.append(f"- span: `{item['span']}`")
        err = item["abs_error"]
        lines.append(
            "- abs error: "
            + ", ".join(f"{name}={err[name]:.4f}" for name in TARGET_NAMES[:6])
        )
        lines.append("")
    lines.append(
        "This smoke validates condition/motion alignment and simple geometry prediction only; it is not a motion-generation result."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-train an AML condition encoder on span geometry.")
    parser.add_argument("--condition-batch-dir", default=str(DEFAULT_CONDITION_BATCH_DIR))
    parser.add_argument("--motion-batch-dir", default=str(DEFAULT_MOTION_BATCH_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["overfit", "case_split", "row_split"], default="overfit")
    parser.add_argument("--token-source", choices=["family", "condition", "structure"], default="family")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--example-rows", type=int, default=12)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--pass-mse", type=float, default=0.05)
    parser.add_argument("--pass-reduction", type=float, default=0.90)
    args = parser.parse_args()

    if args.output_dir is None:
        if args.mode == "case_split":
            out_dir = DEFAULT_SPLIT_OUTPUT_DIR
        elif args.mode == "row_split":
            out_dir = DEFAULT_SPLIT_OUTPUT_DIR.parent / "row_split_smoke"
        else:
            out_dir = DEFAULT_OUTPUT_DIR
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, examples, checkpoint = train_smoke(args)
    _write_json(out_dir / "overfit_metrics.json", summary)
    _write_jsonl(out_dir / "overfit_examples.jsonl", examples)
    torch.save(checkpoint, out_dir / "condition_encoder_smoke.pt")
    write_report(out_dir / "overfit_report.md", summary, examples)
    print(
        "saved={out_dir} mode={mode} status={status} rows={condition_rows} "
        "train_mse={train_mse:.6f} eval_mse={eval_mse:.6f} reduction={reduction:.4f}".format(
            out_dir=out_dir,
            mode=summary["mode"],
            status=summary["status"],
            condition_rows=summary["condition_rows"],
            train_mse=summary["final_train_metrics"]["mse_norm"],
            eval_mse=summary["final_eval_metrics"]["mse_norm"],
            reduction=summary["loss_reduction"],
        )
    )


if __name__ == "__main__":
    main()
