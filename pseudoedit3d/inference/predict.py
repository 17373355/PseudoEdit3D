from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pseudoedit3d.config import load_simple_yaml
from pseudoedit3d.data import MinedMotionEditDataset, PrefixMotionDataset
from pseudoedit3d.edit.schema import EditProgram, load_label_schema
from pseudoedit3d.models import create_model
from pseudoedit3d.training.train_stage1 import _build_condition_inputs, build_dataset, build_model


def load_pair_records(pair_manifest_path: str) -> list[dict]:
    records = []
    with Path(pair_manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_manifest_records(manifest_path: str) -> list[dict]:
    records = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _guess_attribute_cache_from_manifest(manifest_path: str) -> str:
    return str(manifest_path).replace("/scan/manifest_full.jsonl", "/attributes/attribute_cache_full.jsonl")


def salient_case_score(record: dict) -> float:
    program = record["program"]
    delta = abs(float(program.get("delta_value_deg") or 0.0))
    duration = int(program["end_frame"]) - int(program["start_frame"]) + 1
    part = program["part"]
    skill_label = program.get("skill_label", "unknown")
    score = delta * (1.0 + 0.03 * duration)
    if part == "both_arms":
        score += 12.0
    elif part == "torso":
        score += 8.0
    if skill_label == "periodic_arm_motion":
        score += 10.0
    elif skill_label == "locomotion":
        score += 4.0
    return float(score)


def select_case_indices(
    pair_manifest_path: str,
    num_cases: int,
    selection: str = "salient",
    explicit_indices: list[int] | None = None,
    min_delta_deg: float = 12.0,
    min_duration_frames: int = 6,
) -> list[int]:
    if explicit_indices:
        return explicit_indices[:num_cases]
    records = load_pair_records(pair_manifest_path)
    if selection == "first":
        return list(range(min(num_cases, len(records))))
    candidates = []
    for idx, record in enumerate(records):
        program = record["program"]
        delta = abs(float(program.get("delta_value_deg") or 0.0))
        duration = int(program["end_frame"]) - int(program["start_frame"]) + 1
        if delta < min_delta_deg or duration < min_duration_frames:
            continue
        candidates.append((salient_case_score(record), idx))
    if not candidates:
        return list(range(min(num_cases, len(records))))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in candidates[:num_cases]]


def select_prefix_case_indices(
    manifest_path: str,
    num_cases: int,
    selection: str = "salient",
    explicit_indices: list[int] | None = None,
    attribute_cache_path: str | None = None,
) -> list[int]:
    if explicit_indices:
        return explicit_indices[:num_cases]
    records = load_manifest_records(manifest_path)
    if selection == "first":
        return list(range(min(num_cases, len(records))))
    attribute_cache_path = attribute_cache_path or _guess_attribute_cache_from_manifest(manifest_path)
    skill_map = {}
    if Path(attribute_cache_path).exists():
        for rec in load_manifest_records(attribute_cache_path):
            skill = rec.get("skill_context", {})
            skill_map[rec["path"]] = (
                skill.get("skill_label", "unknown"),
                float(skill.get("dominant_attr_range", 0.0)),
            )
    scored = []
    for idx, record in enumerate(records):
        skill_label, dominant_range = skill_map.get(record["path"], ("unknown", 0.0))
        score = dominant_range
        if skill_label == "periodic_arm_motion":
            score += 30.0
        elif skill_label == "locomotion":
            score += 20.0
        elif skill_label == "torso_leaning":
            score += 10.0
        elif skill_label == "static_pose":
            score -= 10.0
        scored.append((score, idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored[:num_cases]]


def _build_legacy_program_vector(program: EditProgram, schema, edit_dim: int) -> np.ndarray:
    span_vec = [program.start_frame / 59.0, program.end_frame / 59.0]
    if edit_dim == 19:
        part_keys = ['left_arm', 'right_arm', 'both_arms', 'torso']
        attr_keys = ['raise', 'lower', 'bend', 'extend', 'lean_left', 'lean_right', 'lean_forward', 'lean_backward']
        part_vec = [1.0 if program.part == value else 0.0 for value in part_keys]
        attr_vec = [1.0 if program.attribute == value else 0.0 for value in attr_keys]
        delta_vec = [1.0 if program.delta_bin == value else 0.0 for value in schema.delta_bin_keys]
        contact_vec = [1.0 if program.contact_policy == value else 0.0 for value in schema.contact_policy_keys]
        vector = part_vec + attr_vec + delta_vec + contact_vec + span_vec
    elif edit_dim == 23:
        part_keys = ['left_arm', 'right_arm', 'both_arms', 'torso']
        attr_keys = ['raise', 'lower', 'bend', 'extend', 'lean_left', 'lean_right', 'lean_forward', 'lean_backward']
        part_vec = [1.0 if program.part == value else 0.0 for value in part_keys]
        attr_vec = [1.0 if program.attribute == value else 0.0 for value in attr_keys]
        delta_vec = [1.0 if program.delta_bin == value else 0.0 for value in schema.delta_bin_keys]
        contact_vec = [1.0 if program.contact_policy == value else 0.0 for value in schema.contact_policy_keys]
        operator_vec = [1.0 if program.operator == value else 0.0 for value in schema.operator_keys]
        reference_vec = [1.0 if program.reference == value else 0.0 for value in schema.reference_keys]
        vector = part_vec + attr_vec + delta_vec + contact_vec + operator_vec + reference_vec + span_vec
    elif edit_dim == 32:
        part_keys = ['left_arm', 'right_arm', 'both_arms', 'torso']
        attr_keys = ['raise', 'lower', 'bend', 'extend', 'lean_left', 'lean_right', 'lean_forward', 'lean_backward']
        part_vec = [1.0 if program.part == value else 0.0 for value in part_keys]
        attr_vec = [1.0 if program.attribute == value else 0.0 for value in attr_keys]
        delta_vec = [1.0 if program.delta_bin == value else 0.0 for value in schema.delta_bin_keys]
        contact_vec = [1.0 if program.contact_policy == value else 0.0 for value in schema.contact_policy_keys]
        operator_vec = [1.0 if program.operator == value else 0.0 for value in schema.operator_keys]
        reference_vec = [1.0 if program.reference == value else 0.0 for value in schema.reference_keys]
        preserve_vec = [1.0 if program.preserve_mode == value else 0.0 for value in schema.preserve_mode_keys]
        skill_vec = [1.0 if (program.skill_label or 'unknown') == value else 0.0 for value in schema.skill_label_keys]
        vector = part_vec + attr_vec + delta_vec + contact_vec + operator_vec + reference_vec + preserve_vec + skill_vec + span_vec
    else:
        raise ValueError(f"Unsupported legacy edit_dim={edit_dim}")
    return np.asarray(vector, dtype=np.float32)


def _build_model_from_checkpoint(cfg, checkpoint_path: str, torch_device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    if 'future_out.weight' in state_dict and 'delta_out.weight' not in state_dict:
        state_dict['delta_out.weight'] = state_dict.pop('future_out.weight')
        state_dict['delta_out.bias'] = state_dict.pop('future_out.bias')
    if 'seq_edit_proj.weight' not in state_dict and 'edit_proj.weight' in state_dict:
        state_dict['seq_edit_proj.weight'] = state_dict['edit_proj.weight'].clone()
        state_dict['seq_edit_proj.bias'] = state_dict['edit_proj.bias'].clone()
    edit_dim = state_dict["edit_proj.weight"].shape[1]
    model = build_model(cfg, torch_device)
    if model.edit_proj.weight.shape[1] != edit_dim:
        text_vocab_size = 0
        if "text_embed.weight" in state_dict:
            text_vocab_size = state_dict["text_embed.weight"].shape[0]
        model = create_model(
            model_arch=cfg.model_arch,
            pose_dim=model.pose_proj.in_features,
            edit_dim=edit_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            text_vocab_size=text_vocab_size,
            max_frames=cfg.max_frames,
        ).to(torch_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, edit_dim


def load_model_for_inference(config_path: str, checkpoint_path: str, device: str = "auto"):
    cfg = load_simple_yaml(config_path)
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    model, edit_dim = _build_model_from_checkpoint(cfg, checkpoint_path, torch_device)
    return cfg, model, torch_device, edit_dim


def _tensorize_case(sample: dict, device: torch.device, condition_mode: str) -> tuple[torch.Tensor, dict]:
    batch = {
        "source_pose": sample["source_pose"].unsqueeze(0),
        "target_pose": sample["target_pose"].unsqueeze(0),
        "joint_mask": sample["joint_mask"].unsqueeze(0),
        "conditioning_frame_mask": sample["conditioning_frame_mask"].unsqueeze(0),
        "edit_vector": sample["edit_vector"].unsqueeze(0),
        "prompt_token_ids": sample["prompt_token_ids"].unsqueeze(0),
        "prompt_attention_mask": sample["prompt_attention_mask"].unsqueeze(0),
        "goal_start_frame": sample["goal_start_frame"].unsqueeze(0),
        "goal_end_frame": sample["goal_end_frame"].unsqueeze(0),
        "goal_delta_deg": sample["goal_delta_deg"].unsqueeze(0),
        "goal_direction_sign": sample["goal_direction_sign"].unsqueeze(0),
        "seq_edit_vectors": sample["seq_edit_vectors"].unsqueeze(0),
        "seq_start_frames": sample["seq_start_frames"].unsqueeze(0),
        "seq_end_frames": sample["seq_end_frames"].unsqueeze(0),
        "seq_delta_deg": sample["seq_delta_deg"].unsqueeze(0),
        "seq_direction_sign": sample["seq_direction_sign"].unsqueeze(0),
        "seq_num_edits": sample["seq_num_edits"].unsqueeze(0),
    }
    source_pose = batch["source_pose"].to(device).reshape(1, sample["source_pose"].shape[0], -1)
    condition = _build_condition_inputs(batch, device, condition_mode)
    return source_pose, condition


def run_prefix_case_inference(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    case_indices: list[int],
    device: str = "auto",
    prompt_style: str = "template",
) -> list[dict]:
    cfg, model, torch_device, checkpoint_edit_dim = load_model_for_inference(config_path, checkpoint_path, device=device)
    cfg.manifest_path = manifest_path
    cfg.prompt_style = prompt_style
    cfg.max_clips = 0
    dataset = build_dataset(cfg)

    results = []
    for case_idx in case_indices:
        sample = dataset[case_idx]
        if sample["edit_vector"].numel() != checkpoint_edit_dim:
            program = json.loads(sample.get("program_json", "{}"))
            schema = load_label_schema(cfg.label_schema_path) if cfg.label_schema_path else load_label_schema()
            if "edits" in program and program["edits"]:
                edit_programs = [EditProgram.from_dict(item) for item in program["edits"]]
                program_obj = edit_programs[0]
                legacy_seq = []
                for item in edit_programs[: sample["seq_edit_vectors"].shape[0]]:
                    legacy_seq.append(_build_legacy_program_vector(item, schema=schema, edit_dim=checkpoint_edit_dim))
                if legacy_seq:
                    stacked = np.zeros((sample["seq_edit_vectors"].shape[0], checkpoint_edit_dim), dtype=np.float32)
                    for i, vec in enumerate(legacy_seq):
                        stacked[i] = vec
                    sample["seq_edit_vectors"] = torch.from_numpy(stacked)
            else:
                program_obj = EditProgram.from_dict(program)
                stacked = np.zeros((sample["seq_edit_vectors"].shape[0], checkpoint_edit_dim), dtype=np.float32)
                first_legacy = _build_legacy_program_vector(program_obj, schema=schema, edit_dim=checkpoint_edit_dim)
                stacked[0] = first_legacy
                sample["seq_edit_vectors"] = torch.from_numpy(stacked)
            legacy_vec = _build_legacy_program_vector(program_obj, schema=schema, edit_dim=checkpoint_edit_dim)
            sample["edit_vector"] = torch.from_numpy(legacy_vec)
        source_pose, condition = _tensorize_case(sample, torch_device, cfg.condition_mode)
        with torch.no_grad():
            pred_pose = model(source_pose, **condition).cpu()[0].view(sample["source_pose"].shape[0], 52, 3)
        target_path = sample.get("source_path", "")
        source_npz = np.load(target_path, allow_pickle=True) if target_path else None
        program = json.loads(sample.get("program_json", "{}"))
        result = {
            "case_idx": case_idx,
            "prompt_text": sample["prompt_text"],
            "source_path": target_path,
            "target_path": target_path,
            "program": program,
            "source_pose": sample["source_pose"].cpu().numpy(),
            "target_pose": sample["target_pose"].cpu().numpy(),
            "pred_pose": pred_pose.numpy(),
            "source_trans": sample["source_trans"].cpu().numpy(),
            "target_trans": sample["target_trans"].cpu().numpy(),
            "betas": np.asarray(source_npz.get("betas", np.zeros((1, 16), dtype=np.float32)), dtype=np.float32)
            if source_npz is not None
            else np.zeros((1, 16), dtype=np.float32),
        }
        results.append(result)
    return results


def run_mined_case_inference(
    config_path: str,
    checkpoint_path: str,
    pair_manifest_path: str,
    case_indices: list[int],
    device: str = "auto",
    prompt_style: str = "template",
) -> list[dict]:
    cfg, model, torch_device, checkpoint_edit_dim = load_model_for_inference(config_path, checkpoint_path, device=device)
    schema = load_label_schema(cfg.label_schema_path) if cfg.label_schema_path else load_label_schema()
    dataset = MinedMotionEditDataset(
        pair_manifest_path=pair_manifest_path,
        max_pairs=0,
        label_schema_path=cfg.label_schema_path,
        prompt_style=prompt_style,
        prompt_max_length=cfg.prompt_max_length,
    )

    results = []
    for case_idx in case_indices:
        sample = dataset[case_idx]
        record = dataset.records[case_idx]
        if sample["edit_vector"].numel() != checkpoint_edit_dim:
            program = EditProgram.from_dict(record["program"])
            legacy_vec = _build_legacy_program_vector(program, schema=schema, edit_dim=checkpoint_edit_dim)
            sample["edit_vector"] = torch.from_numpy(legacy_vec)
        source_pose, condition = _tensorize_case(sample, torch_device, cfg.condition_mode)
        with torch.no_grad():
            pred_pose = model(source_pose, **condition).cpu()[0].view(sample["source_pose"].shape[0], 52, 3)
        source_npz = np.load(record["source_path"], allow_pickle=True)
        result = {
            "case_idx": case_idx,
            "prompt_text": sample["prompt_text"],
            "source_path": record["source_path"],
            "target_path": record["target_path"],
            "program": record["program"],
            "source_pose": sample["source_pose"].cpu().numpy(),
            "target_pose": sample["target_pose"].cpu().numpy(),
            "pred_pose": pred_pose.numpy(),
            "source_trans": sample["source_trans"].cpu().numpy(),
            "target_trans": sample["target_trans"].cpu().numpy(),
            "betas": np.asarray(source_npz.get("betas", np.zeros((1, 16), dtype=np.float32)), dtype=np.float32),
        }
        results.append(result)
    return results
