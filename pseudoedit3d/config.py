from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    seed: int
    data_mode: str
    dataset_root: str
    manifest_path: str
    pair_manifest_path: str
    contact_filter: str
    max_clips: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    num_layers: int
    dropout: float
    max_frames: int
    delta_scale_deg: float
    save_dir: str
    condition_mode: str = "program"
    label_schema_path: str = ""
    prompt_style: str = "template"
    prompt_max_length: int = 96
    prefix_task_mode: str = "relative_edit"
    input_source_mode: str = "source_motion"
    source_prefix_frames: int = 1
    lambda_keep: float = 0.5
    lambda_smooth: float = 0.01
    lambda_condition: float = 0.0
    use_goal_satisfaction_loss: bool = False
    lambda_goal_delta: float = 0.0
    lambda_goal_direction: float = 0.0
    lambda_goal_tolerance: float = 0.0
    lambda_goal_span: float = 0.0
    lambda_goal_offset: float = 0.0
    lambda_goal_amplitude: float = 0.0
    lambda_goal_preserve_attr: float = 0.0


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TrainConfig(**raw)
