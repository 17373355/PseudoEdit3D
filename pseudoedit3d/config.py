from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    raw = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split(":", 1)
        raw[key.strip()] = _parse_scalar(value.strip())
    return TrainConfig(**raw)
