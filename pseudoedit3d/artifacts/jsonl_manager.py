from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _slugify(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def count_jsonl_records(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@dataclass
class JsonlArtifactManager:
    root: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "JsonlArtifactManager":
        return cls(root=Path(project_root) / "artifacts" / "jsonl")

    def registry_path(self) -> Path:
        return self.root / "registry.jsonl"

    def artifact_path(
        self,
        subset: str,
        stage: str,
        purpose: str,
        split: str = "full",
        tag: str = "",
    ) -> Path:
        subset_slug = _slugify(subset)
        stage_slug = _slugify(stage)
        purpose_slug = _slugify(purpose)
        split_slug = _slugify(split)
        filename = f"{purpose_slug}_{split_slug}.jsonl" if not tag else f"{purpose_slug}_{split_slug}_{_slugify(tag)}.jsonl"
        path = self.root / subset_slug / stage_slug / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_registry_entry(
        self,
        subset: str,
        stage: str,
        purpose: str,
        path: str | Path,
        split: str = "full",
        tag: str = "",
        num_records: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        registry_path = self.registry_path()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "subset": subset,
            "stage": stage,
            "purpose": purpose,
            "split": split,
            "tag": tag,
            "path": str(Path(path)),
            "num_records": num_records,
        }
        if extra:
            payload.update(extra)
        with registry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
