from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "label_schema.yaml"


@dataclass
class SchemaEntry:
    key: str
    display: str
    aliases: list[str] = field(default_factory=list)
    applicable_parts: list[str] = field(default_factory=list)
    prompt_variants: list[str] = field(default_factory=list)
    prompt_templates: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    default_degrees: float | None = None
    synthetic_axis: int | None = None
    synthetic_sign: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelSchema:
    schema_version: str
    parts: dict[str, SchemaEntry]
    attributes: dict[str, SchemaEntry]
    delta_bins: dict[str, SchemaEntry]
    contact_policies: dict[str, SchemaEntry]
    operators: dict[str, SchemaEntry]
    references: dict[str, SchemaEntry]
    preserve_modes: dict[str, SchemaEntry]
    skill_labels: dict[str, SchemaEntry]
    prompt_defaults: dict[str, Any]
    path: str

    @property
    def part_keys(self) -> list[str]:
        return list(self.parts.keys())

    @property
    def attribute_keys(self) -> list[str]:
        return list(self.attributes.keys())

    @property
    def delta_bin_keys(self) -> list[str]:
        return list(self.delta_bins.keys())

    @property
    def contact_policy_keys(self) -> list[str]:
        return list(self.contact_policies.keys())

    @property
    def operator_keys(self) -> list[str]:
        return list(self.operators.keys())

    @property
    def reference_keys(self) -> list[str]:
        return list(self.references.keys())

    @property
    def preserve_mode_keys(self) -> list[str]:
        return list(self.preserve_modes.keys())

    @property
    def skill_label_keys(self) -> list[str]:
        return list(self.skill_labels.keys())

    @property
    def vector_dim(self) -> int:
        return (
            len(self.part_keys)
            + len(self.attribute_keys)
            + len(self.delta_bin_keys)
            + len(self.contact_policy_keys)
            + len(self.operator_keys)
            + len(self.reference_keys)
            + len(self.preserve_mode_keys)
            + len(self.skill_label_keys)
            + 2
        )

    def part(self, key: str) -> SchemaEntry:
        return self.parts[key]

    def attribute(self, key: str) -> SchemaEntry:
        return self.attributes[key]

    def delta_bin(self, key: str) -> SchemaEntry:
        return self.delta_bins[key]

    def contact_policy(self, key: str) -> SchemaEntry:
        return self.contact_policies[key]

    def operator(self, key: str) -> SchemaEntry:
        return self.operators[key]

    def reference(self, key: str) -> SchemaEntry:
        return self.references[key]

    def preserve_mode(self, key: str) -> SchemaEntry:
        return self.preserve_modes[key]

    def skill_label(self, key: str) -> SchemaEntry:
        return self.skill_labels[key]

    def attributes_for_part(self, part_key: str) -> list[str]:
        return [
            key for key, entry in self.attributes.items()
            if not entry.applicable_parts or part_key in entry.applicable_parts
        ]

    def encode_program(self, program: "EditProgram") -> list[float]:
        part_vec = [1.0 if program.part == value else 0.0 for value in self.part_keys]
        attr_vec = [1.0 if program.attribute == value else 0.0 for value in self.attribute_keys]
        delta_vec = [1.0 if program.delta_bin == value else 0.0 for value in self.delta_bin_keys]
        contact_vec = [1.0 if program.contact_policy == value else 0.0 for value in self.contact_policy_keys]
        operator_vec = [1.0 if program.operator == value else 0.0 for value in self.operator_keys]
        reference_vec = [1.0 if program.reference == value else 0.0 for value in self.reference_keys]
        preserve_vec = [1.0 if program.preserve_mode == value else 0.0 for value in self.preserve_mode_keys]
        skill_vec = [1.0 if (program.skill_label or "unknown") == value else 0.0 for value in self.skill_label_keys]
        span_vec = [program.start_frame / 59.0, program.end_frame / 59.0]
        return part_vec + attr_vec + delta_vec + contact_vec + operator_vec + reference_vec + preserve_vec + skill_vec + span_vec


@dataclass
class EditProgram:
    part: str
    attribute: str
    delta_bin: str
    start_frame: int
    end_frame: int
    contact_policy: str = "ignore"
    attribute_key: str | None = None
    direction: str | None = None
    delta_value_deg: float | None = None
    source_type: str = "synthetic"
    schema_version: str | None = None
    input_mode: str = "motion"
    operator: str = "add"
    reference: str = "current_state"
    unit: str = "deg"
    preserve_parts: list[str] = field(default_factory=list)
    preserve_mode: str = "all_non_target"
    skill_label: str | None = None
    skill_phase: float | None = None
    tolerance_deg: float | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_vector(self, schema: LabelSchema | None = None) -> list[float]:
        schema = schema or get_default_schema()
        return schema.encode_program(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "attribute": self.attribute,
            "delta_bin": self.delta_bin,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "contact_policy": self.contact_policy,
            "attribute_key": self.attribute_key,
            "direction": self.direction,
            "delta_value_deg": self.delta_value_deg,
            "source_type": self.source_type,
            "schema_version": self.schema_version,
            "input_mode": self.input_mode,
            "operator": self.operator,
            "reference": self.reference,
            "unit": self.unit,
            "preserve_parts": self.preserve_parts,
            "preserve_mode": self.preserve_mode,
            "skill_label": self.skill_label,
            "skill_phase": self.skill_phase,
            "tolerance_deg": self.tolerance_deg,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditProgram":
        return cls(
            part=payload["part"],
            attribute=payload["attribute"],
            delta_bin=payload["delta_bin"],
            start_frame=int(payload["start_frame"]),
            end_frame=int(payload["end_frame"]),
            contact_policy=payload.get("contact_policy", "ignore"),
            attribute_key=payload.get("attribute_key"),
            direction=payload.get("direction"),
            delta_value_deg=payload.get("delta_value_deg"),
            source_type=payload.get("source_type", "synthetic"),
            schema_version=payload.get("schema_version"),
            input_mode=payload.get("input_mode", "motion"),
            operator=payload.get("operator", payload.get("op", "add")),
            reference=payload.get("reference", "current_state"),
            unit=payload.get("unit", "deg"),
            preserve_parts=payload.get("preserve_parts", []) or [],
            preserve_mode=payload.get("preserve_mode", "all_non_target"),
            skill_label=payload.get("skill_label", "unknown"),
            skill_phase=payload.get("skill_phase"),
            tolerance_deg=payload.get("tolerance_deg"),
            constraints=payload.get("constraints", {}) or {},
            metadata=payload.get("metadata", {}) or {},
        )


def _load_entries(payload: dict[str, dict[str, Any]]) -> dict[str, SchemaEntry]:
    entries = {}
    for key, value in payload.items():
        value = dict(value or {})
        known_keys = {
            "display", "aliases", "applicable_parts", "prompt_variants", "prompt_templates",
            "phrases", "default_degrees", "synthetic_axis", "synthetic_sign",
        }
        entries[key] = SchemaEntry(
            key=key,
            display=value.pop("display", key.replace("_", " ")),
            aliases=value.pop("aliases", []) or [],
            applicable_parts=value.pop("applicable_parts", []) or [],
            prompt_variants=value.pop("prompt_variants", []) or [],
            prompt_templates=value.pop("prompt_templates", []) or [],
            phrases=value.pop("phrases", []) or [],
            default_degrees=value.pop("default_degrees", None),
            synthetic_axis=value.pop("synthetic_axis", None),
            synthetic_sign=value.pop("synthetic_sign", None),
            extra={k: v for k, v in value.items() if k not in known_keys},
        )
    return entries


def load_label_schema(path: str | Path | None = None) -> LabelSchema:
    schema_path = Path(path) if path else _default_schema_path()
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    return LabelSchema(
        schema_version=str(payload.get("schema_version", "v0.1")),
        parts=_load_entries(payload.get("parts", {})),
        attributes=_load_entries(payload.get("attributes", {})),
        delta_bins=_load_entries(payload.get("delta_bins", {})),
        contact_policies=_load_entries(payload.get("contact_policies", {})),
        operators=_load_entries(payload.get("operators", {})),
        references=_load_entries(payload.get("references", {})),
        preserve_modes=_load_entries(payload.get("preserve_modes", {})),
        skill_labels=_load_entries(payload.get("skill_labels", {})),
        prompt_defaults=dict(payload.get("prompt_defaults", {})),
        path=str(schema_path),
    )


@lru_cache(maxsize=4)
def get_default_schema() -> LabelSchema:
    return load_label_schema(_default_schema_path())
