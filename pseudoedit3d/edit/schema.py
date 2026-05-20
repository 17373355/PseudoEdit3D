from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EDIT_PARTS = ["left_arm", "right_arm", "both_arms", "torso"]
EDIT_ATTRIBUTES = [
    "raise",
    "lower",
    "bend",
    "extend",
    "lean_left",
    "lean_right",
    "lean_forward",
    "lean_backward",
]
DELTA_BINS = ["small", "medium", "large"]
CONTACT_POLICIES = ["ignore", "keep"]


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

    def to_vector(self) -> list[float]:
        part_vec = [1.0 if self.part == value else 0.0 for value in EDIT_PARTS]
        attr_vec = [1.0 if self.attribute == value else 0.0 for value in EDIT_ATTRIBUTES]
        delta_vec = [1.0 if self.delta_bin == value else 0.0 for value in DELTA_BINS]
        contact_vec = [1.0 if self.contact_policy == value else 0.0 for value in CONTACT_POLICIES]
        span_vec = [self.start_frame / 59.0, self.end_frame / 59.0]
        return part_vec + attr_vec + delta_vec + contact_vec + span_vec

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
        )
