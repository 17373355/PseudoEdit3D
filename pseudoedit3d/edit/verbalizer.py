from __future__ import annotations

from typing import Iterable

from pseudoedit3d.edit.schema import EditProgram, LabelSchema, get_default_schema, load_label_schema
from pseudoedit3d.edit.skill_context import SKILL_LABEL_TO_PROMPT_PHRASE


def _coarse_time_phrase(start_frame: int, end_frame: int, total_frames: int = 60) -> str:
    start_ratio = start_frame / max(total_frames - 1, 1)
    end_ratio = end_frame / max(total_frames - 1, 1)
    center = 0.5 * (start_ratio + end_ratio)
    duration = end_ratio - start_ratio
    if duration >= 0.75:
        return "throughout the motion"
    if center < 0.33:
        return "early in the motion"
    if center > 0.66:
        return "late in the motion"
    return "during the middle of the motion"


def _format_delta(program: EditProgram, schema: LabelSchema, numeric: bool = True) -> str:
    delta_entry = schema.delta_bin(program.delta_bin)
    if numeric:
        value = program.delta_value_deg if program.delta_value_deg is not None else delta_entry.default_degrees
        if value is None:
            return delta_entry.display
        value_str = str(int(round(float(abs(value))))) if abs(float(value) - round(float(value))) < 1e-4 else f"{abs(float(value)):.1f}"
        return f"{value_str} degrees"
    if delta_entry.phrases:
        return delta_entry.phrases[0]
    return delta_entry.display


def _format_target_value(program: EditProgram, schema: LabelSchema) -> str:
    target_value = program.metadata.get("target_value_deg")
    if target_value is None:
        target_value = program.delta_value_deg
    if target_value is None:
        target_value = schema.delta_bin(program.delta_bin).default_degrees
    value = float(abs(target_value))
    value_str = str(int(round(value))) if abs(value - round(value)) < 1e-4 else f"{value:.1f}"
    return f"{value_str} degrees"


def verbalize_program(
    program: EditProgram,
    schema: LabelSchema | None = None,
    style: str = "template",
    variant_index: int = 0,
    total_frames: int = 60,
) -> str:
    schema = schema or get_default_schema()
    attribute_entry = schema.attribute(program.attribute)
    part_entry = schema.part(program.part)

    variants = attribute_entry.prompt_variants or [attribute_entry.display]
    verb = variants[variant_index % len(variants)]
    operator_entry = schema.operator(program.operator)
    templates = operator_entry.prompt_templates or attribute_entry.prompt_templates or schema.prompt_defaults.get("templates", [])
    if not templates:
        templates = ["{verb} the {part} by {delta_value} {time_phrase}"]
    template = templates[variant_index % len(templates)]

    numeric = style != "paraphrase"
    delta_value = _format_delta(program, schema, numeric=numeric)
    delta_phrase = _format_delta(program, schema, numeric=False)
    target_value = _format_target_value(program, schema)
    time_phrase = _coarse_time_phrase(program.start_frame, program.end_frame, total_frames=total_frames)
    skill_phrase = SKILL_LABEL_TO_PROMPT_PHRASE.get(program.skill_label or "unknown", "the current motion")
    prompt = template.format(
        verb=verb,
        part=part_entry.display,
        delta_value=delta_value,
        delta_phrase=delta_phrase,
        target_value=target_value,
        time_phrase=time_phrase,
        attribute=attribute_entry.display,
    )
    task_mode = program.metadata.get("task_mode")
    if (
        program.reference == "current_state"
        and program.skill_label not in {None, "unknown", "static_pose"}
        and task_mode != "semantic_continue"
    ):
        if program.operator == "add":
            prompt = f"{prompt} while continuing {skill_phrase}"
        else:
            prompt = f"{prompt} from {skill_phrase}"
    prompt = " ".join(prompt.strip().split())
    return prompt


def augment_program_prompts(
    program: EditProgram,
    schema: LabelSchema | None = None,
    styles: Iterable[str] = ("template", "paraphrase"),
    max_variants_per_style: int = 3,
    total_frames: int = 60,
) -> list[str]:
    schema = schema or get_default_schema()
    prompts = []
    for style in styles:
        for variant_index in range(max_variants_per_style):
            prompt = verbalize_program(
                program=program,
                schema=schema,
                style=style,
                variant_index=variant_index,
                total_frames=total_frames,
            )
            if prompt not in prompts:
                prompts.append(prompt)
    return prompts


def load_schema_from_path(label_schema_path: str | None) -> LabelSchema:
    if label_schema_path:
        return load_label_schema(label_schema_path)
    return get_default_schema()
