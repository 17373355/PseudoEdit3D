"""Public edit-module exports.

The edit package includes both lightweight JSON/program utilities and heavier
motion-processing code. Keep exports lazy so pure program inspection does not
require optional numeric dependencies at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "EditProgram": ("pseudoedit3d.edit.schema", "EditProgram"),
    "build_goal_spec": ("pseudoedit3d.edit.action_program", "build_goal_spec"),
    "goal_spec_to_numpy": ("pseudoedit3d.edit.action_program", "goal_spec_to_numpy"),
    "get_default_schema": ("pseudoedit3d.edit.schema", "get_default_schema"),
    "load_label_schema": ("pseudoedit3d.edit.schema", "load_label_schema"),
    "augment_program_prompts": ("pseudoedit3d.edit.verbalizer", "augment_program_prompts"),
    "verbalize_program": ("pseudoedit3d.edit.verbalizer", "verbalize_program"),
    "ObservableSequence": ("pseudoedit3d.edit.frame_observables", "ObservableSequence"),
    "FrameObservables": ("pseudoedit3d.edit.frame_observables", "FrameObservables"),
    "extract_layer0_frame_observables": ("pseudoedit3d.edit.frame_observables", "extract_layer0_frame_observables"),
    "MicroEvent": ("pseudoedit3d.edit.micro_events", "MicroEvent"),
    "extract_layer1_micro_events": ("pseudoedit3d.edit.micro_events", "extract_layer1_micro_events"),
    "SubMotionPattern": ("pseudoedit3d.edit.submotion_lexicon", "SubMotionPattern"),
    "SubMotionUnit": ("pseudoedit3d.edit.submotion_lexicon", "SubMotionUnit"),
    "SUBMOTION_LEXICON_V1": ("pseudoedit3d.edit.submotion_lexicon", "SUBMOTION_LEXICON_V1"),
    "merge_micro_events": ("pseudoedit3d.edit.submotion_lexicon", "merge_micro_events"),
    "PhasePattern": ("pseudoedit3d.edit.phase_patterns", "PhasePattern"),
    "detect_repeated_phases": ("pseudoedit3d.edit.phase_patterns", "detect_repeated_phases"),
    "project_units_by_category": ("pseudoedit3d.edit.phase_patterns", "project_units_by_category"),
    "build_layer3_atomic_program": ("pseudoedit3d.edit.aml_atomic_program", "build_layer3_atomic_program"),
    "dedupe_phase_patterns": ("pseudoedit3d.edit.aml_atomic_program", "dedupe_phase_patterns"),
    "aml_event_to_template": ("pseudoedit3d.edit.aml_language", "aml_event_to_template"),
    "aml_program_to_templates": ("pseudoedit3d.edit.aml_language", "aml_program_to_templates"),
    "attach_aml_language": ("pseudoedit3d.edit.aml_language", "attach_aml_language"),
    "event_to_prompt_clause": ("pseudoedit3d.edit.aml_prompt_renderer", "event_to_prompt_clause"),
    "select_prompt_events": ("pseudoedit3d.edit.aml_prompt_renderer", "select_prompt_events"),
    "render_aml_prompt": ("pseudoedit3d.edit.aml_prompt_renderer", "render_aml_prompt"),
    "render_coarse_aml_prompt": ("pseudoedit3d.edit.coarse_prompt_renderer", "render_coarse_aml_prompt"),
    "build_coarse_action_program": ("pseudoedit3d.edit.coarse_signature", "build_coarse_action_program"),
    "assign_seeded_prototype": ("pseudoedit3d.edit.coarse_signature", "assign_seeded_prototype"),
    "build_event_coarse_signature": ("pseudoedit3d.edit.coarse_signature", "build_event_coarse_signature"),
    "build_geometry_signature": ("pseudoedit3d.edit.geometry_sidecar", "build_geometry_signature"),
    "summarize_geometry_sidecars": ("pseudoedit3d.edit.geometry_sidecar", "summarize_geometry_sidecars"),
    "load_composable_pattern_program": (
        "pseudoedit3d.edit.aml_composable_pattern_program",
        "load_composable_pattern_program",
    ),
    "SUPPORT_STATE_V1_PROGRAM_PATH": (
        "pseudoedit3d.edit.aml_composable_pattern_program",
        "SUPPORT_STATE_V1_PROGRAM_PATH",
    ),
    "program_nodes": ("pseudoedit3d.edit.aml_composable_pattern_program", "program_nodes"),
    "child_node_ids": ("pseudoedit3d.edit.aml_composable_pattern_program", "child_node_ids"),
    "child_nodes": ("pseudoedit3d.edit.aml_composable_pattern_program", "child_nodes"),
    "condition_vocabulary": ("pseudoedit3d.edit.aml_composable_pattern_program", "condition_vocabulary"),
    "condition_by_program_node": ("pseudoedit3d.edit.aml_composable_pattern_program", "condition_by_program_node"),
    "edit_handles_for_condition": ("pseudoedit3d.edit.aml_composable_pattern_program", "edit_handles_for_condition"),
    "search_program_nodes": ("pseudoedit3d.edit.aml_composable_pattern_program", "search_program_nodes"),
    "summarize_program": ("pseudoedit3d.edit.aml_composable_pattern_program", "summarize_program"),
}


__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
