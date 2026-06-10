# AML Edit Call Graph

This document records the active `pseudoedit3d/edit` call graph and the current legacy boundary.

## Current Mainline

```text
HumanML3D joints / pose / translation
  -> pseudoedit3d.edit.frame_observables.extract_layer0_frame_observables
       -> pseudoedit3d.edit.attributes.extract_upper_body_proxy_attributes
  -> pseudoedit3d.edit.micro_events.extract_layer1_micro_events
       -> segment_observable
       -> segment_locomotion_state
       -> segment_low_body_state
  -> pseudoedit3d.edit.submotion_lexicon.merge_micro_events
  -> pseudoedit3d.edit.phase_patterns.detect_repeated_phases
       -> phase_patterns.project_units_by_category
  -> pseudoedit3d.edit.aml_atomic_program.build_layer3_atomic_program
       -> _submotion_to_event
       -> _phase_to_event
       -> _build_locomotion_turn_events
       -> _build_terminal_state_events
       -> pseudoedit3d.edit.semantic_events.build_semantic_joint_events
       -> pseudoedit3d.edit.bimanual_split.split_bimanual_events
       -> abstract_atomic_events
  -> pseudoedit3d.edit.coarse_signature.build_coarse_action_program
       -> build_event_coarse_signature
       -> assign_seeded_prototype
       -> _cover_primary_events
       -> _semantic_candidate_actions
       -> _candidate_jump_segments
       -> _recovery_step_segments
       -> _secondary_actions
       -> _apply_semantic_dominance
       -> _attach_action_metadata
            -> _semantic_family_descriptor
            -> _approx_slots
  -> pseudoedit3d.edit.coarse_prompt_renderer.render_coarse_aml_prompt
       -> _action_clause
       -> _residual_clauses
            -> pseudoedit3d.edit.aml_prompt_renderer.event_to_prompt_clause
  -> scripts.export_aml_condition_manifest
       -> pseudoedit3d.edit.aml_condition_schema
       -> conditions.jsonl
```

The main AutoPrompt path is the coarse path:

```text
scripts/run_momask_aml_autoprompt_probe.py --prompt-mode coarse
scripts/build_aml_regression_testset.py
scripts/analyze_aml_semantic_family_status.py
scripts/export_aml_condition_manifest.py
```

Inspection and visualization entrypoints:

```text
scripts/extract_aml_layers.py
scripts/visualize_aml_atomic_program.py
scripts/summarize_aml_family_taxonomy.py
scripts/analyze_coordination_patterns.py
scripts/analyze_bimanual_split_candidates.py
```

## Renderer Boundary

`coarse_prompt_renderer.render_coarse_aml_prompt` is the current MoMask probe-text renderer. It should prefer `canonical_actions`, `semantic_family`, and `approx_slots`.

`aml_prompt_renderer.render_aml_prompt` is retained for event-stream diagnostics and residual clauses only. It should not define the main AML semantics.

## Module Status

| File | Current role | Decision |
| --- | --- | --- |
| `frame_observables.py` | Layer 0 dense signals. | Active AML mainline. |
| `micro_events.py` | Layer 1 local events and state events. | Active AML mainline. |
| `submotion_lexicon.py` | Layer 2 event composition. | Active AML mainline. |
| `phase_patterns.py` | Layer 2.5 repeated/alternating patterns. | Active AML mainline. |
| `aml_atomic_program.py` | Layer 3 event family abstraction. | Active AML mainline. |
| `semantic_events.py` | Joint-derived semantic proxy events used by Layer 3. | Active AML mainline. |
| `bimanual_split.py` | Bimanual periodic split and feature diagnostics. | Active AML mainline. |
| `coordination_patterns.py` | Layer 4 coordination diagnostics for regression selection. | Active diagnostic path. |
| `aml_language.py` | Deterministic AML text for inspection. | Active inspection path. |
| `coarse_signature.py` | Layer 3.5/4 coarse signatures, canonical actions, semantic family, approximate slots. | Active AutoPrompt and future conditioning path. |
| `coarse_prompt_renderer.py` | Budgeted probe text from canonical actions. | Active AutoPrompt path. |
| `aml_condition_schema.py` | Required-slot schema and condition weights shared by diagnostics/exporters. | Active future conditioning path. |
| `aml_prompt_renderer.py` | Event-stream naturalizer and residual fallback. | Keep active, but not main semantics. |
| `attributes.py` | Shared low-level proxy attributes. | Keep active; used by Layer 0 and training/data paths. |
| `action_program.py` | Old Stage 1 goal-vector conversion. | Keep active; imported by `pseudoedit3d/data`. |
| `schema.py` | Old Stage 1 edit schema and program dataclasses. | Keep active; imported by data/training/inference. |
| `verbalizer.py` | Old Stage 1 prompt verbalizer. | Keep active; imported by data paths. |
| `skill_context.py` | Old Stage 1 skill context summaries. | Keep active; imported by mined/prefix datasets. |
| `synthetic.py` | Synthetic edit sample construction. | Keep active; imported by `pseudoedit3d/data/dataset.py`. |
| `mining.py` | Mined pair construction. | Keep active; imported by mining scripts and artifact builders. |
| `segmentation.py` | Active-span helper for mined pairs. | Keep active through `mining.py`. |
| `hierarchical_atomic.py` | Old hierarchical atomic candidates. | Keep active until `prefix_dataset.py` dependency is migrated. |
| `iterative.py` | Old mined-pair iterative refinement loop. | Moved to legacy; no active AML caller. |

## Legacy Boundary

Moved on 2026-06-10:

```text
legacy/aml_demos/scripts/
  dump_micro_events_demo.py
  visualize_micro_events.py
  detect_phase_patterns_demo.py
  visualize_phase_patterns.py
  merge_submotion_demo.py
  visualize_submotions.py
  summarize_phase_hierarchy.py
  summarize_aml_hierarchy.py

legacy/old_stage1_iterative_refinement/
  pseudoedit3d/edit/iterative.py
  scripts/iterate_mined_pairs.py
```

Reason:

- the demo scripts are superseded by `scripts/extract_aml_layers.py` and `scripts/visualize_aml_atomic_program.py`;
- `iterative.py` is only used by the old mined-pair iterative refinement script and is not part of the AML coarse action path.

Not moved yet:

- `hierarchical_atomic.py`, because `pseudoedit3d/data/prefix_dataset.py` imports it;
- `action_program.py`, `schema.py`, `verbalizer.py`, `skill_context.py`, `synthetic.py`, `mining.py`, and `segmentation.py`, because data/training/mining code still imports them;
- `aml_prompt_renderer.py`, because `coarse_prompt_renderer.py` uses `event_to_prompt_clause` for residual event fallback.

## Next Plan

1. Unknown semantic family audit:
   - run `scripts/analyze_aml_semantic_family_status.py` on the 250-case set and gap cases;
   - group `unknown` actions by `source_event_family_counts`, `source_event_cluster_counts`, span, salience, and selected HML3D prompt only for diagnosis;
   - split real unknowns from missing-metadata bugs.

2. Add conservative semantic families:
   - promote only high-confidence motion signatures to `stable`;
   - use `candidate` for plausible action classes such as arm mime, dance/acrobatic, or leg-pose families;
   - use `proxy` for observable states such as subtle pose hold, hand high, squat hold, climb-over proxy, or support-like contact.

3. Calibrate approximate slots:
   - define required slots per family, such as `span`, `count`, `direction`, `distance_m`, `angle_deg`, `vertical_amplitude_m`, `dominant_side`;
   - report missing or low-quality slots per family;
   - keep old flat `slots` for compatibility, but treat `approx_slots` as the uncertainty-aware condition contract.

4. Renderer policy after family cleanup:
   - render `stable` families directly;
   - render `candidate` and `proxy` families conservatively;
   - keep `unknown` out of probe text unless it contributes a safe residual clause;
   - never drop structured `canonical_actions` from the saved coarse program.

5. Only after the above:
   - export a training-oriented AML condition manifest from `canonical_actions`, `semantic_family`, and `approx_slots`;
   - connect the manifest to the feedback-conditioned revision / Language-Guided Action Regulation training stack.

Current manifest exporter:

```text
scripts/export_aml_condition_manifest.py
  -> conditions.jsonl
  -> summary.json
  -> summary.md
```
