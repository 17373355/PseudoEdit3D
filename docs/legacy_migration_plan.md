# Legacy Migration Plan

This document separates the current AML mainline from earlier auto-prompt / pattern-repair mechanisms.

## Current AML Mainline

Keep these as active components:

- `pseudoedit3d/edit/frame_observables.py`
- `pseudoedit3d/edit/micro_events.py`
- `pseudoedit3d/edit/submotion_lexicon.py`
- `pseudoedit3d/edit/phase_patterns.py`
- `pseudoedit3d/edit/aml_atomic_program.py`
- `pseudoedit3d/edit/aml_language.py`
- `pseudoedit3d/edit/coarse_signature.py`
- `pseudoedit3d/edit/coarse_prompt_renderer.py`
- `scripts/build_aml_mining_corpus.py`
- `scripts/extract_aml_layers.py`
- `scripts/summarize_aml_family_taxonomy.py`
- `scripts/select_aml_cluster_representatives.py`
- `scripts/select_aml_split_candidate_representatives.py`
- `scripts/visualize_aml_atomic_program.py`

These implement the current architecture:

```text
Layer 0 frame observables
-> Layer 1 micro-events and state-events
-> Layer 2 sub-motion units
-> Layer 2.5 phase/repetition patterns
-> Layer 3 atomic program
-> Layer 3.5 coarse signature, semantic family, approximate slots
-> Layer 4 canonical action program
-> AML template language
```

## Active Probe / Dependency, Not Main Representation

Keep available for now, but treat as probe infrastructure rather than the AML representation:

- `scripts/run_momask_case_study.py`
- `scripts/visualize_momask_case_study.py`

Reason:

- useful for MoMask generation probes and selected-HML3D-vs-auto-prompt visualization
- should no longer define AML semantics
- same-case HumanML3D text must not be used to generate AML labels

## Moved Legacy: Earlier Pattern-Repair Batch Workflow

Moved to `legacy/auto_prompt_pattern_batches/` on 2026-06-16 after confirming
no active script imports them:

- `legacy/auto_prompt_pattern_batches/scripts/build_hml3d_pattern_batch.py`
- `legacy/auto_prompt_pattern_batches/scripts/analyze_hml3d_pattern_batch.py`
- `legacy/auto_prompt_pattern_batches/scripts/report_hml3d_batch_sizes.py`
- `legacy/auto_prompt_pattern_batches/scripts/mine_hml3d_missing_patterns.py`
- `docs/humanml3d_pattern_iteration_plan.md`
- `docs/hml3d_experiment_pipeline.md`
- `docs/experiments/regression_check_protocol.md`
- `docs/logs/experiment_registry.md`
- `docs/experiments/hml3d_benchmark_plan.md`

Reason:

- these belong to the earlier batch-wise auto-prompt correction workflow
- useful historical evidence, but no longer the core architecture

## Moved Legacy: Demo-Only Layer Dumps / Visualizers

Moved to `legacy/aml_demos/` on 2026-06-10 after confirming they are superseded by `extract_aml_layers.py` and `visualize_aml_atomic_program.py`:

- `legacy/aml_demos/scripts/dump_micro_events_demo.py`
- `legacy/aml_demos/scripts/visualize_micro_events.py`
- `legacy/aml_demos/scripts/detect_phase_patterns_demo.py`
- `legacy/aml_demos/scripts/visualize_phase_patterns.py`
- `legacy/aml_demos/scripts/merge_submotion_demo.py`
- `legacy/aml_demos/scripts/visualize_submotions.py`
- `legacy/aml_demos/scripts/summarize_phase_hierarchy.py`
- `legacy/aml_demos/scripts/summarize_aml_hierarchy.py`

Reason:

- still useful for debugging individual layers
- but should be marked as demo/legacy if the unified AML export/visualization scripts cover the same use cases

## Moved Legacy: Old Mined-Pair Iterative Refinement

Moved to `legacy/old_stage1_iterative_refinement/` on 2026-06-10:

- `legacy/old_stage1_iterative_refinement/pseudoedit3d/edit/iterative.py`
- `legacy/old_stage1_iterative_refinement/scripts/iterate_mined_pairs.py`

Reason:

- `pseudoedit3d/edit/iterative.py` was only imported by `scripts/iterate_mined_pairs.py`
- the loop belongs to the old mined-pair Stage 1 scaffold, not the current AML coarse action path
- current unknown-family / approximate-slot work lives in `pseudoedit3d/edit/coarse_signature.py` and `scripts/analyze_aml_semantic_family_status.py`

## Moved Legacy: Hard-Coded Text Phrase Mining Probe

Moved to `legacy/text_phrase_mining/` on 2026-06-23:

- `legacy/text_phrase_mining/scripts/mine_hml3d_upperbody_phrases.py`

Reason:

- this script used a Python-local hard-coded word-family list for earlier upper-body phrase mining;
- current Motion-BPE mainline separates text pseudo-GT target definitions into `configs/motion_pattern_text_targets.json`;
- generic audit scripts consume those target definitions only for evaluation/naming diagnostics, not for Motion-BPE merge learning or runtime AML rules.

## Candidate Legacy: Old Hierarchical Atomic Training Scaffold

Review before moving:

- `pseudoedit3d/edit/hierarchical_atomic.py`
- `pseudoedit3d/data/prefix_dataset.py`
- old stage1 configs that depend on `hierarchical_atomic.py`

Reason:

- this may still be used by earlier Stage1 training code
- do not move until dependencies are audited

## Migration Rule

Do not delete research artifacts directly.

Preferred migration:

```text
legacy/
  README.md
  auto_prompt_pattern_batches/
  aml_demos/
  old_stage1_iterative_refinement/
  old_stage1_atomic_scaffold/
```

For each moved file:

- keep original relative path in `legacy/README.md`
- record why it moved
- record which active AML file supersedes it
- do not move files imported by active scripts

Current call graph and file decisions are documented in `docs/design/aml_edit_call_graph.md`.

## Current Architecture Coverage Status

As of 2026-06-07:

- change events: covered by Layer 1 micro-events
- sustained states: initial coverage via `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_*`
- repeated phases: covered by Layer 2.5 phase patterns
- numeric residue: preserved through magnitude, signed_delta, count, frame span, path_length, mean_speed
- template language: covered by `pseudoedit3d/edit/aml_language.py`

Known gaps:

- locomotion needs semantic splitting: forward/backward/sideways, walk/run, turn-in-place, stop/pause, step count
- contact/support needs stronger observables: wall, rail, hand support, foot contact phases
- hand and fine-arm semantics remain coarse
- quantity/repetition/angle handling needs literature-inspired design from LLM/VLM token and numeric representation work
