# PseudoEdit3D Docs Index

This docs tree is organized for paper-driven research iteration.

## Recommended Entry Points

- `design_brief.md`
- `design/design_overview.md`
- `experiments/evaluation_experiments.md`
- `experiments/hml3d_benchmark_plan.md`
- `logs/experiment_registry.md`
- `paper/paper_workspace.md`
- `paper/paper_current_proposal.md`

## Current Research Goal

Build a motion-derived annotation layer that is:
- more atomic than raw HumanML3D captions
- more accurate than raw HumanML3D captions
- suitable for training a structured condition model later

HumanML3D raw text is treated as a noisy prior, not a strict ground-truth target.

## New Paper-Oriented Structure

### `design/`
Research framing, system design, task definition, and pipeline design.

### `experiments/`
Evaluation design, benchmark protocols, batch-iteration plans, and representative findings.

### `logs/`
Running experiment logs, milestone notes, and artifact indexes.

### `paper/`
Paper outline, claim structure, experiment-to-paper mapping, and draft planning.

### `notes/`
Discussion notes, unresolved questions, and distilled insights.

### `legacy/`
Older proposal drafts and superseded design documents kept for historical reference.

## Legacy Top-Level Docs Still In Use

These files remain valid and are linked into the new structure rather than moved immediately:

- `experiment_log.md`
- `planning.md`
- `hml3d_experiment_pipeline.md`
- `humanml3d_pattern_iteration_plan.md`
- `paper_v3_proposal.md`
- `stage1_upgrade_plan.md`
- `stage_1_goal.md`
- `stage_1_goal_template.md`
- `memory.md`

## Suggested Reading Order

1. `design/design_overview.md`
2. `experiments/evaluation_experiments.md`
3. `experiments/hml3d_benchmark_plan.md`
4. `logs/experiment_registry.md`
5. `paper/paper_workspace.md`
