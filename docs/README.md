# PseudoEdit3D Docs Index

This docs tree is organized for paper-driven research iteration.

## Recommended Entry Points

- `design/motion_corpus_pattern_tree_mainline.md`
- `design/motion_cluster_bpe_tree_induction.md`
- `design/multi_channel_motion_bpe_extraction.md`
- `design/text_bpe_wordnet_naming_layer.md`
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
HumanML3D motion is treated as a motion corpus; motion clusters and motion-BPE
motifs should induce pattern-tree structure, while text-BPE and WordNet provide
names and semantic grouping.

The current Motion-BPE direction is moving from a single flattened Layer3 event
sequence to a multi-channel representation with channel events, temporal
overlap graphs, and parallel packets.

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
2. `design/motion_corpus_pattern_tree_mainline.md`
3. `design/motion_cluster_bpe_tree_induction.md`
4. `design/multi_channel_motion_bpe_extraction.md`
5. `design/text_bpe_wordnet_naming_layer.md`
6. `experiments/evaluation_experiments.md`
7. `experiments/hml3d_benchmark_plan.md`
8. `logs/experiment_registry.md`
9. `paper/paper_workspace.md`
