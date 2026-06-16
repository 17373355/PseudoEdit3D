# Design Overview

## Main Framing

The current project is not only a text-to-motion task.
It is a motion-derived semantic annotation project whose long-term target is a structured condition model.

## Core Principle

- motion first
- structure first
- text second

This means:
- `auto_program` is the primary output
- `auto_prompt` is a rendered view for probing and visualization
- HumanML3D captions are noisy semantic priors only
- motion clusters and motion-BPE motifs should induce the pattern tree
- HumanML3D text-BPE and WordNet should name motion-derived nodes, not create them

## Current Mainline

The current design direction is documented in:

- `motion_corpus_pattern_tree_mainline.md`
- `motion_cluster_bpe_tree_induction.md`
- `multi_channel_motion_bpe_extraction.md`
- `text_bpe_wordnet_naming_layer.md`

The previous hand-built AML tree is now a bootstrap/reference/evaluation
artifact. It should not be expanded by case-specific action-name logic unless
the same pattern is supported by corpus-level motion evidence.

## Current Structured Target

Each event should ideally include:
- `type`
- `part`
- `direction`
- `magnitude`
- `unit`
- `count`
- `start_frame`
- `end_frame`
- `confidence`

## Current Main Issues

- false squat / false crouch
- missing backward-walk semantics
- weak stop detection in some edge cases
- missing limb-level repeated patterns
- missing number / angle awareness
- too-coarse whole-body descriptions
- single-sequence Motion-BPE flattens concurrent upper-body / lower-body /
  root events into accidental adjacency

## Current Entry Point

- `motion_corpus_pattern_tree_mainline.md`
- `../design_brief.md`

## Legacy References

- `../hml3d_experiment_pipeline.md`
- `../stage1_upgrade_plan.md`
