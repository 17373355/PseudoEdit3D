# Paper V1 Proposal

## Title

PseudoEdit3D: Bootstrapping Fine-Grained Motion Editing from Unlabeled SMPL-H Clips

## Main goal

Show that fine-grained 3D motion editing supervision can be induced from unlabeled SMPL-H motion by combining:
- proxy kinematic attribute extraction
- pseudo edit program induction
- synthetic and mined motion triplets

## Core question

Can a motion editor learn localized, compositional edits without manual text labels if supervision is built from geometry-driven pseudo edit programs?

## Main contribution targets

1. a structured pseudo edit program space for upper-body motion edits
2. a self-bootstrapped training set from unlabeled clips
3. a localized masked motion editor with edit-preservation objectives
4. evaluation centered on edit success, locality, reversibility, and composition

## Scope

Included:
- SMPL-H clips only
- no dialogue requirement
- no full scene modeling
- upper-body first

Deferred:
- language grounding
- object interaction
- hard contact repair

## Deliverables

- reproducible pseudo triplet mining pipeline
- trainable baseline editor
- ablation between synthetic-only and mined-pair training

## Success criteria

- attribute change accuracy is significantly above identity baseline
- non-edited joints remain stable
- composed edits behave more consistently than plain generative baselines
