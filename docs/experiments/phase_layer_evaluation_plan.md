# Phase-Layer Evaluation Plan

## Goal

Evaluate whether the newly introduced phase-pattern layer is a meaningful motion-semantic layer, rather than just another visualization artifact.

## Main Question

Does the hierarchy

```text
Layer 0 observables -> Layer 1 micro-events -> Layer 2 sub-motion units -> Layer 2.5 phase patterns
```

provide a better semantic representation of motion than flat auto-prompts alone?

## Evaluation Principle

The phase layer should be evaluated first as a representation layer, not immediately as a text prompt.

So the next evaluation stage should focus on:

1. whether phase patterns compress motion semantics better
2. whether phase patterns separate confusing actions better
3. whether phase patterns improve interpretability
4. only then whether their verbalization improves MoMask probing

## Phase-Layer Evaluation Axes

### A. Compression / abstraction

For each case, compare:

- number of micro-events
- number of sub-motion units
- number of phase patterns

Goal:
- see whether the hierarchy produces a more compact but still interpretable representation

### B. Necessity by action type

Identify which motion classes truly need the phase layer.

Expected candidates:
- repeated bounce
- repeated squat
- repeated hop
- walk-stop-walk
- repeated arm up/down
- elbow flap

Question:
- can these be adequately described without phase patterns?
- if not, the phase layer is justified

### C. Confusion reduction

Compare before/after phase abstraction on ambiguous pairs:

- bounce vs crouch_repeated
- hop vs jump_up
- stairs vs crouch
- raise vs elbow flap

Goal:
- show whether phase patterns reduce ambiguity better than flat event lists

### D. Human interpretability

For a representative subset of cases, show side-by-side:

- selected HML3D prompt
- micro-event sequence
- sub-motion sequence
- phase-pattern sequence

Question:
- which layer is closest to the actual motion semantics?

### E. Optional generation probe

As a supporting experiment, not the main one:

- selected HML3D prompt -> MoMask
- auto prompt -> MoMask
- phase verbalization -> MoMask

Goal:
- test whether phase verbalization is a stronger semantic probe than flat prompt text

## Immediate Recommended Metrics

### 1. Per-case count summary

- `num_micro_events`
- `num_submotions`
- `num_phase_patterns`
- `submotion_per_micro_ratio`
- `phase_per_submotion_ratio`

### 2. Category-specific phase coverage

For target categories such as:
- bounce
- hop
- crouch
- stop
- arm up/down

report whether a phase-level motif is detected.

### 3. Representative qualitative tables

Suggested columns:
- case id
- selected HML3D prompt
- auto prompt
- top micro-events
- top sub-motions
- detected phase patterns
- notes

## Execution Order

1. compute hierarchy counts on the current representative set
2. identify categories that depend on phase patterns
3. build a qualitative table for representative cases
4. if phase layer looks useful, run a small MoMask verbalization probe

## Current Hypothesis

The phase layer is especially important for motions that are:
- repeated
- cyclic
- alternating
- or composed of repeated local motifs

For one-shot actions, micro-events or sub-motions may already be enough.
