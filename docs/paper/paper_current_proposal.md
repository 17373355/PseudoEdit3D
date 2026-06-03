# Current Paper Proposal

## Working Title

PseudoEdit3D: Refining Noisy Human Motion Captions into Motion-Derived Atomic Programs for Structured Motion Conditioning

## One-Sentence Positioning

We do not treat HumanML3D captions as strict ground truth; instead, we treat them as noisy semantic priors and iteratively build a cleaner, more atomic, motion-derived annotation layer that can later support structured motion conditioning.

## Core Problem

HumanML3D captions are useful but limited:

- multiple captions for the same motion can conflict
- granularity is inconsistent across samples
- many descriptions are too coarse for structured control
- some descriptions are partially wrong relative to the motion itself

This makes raw HumanML3D text a weak target for atomic conditioning.

## Main Hypothesis

A motion-derived auto-annotation layer can be:

- more atomic
- more consistent
- more motion-faithful
- more suitable for structured conditioning

than the original HumanML3D captions, while still remaining close enough to natural language to be probed through a text-conditioned motion generator.

## Current Method Stage

### Stage 1: Build a Motion-Derived Annotation Layer

Input:
- HumanML3D motion
- all captions attached to the same motion

Output:
- `auto_program` (primary structured output)
- `auto_prompt` (secondary textual rendering)

Key principle:
- `auto_program` is the real output
- `auto_prompt` is only a probe/rendering layer

### Stage 2: Use MoMask as a Semantic Probe

Use MoMask only to test whether the extracted annotation semantics are strong enough to drive generation.

This stage is not the final structured model.

### Stage 3: Train a Future Structured Condition Model

Later, replace text-only conditioning with the structured atomic program itself.

## Current Structured Target

Each event should include:

- `type`
- `part`
- `direction`
- `magnitude`
- `unit`
- `count`
- `start_frame`
- `end_frame`
- `confidence`

The long-term target is a condition space with:

- direction awareness
- angle awareness
- number awareness
- part-level atomic detail

## Current Evaluation Design

### Experiment 1: HumanML3D Re-annotation Benchmark

Compare under the same MoMask architecture:

- original HumanML3D captions + MoMask
- AutoPrompt-HumanML3D + MoMask

Goal:
- test whether the new annotation layer forms a better supervision space than raw HumanML3D text

### Experiment 2: Cross-Dataset Generalization

On unlabeled dataset B:

- motion_B -> AutoPrompt
- AutoPrompt-conditioned generator trained on AutoPrompt-HumanML3D
- generated motion vs GT_B

Goal:
- test whether AutoPrompt behaves like a more transferable motion semantic interface rather than a benchmark-specific text style

### Annotation-Layer Regression

Current active regression set:

- Batch 1: 100 cases
- Batch 2: 500 disjoint cases

The current 600-case set is the main debugging benchmark before expanding further.

## Current Failure Taxonomy

We currently split cases into:

- `good`
- `soft_bad`
- `hard_bad`

Typical remaining hard problems:

- false crouch / false squat
- missing backward-walk semantics
- repeated hop vs repeated squat vs repeated bounce
- weak limb-level repeated patterns
- missing hand-support / rail-support detail

## Main Scientific Claim Direction

The central claim is not merely that a better prompt can be written.
The stronger claim is:

> a motion-derived atomic annotation layer can be iteratively refined from noisy human captions and motion evidence, yielding a cleaner supervision interface for future structured motion generation and editing.

## Non-Claims

This current stage does not claim:

- final structured conditioning has already been solved
- MoMask text conditioning is the final method
- FID alone proves full semantic correctness

## Immediate Next Steps

1. stabilize the current 600-case annotation benchmark
2. reduce hard-bad categories, especially:
   - `crouch_bend`
   - `walk_backward`
   - `turn`
   - `bounce/hop/squat` distinctions
3. improve limb-level atomic patterns
4. introduce stronger number / angle / count fields
5. only then expand to more HumanML3D cases and later train the structured-condition model
