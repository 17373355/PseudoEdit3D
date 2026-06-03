# ICLR 4-Month Goal Plan

## Current Position

The current project is no longer framed as plain text-to-motion prompting.
The active research target is:

> build a motion-derived, more atomic, more accurate, and more transferable semantic annotation layer than raw HumanML3D captions, and use it as the supervision/conditioning interface for a future structured motion model.

Current facts:

- HumanML3D raw captions are treated as noisy priors, not strict ground truth.
- `auto_program` is the primary object.
- `auto_prompt` is only a rendered textual probe.
- MoMask is currently used as a motion prior / semantic probe, not the final model.
- A 600-case regression set is already fixed:
  - Batch 1: 100
  - Batch 2: 500 disjoint

## 4-Month Top-Level Objective

## Explicit Deadlines

- **By 2026-06-10**: Phase 1 must produce a paper-writable abstract mechanism for AutoPrompt refinement, not just a collection of heuristic patches.
- **From 2026-09-10**: start formal paper writing.

This means the period from **2026-06-10 to 2026-09-10** should mainly be used for:
- stabilizing the 600-case benchmark
- running benchmark and transfer experiments
- building the structured-condition prototype
- preparing figures, tables, and artifact summaries


In about 4 months, the project should reach a paper-ready ICLR-level form with:

1. a clear problem statement
2. a stable motion-derived annotation layer
3. convincing quantitative and qualitative evidence that the new annotation layer is better than raw HumanML3D captions
4. at least one structured-condition model or strong structured-condition prototype
5. a complete paper narrative, experiment suite, and release-ready assets

---

## Paper-Level Claim We Want To Reach

### Main claim

We refine noisy motion-language supervision into a structured, atomic motion semantic layer that is more consistent, more motion-faithful, and more transferable than raw human captions.

### Secondary claim

This refined semantic layer improves downstream motion generation/conditioning under the same backbone, and forms a better interface for structured motion control than free-form benchmark captions.

---

## Success Criteria For The Whole 4-Month Window

By the end of the 4-month period, we want all of the following to be true:

### A. Annotation layer

- the current 600-case regression set is mostly stabilized
- false positives like false crouch / false stairs / false bounce are substantially reduced
- number / angle / direction / part detail are explicitly represented in `auto_program`
- `auto_prompt` is derived from motion only
- `auto_prompt` no longer copies same-case HumanML3D text

### B. Benchmark / data product

- an `AutoPrompt-HumanML3D` benchmark exists
- its extraction pipeline is reproducible
- representative good / soft-bad / hard-bad splits are saved
- benchmark statistics and consistency analyses are saved

### C. Experimental proof

- same-architecture comparison on HumanML3D shows improved learning signal under AutoPrompt supervision
- cross-dataset experiment on unlabeled dataset B shows usable transfer
- representative qualitative comparisons are organized and publishable

### D. Model direction

- at least one structured-condition model or structured-condition prototype exists
- this model does not depend on case-specific caption retrieval
- condition format explicitly supports:
  - direction awareness
  - angle awareness
  - number awareness
  - part-level control

### E. Paper package

- problem statement, method, benchmark, experiments, and limitations are written coherently
- figures/tables are mostly prepared
- code and data artifacts are organized enough for release planning

---

## Phase Plan

## Phase 1 (Until 2026-06-10)
### Theme: Turn the current heuristic AutoPrompt pipeline into a paper-writable semantic refinement mechanism

### Core goals

- finish the motion-only `auto_prompt` correctness pass on the current 600 cases
- reduce the biggest hard-bad categories
- freeze a stable triage set: `good / soft_bad / hard_bad`
- elevate the current engineering process into an abstract mechanism, such as:
  - iterative refinement framework
  - confidence-based label denoising
  - consistency-driven relabeling
  - motion-text agreement modeling
  - structured latent induction from motion

### Priority error classes

1. false crouch / false squat
2. walk_backward missing
3. stop_pause edge cases
4. repeated hop / bounce / crouch separation
5. stairs up/down vs crouch / bounce
6. torso bend vs bounce
7. arm up/down and elbow-flap primitives

### Deliverables

- updated `triage_600/summary.json`
- updated bad-category rankings
- representative visualization bundles
- stable `auto_program` schema draft

### Gate to exit Phase 1

- major hard-bad categories reduced enough that new failures are mostly finer-grained rather than gross semantic mismatches
- the method can be written as a coherent mechanism rather than a list of isolated prompt fixes
- `auto_program` is clearly defined as the primary structured output, with `auto_prompt` only as a rendered probe layer

---

## Phase 2 (Weeks 5-8)
### Theme: Build the annotation-layer paper core

### Core goals

- turn the current rule/refinement process into a clearer method section
- add annotation-layer metrics beyond anecdotal case studies
- define `AutoPrompt-HumanML3D` formally as a benchmark/data product

### Experiments to finish

#### Annotation-layer consistency analysis

Compare raw HumanML3D vs AutoPrompt-HumanML3D:

- caption conflict rate
- semantic consistency across similar motions
- normalized vocabulary / expression entropy
- atomic slot coverage
- direction / angle / count field coverage

#### Regression diagnostics

- hard-bad reduction curve over iterations
- soft-bad reduction curve over iterations
- per-pattern accuracy statistics

### Deliverables

- benchmark definition doc
- annotation-layer consistency results
- reusable evaluation scripts
- updated paper figures for benchmark analysis

### Gate to exit Phase 2

- the annotation-layer paper story is already independently meaningful even before the final structured model is perfect

---

## Phase 3 (Between 2026-06-10 and 2026-09-10)
### Theme: Train benchmark-level baselines and cross-dataset validation

### Core goals

- use the refined annotation layer to retrain text-conditioned baselines under the same backbone
- validate transferability on dataset B

### Main experiments

#### Experiment 1

- original HumanML3D captions + MoMask text-conditioned model trained on original captions
- AutoPrompt-HumanML3D + same MoMask text-conditioned model retrained on AutoPrompt-HumanML3D
- compare FID and semantic consistency results

#### Experiment 2

- unlabeled dataset B -> AutoPrompt
- AutoPrompt-conditioned model trained on AutoPrompt-HumanML3D
- compare generation to GT_B

#### Recommended supporting analyses

- zero-shot baseline comparison
- round-trip consistency
- representative good / bad qualitative cases

### Deliverables

- benchmark baseline training results
- dataset-B transfer results
- tables and figures for the paper

### Gate to exit Phase 3

- enough evidence exists that AutoPrompt is not just cleaner wording, but a better supervision interface

---

## Phase 4 (Starting from 2026-09-10)
### Theme: Structured-condition prototype + paper closure

### Core goals

- build at least a minimal structured-condition model/prototype
- connect the annotation layer to the future model story
- finish paper packaging

### Minimum acceptable structured-condition prototype

Reuse MoMask tokenizer/detokenizer if needed, but replace free-form text dependency with a structured interface or semi-structured encoder that consumes:

- event type
- part
- direction
- magnitude
- count
- angle
- time span
- confidence

### Final writing tasks

- final title / abstract / intro / related work alignment
- experiment section polishing
- discussion of limitations and scope
- appendix / artifact summary

### Deliverables

- submission-ready draft
- figure package
- benchmark + code artifact plan

---

## What We Should Explicitly Avoid

- treating HumanML3D captions as absolute ground truth
- mixing benchmark-cleanup contributions with final structured-model claims in one vague story
- overclaiming semantic correctness from FID alone
- expanding to many more datasets before the current 600-case benchmark is stable
- relying on same-case text retrieval to make `auto_prompt` look better

---

## Immediate Next Tasks

1. continue reducing hard-bad categories on the current 600 cases
2. formalize `auto_program` as the primary structured output
3. finish annotation-layer consistency metrics
4. cleanly separate:
   - annotation-layer experiments
   - MoMask semantic-probe experiments
   - final structured-model experiments

---

## Expected Paper Structure (Working)

1. Introduction
2. Related Work
3. Problem: noisy motion-language supervision
4. Method: motion-derived atomic annotation refinement
5. AutoPrompt-HumanML3D benchmark
6. HumanML3D benchmark experiments
7. Cross-dataset transfer experiments
8. Structured-condition prototype
9. Limitations and future work

---

## Working Summary

The next 4 months should not be spent on random expansion.
They should be spent on turning the current iterative engineering line into:

- a benchmark
- a representation-learning story
- a reproducible evaluation suite
- and a structured-condition bridge to the next model stage.
