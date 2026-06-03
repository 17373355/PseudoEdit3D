# Planning

This file is the **forward plan**, not the project definition and not the experiment history.

Use it to answer only:

- what is the current next-step question?
- what will be tried next?
- what decision will each experiment support?

Do **not** use this file to duplicate:

- core concepts from `docs/design_brief.md`
- historical runs from `docs/experiment_log.md`
- local commands from `Test.md`

## Role of each doc

- `docs/design_brief.md`
  - stable project framing
  - task setting
  - representation / model / loss concepts

- `docs/experiment_log.md`
  - what was actually run
  - configs, key losses, output dirs

- `docs/planning.md`
  - next experiments only
  - open decisions only
  - short and disposable

- `docs/memory.md`
  - compact repo working memory
  - how to quickly re-enter the project

## Current planning focus

### Question 1

Can `prefix-conditioned action completion` be trained as a real completion model, rather than collapsing into a static future pose?

### Question 2

Is the current failure mainly caused by:

- loss design
- condition injection
- target construction
- or action-boundary / prompt-target mismatch

## Current active task definition

- input:
  - first 20 frames as motion prefix
  - `EditProgram` as structured condition
- output:
  - complete 60-frame motion
- intended constraint:
  - prefix provides motion context
  - model should complete future frames under the program

## Immediate next experiments

1. prefix locked + future-only regression
   - purpose:
     - verify whether the model can first learn plain future completion
   - decision:
     - if this still freezes, the issue is likely structural rather than goal-loss-specific

2. prefix locked + future regression + active-region extra weight
   - purpose:
     - keep whole-future supervision while preserving program sensitivity
   - decision:
     - if this is better than future-only, local program signal is useful

3. inspect condition injection
   - purpose:
     - determine whether `EditProgram -> edit_proj -> broadcast` is too weak
   - decision:
     - if future-only works but program-guided variants fail, condition injection becomes the next bottleneck

## Current stop conditions

Pause new semantic or goal-loss experiments until:

- completion no longer collapses into a static pose
- held-out visualization no longer looks like mixed future fragments
- prefix panel / predicted panel semantics are confirmed correct

## Update rule

When the next-step question changes, rewrite this file.
When an experiment finishes, record it in `docs/experiment_log.md` instead of here.
