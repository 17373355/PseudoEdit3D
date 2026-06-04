# Regression Check Protocol

This document defines the minimum coarse but effective checks that must be run after each meaningful AutoPrompt / auto_program update.

## Purpose

The goal is to avoid the illusion of progress. Each update should be judged by:

1. local correctness on representative cases
2. global regression behavior on the fixed benchmark
3. qualitative motion-level sanity

## Three-Level Check

## Level 1: Local correctness check

After each new rule or pattern update, inspect at least:

- 3 positive cases where the pattern should trigger
- 3 negative cases where it should not trigger

For each case, log:

- selected HML3D prompt
- auto_prompt
- auto_program / program.edits
- whether the trigger is correct

## Level 2: Fixed regression-set check

Run on the current fixed benchmark (currently 600 cases):

- `good`
- `soft_bad`
- `hard_bad`

Also record:

- top missing categories in `soft_bad`
- top missing categories in `hard_bad`

The purpose is to ensure that a local fix does not globally regress the benchmark.

## Level 3: Qualitative visualization check

For each regression cycle, generate a small representative visualization set:

- 5-10 good cases
- 5-10 hard-bad cases
- 5-10 soft-bad cases if needed

The purpose is to catch:

- semantically wrong prompts that look plausible numerically
- prompts that are still too coarse
- generated motions that do not match the intended semantics

## Minimum Pass Condition

A patch is considered acceptable only if:

- the intended representative cases are improved
- the fixed regression benchmark does not get worse globally
- the qualitative motion results do not become obviously less plausible

## Current Fixed Benchmark

- Batch 1: 100 cases
- Batch 2: 500 disjoint cases
- Combined active regression set: 600 cases

## Current Priority Order

1. hard-bad `crouch_bend`
2. `walk_backward`
3. `turn`
4. `bounce_repeated / hop_repeated / crouch_repeated`
5. limb-level repeated motions
6. hand support / rail support
7. number / angle / count awareness
