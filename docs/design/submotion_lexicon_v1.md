# Sub-Motion Lexicon V1

This document records the first manually curated merge baseline built on top of Layer 1 micro-events.

## Purpose

The goal of V1 is not to finalize the motion vocabulary, but to test whether the most frequent local event pairs/triples can already be grouped into interpretable sub-motion units.

## Current Sources

The lexicon is derived from:

- `outputs/submotion_pairs_batch1_v1.json`
- representative 600-case benchmark inspection
- current hard-bad / soft-bad error clusters

## Whole-Body Candidates

- `crouch_descent`
- `crouch_descent_strong`
- `hop_ascent`
- `hop_ascent_variant`
- `leg_bounce_cycle`
- `hop_unit`

## Torso Candidates

- `torso_rise_back`

## Arm Candidates

- `left_arm_lowering`
- `right_arm_lowering`
- `both_arms_lift`
- `hands_move_away_from_chest`
- `arm_lift_front`

## Interpretation

These units should be understood as a first BPE-style merge baseline over micro-event symbols.
They are not yet the final AML vocabulary.

## Next Step

Use these merge rules to:

1. inspect whether representative cases become easier to read
2. identify which units are stable enough to keep
3. compare this manual-frequency baseline with later automatic merge strategies
