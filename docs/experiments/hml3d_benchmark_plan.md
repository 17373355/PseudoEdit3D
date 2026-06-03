# HumanML3D Benchmark Plan

## Current Fixed Regression Set

The current active regression set is fixed to 600 cases:

- Batch 1: 100 cases
- Batch 2: 500 disjoint cases

No new HumanML3D data should be added before this 600-case set is stabilized.

## Triage

The 600-case set is split into:
- `good`
- `soft_bad`
- `hard_bad`

## Current Counts

See:
- `../..//outputs/hml3d_pattern_batches/triage_600/summary.json`
- `../..//outputs/hml3d_pattern_batches/triage_600/missing_category_stats.json`

## Iteration Policy

1. fix `hard_bad`
2. regress on all 600 cases
3. reduce `soft_bad`
4. only after the 600-case set is satisfactory, expand to more HumanML3D data

## Current Priority Order

1. `crouch_bend`
2. `walk_backward`
3. `turn`
4. `bounce_repeated / hop_repeated / crouch_repeated`
5. limb-level repeated patterns
6. number / angle / step count
