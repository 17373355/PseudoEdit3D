# HumanML3D Pattern Iteration Plan

## Goal
Use HumanML3D multi-caption annotations to iteratively improve auto-prompt extraction with one 100-case seed batch followed by larger 500-case generalization batches.

## Loop
1. Build a 100-case seed manifest from HumanML3D caption priors.
2. Run `run_momask_case_study.py` on the seed batch.
3. Compare `selected_hml3d_prompt` vs `auto_prompt` semantics.
4. Split results into `good_cases` and `bad_cases`.
5. Patch extraction rules or caption-prior patterns using seed-batch bad cases.
6. Freeze the updated rules and test on the next 500-case batch.
7. Repeat the patch-and-freeze process on the next two 500-case batches.
8. After 1x100 + 3x500 are finished, merge all accumulated bad cases and rerun them as a global regression set.
9. Use any newly exposed bad cases to seed the next iteration cycle.

## Current priority patterns
- `bounce_up_down`
- `stair_descent`
- `stair_ascent`
- `walk_backward`
- `stop_pause`
- `turn`
- `crouch_bend`

## Current good/bad criteria
- `stairs down` captions should produce `stair_descent`
- `stairs up` captions should produce `stair_ascent`
- `up and down` / `bounce` captions should produce repeated-bounce semantics, not single jump
- `walk back/backward` captions should produce `walk_backward`
- `turn` captions should keep `turn_left/right`
- `stop` captions should produce `stop_pause`

## Artifacts
- Batch manifest: `outputs/hml3d_pattern_batches/batch_XXXX_manifest.jsonl`
- Batch summary: `outputs/hml3d_pattern_batches/batch_XXXX_manifest_summary.json`
- Good cases: `outputs/hml3d_pattern_batches/batch_XXXX_good.jsonl`
- Bad cases: `outputs/hml3d_pattern_batches/batch_XXXX_bad.jsonl`
- Aggregate resolved cache: `outputs/hml3d_pattern_batches/resolved_case_ids.txt`
- Aggregate bad cache: `outputs/hml3d_pattern_batches/bad_case_ids.txt`

## Batch Generalization Policy
- Batch 1: iterate on the first 100 cases until the biggest bad-case modes are reduced.
- Batch 2: freeze the updated patterns from Batch 1 and test on a disjoint 500-case batch.
- Batch 3: freeze the updated patterns from Batch 2 and test on another disjoint 500-case batch.
- Batch 4: freeze the updated patterns from Batch 3 and test on a third disjoint 500-case batch.
- After these 4 batches, merge all accumulated bad cases and rerun them with the latest patterns as a global regression pass.
- Any newly exposed bad cases become the seed set for the next 8-10 iteration cycle.

## Cache Policy
- `resolved_case_ids.txt`: cases judged good enough and excluded from later discovery batches
- `bad_case_ids.txt`: accumulated hard cases for regression testing after each major pattern update
- `batch_XXXX_bad.jsonl`: per-batch failure snapshot
- `batch_XXXX_good.jsonl`: per-batch solved snapshot


## Output Size Monitoring
- After each batch, record the disk usage of the batch output directory.
- Keep track of: case-study summary, generated MoMask directories, visualization GIF directories, and mined reports.
- If a batch output grows unexpectedly, inspect whether redundant videos or summaries are being duplicated.
