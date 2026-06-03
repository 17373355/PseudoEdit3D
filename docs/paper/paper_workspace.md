# Paper Workspace

## Proposed Paper Logic

### Part A. Problem
HumanML3D captions are noisy, inconsistent, and too coarse for atomic conditioning.

### Part B. Method
Extract a motion-derived structured annotation layer:
- more atomic
- more consistent
- more accurate
- more suitable for structured conditioning

### Part C. Evaluation
1. annotation-layer evaluation on HumanML3D
2. generation-probe evaluation with MoMask
3. later structured-condition model evaluation

## Current Main Claims

- AutoPrompt is better viewed as a motion-derived annotation layer than a plain text rewrite.
- The annotation layer can be more consistent than raw HumanML3D captions.
- The annotation layer can transfer across datasets more naturally than benchmark-specific human captions.

## Current Open Risks

- overclaiming semantic correctness from FID alone
- mixing annotation experiments with final model experiments
- letting MoMask text conditioning appear as the final method instead of the probe baseline

## Legacy Proposal Files

- `../paper_v1_proposal.md`
- `../paper_v2_proposal.md`
- `../paper_v3_proposal.md`


## Current Proposal

- `paper_current_proposal.md`
