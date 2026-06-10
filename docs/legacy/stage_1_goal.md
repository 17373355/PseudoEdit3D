# Stage 1 Goal

## 1. Stage 1 one-line objective

- Train a stable `prefix-conditioned action completion` baseline that uses a 20-frame same-clip prefix plus `EditProgram` to complete the remaining 40 frames without freezing.

## 2. Current task definition

- input:
  - first 20 frames of the same 60-frame clip
  - `EditProgram`
- output:
  - complete 60-frame motion

## 3. What Stage 1 must prove

- prefix is preserved exactly
- future remains dynamic rather than collapsing to a static pose
- `EditProgram`-conditioned completion is at least sensitive to the active future region
- held-out visualizations and frozen-motion diagnostics agree with each other

## 4. What Stage 1 does not try to prove

- free-form natural language understanding
- dialogue-based regulation
- scene / video feedback
- long-horizon editing or memory

## 5. Success criteria

### Visual criteria

- [x] prefix frames remain locked
- [ ] future is not a frozen pose
- [ ] no obvious mixed-action fragment in salient held-out cases
- [ ] GT / pred comparison is interpretable without translation leakage

### Training criteria

- [x] training code runs with the current prefix pipeline
- [ ] fresh checkpoint is trained after the structural fixes
- [ ] held-out export works from the fresh checkpoint

### Experimental criteria

- [x] one clean regression-only diagnosis path exists: `future-only`
- [x] one clean active-region comparison exists: `future+active`
- [ ] at least one fresh run explains whether collapse is structural or supervisory

## 6. Primary experiment loop

| ID | Question | Config / change | Expected signal | Result |
|---|---|---|---|---|
| E0 | Is the current pipeline structurally consistent? | code audit + bug fix | no time-window misalignment; future tokens become time-distinguishable | fixed `atomic_realize.start_frame`; added positional + prefix/future frame-type embeddings |
| E1 | After the structural fixes, does `future-only` still freeze? | `configs/stage1_prefix_completion_cmu_futureonly_structfix.yaml` | if this works, collapse was partly structural | pending |
| E2 | Does active-region extra weight improve over `future-only`? | `configs/stage1_prefix_completion_cmu_futureplusactive_structfix.yaml` | better active future velocity ratio and better held-out focus | pending |
| E3 | If E1 still freezes, is the failure visible in the diagnostics? | `scripts/eval_prefix_completion.py` + held-out GIFs | frozen cases should show low future velocity ratio, not just bad visuals | pending |

## 7. Current default baseline

Leave blank until E1 finishes and is trusted.

## 8. Current most likely failure modes

- future tokens are too homogeneous, so the network learns a low-motion average future
- `EditProgram` still describes only one dominant attribute inside a clip that may contain multiple phases
- active-region supervision is aligned to the chosen attribute, but the clip may contain another comparably strong motion phase

## 9. Next decision points

- if `future-only` still freezes after E0, inspect the sequence model and source representation before adding more semantic loss
- if `future-only` works but `future+active` degrades, inspect `EditProgram` injection and active span quality
- if visuals still look mixed while velocity ratios look healthy, inspect action-boundary construction rather than loss weights
