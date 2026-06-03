# HumanML3D Auto-Prompt Experiment Pipeline

## Goal

Build a motion-derived annotation layer that is:

- more atomic than raw HumanML3D captions
- more accurate than raw HumanML3D captions
- suitable for training a future structured-condition model

The final target is not to imitate HumanML3D text, but to produce a better intermediate supervision layer from motion itself.


## Core Hypothesis

HumanML3D captions are useful as a noisy semantic prior, but they are:

- inconsistent across captions of the same motion
- often too coarse
- sometimes partially wrong
- not structured enough for atomic conditioning

So the working strategy is:

1. use HumanML3D captions to discover language patterns
2. use motion to verify and refine those patterns
3. iteratively produce a cleaner auto-prompt / atomic program layer
4. later train a structured-condition model on top of the cleaned layer


## Current Experiment Design

### Stage A. Pattern Mining From HumanML3D

Input:

- full HumanML3D motion
- all captions attached to the same case

Output:

- caption prior categories
- motion-derived atomic edits
- auto-prompt

Current caption prior categories include:

- `stair_descent`
- `stair_ascent`
- `walk_forward`
- `walk_backward`
- `turn`
- `jump_up`
- `crouch_bend`
- `arm_support`
- `stop_pause`


### Stage B. Motion-Only Auto-Prompt Extraction

Important constraint:

- `auto_prompt` must be inferred from motion only
- same-case HumanML3D text is not allowed to generate the final `auto_prompt`
- HumanML3D text is only used offline to build pattern inventory and for later comparison

Motion signals currently used:

- root heading change
- root forward/backward progress
- root step speed
- root height change
- repeated height peaks / valleys
- pelvis-to-ankle compression
- bilateral arm raise proxy


### Stage C. MoMask-Based Semantic Probe

We use MoMask only as a motion tokenizer / decoder probe.

We compare:

- `selected_hml3d_prompt -> MoMask generation`
- `auto_prompt -> MoMask generation`

This stage is used only to test whether the extracted semantics are meaningful enough to drive motion generation.

It is not the final structured-condition model.


## Current Pipeline

```text
HumanML3D motion
    + all captions
        ->
caption prior inventory
        +
motion kinematic analysis
        ->
multi-atomic program
        ->
motion-only auto-prompt
        ->
MoMask generation probe
        ->
qualitative / bad-case analysis
        ->
pattern update
```


## Batch Iteration Strategy

Current large-scale strategy:

```text
Batch 1: 100 cases
    ->
diagnose good / bad
    ->
update rules

Batch 2: 500 disjoint cases
    ->
diagnose good / bad
    ->
update rules

Batch 3: 500 disjoint cases
    ->
diagnose good / bad
    ->
update rules

Batch 4: 500 disjoint cases
    ->
diagnose good / bad
    ->
update rules

Merge all bad cases
    ->
global regression pass
```

Large batches currently run in fast mode:

- prompt/program extraction only
- no full MoMask generation for every case
- visualizations only for selected representative good/bad cases


## Good / Bad Definition

A case is considered good when the extracted `auto_prompt` preserves the main semantics required by the case.

Examples:

- stairs down -> should contain `stair_descent`
- stairs up -> should contain `stair_ascent`
- repeated bouncing -> should not collapse into single jump
- repeated squat -> should not collapse into bounce
- walk back -> should preserve backward semantics
- walk then stop -> should preserve stop semantics


## Current Important Corrections

Already fixed:

- repeated bounce no longer always collapses to single jump
- stair up/down no longer always collapses to bounce
- some stop cases now emit `stop_pause`
- repeated squat can be separated from repeated bounce
- crouch is no longer trivially confused with stair descent in key cases

Still under iteration:

- `bend/stoop` vs `bounce`
- `run-hop-land` vs `stairs`
- `walk_backward` variants
- `stop_pause` edge cases
- hand / torso fine-grained semantics


## Why This Matters

The real objective is to produce a new supervision layer with:

- direction awareness
  - front / back / left / right / up / down
- angle awareness
  - turn angle, lift angle, bend angle
- number awareness
  - count, duration, span, height, distance

This cleaned layer will later replace raw HumanML3D text as the conditioning target for a structured atomic model.


## Current Deliverables

Current outputs include:

- batch manifests
- batch summaries
- good / bad splits
- missing-phrase reports
- selected representative visualizations

These are used to iteratively upgrade the auto-prompt layer before retraining a structured-condition model.


## Auto-Prompt To MoMask Generation

This is the current probing path used after we extract a motion-only `auto_prompt`.

### High-Level Flow

```text
HumanML3D motion
    ->
motion analysis
    ->
multi-atomic program
    ->
motion-only auto-prompt
    ->
MoMask text-conditioned generator
    ->
motion tokens
    ->
MoMask decoder
    ->
generated motion
```

### Detailed Network Flow

```text
auto_prompt (natural language)
    ->
CLIP text encoder inside MoMask
    ->
text embedding
    ->
MaskTransformer.generate(...)
    ->
coarse motion token sequence
    ->
Residual Transformer refine(...)
    ->
refined motion token sequence
    ->
RVQ-VAE decoder
    ->
263-d HumanML3D motion feature
    ->
recover_from_ric(...)
    ->
22-joint motion
    ->
video / joints / visualization
```

### Stage Breakdown

#### 1. Auto-prompt creation

Input:

- full HumanML3D motion

Output:

- motion-only `auto_prompt`
- `program.edits`

This stage is produced by `run_momask_case_study.py`.

#### 2. Text encoding

MoMask does not directly read the structured program.
It only receives:

- `auto_prompt`
- motion length

Then:

- `auto_prompt`
  ->
- internal CLIP text encoder
  ->
- text feature

#### 3. Motion token generation

Using the text feature:

- `MaskTransformer.generate(...)`
  predicts a first-stage token sequence

Then:

- `Residual Transformer`
  refines the token sequence

This is still fully text-conditioned generation.

#### 4. Motion decoding

Refined token sequence
  ->
`RVQ-VAE decoder`
  ->
HumanML3D `new_joint_vecs`
  ->
joint recovery
  ->
motion output

### Important Limitation

At this stage, MoMask only consumes:

- natural-language `auto_prompt`

It does **not** consume:

- structured direction slots
- explicit angle slots
- explicit count slots
- explicit frame spans

So the current path is:

```text
structured motion semantics
    ->
collapse to text
    ->
MoMask text-conditioned generation
```

This is why the current MoMask probe is useful for semantic testing, but not the final structured-condition solution.
