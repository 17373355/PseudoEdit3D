# PseudoEdit3D Planning

## Working title

PseudoEdit3D: Bootstrapping Fine-Grained 3D Human Motion Editing from Unlabeled SMPL-H Motion

## Objective

Build a Paper v1 system around:

`unlabeled motion -> pseudo edit program -> fine-grained motion editing`

The core claim for v1 is not language understanding. The core claim is that fine-grained edit supervision can be self-bootstrapped from unlabeled motion geometry, and that an editor trained on these pseudo triplets learns localized controllable motion transformations.

## Research question

Can we learn a fine-grained motion editor for SMPL-H clips without manual text labels by:

1. deriving structured edit programs from kinematic change patterns,
2. synthesizing source-target edit pairs automatically, and
3. training a localized motion editor that preserves untouched content?

## V1 boundaries

Included:
- SMPL-H motion clips
- pseudo edit program induction
- upper-body centered operators first
- synthetic source-target pair construction
- localized motion editing
- edit/preservation/composition evaluation

Deferred:
- natural dialogue
- free-form language grounding
- scene/object interaction
- hard contact repair with a body model
- object-aware affordance constraints

## Dataset assumption

Primary source for bootstrapping:
- `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset`

Observed format from one sample:
- `poses`: `(60, 156)`
- `trans`: `(60, 3)`
- `betas`: `(1, 16)`
- contact masks may exist at vertex level

This dataset is already segmented into 60-frame clips and grouped by coarse contact type:
- `*_contact_sequences`
- `*_neutral_sequences`
- `*_non_contact_sequences`

That segmentation is enough for a first pass. We should not wait for the full transferred dataset before building the pipeline.

## Pseudo edit program design

First version uses a structured program rather than natural language:

```text
EditProgram = {
  part: one of {left_arm, right_arm, both_arms, torso},
  attribute: one of {raise, lower, bend, extend, lean_left, lean_right, lean_forward, lean_backward},
  delta_bin: discrete magnitude bucket,
  time_mask: frame span or soft mask,
  source_state: optional coarse precondition,
  contact_policy: keep|ignore
}
```

Rationale:
- this is aligned with motion editing rather than captioning
- it can be generated from geometry
- it can later be verbalized into template text
- it is a stable interface for a future dialogue parser

## Pseudo supervision strategy

### Stage A: motion statistics and filtering

- scan all clips and record available fields
- keep 60-frame clips with valid `poses` and `trans`
- bucket clips by sequence family and contact category

### Stage B: attribute extraction

For upper-body first-pass attributes, estimate from joint rotations or proxy heights:
- left/right wrist height relative to pelvis and shoulder
- elbow flexion
- shoulder elevation
- torso lean
- bilateral symmetry/asymmetry

Important note:
- do not equate language labels directly with raw axis-angle channels
- derive interpretable scalar attributes per frame and per clip

### Stage C: pseudo edit mining

Two complementary routes:

1. nearest-neighbor edit triplets
- find clip pairs with similar global context but controlled difference in one attribute
- example: same coarse motion cluster, but right-arm elevation differs by one or two bins

2. operator-based synthetic edits
- apply simple local transformations to the source clip in attribute space
- first pass can be coarse and only used for supervision bootstrapping

The initial scaffold implements the second route in simplified form.

## Model plan

Baseline model family for v1:
- masked motion editor

Why:
- the task is edit-local rather than full generation
- unchanged regions should be preserved explicitly
- later this can absorb better tokenizers or diffusion decoders

First implementation:
- input: source motion features + edit program embedding
- predict: residual motion delta
- preserve untouched frames/joints with a mask-aware objective

Upgrade path:
- part-aware tokenization
- discrete pose codes
- contact refinement head
- edit composition consistency

## Evaluation plan

V1 should not rely only on generative quality metrics.

Primary metrics:
- edit success on target attribute
- locality outside edited part/span
- motion smoothness
- reconstruction on identity edit
- compositional consistency
- reversibility

Future embodied metrics:
- foot skating
- hand/object contact preservation
- penetration

## Immediate implementation plan

1. create dataset manifest tooling
2. implement robust SMPL-H clip loader
3. define pseudo edit schema
4. implement a first synthetic pair generator for upper-body edits
5. add proxy-attribute extraction and real pair mining
6. train a minimal editor baseline
7. inspect saved samples and failure modes before adding language

## Risks

- axis-angle channels are not semantically aligned with natural edit intent
- pure change-point segmentation may over-segment periodic motion
- naive synthetic edits can violate kinematics or balance
- contact labels are vertex-level, so a clean joint/contact abstraction is still needed

## Near-term next steps

- replace naive joint-channel edits with stronger FK-based attributes after the proxy stage
- add sequence-level retrieval to mine real pseudo triplets
- introduce simple template text from edit programs
- add contact-aware penalties for hands and feet
