# Design Brief

This document is the compact working note for the project. Update this file as the data representation, network, and loss design evolve.

## Project intent

The project is moving from plain motion editing toward embodied action understanding.

Current core target phrase:

- **Language-Guided Action Regulation**

Interpretation:

- the repository name `PseudoEdit3D` stays unchanged for engineering stability
- the research target is no longer framed as third-person motion editing
- the main goal is to let an embodied agent execute an action once, receive language feedback, and produce a corrected second attempt
- future visual and scene feedback extensions should still be interpreted under the same action-regulation and revision framing

Long-term view:

- learn a body-centric action representation from unlabeled motion
- map text or video into an internal body program
- realize or revise motion through structured state change, not only trajectory regression
- eventually support memory, refinement, and multimodal skill acquisition

## Current data source

- dataset root: `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset`
- current working subset for experiments: `HumanML3D-CMU`
- base clip format:
  - `poses`: `(60, 156)` SMPL-H axis-angle
  - `trans`: `(60, 3)`
- optional contact masks at vertex level

## Current data management

The current Stage 1 engineering path uses:

- one fixed 60-frame clip as the base unit
- same-clip prefix conditioning
- on-the-fly `EditProgram` extraction inside the dataset loader

Important:

- there is **no pre-saved per-clip program file** yet
- the current `EditProgram`, masks, and prompt are generated at `__getitem__` time
- this keeps iteration fast, but also means the current supervision logic is defined by code rather than by a frozen annotation artifact

Current loader entry:

- `pseudoedit3d/data/prefix_dataset.py`

Current training entry:

- `pseudoedit3d/training/train_stage1.py`

## Current Stage 1 sample construction

Current main setting:

- `prefix-conditioned action completion`

One training sample is built from **one 60-frame clip**:

- input:
  - first 20 frames as prefix motion context
  - `EditProgram` as structured condition
- target:
  - full 60-frame motion

Current source construction for this setting:

- `source_pose`
  - frames `0:19` = real clip prefix
  - frames `20:59` = zero-masked
- `source_trans`
  - frames `0:19` = real prefix translation
  - frames `20:59` = zero-masked
- `conditioning_frame_mask`
  - first 20 frames = `1`
  - remaining 40 frames = `0`

Current target construction:

- `target_pose = poses.copy()`
- `target_trans = trans.copy()`

So:

- source = masked prefix view of the same clip
- target = original full clip

## How the current EditProgram is extracted

Current path for the main prefix-completion experiments:

- task mode: `atomic_realize`
- file: `pseudoedit3d/data/prefix_dataset.py`
- function: `_build_atomic_program(...)`

Current extraction logic:

1. compute proxy attributes from the clip
2. use frame `19` as the prefix anchor state
3. inspect future attribute trajectories from frame `20` onward
4. choose the attribute with the **largest absolute change** relative to the prefix anchor
5. convert that chosen proxy attribute into:
   - `part`
   - `attribute`
   - `direction`
6. derive:
   - `delta_value_deg`
   - `delta_bin`
   - `start_frame`
   - `end_frame`

Current candidate proxy attributes:

- `left_shoulder_pitch_proxy_deg`
- `right_shoulder_pitch_proxy_deg`
- `both_shoulder_pitch_proxy_deg`
- `left_elbow_flex_proxy_deg`
- `right_elbow_flex_proxy_deg`
- `both_elbow_flex_proxy_deg`
- `torso_pitch_proxy_deg`
- `torso_roll_proxy_deg`

Current start/end extraction details:

- `start_frame`
  - first future frame whose attribute change exceeds `0.2 * |delta|`
- `end_frame`
  - based on the detected future peak
- `valid_end_frame`
  - an extra heuristic stop frame inferred from active-joint motion energy

Important limitation:

- this is still a **single best-attribute heuristic**
- it is not a clean action-boundary parser
- it can still pick a local attribute change inside a clip that contains more than one motion phase

## Current EditProgram structure

Current Python structure:

- file: `pseudoedit3d/edit/schema.py`
- class: `EditProgram`

Main fields:

- `part`
- `attribute`
- `delta_bin`
- `start_frame`
- `end_frame`
- `contact_policy`
- `attribute_key`
- `direction`
- `delta_value_deg`
- `source_type`
- `schema_version`
- `input_mode`
- `operator`
- `reference`
- `preserve_parts`
- `preserve_mode`
- `skill_label`
- `skill_phase`
- `tolerance_deg`
- `constraints`
- `metadata`

## Current EditProgram vector encoding

Current model-side condition is a fixed-length vector:

- file: `pseudoedit3d/edit/schema.py`
- method: `LabelSchema.encode_program(...)`

Current dimension:

- `part`: 4
- `attribute`: 8
- `delta_bin`: 3
- `contact_policy`: 2
- `operator`: 2
- `reference`: 2
- `preserve_mode`: 3
- `skill_label`: 6
- normalized `start_frame`, `end_frame`: 2

Total:

- **32-d `edit_vector`**

Important:

- most continuous motion semantics are **not** directly encoded into this 32-d condition
- fields like:
  - `delta_value_deg`
  - `valid_end_frame`
  - `skill_phase`
  - other metadata
  are currently used more on the supervision / visualization side than in the main model condition

## What exactly gets loaded during training

Each current Stage 1 prefix sample returns:

- `source_pose`
- `target_pose`
- `source_trans`
- `target_trans`
- `joint_mask`
- `time_mask`
- `conditioning_frame_mask`
- `edit_vector`
- `prompt_token_ids`
- `prompt_attention_mask`
- `prompt_text`
- `program_json`
- `source_path`
- `betas`
- goal-spec tensors

Current model actually uses:

- `source_pose`
- `edit_vector`
- `conditioning_frame_mask`

Text tokens may be present in the batch, but the current default baseline is:

- `condition_mode: program`

so the active condition is the structured program vector, not text.

## Current part mask behavior

Current part supervision is defined by:

- `BODY_PART_TO_JOINTS`
- `joint_mask`

Current `joint_mask` logic for `atomic_realize`:

- pick active joint ids from `program.part`
- set mask `= 1` only for:
  - those active joints
  - between `program.start_frame` and `valid_end_frame`

So:

- the mask is **part-local**
- and **time-local**

Current `time_mask` is the same time span collapsed to a frame-only mask.

## Does one clip produce multiple part-mask / program combinations?

**Currently: no.**

Current behavior is:

- one 60-frame clip
- one sampled `EditProgram`
- one `joint_mask`
- one training sample

So a clip is **not yet expanded** into:

- left-arm version
- right-arm version
- both-arms version
- torso version

all at once.

That is an important current limitation.

It means:

- current supervision coverage per clip is sparse
- and the dataset does not yet explicitly enumerate multiple local action views from the same base clip

## Current anti-mixing status

We are trying to reduce mixed-motion supervision by adding:

- `valid_end_frame`

but this is **not fully solved yet**.

Current reality:

- fixed 60-frame clips can still contain:
  - one motion phase
  - then recovery
  - then another phase
- the current `valid_end_frame` heuristic is only a first pass
- it does not yet guarantee “one action only”

So if a held-out visualization looks like:

- one action followed by another fragment

that is currently more likely due to:

- base clip segmentation
- and weak action-boundary inference

than to source/target file mismatch.

## Task settings

The project now distinguishes three related but different tasks.

### A. Canonical-pose to motion

Form:

- `canonical pose + detailed body program -> target motion`

Use:

- learn atomic body programs
- study action composition from a stable body reference
- future skill-library learning

### B. Start-pose to motion

Form:

- `target motion first frame + absolute action description -> target motion`

Use:

- verify that the model can realize a motion from a natural initial body state
- keep source and target aligned

Important:

- prompts in this setting should not imply ongoing skill continuation

### C. Attempt-motion to revised motion

Form:

- `first-attempt motion + corrective language feedback -> revised motion`

Use:

- best match to the robot-does-it-once, then gets correction, then retries setting
- supports instructions such as:
  - "raise the arm a bit higher than before"
  - "keep the overall action but reduce arm swing"
  - "do the second attempt with less torso leaning"

Current priority:

- this is now the highest-priority data setting for the project
- source and target should describe two executions of the same intended task whenever possible
- the first attempt should provide error context and skill realization better than a single start pose
- same-clip prefix conditioning can still be used as an engineering scaffold, but it is not the main scientific claim

Scientific question for the current direction:

- Can a feedback-conditioned motion revision model learn atomic body-action factors from unlabeled 3D motion, using a first attempt plus corrective language to produce a better second attempt, while letting whole-body coordination and compensatory reactions emerge implicitly?

Working intuition:

- prompt/program should only specify the active intended correction relative to the previous attempt
- the first attempt should supply the skill prior, intent context, and visible error pattern
- support-region behavior such as balance, compensation, and coordination should be learned rather than manually scripted

## Jsonl artifacts

Managed under:

- `artifacts/jsonl/<subset>/<stage>/<purpose>_<split>.jsonl`

Examples:

- `artifacts/jsonl/HumanML3D-CMU/scan/manifest_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/attributes/attribute_cache_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/mining/mined_pairs_full.jsonl`

Registry:

- `artifacts/jsonl/registry.jsonl`

## Representation stack

### 1. Proxy attributes

Current proxy attributes are upper-body focused:

- `left_shoulder_pitch_proxy_deg`
- `right_shoulder_pitch_proxy_deg`
- `both_shoulder_pitch_proxy_deg`
- `left_elbow_flex_proxy_deg`
- `right_elbow_flex_proxy_deg`
- `both_elbow_flex_proxy_deg`
- `torso_pitch_proxy_deg`
- `torso_roll_proxy_deg`

These are still kinematic proxies, not full FK/body-space semantics.

### 2. Action program

Main structure: `EditProgram`

Core fields:

- `part`
- `attribute`
- `delta_bin`
- `start_frame`, `end_frame`
- `operator`
- `reference`
- `contact_policy`
- `preserve_mode`
- `skill_label`
- `skill_phase`
- `tolerance_deg`
- `metadata`

Current intent:

- `operator=set` means move toward a target value
- `operator=add` means apply a relative correction with respect to the previous attempt
- `reference=source_attempt` is the main path toward feedback-conditioned revision modeling

### 3. Skill context

Current lightweight skill labels:

- `static_pose`
- `locomotion`
- `periodic_arm_motion`
- `torso_leaning`
- `arm_reaching_or_repositioning`
- `unknown`

Current periodic-arm state approximation:

- dominant periodic limb
- dominant periodic attribute
- mean value as offset proxy
- half range as amplitude proxy
- phase estimated from attribute value + velocity

Current relative-action interpretation for periodic arm motion:

- `operator=add`
- `reference=source_attempt`
- `relative_skill_parameter=offset_deg`
- preserve amplitude when possible

This means an instruction like "raise it a bit more than before while keeping the same motion" is currently approximated as:

- shift the periodic offset relative to the first attempt
- keep the periodic amplitude
- keep the rest of the motion structure as much as possible

This is only a first approximation for feedback-conditioned relative revision.

Current implementation direction:

- `target_start_pose` is used for the start-pose task and first-attempt baselines
- `source_attempt_motion` is the next main setting for feedback-conditioned revision

### 4. Goal spec

Programs are converted into compact training-side goal fields:

- target attribute index
- operator index
- reference index
- preserve mode index
- skill label index
- active span
- desired delta
- target absolute value if available
- target offset if periodic skill editing is active
- preserve amplitude flag
- tolerance
- skill phase

## Current model

Main baseline:

- `pseudoedit3d/models/masked_editor.py`

Conditioning modes:

- `program`
- `text`
- `hybrid`

Current architecture role:

- encode source pose/motion
- inject program and/or prompt condition
- predict edited motion residual

Important limitation:

- current model is still an editor/generator, not yet a dedicated revision model with an explicit attempt-level latent state predictor

## Current losses

### Existing motion-space losses

- `edit_loss`
- `keep_loss`
- `smooth_loss`

### Goal-satisfaction losses

- `goal_delta_loss`
- `goal_direction_loss`
- `goal_tolerance_loss`
- `goal_span_consistency_loss`
- `goal_offset_loss`
- `goal_amplitude_preserve_loss`
- `goal_preserve_attr_loss`

Design principle:

- do not require one exact target trajectory
- require the motion to satisfy the intended action goal
- keep non-target content stable when editing an existing motion

## Current design philosophy

The project should gradually move from:

- `state + prompt -> target trajectory`

toward:

- `first-attempt motion + corrective feedback -> revised motion`

and later:

- `first-attempt motion + feedback -> updated skill state -> revised motion`

Potential future extension:

- `current skill state + relative action -> desired state transition`

This is why `operator`, `reference`, `preserve_mode`, `skill_label`, `skill_phase`, and periodic `offset/amplitude` metadata are being added now.

## Current experiment track

Active subset:

- `HumanML3D-CMU`

Baseline completed:

- program-conditioned CMU run without goal loss
- train-split CMU start-pose goal-loss run

Stage-1 code path now available:

- `configs/stage1_prefix_cmu_train_continue.yaml`
  - same-clip prefix continuation baseline
- `configs/stage2_prefix_cmu_train_relative.yaml`
  - same-clip prefix relative-action baseline, currently a proxy for the later full revision task

Next focus:

- build a full-attempt feedback-revision setting as the main motion-conditioned task
- decide how to construct or approximate first-attempt and revised-attempt pairs on CMU clips
- train a first CMU revision-oriented model or a strong proxy baseline
- inspect held-out salient visualizations for iterative debugging

Current refreshed CMU artifacts:

- `artifacts/jsonl/HumanML3D-CMU/scan/manifest_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/attributes/attribute_cache_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/mining/mined_pairs_full.jsonl`

Current goal-loss experiment config:

- `configs/stage1_mined_cmu_program_goal_clean.yaml`

## Open questions

- how to better identify true skill structure from short 60-frame clips
- how to construct reliable `first attempt -> feedback -> revised attempt` supervision from mostly single-clip data
- how to separate `set absolute target` from `relative correction over a previous attempt`
- how much of the first attempt the model should see: full clip, compressed summary, or selected salient subsegment
- how to represent skill memory beyond per-clip metadata
- when to introduce explicit attempt-level latent state modeling instead of only edited trajectory prediction
