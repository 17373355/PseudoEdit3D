# Design Brief

This document is the compact working note for the project. Update this file as the data representation, network, and loss design evolve.

## Project intent

The project is moving from plain motion editing toward embodied action understanding.

Current core target phrase:

- **Language-Guided Action Regulation**

Interpretation:

- the repository name `PseudoEdit3D` stays unchanged for engineering stability
- the research target is no longer framed as third-person motion editing
- the main goal is to let an embodied agent regulate its own ongoing actions under language feedback
- future visual and scene feedback extensions should still be interpreted under the same action-regulation framing

Long-term view:

- learn a body-centric action representation from unlabeled motion
- map text or video into an internal body program
- realize or edit motion through state transitions, not only trajectory regression
- eventually support memory, refinement, and multimodal skill acquisition

## Current data source

- dataset root: `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset`
- current working subset for experiments: `HumanML3D-CMU`
- base clip format:
  - `poses`: `(60, 156)` SMPL-H axis-angle
  - `trans`: `(60, 3)`
  - optional contact masks at vertex level

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

### C. Motion-prefix to continuation

Form:

- `same-clip motion prefix + relative action instruction -> edited future continuation`

Use:

- best match to natural-language robot motion adjustment
- supports instructions such as:
  - "raise it a bit more"
  - "keep waving but higher"
  - "continue walking and reduce arm swing"

Current priority:

- this is now the highest-priority data setting for the project
- source and target should come from the same clip whenever possible
- the prefix should encode ongoing motion state better than a single start pose

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
- `operator=add` means apply a relative change
- `reference=current_state` is the main path toward action-conditioned transition modeling

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
- `reference=current_state`
- `relative_skill_parameter=offset_deg`
- preserve amplitude when possible

This means an instruction like "raise it a bit more while continuing the motion" is currently approximated as:

- shift the periodic offset
- keep the periodic amplitude
- keep the rest of the motion structure as much as possible

This is only a first approximation for relative-action editing.

Current implementation direction:

- `target_start_pose` is used for the start-pose task
- `target_prefix` is the next main setting for same-clip motion-conditioned adjustment

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

- current model is still an editor/generator, not yet a dedicated transition model with an explicit latent state predictor

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

- `current state + action program -> desired state transition`

and later:

- `current skill state + relative action -> updated skill state -> motion`

This is why `operator`, `reference`, `preserve_mode`, `skill_label`, `skill_phase`, and periodic `offset/amplitude` metadata are being added now.

## Current experiment track

Active subset:

- `HumanML3D-CMU`

Baseline completed:

- program-conditioned CMU run without goal loss
- train-split CMU start-pose goal-loss run

Next focus:

- implement same-clip prefix task as the main motion-conditioned setting
- train a first CMU prefix model
- inspect held-out salient visualizations for iterative debugging

Current refreshed CMU artifacts:

- `artifacts/jsonl/HumanML3D-CMU/scan/manifest_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/attributes/attribute_cache_full.jsonl`
- `artifacts/jsonl/HumanML3D-CMU/mining/mined_pairs_full.jsonl`

Current goal-loss experiment config:

- `configs/stage1_mined_cmu_program_goal_clean.yaml`

## Open questions

- how to better identify true skill structure from short 60-frame clips
- how to separate `set absolute target` from `relative edit on ongoing skill`
- how to represent skill memory beyond per-clip metadata
- when to introduce explicit latent transition modeling instead of only edited trajectory prediction
