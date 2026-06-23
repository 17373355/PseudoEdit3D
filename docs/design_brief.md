# Design Brief

This is the current working design for the Atomic Motion Language (AML) / AutoPrompt-HumanML3D project.

The previous Stage 1 prefix-completion and Language-Guided Action Regulation brief has been archived at:

- `docs/legacy/design_brief_stage1_action_regulation_legacy_2026-06-08.md`

## Current Title

Working title:

- **Learning an Atomic Motion Language for Structured Motion Annotation, Control, and Transfer**

Short internal name:

- **Atomic Motion Language (AML)**

## Current Research Target

The project is now centered on learning a motion-derived semantic annotation layer.

The immediate goal is to construct an AutoPrompt-HumanML3D benchmark:

- input: HumanML3D motion only
- output: structured AML program plus rendered auto-prompt
- constraint: the auto-prompt must not copy the same-case HumanML3D caption
- use of HumanML3D text: allowed only as a global wording inventory / noisy semantic prior / evaluation reference

The long-term goal is to make motion controllable in a language-like way:

- decompose motion into atomic and sub-motion units
- merge repeated units into higher-level phase patterns
- render the motion program into a stable motion language
- train condition models on this structured language
- support future commands such as `move the right hand outward by 30 degrees` or `walk forward 2 steps, then jump right 3 steps without turning`

## Why AML Instead of Direct Motion Tokenization Only

Direct motion tokenization is still useful, but it is not sufficient as the project claim.

Motion tokenization answers:

- how to discretize continuous pose or motion trajectories for generation

AML answers:

- what the motion means structurally
- which body part changes
- when it changes
- in what direction
- by how much
- how many times
- under what support/contact/locomotion context

The dependency is:

```text
pose / motion observables
-> atomic motion events
-> sub-motion units
-> phase patterns
-> AML program
-> rendered auto-prompt / condition language
```

So AML is a semantic layer over motion-derived tokens, not a replacement for low-level tokenization.

## Core Principle

Use a motion-first pipeline.

```text
HumanML3D motion
-> frame observables
-> micro-events
-> sub-motion units
-> repeated phase patterns
-> AML Layer3 program
-> motion-only auto-prompt
-> MoMask generation probe / qualitative audit
-> full-corpus regression
```

HumanML3D captions are not the auto-prompt source. They are used only for:

- global wording inventory
- rough cluster naming reference
- finding disagreement cases
- evaluating whether AML is more consistent than raw captions

## Current Data Source

Current active data source:

- HumanML3D packed joints: `/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/joints3d.pth`
- HumanML3D text files: `/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/texts/`
- full AML manifest: `outputs/aml_mining_corpus_full/hml3d_mining_30000.jsonl`

Current full-corpus scale:

- manifest cases: `29048`
- full AML scan with phase and low-body state: `outputs/aml_full_cluster_scan_with_phase_lowbody_v2.json`
- report: `outputs/aml_full_cluster_scan_with_phase_lowbody_v2_report.md`

## AML Layer Design

### Layer 0: Frame Observables

Layer 0 converts raw joints into continuous frame-level channels.

Current examples:

- root x/z motion
- root body-frame forward/lateral velocity
- root yaw from joints-derived heading
- root height proxy
- pelvis-to-ankle height
- torso bend/drop/forward extent
- arm raise / elbow lift
- wrist-to-chest distance

Design target:

- keep numeric values recoverable
- preserve frame alignment
- avoid early semantic overcommitment

### Layer 1: Micro-Events

Layer 1 segments each observable into atomic changes or sustained states.

Current event types:

- directional changes: up/down, near/far, turn left/right, locomotion active
- sustained states: locomotion state, low-body hold

Recent important addition:

- `WHOLE_BODY_POSTURE/WB_LOW_BODY_HOLD`
- rendered conservatively as `keeps the body low`
- fixes the failure mode where static low poses produce no event

### Layer 2: Sub-Motion Units

Layer 2 merges local micro-event sequences into interpretable units.

Examples:

- crouch descent
- hop ascent
- leg compress-release cycle
- torso bend recover
- hand near/far cycle
- both arms lift
- hands move away from chest

Layer 2 is still intentionally conservative. It should not become a collection of one-off hard-coded HumanML3D action names.

### Layer 2.5: Phase Patterns

Layer 2.5 detects repeated sub-motion phases.

Examples:

- repeated arm swing
- repeated vertical phases
- alternating repeated phases
- torso oscillation

Full-scan evidence shows phase detection should remain in the mainline:

- no-phase full scan: `avg_layer3=7.030`, `low_event_case_count_le2=6726`
- with-phase full scan: `avg_layer3=10.125`, `low_event_case_count_le2=4515`

### Layer 3: AML Atomic Program

Layer 3 exports the structured event program.

Each event should ideally contain:

- `super_family`
- `cluster_id`
- `part`
- `direction`
- `role`
- `start_frame`
- `end_frame`
- `magnitude`
- `unit`
- `count`
- `confidence`
- `source`
- `motion_signature`
- `metadata`

Current major families:

- `WHOLE_BODY_LOCOMOTION`
- `WHOLE_BODY_ROTATION`
- `WHOLE_BODY_VERTICAL`
- `WHOLE_BODY_POSTURE`
- `TORSO_PERIODIC`
- `BIMANUAL_PERIODIC`
- `LEFT_ARM_PERIODIC`
- `RIGHT_ARM_PERIODIC`

## Current Renderer Design

The renderer turns the AML program into a natural-language auto-prompt.

Important distinction:

- AML program: complete, structured, numeric, searchable
- auto-prompt: compressed, human-readable, suitable for MoMask probing and qualitative audit

The renderer should not expose internal technical names such as `arm cycle`.

Current renderer improvements:

- full rotation wording uses explicit 360-degree language
- jump-spin cases merge into `jumps and does one complete 360-degree ... spin`
- vertical salience gate suppresses walking root oscillation from becoming false jump/squat language
- recovery from down to up renders as `rises back up`, not `jumps upward`
- locomotion-coupled arm repeats render as `swings both arms while walking`
- non-locomotion arm repeats render as `moves the left/right arm repeatedly N times`

Current regression artifacts:

- vertical salience report: `outputs/aml_vertical_salience_v2_report.md`
- arm family abstraction report: `outputs/aml_arm_family_abstraction_v2_report.md`
- phrase count full regression: `outputs/aml_prompt_phrase_counts_full_arm_v2.json`

## HumanML3D Wording Inventory Mining

Upper-body motion is currently ambiguous. We therefore mine HumanML3D captions as a global wording inventory.

Important constraint:

- same-case captions are not used to generate that case's auto-prompt
- mined phrases are used only for cluster naming, word-bank construction, and confidence analysis

Current mining output:

- full JSON: `outputs/hml3d_upperbody_phrase_mining_full_v2.json`
- report: `outputs/hml3d_upperbody_phrase_mining_full_v2_report.md`

Current mined word-family groups include:

- `support_contact`
- `object_hold_or_manipulate`
- `arm_raise_lift`
- `arm_extend_spread`
- `arm_swing_walk`
- `wave_or_gesture`
- `clap_or_hands_together`
- `touch_body`
- `punch_boxing`
- `dance_or_circular_gesture`

These are candidate semantic families. They should guide subclustering, not directly overwrite motion-derived labels.

## Current Known Bottlenecks

### 1. Bimanual Coarse Events

Current issue:

- `BI_OUT` and `BI_UP` are too broad
- they may correspond to support/contact, object manipulation, clap-like motion, arm raise, or ordinary walking arm motion

Current full regression:

- `bimanual_coarse` prompt cases: `13103/29048`

Next design step:

- add hand-hand distance
- add hand speed / anchoring
- add hand-to-root stability
- add hand extension during locomotion
- split `BI_OUT/BI_UP` into support-like, object-like, clap-like, free-raise, and unknown bimanual families

### 2. Stairs / Steps / Obstacles

Current issue:

- vertical motion while moving can mean stairs, stepping, obstacle crossing, hopping, or jumping
- thresholding alone cannot separate them reliably

Next design step:

- add foot clearance
- add alternating foot height
- add step count estimation
- add airborne/contact proxy
- distinguish stair-like repeated low vertical displacement from jump-like airborne events

### 3. Support / Contact Proxy

Current issue:

- HumanML3D motion has no explicit scene geometry
- AML cannot honestly say `wall` or `rail` from motion alone unless a support-like hand signal is detected

Next design step:

- infer support-like hand behavior from hand extension + low hand velocity + root movement + body height change
- render conservatively as `keeps one hand extended for support`
- do not directly say `wall` / `rail` / `surface` without scene evidence

### 4. Numeric and Repetition Semantics

Current issue:

- counts, steps, degrees, and repetition are partially represented but not yet uniformly rendered

Existing design note:

- `docs/design/aml_numeric_repetition_angle_design.md`

Next design step:

- standardize count confidence
- expose exact angle/step values in AML metadata
- render natural language with numeric bins only when confidence is high

## Evaluation Design

### Experiment 1: AutoPrompt-HumanML3D Benchmark

Compare under the same MoMask architecture:

- original HumanML3D captions + MoMask text-conditioned model -> FID
- AutoPrompt-HumanML3D captions + MoMask text-conditioned model -> FID

What it can support:

- annotation layer matters
- AML auto-prompts produce a more consistent conditioning space
- motion-derived annotations can reduce noise from incorrect or inconsistent HumanML3D captions

### Experiment 2: Cross-Dataset Generalization

On unlabeled dataset B:

```text
motion_B
-> AML / AutoPrompt
-> AutoPrompt-HML3D-trained MoMask
-> generated motion
-> compare with GT motion
```

What it can support:

- AML is not just overfitting HumanML3D text style
- the learned motion language can transfer across motion datasets
- a model trained on AML supervision learns reusable motion patterns rather than only benchmark caption artifacts

### Supporting Analyses

Needed alongside FID:

- caption conflict rate in raw HumanML3D
- AML prompt consistency for similar motions
- phrase entropy per motion cluster
- qualitative GT / selected HML3D / auto-prompt / generated motion comparisons
- bad-case regression suite
- round-trip consistency: motion -> AML -> generation -> AML

## Current Active Scripts

Main extraction / scan:

- `scripts/scan_aml_full_clusters.py`
- `scripts/summarize_aml_cluster_scan.py`
- `scripts/extract_aml_layers.py`

Prompt and MoMask probe:

- `scripts/run_momask_aml_prompt_probe.py`
- `scripts/visualize_momask_case_study.py`

Phrase / wording mining:

- `scripts/analyze_aml_prompt_phrases.py`
- `scripts/analyze_vertical_salience.py`
- `configs/motion_pattern_text_targets.json`

Report visualization:

- `scripts/visualize_aml_report_artifacts.py`

Note: report visualization currently requires a plotting backend such as `matplotlib` in the active environment.

## Current Report Artifacts

Current key outputs:

- `outputs/aml_full_cluster_scan_with_phase_lowbody_v2.json`
- `outputs/aml_full_cluster_scan_with_phase_lowbody_v2_report.md`
- `outputs/aml_vertical_salience_v2_report.md`
- `outputs/aml_arm_family_abstraction_v2_report.md`
- `outputs/aml_prompt_phrase_counts_full_arm_v2.json`
- `outputs/hml3d_upperbody_phrase_mining_full_v2.json`
- `outputs/hml3d_upperbody_phrase_mining_full_v2_report.md`

Experiment log:

- `docs/experiment_log.md`

## Current Design Invariants

1. Auto-prompt is motion-only at case level.
2. Captions can be mined globally but cannot be copied into the same case's auto-prompt.
3. Layer3 should preserve numeric evidence even when renderer suppresses a phrase.
4. Prompt renderer should be conservative when evidence is ambiguous.
5. Full HumanML3D regression should be run after mechanism changes.
6. MoMask generation is a probe, not the definition of AML quality.
7. The final AML benchmark should be more atomic, more consistent, and less noisy than raw HumanML3D captions.

## Immediate Next Steps

1. Generate report visualizations for extraction layers, cluster distribution, phrase distribution, and upper-body word-family heatmap.
2. Split `BI_OUT` and `BI_UP` using support/contact/object/clap/free-raise proxies.
3. Add step/stair/obstacle observables based on foot height, contact, and root vertical trajectory.
4. Re-run full AML scan and prompt phrase regression.
5. Regenerate representative qualitative cases and MoMask probe visualizations.
6. Prepare AutoPrompt-HumanML3D training data export format.
