# Experiment Log

Compact R&D record for this project. Keep this file updated as new training designs, settings, and held-out outputs are added.

## Summary Table

| Name | Main idea | Key config / setting | Final key losses | Held-out output |
|---|---|---|---|---|
| `stage1_mined_cmu_program_clean` | cross-clip mined pairs, program-only baseline | `configs/stage1_mined_cmu_program_clean.yaml` | `loss=0.2040`, `edit=0.1948`, `keep=0.0184`, `smooth=0.0070` | none exported |
| `stage1_mined_cmu_program_goal_clean` | cross-clip mined pairs + goal losses | `configs/stage1_mined_cmu_program_goal_clean.yaml` | `loss=0.3042`, `goal_delta=0.1351`, `goal_offset=0.2476`, `goal_amp=0.4517` | `outputs/HumanML3D-CMU-test/vis_program_goal_clean_heldout/` |
| `stage1_mined_startpose_goal_clean` | use target first frame as source condition | `configs/stage1_mined_cmu_train_startpose_goal_clean.yaml` | `loss=0.2506`, `condition=0.0057`, `goal_delta=0.1555`, `goal_offset=0.3577` | `outputs/HumanML3D-CMU-test/vis_program_startpose_goal_clean_heldout/`, `..._salient/` |
| `stage1_prefix_continue` | same-clip prefix continuation, no language semantics | `configs/stage1_prefix_cmu_train_continue.yaml` | `loss=0.0428`, `edit=0.0393`, `condition=0.0035`, `smooth=0.0045` | `outputs/HumanML3D-CMU-test/vis_stage1_prefix_continue_heldout/` |
| `stage1_prefix_semantic_continue` | same-clip prefix + semantic continue prompt | `configs/stage1_prefix_cmu_train_semantic_continue.yaml` | `loss=0.1354`, `condition=0.0146`, `goal_delta=0.2859`, `goal_span=0.3008` | `outputs/HumanML3D-CMU-test/vis_stage1_prefix_semantic_continue_heldout/` |
| `stage1_prefix_semantic_continue_v2` | semantic continue + explicit velocity / acceleration losses | `configs/stage1_prefix_cmu_train_semantic_continue_v2.yaml` | `loss=0.1369`, `velocity=0.0154`, `acceleration=0.0241`, `goal_delta=0.2858` | not yet exported |
| `stage1_atomic_realize` | start-pose atomic action realization | `configs/stage1_atomic_cmu_train.yaml` | `loss=0.2680`, `future_all=0.0497`, `velocity=0.0222`, `goal_delta=0.3097` | `outputs/HumanML3D-CMU-test/vis_stage1_atomic_heldout/` |
| `stage1_prefix_completion_regonly` | prefix20 + EditProgram -> full completion, regression-only ablation | `configs/stage1_prefix_completion_cmu_regonly.yaml` | `loss=0.1287`, `edit=0.1287` | `outputs/HumanML3D-CMU-test/vis_stage1_prefix_completion_regonly_heldout/` |
| `stage1_prefix_completion_reg_dynamics` | prefix20 + EditProgram -> full completion, regression + dynamics ablation | `configs/stage1_prefix_completion_cmu_reg_dynamics.yaml` | `loss=0.1597`, `edit=0.1266`, `future_all=0.0576`, `velocity=0.0303`, `acceleration=0.0515` | `outputs/HumanML3D-CMU-test/vis_stage1_prefix_completion_reg_dynamics_heldout/` |
| `stage1_prefix_completion_v1` | prefix20 + EditProgram -> full completion, full loss setting | `configs/stage1_prefix_completion_cmu_v1.yaml` | `loss=0.2528`, `edit=0.1520`, `future_all=0.0679`, `velocity=0.0303`, `goal_delta=0.3287` | `outputs/HumanML3D-CMU-test/vis_stage1_prefix_completion_v1_heldout/` |
| `stage1_atomic_prefix20_clean` | prefix20 + atomic prompt -> full completion, earlier prefix-conditioned atomic version | `configs/stage1_atomic_cmu_prefix20_clean.yaml` | `loss=0.3082`, `edit=0.1711`, `future_all=0.0812`, `velocity=0.0382`, `goal_delta=0.3325` | `outputs/HumanML3D-CMU-test/vis_stage1_atomic_prefix20_clean_heldout/` |

## Data and split notes

- Base subset: `HumanML3D-CMU`
- Split method: deterministic split by original clip group, not by individual 60-frame chunk
- Current split sizes:
  - train clips: `6704`
  - test clips: `1685`
- Split-specific mined pairs:
  - train mined pairs: `26782`
  - test mined pairs: `6708`

## Experiment notes

### 1. `stage1_mined_cmu_program_clean`

Goal:

- establish a simple mined-pair baseline with program-only conditioning

What was trained:

- cross-clip mined pairs
- source motion to target motion
- no goal losses

Takeaway:

- easy baseline to run
- not aligned with the intended “self-regulation” framing

### 2. `stage1_mined_cmu_program_goal_clean`

Goal:

- add goal-satisfaction losses on top of the mined baseline

What changed:

- `goal_delta`
- `goal_direction`
- `goal_tolerance`
- `goal_span`
- `goal_offset`
- `goal_amplitude_preserve`

Takeaway:

- cleaner than the pure baseline
- still tied to cross-clip mined supervision

### 3. `stage1_mined_startpose_goal_clean`

Goal:

- fix the source/target mismatch by conditioning on the target motion’s own first frame

What changed:

- `input_source_mode=target_start_pose`
- explicit `condition_loss`

Takeaway:

- much better aligned with “start pose + prompt -> motion”
- still tends to produce static or weak dynamics when prompt semantics are posture-like

### 4. `stage1_prefix_continue`

Goal:

- learn a pure same-clip continuation prior

What changed:

- `data_mode=prefix`
- `prefix_task_mode=continue`
- prefix/future split on the same clip

Takeaway:

- stable training
- useful as a motion prior
- not enough as the main scientific task because `continue` alone lacks action semantics

### 5. `stage1_prefix_semantic_continue`

Goal:

- add explicit semantic content to continuation, e.g. “continue the ongoing arm motion and raise...”

What changed:

- same-clip prefix input
- semantic continuation prompts
- goal losses kept on

Takeaway:

- trains stably
- still often collapses to static posture-like solutions
- semantic prompt and dynamic target are not fully aligned

### 6. `stage1_prefix_semantic_continue_v2`

Goal:

- encourage dynamics more strongly than v1

What changed:

- add explicit `velocity` and `acceleration` losses

Takeaway:

- numerically stable
- may still be semantically limited if prompts remain posture-oriented rather than primitive-oriented

### 7. `stage1_atomic_realize`

Goal:

- shift Stage 1 toward atomic action realization instead of continuation

What changed:

- start-pose style conditioning
- atomic prompts like `raise`, `bend`, `extend`
- weak full-future supervision added via `future_all_loss`

Takeaway:

- better aligned with explicit action semantics than continuation
- current prompts are still somewhat posture-like rather than truly dynamic primitives

### 8. `stage1_prefix_completion_regonly`

Goal:

- isolate whether the current prefix-completion failure is already present under plain regression supervision

What changed:

- same `prefix20 + EditProgram -> 60-frame completion` setting
- keep only `edit_loss`

Takeaway:

- useful for debugging whether complex losses are the main cause of frozen-pose collapse

### 9. `stage1_prefix_completion_reg_dynamics`

Goal:

- test whether adding future dynamics supervision improves prefix completion over pure regression

What changed:

- `future_all_loss`
- `velocity_loss`
- `acceleration_loss`
- no goal losses

Takeaway:

- this is the cleanest ablation to compare against `reg-only` before blaming the full semantic loss stack

### 10. `stage1_prefix_completion_v1`

Goal:

- treat the current main task as prefix-conditioned action completion instead of continuation

What changed:

- prefix20 as motion context
- `EditProgram` as completion condition
- full loss stack

Takeaway:

- better aligned with the intended task framing than blind continuation
- still needs careful inspection because GT may include extra motion phases beyond the intended atomic action

### 11. `stage1_atomic_prefix20_clean`

Goal:

- first pass at prefix-conditioned atomic completion before the cleaner `prefix_completion` naming and setup

What changed:

- prefix20 + atomic prompt
- source panel masked after prefix

Takeaway:

- served as the transitional version before the explicit `prefix-conditioned action completion` framing

## Current interpretation

- `continue`-only training is useful as a prior but weak as a headline task
- `semantic continue` shows that adding words is not enough if the prompt still mainly describes posture rather than dynamic motion primitives
- `atomic realization` is currently the cleaner main direction, but likely needs a shift from posture prompts to dynamic primitive prompts such as `wave`, `swing`, `lift-and-lower`, and `step`


### Annotation-layer regression milestone (current 600-case benchmark)

Goal:

- move from ad hoc prompt repair toward a stable motion-derived annotation layer on a fixed 600-case benchmark
- require every meaningful update to pass local checks, global regression checks, and small qualitative checks

Current fixed benchmark:

- Batch 1: 100 cases
- Batch 2: 500 disjoint cases
- Combined triage: `good / soft_bad / hard_bad`

Current triage summary:

- `good = 276`
- `soft_bad = 207`
- `hard_bad = 117`

Top soft-bad categories:

- `walk_backward = 144`
- `stop_pause = 32`
- `turn = 16`
- `stair_descent = 15`
- `bounce_repeated = 15`

Top hard-bad categories:

- `crouch_bend = 46`
- `turn = 33`
- `walk_backward = 25`
- `bounce_repeated = 22`
- `stop_pause = 13`

Representative fixes already achieved:

- `000270 / 000526`: stairs up/down + turn no longer collapse to bounce/jump
- `000324`: repeated bounce preserved
- `000890`: repeated squat separated from repeated bounce
- `000006`: crouch no longer collapses to stair descent
- `000004`: torso bend no longer collapses to repeated bounce
- `000028 / 000144 / 008344`: stop semantics largely recovered
- `000179`: left-arm up/down recovered
- `000230`: repeated hop recovered from repeated squat

Main unresolved gap now:

- finer limb-level patterns remain weaker than whole-body patterns, especially:
  - `both elbows flap`
  - `arm circling`
  - `hand / rail support`

Artifacts:

- `outputs/hml3d_pattern_batches/triage_600/summary.json`
- `outputs/hml3d_pattern_batches/triage_600/missing_category_stats.json`
- `outputs/hml3d_pattern_batches/triage_600/regression_check_report.json`
- `docs/experiments/regression_check_protocol.md`

## AML family taxonomy v1 - 2026-06-05

Goal:

- move Layer 3 from semantic-name-first events toward family-first kinematic taxonomy
- validate whether current `super_family` and `cluster_id` assignments are stable on the 1000-case AML mining corpus

Artifacts:

- script: `scripts/summarize_aml_family_taxonomy.py`
- smoke report: `outputs/aml_family_taxonomy_hml3d20_smoke_v2.json`
- full report: `outputs/aml_family_taxonomy_hml3d1000_v2.json`

Main results on 1000 cases:

- `num_cases = 1000`
- `total_layer3_events = 10565`
- `avg_layer1_count = 82.203`
- `avg_layer2_count = 72.123`
- `avg_layer25_count = 7.63`
- `avg_layer3_count = 10.565`

Family support:

- `WHOLE_BODY_VERTICAL`: 980 cases
- `LEFT_ARM_PERIODIC`: 688 cases
- `RIGHT_ARM_PERIODIC`: 678 cases
- `TORSO_PERIODIC`: 499 cases
- `BIMANUAL_PERIODIC`: 443 cases

Interpretation:

- most clusters have `core_purity = 1.0` after removing `tempo_bucket` from the core signature
- this supports treating `tempo_bucket` as a control field rather than a cluster-splitting dimension
- stable clusters include `WB_VERT_UP`, `WB_VERT_DOWN`, `LA_REPEAT`, `RA_REPEAT`, `BI_OUT`, `LA_REPEAT_ALT`, `RA_REPEAT_ALT`, `TORSO_BEND_RECOVER`, `BI_UP`, `LA_NEAR_FAR`, `RA_NEAR_FAR`, `LA_COMPOSITE`, and `RA_COMPOSITE`
- `WHOLE_BODY_VERTICAL/WB_VERT_REP` is the only split candidate in this pass: `core_purity = 0.508`, with near-balanced non-alternating and alternating vertical repeated signatures

Next action:

- split `WB_VERT_REP` into two kinematic clusters, likely `WB_VERT_REP` and `WB_VERT_REP_ALT` or equivalent naming
- keep tempo as a parameter for control, not a category label

## AML family taxonomy v2 - 2026-06-05

Goal:

- apply the first signature-driven split inside `WHOLE_BODY_VERTICAL`
- separate alternating repeated vertical phases from non-alternating repeated vertical phases

Code/doc changes:

- `pseudoedit3d/edit/aml_atomic_program.py`: repeated vertical hop-like phases now map to `WB_VERT_REP_ALT` when `alternation=True`; otherwise they remain `WB_VERT_REP`
- `docs/design/motion_subword_design.md`: clarified that `tempo_bucket` is a control parameter, while `alternation` is a cluster split signal for repeated vertical motion

Artifacts:

- 20-case smoke report: `outputs/aml_family_taxonomy_hml3d20_smoke_v3.json`
- 1000-case full report: `outputs/aml_family_taxonomy_hml3d1000_v3.json`

Results:

- `split_candidates = []` after the split
- `WB_VERT_REP`: 248 events, 198 supporting cases, `core_purity = 1.0`
- `WB_VERT_REP_ALT`: 128 events, 106 supporting cases, `core_purity = 1.0`
- `avg_layer3_count` changed from `10.565` to `10.693`, because alternating and non-alternating repeated phases can coexist in the same motion rather than being merged as one cluster

Interpretation:

- this is the first successful family-internal split driven by motion signature rather than action name
- current taxonomy is more stable after separating alternation from the repeated vertical family
- tempo remains a control field and should not drive category splitting at this stage

## AML representative sampling and generalization check - 2026-06-05

Goal:

- make cluster inspection compatible with full HumanML3D scanning rather than only high-count examples
- avoid selecting representatives solely by event count, which biases whole-body clusters toward long walking/running sequences

Code change:

- `scripts/select_aml_cluster_representatives.py` now supports `--mode diverse`
- diverse mode ranks examples by cluster dominance, removes mirror duplicates before filling examples, and records coarse caption tags for inspection

Artifacts:

- top-count baseline: `outputs/aml_cluster_representatives_hml3d1000_v1.json`
- diverse representatives: `outputs/aml_cluster_representatives_hml3d1000_diverse_v1.json`

Generalization notes:

- representative selection is only for human inspection; it must not define the taxonomy
- full-corpus taxonomy decisions should be based on support, core signature purity, and split candidates
- `tempo_bucket` remains a control field
- some arm periodic clusters still appear in ordinary locomotion cases, suggesting a future split between locomotion-driven arm swing and intentional arm action

Next action:

- add an explicit signature or context field for `coupled_with_locomotion` versus isolated/intentional limb action
- use this to refine `LEFT_ARM_PERIODIC` and `RIGHT_ARM_PERIODIC` without overfitting to caption names

## AML context-coupled arm split v1 - 2026-06-05

Goal:

- separate ordinary locomotion-coupled arm swing from isolated or intentional arm periodic motion
- keep the split driven by motion context, not by HumanML3D caption names
- validate the change on the 1000-case AML mining corpus before treating it as a taxonomy update

Code change:

- `pseudoedit3d/edit/aml_atomic_program.py`: added `context_mode` and `coupled_with_locomotion` to Layer 3 motion signatures
- repeated arm clusters are split into `_LOCO` variants when their temporal span overlaps strongly with whole-body vertical motion:
  - `LA_REPEAT -> LA_REPEAT_LOCO`
  - `RA_REPEAT -> RA_REPEAT_LOCO`
  - `LA_REPEAT_ALT -> LA_REPEAT_ALT_LOCO`
  - `RA_REPEAT_ALT -> RA_REPEAT_ALT_LOCO`
- `scripts/summarize_aml_family_taxonomy.py`: includes `context_mode` in the core signature used for split-candidate detection

Artifacts:

- smoke report: `outputs/aml_family_taxonomy_hml3d20_context_split_smoke_v1.json`
- full report: `outputs/aml_family_taxonomy_hml3d1000_context_split_v1.json`
- schema-v2 full report with case-level core signatures: `outputs/aml_family_taxonomy_hml3d1000_context_split_v2.json`
- stable-cluster representative index: `outputs/aml_cluster_representatives_hml3d1000_context_split_v2.json`
- split-candidate representative index: `outputs/aml_split_candidate_representatives_hml3d1000_context_split_v1.json`
- split-candidate case-id list: `outputs/aml_split_candidate_representatives_hml3d1000_context_split_case_ids.txt`
- split-candidate AML visualization: `outputs/aml_vis_split_candidates_hml3d1000_context_split_v1/`

Results on 1000 cases:

- `num_cases = 1000`
- `total_layer3_events = 10693`
- `avg_layer3_count = 10.693`
- base arm repeat clusters become single-core-signature clusters after the split:
  - `LEFT_ARM_PERIODIC/LA_REPEAT`: 654 events, 414 cases, `core_purity = 1.0`
  - `RIGHT_ARM_PERIODIC/RA_REPEAT`: 633 events, 408 cases, `core_purity = 1.0`
  - `LEFT_ARM_PERIODIC/LA_REPEAT_ALT`: 126 events, 114 cases, `core_purity = 1.0`
  - `RIGHT_ARM_PERIODIC/RA_REPEAT_ALT`: 116 events, 106 cases, `core_purity = 1.0`
- locomotion-coupled arm repeat clusters are also stable:
  - `LEFT_ARM_PERIODIC/LA_REPEAT_LOCO`: 551 events, 394 cases, `core_purity = 1.0`
  - `RIGHT_ARM_PERIODIC/RA_REPEAT_LOCO`: 549 events, 386 cases, `core_purity = 1.0`
  - `LEFT_ARM_PERIODIC/LA_REPEAT_ALT_LOCO`: 181 events, 160 cases, `core_purity = 1.0`
  - `RIGHT_ARM_PERIODIC/RA_REPEAT_ALT_LOCO`: 175 events, 153 cases, `core_purity = 1.0`

Remaining split candidates:

- `TORSO_PERIODIC/TORSO_BEND_RECOVER`
- `BIMANUAL_PERIODIC/BI_UP`
- `TORSO_PERIODIC/TORSO_OSC_FB`
- `BIMANUAL_PERIODIC/BI_OUT`

Reusable diagnosis update:

- `scripts/summarize_aml_family_taxonomy.py` now records per-case `cluster_core_signatures`, enabling signature-level inspection for any future split candidate
- `scripts/select_aml_split_candidate_representatives.py` selects examples by split-candidate core signature rather than caption name
- `scripts/visualize_aml_atomic_program.py` renders full-length GT motion with active Layer 3 AML events, cluster ids, context mode, and event timeline
- current split-candidate representative set has 24 unique cases: `M002046`, `004381`, `001412`, `007344`, `M004950`, `004971`, `010780`, `000008`, `009585`, `M004289`, `M002463`, `M004334`, `004920`, `005716`, `014086`, `000090`, `M008251`, `001253`, `000584`, `002965`, `002534`, `000183`, `001878`, `M006848`
- all 24 representatives were rendered to `outputs/aml_vis_split_candidates_hml3d1000_context_split_v1/`; `summary.json` confirms full original GT lengths are used

Interpretation:

- context coupling is a valid family-internal split signal for arm periodic motion
- this reduces a known full-corpus failure mode where ordinary walking arm swing looked like intentional arm action
- torso and bimanual context coupling should not be hard-split yet; those families need representative-case inspection because coupled torso/arm motion can be either locomotion byproduct or intended action

## AML rotation recovery and template language v1 - 2026-06-07

Issue:

- case `000183` visibly contains a mid-motion spin, but the AML visualization showed no active event during the spin interval
- diagnosis showed `root_yaw_proxy_deg` was all zeros because HumanML3D AML extraction used joints but passed zero pose rotations, while the yaw proxy was still pose-derived

Fix:

- `pseudoedit3d/edit/frame_observables.py`: recompute `root_yaw_proxy_deg` from joints using hips + shoulders projected to the xz plane
- `pseudoedit3d/edit/aml_atomic_program.py`: map turn micro-events into a new `WHOLE_BODY_ROTATION` family with `WB_ROT_LEFT` / `WB_ROT_RIGHT` clusters
- `pseudoedit3d/edit/aml_language.py`: add deterministic template-language rendering for AML events
- `scripts/extract_aml_layers.py` and `scripts/visualize_aml_atomic_program.py`: attach compact and detailed AML template strings to exported outputs

Verification:

- `000183` now has `WHOLE_BODY_ROTATION/WB_ROT_LEFT` at frames `8-28`
- estimated signed yaw delta is about `383deg`, matching the HumanML3D caption phrase `spins completely around`
- updated visualization: `outputs/aml_vis_000183_turn_lang_v1/case_000183.gif`
- debug dump: `outputs/debug_aml_000183_with_turn_lang.json`

Template example:

- `frames 8-28 | whole_body.rotation | turn left | angle=383deg, signed_delta=383deg | context=body_driver`

Interpretation:

- this confirms AML should include a template language layer for inspection before free natural-language rendering
- rotation is kept as a motion family with numeric angle retained, not prematurely converted into an action phrase such as `spin around`

## AML rotation salience gate v1 - 2026-06-07

Goal:

- prevent small or noisy heading changes from being promoted into Layer 3 atomic rotation events
- keep precise signed yaw delta for numeric control while using coarse rotation clusters for inspection

Code change:

- `pseudoedit3d/edit/aml_atomic_program.py`: Layer 3 rotation now requires `abs_angle >= 45deg` and `duration >= 4`
- rotation clusters are split by angle family:
  - `SMALL`: 45-75deg
  - `QTR`: 75-135deg
  - `HALF`: 135-225deg
  - `THREE_QTR`: 225-315deg
  - `FULL`: 315-450deg
  - `MULTI`: over 450deg
- `pseudoedit3d/edit/aml_language.py`: template language now verbalizes these coarse angle families, while retaining exact `signed_delta`

Verification on `000183`:

- `WHOLE_BODY_ROTATION/WB_ROT_LEFT_FULL` at frames `8-28`
- compact template: `frames 8-28 | whole_body.rotation | turn left about a full turn | angle=383deg, signed_delta=383deg | context=body_driver`
- debug dump: `outputs/debug_aml_000183_rotation_gated_lang.json`

1000-case taxonomy result:

- report: `outputs/aml_family_taxonomy_hml3d1000_rotation_gated_lang_v1.json`
- `WHOLE_BODY_ROTATION`: 1153 events, 814 supporting cases
- full-turn clusters are rare and stable:
  - `WB_ROT_LEFT_FULL`: 14 events, 14 cases
  - `WB_ROT_RIGHT_FULL`: 13 events, 13 cases
- half-turn clusters dominate:
  - `WB_ROT_LEFT_HALF`: 359 events, 334 cases
  - `WB_ROT_RIGHT_HALF`: 364 events, 340 cases

Interpretation:

- the gate removes very small yaw fragments, but rotation support is still broad
- next diagnostic step is representative visualization for `SMALL / HALF / FULL / MULTI` rotation clusters to decide whether broad half-turn support reflects true turning or heading-estimation flips during ordinary motion

## AML locomotion state coverage v1 - 2026-06-07

Issue:

- M004289 has a long middle walking interval, but the AML visualization had no active event during most of that interval
- root-speed change events existed, but AML lacked a sustained state event for ongoing locomotion
- this exposed a coverage gap: Layer3 represented many changes and repeated phases, but not stable whole-body states

Fix:

- `pseudoedit3d/edit/micro_events.py`: added state-event extraction for sustained `root_xz_speed_proxy` activity
- `pseudoedit3d/edit/submotion_lexicon.py`: fallback submotion units now preserve state-event metadata
- `pseudoedit3d/edit/aml_atomic_program.py`: added `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_{SLOW,MEDIUM,FAST}` Layer3 events
- context coupling now considers both `WHOLE_BODY_VERTICAL` and `WHOLE_BODY_LOCOMOTION` as body-driver events
- `pseudoedit3d/edit/aml_language.py`: added template labels for locomotion state events

Verification on M004289:

- debug dump: `outputs/debug_aml_M004289_locomotion_state_v1.json`
- visualization: `outputs/aml_vis_M004289_locomotion_state_v1/case_M004289.gif`
- recovered event: `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_MEDIUM` at frames `18-138`
- compact template: `frames 18-138 | whole_body.locomotion | move through space | amplitude=4.05m, signed_delta=4.05m | context=body_driver`
- arm periodic events in the same span now become locomotion-coupled variants

1000-case coverage result:

- report: `outputs/aml_family_taxonomy_hml3d1000_locomotion_state_v1.json`
- `num_cases = 1000`
- `total_layer3_events = 13329`
- `avg_layer3_count = 13.329`
- no zero-event cases remain
- only 2 cases have `layer3_count <= 2`
- `WHOLE_BODY_LOCOMOTION`: 1936 events, 955 supporting cases
- locomotion state cluster support:
  - `LOCO_ACTIVE_SLOW`: 323 events, 216 cases
  - `LOCO_ACTIVE_MEDIUM`: 1249 events, 729 cases
  - `LOCO_ACTIVE_FAST`: 364 events, 246 cases

Interpretation:

- AML now covers both change-like events and sustained state-like intervals
- current `LOCO_ACTIVE_*` is intentionally broad; it fixes coverage but is not yet a complete locomotion taxonomy
- next required refinement is to split locomotion by trajectory and gait semantics, such as forward/backward/sideways, walk/run, turn-in-place, stop/pause, and path length / step count

## AML architecture coverage and numeric design docs - 2026-06-07

Goal:

- consolidate the current AML architecture before deeper LLM/VLM-inspired refinement
- separate active AML mainline from older auto-prompt pattern-repair mechanisms
- define how quantity, repetition, angle, and continuous numeric residue should be represented

New documents:

- `docs/design/aml_architecture_coverage.md`
- `docs/design/aml_numeric_repetition_angle_design.md`
- `docs/legacy_migration_plan.md`
- `legacy/README.md`

Key architecture status:

- change events: Layer 1 micro-events and Layer 3 atomic events
- sustained states: initial `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_*`
- repeated phases: Layer 2.5 phase patterns
- numeric residue: `magnitude`, `signed_delta`, `unit`, `path_length`, `mean_speed`, `count`, frame spans
- template language: `pseudoedit3d/edit/aml_language.py`

Legacy cleanup:

- moved backup files only:
  - `scripts/run_momask_case_study.py.bak` -> `legacy/auto_prompt_pattern_batches/scripts/run_momask_case_study.py.bak`
  - `scripts/visualize_momask_case_study.py.bak` -> `legacy/auto_prompt_pattern_batches/scripts/visualize_momask_case_study.py.bak`
- did not move active probe scripts or training scaffold yet

Next design targets:

- explicit repetition evidence: `phase_spans`, `period_frames`, `count_confidence`
- rotation geometry metadata: `axis`, `reference_frame`, `sign_convention`, `unwrap_policy`, `angle_bin`
- locomotion direction split: forward/backward/sideways/mixed/turning/stationary
- deterministic AML-Lang grouped rendering
- targeted numeric evaluation: count accuracy, signed-angle error, locomotion-direction accuracy, round-trip consistency

### AML MoMask prompt probe v2 - 2026-06-07

Goal:

- quickly test motion-derived AML auto prompts through the existing MoMask T2M generator.
- keep auto prompts motion-only: the renderer uses Layer3 AML events, not the same-case HumanML3D captions.

What changed:

- added `pseudoedit3d/edit/aml_prompt_renderer.py` for shared AML-to-natural-prompt rendering.
- added salience filtering so short bimanual/noisy movement and tiny locomotion do not dominate full-body spin cases.
- added Layer3 aggregate `LOCO_TURN_*` events from cumulative yaw evidence inside locomotion states, recovering turn-while-walking cases.
- added `--ext-prefix` to `scripts/run_momask_aml_prompt_probe.py` to avoid overwriting prior MoMask generations.

Probe cases:

- `000183`: selected HML3D prompt=`a person jumps, spins completely around in a circle an lands.`; AML auto prompt=`a person jumps and spins left around once`.
- `M004289`: selected HML3D prompt=`a person walks forward then turns while using their arms to press against the wall`; AML auto prompt=`a person moves through space for about 4.0 meters, then turns right while moving, then moves both hands outward, then raises both arms, then makes a small up-and-down body motion`.

Outputs:

- summary: `outputs/momask_aml_prompt_probe_v2/summary.json`
- visualization: `outputs/momask_aml_prompt_probe_v2_vis/case_000183.gif`
- visualization: `outputs/momask_aml_prompt_probe_v2_vis/case_M004289.gif`

Known gaps:

- locomotion is still generic `moves through space`; forward/backward/sideways direction is not yet inferred.
- wall/support semantics are not yet recoverable without a hand-contact/support proxy.
- M004289 still includes coarse arm clauses and a small vertical clause; a second renderer pass should merge this into a cleaner support-like phrase once support detection exists.
- MoMask rounds generation length to unit length multiples, so `154` requested frames becomes `152` generated frames for M004289.

### AML locomotion direction v1 - 2026-06-07

Goal:

- recover motion-only locomotion direction instead of rendering every state as `moves through space`.

What changed:

- added body-frame root velocity at Layer0 using HumanML3D joints-derived heading.
- attached `root_forward_velocity` and `root_lateral_velocity` to `root_xz_speed_proxy` metadata.
- summarized locomotion states with `trajectory_direction`, `forward_displacement`, `lateral_displacement`, and absolute displacement diagnostics.
- mapped locomotion states to `LOCO_FORWARD/BACKWARD/LEFT/RIGHT/MIXED_*` Layer3 clusters.
- calibrated the HumanML3D heading sign using a small set of clearly forward/backward captioned clips; the raw geometry sign was reversed, so the forward vector is negated in `frame_observables.py`. This calibration fixes the coordinate convention only and is not used as per-case prompt supervision.

Smoke result:

- `M004289` auto prompt changed from generic movement to `a person walks forward for about 4.0 meters, then turns right while moving, then moves both hands outward, then raises both arms, then makes a small up-and-down body motion`.
- `000183` remains `a person jumps and spins left around once`; its small locomotion state keeps `trajectory_direction=unknown`, so spin cases are not mislabeled as walking.

Known gaps:

- direction is single-state coarse direction; multi-segment forward/backward sequences still need temporal splitting.
- support/wall semantics are still missing.
- step count is not yet estimated.

### MoMask case-study visualization prompt-wide update - 2026-06-07

Goal:

- make selected HML3D prompt, AML auto prompt, and all HML3D captions readable in case-study GIFs.

What changed:

- increased canvas from `1600x460` to `1900x560`.
- widened the prompt/program panel from `380px` to `520px`.
- reduced prompt font from `16` to `14` and raw-caption font from `13` to `11`.
- removed fixed selected/auto prompt truncation; text now fills the prompt panel until the panel bottom.
- raw captions now include all `raw_prompt_segments` rather than only the first three, subject to available panel space.

Outputs:

- `outputs/momask_aml_prompt_probe_v3_vis_promptwide/case_000183.gif`
- `outputs/momask_aml_prompt_probe_v3_vis_promptwide/case_M004289.gif`

### AML full HumanML3D cluster scan v1 - 2026-06-07

Goal:

- move AML mechanism validation from 600/1000-case slices to full HumanML3D coverage.
- inspect current `super_family` / `cluster_id` distribution before further mechanism changes.
- verify whether Layer 2.5 phase detection should be part of the mainline full-scan regression.

Outputs:

- full manifest: `outputs/aml_mining_corpus_full/hml3d_mining_30000.jsonl` (`29048` cases).
- no-phase scan: `outputs/aml_full_cluster_scan_no_phase_v1.json`.
- with-phase scan: `outputs/aml_full_cluster_scan_with_phase_v1.json`.
- readable report: `outputs/aml_full_cluster_scan_with_phase_v1_report.md`.
- report generator: `scripts/summarize_aml_cluster_scan.py`.

Key results:

- no-phase full scan: `elapsed=117.35s`, `avg_layer3=7.030`, `low_event_case_count_le2=6726`.
- with-phase full scan: `elapsed=133.02s`, `avg_layer25=6.195`, `avg_layer3=10.125`, `low_event_case_count_le2=4515`.
- phase detection reduces low-event cases and adds stable repeated arm / torso / vertical structures, so it should stay in the AML mainline.

Mechanism readout:

- full HumanML3D AML scan is now fast enough to use as a regular regression check after mechanism changes.
- high-support `WHOLE_BODY_VERTICAL` clusters are the first salience-control target; ordinary walking root oscillation can otherwise become false jump/squat wording.
- high-support bimanual and arm-repeat clusters should be treated as family candidates, not direct action names; walking arm swing, support-like poses, hand-object actions, clapping, and waving need subclustering.
- support/contact and scene-dependent language should stay conservative: motion-only AML can infer hand anchoring or support-like extension, but should not directly say wall/rail/surface without scene evidence.
- low-event examples concentrate around stairs/rail, object/hold, sit/lie/kneel, jump/hop, and step-count cases, which are the next observables/rendering targets.


### AML sustained low-body state v2 - 2026-06-08

Goal:

- address a full-scan coverage gap where static low poses can produce no Layer3 event because the earlier AML mainline focused on changes rather than sustained states.
- add a conservative motion-only posture state without naming it as squat/kneel/crawl/lie too early.

What changed:

- `pseudoedit3d/edit/micro_events.py`: added `SUSTAINED_STATE_CHANNEL_CONFIG` and `segment_low_body_state` on `pelvis_to_ankle_height`.
- `pseudoedit3d/edit/aml_atomic_program.py`: maps sustained low body state to `WHOLE_BODY_POSTURE/WB_LOW_BODY_HOLD`.
- `pseudoedit3d/edit/aml_prompt_renderer.py`: renders this conservatively as `keeps the body low`.

Smoke result:

- `013356` and `M013356` changed from `a person moves naturally` to `a person keeps the body low`.
- `002469`, `004775`, `000183`, and `M004289` were not falsely assigned `WB_LOW_BODY_HOLD`.

Full-scan result:

- v1 with-phase scan: `outputs/aml_full_cluster_scan_with_phase_v1.json`.
- v2 low-body scan: `outputs/aml_full_cluster_scan_with_phase_lowbody_v2.json`.
- v2 report: `outputs/aml_full_cluster_scan_with_phase_lowbody_v2_report.md`.
- new `WHOLE_BODY_POSTURE/WB_LOW_BODY_HOLD` support: `1848` cases.
- `avg_layer3` changed from `10.125` to `10.196`.
- `low_event_case_count_le2` changed from `4515` to `4460`.

Interpretation:

- this is not mainly a low-event-count optimization; it fixes a representation hole where sustained postures are invisible to a pure change-event tokenizer.
- next steps should split low-body posture into squat/kneel/crawl/lie only after adding more observables such as torso orientation, hand/foot support, and body-horizontal extent.


### AML vertical salience gate v2 - 2026-06-08

Goal:

- reduce false `jumps upward`, `lowers the body`, `hop-like`, and tiny `up-and-down` wording caused by ordinary locomotion root oscillation.
- keep vertical Layer3 events searchable; only gate natural prompt rendering.

What changed:

- `pseudoedit3d/edit/aml_prompt_renderer.py`: added locomotion-overlap-aware vertical salience filtering.
- high-overlap `WB_VERT_CYCLE`, `WB_VERT_REP`, and `WB_VERT_REP_ALT` are suppressed from prompt rendering.
- high-overlap small `WB_VERT_UP/DOWN` is suppressed; larger high-overlap `UP/DOWN` is rendered neutrally as `changes body height while moving`.
- `WB_VERT_UP` closely following `WB_VERT_DOWN` is rendered as `rises back up` instead of `jumps upward`.
- tiny `WB_VERT_CYCLE` with magnitude below `0.04` is suppressed as body-bounce/noise.

Smoke result:

- `004775`: now renders only walking, removing false small up/down wording.
- `008939`: false `jumps upward` changed to neutral `changes body height while moving`.
- `011473`: bend-knee recovery changed from false `jumps upward` to `rises back up`.
- `006761` and `000183` still preserve isolated jump / jump-spin wording.
- `013356` still renders `keeps the body low`.

Regression outputs:

- 5k before gate: `outputs/aml_vertical_salience_5k_v1.json`, problematic vertical wording `4548/5000`.
- 5k after gate: `outputs/aml_vertical_salience_5k_v2.json`, problematic vertical wording `1692/5000`.
- full after gate: `outputs/aml_vertical_salience_full_v2.json`, problematic vertical wording `6965/29048`.
- readable report: `outputs/aml_vertical_salience_v2_report.md`.

Interpretation:

- this gate fixes a renderer-level failure mode, not a Layer3 event deletion; numeric vertical evidence remains available for later retrieval/control.
- remaining stair/upstairs/downstairs and obstacle cases need step-height, foot-clearance/airborne, and support/contact proxies rather than more threshold-only suppression.
- arm repeat noise is now a more visible next bottleneck and should be handled by family abstraction next.


### AML arm family abstraction v2 - 2026-06-08

Goal:

- reduce technical prompt wording such as `repeats a left arm cycle`.
- introduce a family-level renderer for locomotion-coupled arm repeats.

What changed:

- `pseudoedit3d/edit/aml_prompt_renderer.py`: locomotion-coupled arm repeats now render as `swings the left/right arm while walking`.
- left and right walking arm swing clauses merge into `swings both arms while walking`.
- remaining non-locomotion arm repeats render as `moves the left/right arm repeatedly N times` instead of `repeats arm cycle`.
- added prompt phrase regression script: `scripts/analyze_aml_prompt_phrases.py`.

Regression outputs:

- 5k before wording cleanup: `outputs/aml_prompt_phrase_counts_5k_arm_v1.json`, raw arm-cycle prompts `1472/5000`.
- 5k after cleanup: `outputs/aml_prompt_phrase_counts_5k_arm_v2.json`, raw arm-cycle prompts `0/5000`.
- full after cleanup: `outputs/aml_prompt_phrase_counts_full_arm_v2.json`, raw arm-cycle prompts `0/29048`, arm-swing family prompts `9826/29048`, bimanual coarse prompts `13103/29048`.
- readable report: `outputs/aml_arm_family_abstraction_v2_report.md`.

Interpretation:

- this is a prompt-layer family abstraction: Layer3 still stores the repeat/count evidence, but the natural language no longer exposes technical `cycle` names.
- the next major bottleneck is `BIMANUAL_PERIODIC/BI_OUT` and `BI_UP`; these need support-like, object-like, clap-like, and free-raise subfamilies rather than direct `moves both hands outward` / `raises both arms` wording.
