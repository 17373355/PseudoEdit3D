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


### AML report visualization artifacts v1 - 2026-06-08

Goal:

- create report-ready visualizations and tables for the current AML extraction pipeline, full-corpus cluster distribution, prompt phrase distribution, and upper-body wording mining.

Inputs:

- cluster scan: `outputs/aml_full_cluster_scan_with_phase_lowbody_v2.json`.
- previous scan for delta: `outputs/aml_full_cluster_scan_with_phase_v1.json`.
- prompt phrase counts: `outputs/aml_prompt_phrase_counts_full_arm_v2.json`.
- upper-body phrase mining: `outputs/hml3d_upperbody_phrase_mining_full_v2.json`.
- vertical salience regression: `outputs/aml_vertical_salience_full_v2.json`.

Outputs:

- artifact root: `outputs/aml_report_artifacts_v1/`.
- browser index: `outputs/aml_report_artifacts_v1/index.html`.
- markdown index: `outputs/aml_report_artifacts_v1/README.md`.
- figures: `outputs/aml_report_artifacts_v1/figures/`.
- tables: `outputs/aml_report_artifacts_v1/tables/`.

Generated figures:

- `01_layer_average_counts.png`: average Layer1 / Layer2 / Layer2.5 / Layer3 counts per case.
- `02_super_family_event_vs_support.png`: AML super-family event count vs case support.
- `03_top_aml_clusters.png`: top AML family/cluster distribution.
- `05_prompt_phrase_distribution.png`: prompt phrase group distribution after current renderer.
- `06_upperbody_motion_word_family_heatmap.png`: motion-cluster to HML3D word-family coverage heatmap.

Key readout:

- processed cases: `29048`.
- avg Layer3 events: `10.196`.
- low-event cases <=2: `4460`.
- raw arm-cycle prompt cases: `0`.
- bimanual coarse prompt cases: `13103`.
- problematic vertical prompt cases after gate: `6965`.

Note:

- HML3D captions are used as a global wording inventory for cluster naming/reference only, not as same-case auto-prompt input.


### AML bimanual split v1 - 2026-06-08

Goal:

- split the coarse `BIMANUAL_PERIODIC/BI_OUT` and `BIMANUAL_PERIODIC/BI_UP` clusters into motion-checkable subclasses.
- keep the split motion-only at case level; HumanML3D captions are used only for diagnostic examples and global wording analysis.

What changed:

- added `pseudoedit3d/edit/bimanual_split.py` for joints-derived bimanual feature extraction and split classification.
- `build_layer3_atomic_program(..., joints=...)` can now relabel coarse bimanual events into `BI_SPREAD`, `BI_RAISE_SPREAD`, `BI_RAISE`, `BI_HANDS_CLOSE`, and `BI_HANDS_CLOSE_RAISE`.
- updated AML renderer/language mappings so auto-prompts can use split-level bimanual wording.
- updated scan / prompt phrase / AML visualization / MoMask probe scripts to pass `joints` into Layer3 construction.
- added diagnostic split visualization script: `scripts/visualize_bimanual_split_report.py`.

Regression outputs:

- diagnostic 5k split: `outputs/bimanual_split_candidates_5k_v2.json`.
- diagnostic 5k split report: `outputs/bimanual_split_candidates_5k_v2_report.md`.
- diagnostic 5k split figures: `outputs/bimanual_split_candidates_5k_v2_figures/`.
- full HumanML3D scan after split: `outputs/aml_full_cluster_scan_bimanual_split_v1.json`.
- bimanual report root: `outputs/bimanual_split_report_v1/`.
- bimanual report figures: `outputs/bimanual_split_report_v1/figures/`.
- bimanual report tables: `outputs/bimanual_split_report_v1/tables/`.
- prompt phrase counts after split: `outputs/aml_prompt_phrase_counts_full_bimanual_split_v2.json`.

Key readout:

- processed cases: `29048`.
- avg Layer3 events: `10.196`.
- low-event cases <=2: `4460`.
- full bimanual event split:
  - `BI_SPREAD`: events `10898`, cases `7451`.
  - `BI_RAISE_SPREAD`: events `8497`, cases `3873`.
  - `BI_RAISE`: events `4822`, cases `3347`.
  - `BI_HANDS_CLOSE`: events `5148`, cases `2873`.
  - `BI_HANDS_CLOSE_RAISE`: events `3022`, cases `2059`.
- 5k prompt phrase counts after split:
  - `bimanual_spread`: `1457/5000`.
  - `bimanual_raise`: `738/5000`.
  - `bimanual_raise_spread`: `703/5000`.
  - `bimanual_hands_close`: `475/5000`.

Interpretation:

- the old bimanual bottleneck is now structurally split into multiple motion-derived subclasses rather than one `moves both hands outward` / `raises both arms` bucket.
- object / support / clap labels are intentionally not used yet because joints-only HumanML3D evidence does not prove object contact, wall contact, or palm contact.
- the next acceptance standard should visually test whether each split cluster is internally coherent, whether split clusters remain overloaded, and whether all high-support bimanual clusters have distinguishable feature distributions.


### AML bimanual cluster contact sheets v1 - 2026-06-08

Goal:

- provide a lightweight visual acceptance artifact for checking whether bimanual split clusters are actually separable.
- avoid generating many large GIFs by using static before/middle/after event frames.

Outputs:

- contact sheet root: `outputs/bimanual_cluster_contact_sheets_v1/`.
- contact sheet PNGs: `outputs/bimanual_cluster_contact_sheets_v1/contact_sheets/`.
- index table: `outputs/bimanual_cluster_contact_sheets_v1/bimanual_contact_sheet_index.csv`.
- acceptance criteria: `outputs/bimanual_cluster_contact_sheets_v1/acceptance_criteria.md`.

Key readout:

- generated `5` contact sheets, one per bimanual split cluster.
- each sheet contains `6` representative samples from the full scan examples.
- total PNG size is under `1MB`, much lighter than full GIF visualization.
- all current split clusters are covered: `BI_SPREAD`, `BI_RAISE_SPREAD`, `BI_RAISE`, `BI_HANDS_CLOSE`, `BI_HANDS_CLOSE_RAISE`.

Interpretation:

- this artifact is intended for human acceptance checks, not for quantitative scoring.
- the first acceptance standard is intra-cluster visual coherence and inter-cluster separability by hand distance, wrist height, and timing.
- semantic labels such as clap / support / object-hold remain blocked until stronger contact/object evidence is added.


### AML coordination pattern layer v1 - 2026-06-08

Motivation:

- bimanual split clusters are local arm realizations, not final action semantics.
- semantic actions such as standing long jump require whole-body coordination: leg compression/release, forward root displacement, and bimanual arm timing.
- different people may realize the arm slot differently: `BI_SPREAD`, `BI_RAISE_SPREAD`, `BI_RAISE`, `BI_HANDS_CLOSE`, or combinations.

What changed:

- added `pseudoedit3d/edit/coordination_patterns.py` as a Layer4 diagnostic above atomic AML events.
- added `scripts/analyze_coordination_patterns.py` to scan jump/body-arm coordination patterns.
- Layer4 patterns preserve `coordination_slots.arms.realization_clusters` instead of replacing them with a single action label.

Smoke outputs:

- 1k diagnostic v1: `outputs/coordination_patterns_1k_v1.json` and `outputs/coordination_patterns_1k_v1_report.md`.
- 1k diagnostic v2: `outputs/coordination_patterns_1k_v2.json` and `outputs/coordination_patterns_1k_v2_report.md`.
- v2 counts on 1000 cases:
  - `COORD_FORWARD_JUMP_ARM_COORDINATION`: events `417`, cases `230`.
  - `COORD_VERTICAL_JUMP_ARM_COORDINATION`: events `129`, cases `85`.
  - `COORD_STANDING_FORWARD_JUMP_CANDIDATE`: events `30`, cases `29`.
- top arm realization variants include `BI_SPREAD`, `BI_RAISE`, `BI_RAISE_SPREAD`, `BI_HANDS_CLOSE`, and combinations.

Important correction:

- `COORD_STANDING_FORWARD_JUMP_CANDIDATE` is not yet a reliable semantic label. v2 examples still include pick-up / step-up / crouch-walk cases.
- the detector was tightened after v2: `WB_VERT_CYCLE` no longer triggers jump coordination, and standing candidate now requires low-body preparation, small pre-path, enough forward displacement, enough vertical magnitude, and arm preparation/takeoff timing.
- the tightened rule passed a synthetic smoke test but was not full-scanned due a temporary I/O stall; do not report v3 corpus numbers yet.

Current conclusion:

- Layer4 coordination is the right abstraction: action semantics should be built from whole-body slots plus local realization variants.
- local bimanual splits should remain as implementation details / slots, not final semantic action labels.
- next step is to build a visual/quantitative acceptance set for coordination patterns before using names like standing long jump, support, object-hold, or clap.

## 2026-06-08 AML regression test set v2

- Built a fixed HumanML3D test-split AML regression set for repeated AutoPrompt visual checks.
- Command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_aml_regression_testset.py --output-dir outputs/aml_regression_testset_v2 --total-cases 250 --group-size 50 --strategy stratified --progress-every 500`
- Scanned 4384 test IDs, 4358 valid motions; selected 250 cases in 5 groups of 50.
- Selection uses per-group buckets: coordination 6, locomotion 8, rotation 7, vertical 7, bimanual 7, unilateral_arm 5, torso_posture 5, simple_other 5.
- Main artifacts: `outputs/aml_regression_testset_v2/aml_regression_testset_v2.csv`, `outputs/aml_regression_testset_v2/aml_regression_testset_v2.json`, `outputs/aml_regression_testset_v2/group_01_case_ids.txt` ... `group_05_case_ids.txt`.
- Generated group 01 GIFs with `frame_stride=4`, `fps=10`: `outputs/aml_regression_testset_v2/group_01_aml_gifs_stride4/`.
- Group 01 validation: 50 GIFs, 50 summary rows, no missing or empty files, total directory size about 85.77 MB.
- Visualization panel now prints HML3D prompt in blue and motion-only `auto_prompt` in orange.

## 2026-06-09 Group 01 first10 MoMask AutoPrompt-vs-GT probe

- Added auto-only MoMask probe: `scripts/run_momask_aml_autoprompt_probe.py`.
- Added GT-vs-AutoPrompt generation visualizer: `scripts/visualize_momask_auto_gt.py`.
- Generated first 10 cases from `outputs/aml_regression_testset_v2/group_01_case_ids.txt` with motion-only AutoPrompt as MoMask text condition.
- Probe summary: `outputs/aml_regression_testset_v2/group_01_momask_auto_probe_first10/summary.json`.
- GIF output: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_gifs_stride4/`.
- Validation: 10 GIFs produced; directory size about 9.9 MB; visualization has GT motion, AutoPrompt text, and MoMask from AutoPrompt only.

## 2026-06-09 AutoPrompt MoMask probe diagnosis from group 01 first10

- User inspection found that several ordinary motions are over-rendered into complex event-stream AutoPrompts: `000189` and `002755` should mostly read as walking/jogging/running-like motions, but the rendered prompt becomes a long sequence of hops, arm swings, raising/spreading arms, and body-height changes.
- `001082` shows another failure mode: upper-body details are partially captured, but root motion such as slow leftward movement and turning is lost or drowned by the upper-body phrase stream in MoMask generation.
- `M000106`, `M010032`, and related cases show that forward jump / jump-ahead should be represented as a whole-body coordinated action family, not just separated arm, vertical, and locomotion events.
- `M002798`, `M008014`, and `M008235` show that current wording such as `change body height` and broad `spread arms` is not a stable common-sense action phrase for T2M, and it also misses rhythm/amplitude distinctions.
- Interpretation: the current AutoPrompt captures local details but lacks coarse action-family abstraction and coordination-level semantics; a pure streaming text condition is not adequate for program-like motion labels.
- Next design direction: build a hierarchical renderer/conditioner where coarse common actions such as walk, run, jog-in-place, jump-forward, jump-up, jumping-jack, and turn are inferred first, then residual atomic details are injected as structured program slots rather than as a long natural-language stream.
- Visualization update: `scripts/visualize_momask_auto_gt.py` now shows HML3D captions in blue and AutoPrompt in orange; regenerated first10 GIFs at `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_gifs_hml3d_stride4/`.



## 2026-06-09 - AML coarse signature renderer v2

- Separated MoMask probe aliases from canonical AML condition encoding. Probe text remains natural language; `canonical_actions` stores structured ids and numeric slots.
- Added conservative Layer-3 terminal-state event `WHOLE_BODY_STATE/WB_TERMINAL_STILL`; coarse signature consumes it rather than rescanning raw motion.
- Improved repeat counting for jumping-jack-like motions using vertical down/up evidence and bimanual raise-spread evidence; `M008014` now renders `does jumping jacks 8 times`.
- Tightened `JUMPING_JACK` matching to avoid misclassifying short jump/backward cases; `M008235` is no longer rendered as jumping jacks.
- Added `WEAK_BALLISTIC_CANDIDATE` canonical actions for low-confidence later jumps; these are kept for future program conditioning but hidden from MoMask text probe by default.
- Current preview summary: `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v5/summary.json`.
- Design note: `docs/design/aml_coarse_signature_pipeline.md`.

## 2026-06-09 - Upper-body global alias evidence scan v2

- Updated `scripts/mine_hml3d_upperbody_phrases.py` to scan all HumanML3D text files directly when no manifest is provided, so the current compact `outputs/` layout no longer depends on the removed mining manifest.
- Added global wording families for `jumping_jack`, `clap_or_hands_together`, `overhead_clap_or_cheer`, `martial_strike`, `push_shove`, `dance_or_rhythm`, `instrument_or_tool_mime`, `throw_catch`, and related upper-body patterns.
- Full run: `29232` requested cases, `23999` upper-body-valid cases, elapsed `180.65s`.
- Outputs: `outputs/hml3d_upperbody_phrase_mining_full_v2.json` and `outputs/hml3d_upperbody_phrase_mining_full_v2_report.md`; total size remains small enough to keep as a milestone artifact.
- Strong evidence: `BIMANUAL_PERIODIC/BI_RAISE_SPREAD|nonloco+vertical -> jumping_jack` has support `329`, coverage `0.333`, precision `0.906`, lift `22.04`.
- Strong evidence: `BIMANUAL_PERIODIC/BI_HANDS_CLOSE|nonloco -> clap_or_hands_together` has support `290`, coverage `0.249`, precision `0.295`, lift `6.09`.
- Weak/mixed evidence: `overhead_clap_or_cheer`, `martial_strike`, `push_shove`, `support_contact`, and `instrument_or_tool_mime` produce plausible aliases but are not clean enough to become motion-only action names yet.
- Interpretation: global HML3D text can be used as an alias-bank / naming-evidence layer, but weak upper-body semantics need finer event signatures before entering AutoPrompt rendering. Same-case HML3D captions remain forbidden for motion-only AutoPrompt generation.

## 2026-06-09 - Conservative hands-close canonical action

- Added `BIMANUAL_HANDS_CLOSE` as a conservative coarse canonical action for `BI_HANDS_CLOSE` and `BI_HANDS_CLOSE_RAISE` events.
- The action stores `global_alias_evidence=clap_or_hands_together` in slots, but the MoMask probe phrase stays conservative: `brings both hands together`.
- Nearby hands-close events are merged into one action with `count` and `source_event_clusters`, so canonical actions do not fragment into repeated identical clauses.
- Refreshed the first-10 coarse preview metadata in `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v5/` without running MoMask generation.
- `001082` now includes `BIMANUAL_HANDS_CLOSE` in canonical actions; this confirms the global alias evidence is connected to structured AML without using same-case HML3D captions.
- Regression note: natural-language probe prompts can still become too long when residual events are appended. This should be addressed by a salience/budgeted probe renderer, while keeping the full canonical action program intact for future AML-conditioned training.

## 2026-06-09 - Salience-budgeted MoMask probe renderer

- Updated `pseudoedit3d/edit/coarse_prompt_renderer.py` so MoMask probe text is budgeted separately from the full canonical AML program.
- Default probe policy: at most `5` coarse clauses, at most `1` residual clause, and about `34` words.
- The renderer keeps temporal order for selected clauses; it does not reorder actions by salience because that can corrupt motion sequence semantics.
- Low-value residual phrases such as repeated arm cycles, torso oscillation, and generic height changes are filtered from probe text.
- Full canonical actions and residual events remain in `coarse_action_program`; only the natural-language `auto_prompt` is shortened.
- Refreshed `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v5/summary.json` and `index.md`.
- First-10 prompt lengths after budget: `M008014=7`, `M011732=28`, `M002798=30`, `M010032=32`, `000189=5`, `002755=5`, `009961=34`, `001082=30`, `M008235=11`, `M000106=32` words.
- Interpretation: this makes MoMask probing less misleading by avoiding long event streams, while preserving the richer AML program for future program-conditioned training.

## 2026-06-09 - Focus5 budgeted MoMask visualization refresh

- Regenerated focus5 MoMask generations with new ext prefix `aml_reg_v2_focus5_budget_v1` so old focus5 generations are not reused.
- Covered cases: `M008014`, `M008235`, `M010032`, `M000106`, `002755`.
- Probe summary overwritten at `outputs/aml_regression_testset_v2/group_01_momask_auto_probe_focus5_coarse_v2/summary.json`.
- GIFs overwritten at `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_focus5_coarse_v2_gifs_hml3d_stride4/`.
- Index: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_focus5_coarse_v2_gifs_hml3d_stride4/index.md`.
- Prompt correction: `M008235` now probes `jumps straight up, then steps backward to regain balance`, not backward jump.
- Prompt budget check: focus5 word counts are `M008014=7`, `M008235=11`, `M010032=32`, `M000106=32`, `002755=5`.

## 2026-06-09 - Focus5 kinematic sanity check

- Added `scripts/analyze_momask_probe_kinematics.py` for lightweight regression checks on MoMask probe outputs.
- The script compares GT and generated root XZ path length, root net displacement, root vertical amplitude, mean speed, hand-distance statistics, and length delta.
- Outputs for focus5:
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_focus5_coarse_v2_gifs_hml3d_stride4/kinematic_sanity.json`
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_focus5_coarse_v2_gifs_hml3d_stride4/kinematic_sanity.md`
- No automatic mismatch flags were triggered under the current loose thresholds.
- Notable case for manual GIF inspection: `M000106` generated root path is about `1.95x` GT path, suggesting possible over-translation despite no threshold violation.
- This is a rough per-iteration sanity check, not a perceptual metric and not a replacement for FID or human visual review.

## 2026-06-09 - Group01 first10 budgeted MoMask probe refresh

- Ran the full first10 group with the salience-budgeted coarse renderer, using `scripts/run_momask_aml_autoprompt_probe.py --prompt-mode coarse --time-steps 10 --cond-scale 4 --ext-prefix aml_reg_v2_first10_budget_v1`.
- Cases: `M008014`, `M011732`, `M002798`, `M010032`, `000189`, `002755`, `009961`, `001082`, `M008235`, `M000106`.
- Probe summary: `outputs/aml_regression_testset_v2/group_01_momask_auto_probe_first10_budget_v1/summary.json`.
- GIF/index output: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_budget_v1_gifs_hml3d_stride4/`.
- Kinematic sanity output:
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_budget_v1_gifs_hml3d_stride4/kinematic_sanity.json`
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_budget_v1_gifs_hml3d_stride4/kinematic_sanity.md`
- Output check: `10` GIFs produced; directory size about `10.88 MB`; index and sanity report both present.
- Prompt check: `000189` and `002755` now probe `a person runs in place`, avoiding the previous long local-event stream.
- Prompt check: `M008014` now probes `a person does jumping jacks 8 times`.
- Prompt check: `M008235` now probes `a person jumps straight up, then steps backward to regain balance`, not backward jumping.
- Automatic sanity flag: `000189` has `vertical_amp_mismatch` because generated root vertical amplitude is about `3.25x` GT.
- Manual watchlist: `M000106` has generated root path about `1.95x` GT; `M008235` generated root path is about `0.46x` GT, which may indicate under-translation / weak recovery-step realization.
- Interpretation: the budgeted probe is less misleading than the previous long stream, but it still exposes the limitation of pure natural-language MoMask probing. Canonical AML actions and slots should remain the primary future condition representation; the natural prompt is only a compatibility probe.

## 2026-06-09 - Group01 prompt-level correction after user inspection

- User inspection flagged two prompt-level errors before rerunning MoMask:
  - `000189`: the motion is walking-like, but the AutoPrompt said `runs in place`.
  - `M002798`: the motion should expose a cheering/dancing-like whole-body arm gesture, but AutoPrompt only listed turns, small walking, and hand movement.
- Updated `pseudoedit3d/edit/coarse_signature.py`:
  - `IN_PLACE_GAIT` now separates `walk_in_place`, `jog_in_place`, and `run_in_place` using event-derived intensity: vertical amplitude, phase repeat count, and arm-locomotion proxy count.
  - Added conservative canonical family `CELEBRATORY_DANCE_GESTURE`, triggered by repeated bimanual raise-spread, low vertical bounce, multiple small turns / side movements, and no jumping-jack evidence.
- Updated `pseudoedit3d/edit/coarse_prompt_renderer.py`:
  - `walk_in_place -> walks in place`, `jog_in_place -> jogs in place`, `run_in_place -> runs in place`.
  - `CELEBRATORY_DANCE_GESTURE -> makes a cheer-like dance gesture with repeated arm raises`.
- Preview-only refresh, no MoMask generation: `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v6/summary.json`.
- Preview index: `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v6/index.md`.
- Corrected prompt checks:
  - `000189 -> a person walks in place`.
  - `002755 -> a person jogs in place`.
  - `M002798 -> a person makes a cheer-like dance gesture with repeated arm raises`.
- Important caveat: `CELEBRATORY_DANCE_GESTURE` is still a motion-signature candidate family, not a guaranteed action label. It should be validated by full-corpus cluster examples before being used as a stable benchmark label.

## 2026-06-09 - Group01 random10 current-rule MoMask probe v1

- Sampled 10 new cases from group 01 excluding the inspected first10 with seed `20260609`.
- Case manifest: `outputs/aml_regression_testset_v2/group_01_random10_current_rule_v1_case_ids.txt`.
- Cases: `007232`, `M000266`, `M009712`, `M010447`, `M013562`, `012388`, `004163`, `014448`, `M004684`, `M001969`.
- Probe summary: `outputs/aml_regression_testset_v2/group_01_momask_auto_probe_random10_current_rule_v1/summary.json`.
- GIF/index output: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_random10_current_rule_v1_gifs_hml3d_stride4/`.
- Kinematic sanity output:
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_random10_current_rule_v1_gifs_hml3d_stride4/kinematic_sanity.json`
  - `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_random10_current_rule_v1_gifs_hml3d_stride4/kinematic_sanity.md`
- Output check: `10` GIFs produced; GIF/index directory size about `13 MB`; total `outputs/` size about `162 MB`.
- Automatic sanity flag: `012388` has `vertical_amp_mismatch`; generated path is about `2.47x` GT while generated vertical amplitude is about `0.32x` GT, suggesting the forward-jump / climb-over motion was weakened into translation.
- Prompt-level watchlist from random sampling:
  - `M010447`: HML3D says cartwheel then jumping; current AutoPrompt decomposes it into jumps, turns, walking, and low body state, so a cartwheel-like whole-body inverted/side-rotation family is missing.
  - `M013562`: HML3D says hops left twice then right; current AutoPrompt emphasizes turns and straight jump, so side-hop repetition / lateral hopping family is missing.
  - `M004684`: HML3D says bends legs and lifts dumbbells; current AutoPrompt says jumps up and down 3 times, so low-body bend + arm-lift / object-lift proxy is missing and vertical cycles are over-interpreted as jumping.
  - `004163`: HML3D focuses on arm stretching; current AutoPrompt includes large forward walk and spin, so root-motion salience may be too high for arm-dominant motions.
- Interpretation: random sampling exposes missing coarse action families beyond the first10 fixes. The next productive step is not more MoMask generation, but adding/validating candidate families for lateral hop repetition, cartwheel/inverted rotation, low-body bend + arm lift, and arm-dominant stretch/move before another probe batch.

## 2026-06-10 - Semantic-family AML upgrade and gap8 MoMask probe v8

- Added motion-only semantic event layer `pseudoedit3d/edit/semantic_events.py` for general observables rather than case-specific rules: bounded torso pitch, wrist height, squat/low-body state, leg forward extension, root circular path, root-height climb proxy, and inverted-body acrobatics proxy.
- Extended `pseudoedit3d/edit/coarse_signature.py` with higher-level candidate families and dominance filtering:
  - `CLIMB_UP_OVER_PROXY`
  - `SQUAT_REPETITION`
  - `SQUAT_ARM_LIFT`
  - `CIRCULAR_WALK_PATH`
  - `ACROBATIC_SEQUENCE_CANDIDATE`
  - `DANCE_LEG_POSE_CANDIDATE`
- Updated `pseudoedit3d/edit/coarse_prompt_renderer.py` so probe prompts keep high-salience semantic families under the word budget instead of simply truncating later clauses.
- Preview-only output: `outputs/aml_regression_testset_v2/semantic_gap8_preview_v8/index.md`.
- MoMask probe output: `outputs/aml_regression_testset_v2/semantic_gap8_momask_v8/summary.json`.
- GIF/index output: `outputs/aml_regression_testset_v2/semantic_gap8_momask_v8_gifs/index.md`.
- Output size check: prompt preview about `1.6 MB`; GIF folder about `11 MB`; `outputs/aml_regression_testset_v2` about `135 MB`.
- Prompt checks:
  - `012388` now includes `climbs upward and over`.
  - `014448` now includes kick-like leg actions and no longer starts with false circular walking.
  - `M000266` now includes `walks in a circular path` without gait-leg false kick.
  - `M004684` now probes `repeatedly squats low 3 times`, not jumping.
  - `M009712` now includes a `dance-like pose with the left leg extended`.
  - `M010447` now includes `repeated inverted acrobatic motions 7 times`.
- Remaining caveat: these are still candidate semantic families inferred from motion signatures. The MoMask natural-language probe is only a compatibility check; the canonical AML program/slots should remain the target conditioning representation for future training.

## 2026-06-10 - Unknown semantic family and approximate slot metadata v1

- Updated `pseudoedit3d/edit/coarse_signature.py` so every coarse/canonical action now carries:
  - `semantic_family`: `{family_id, source_family, status, label_confidence, motion_only, source, probe_visible}`.
  - `approx_slots`: uncertainty-aware slot values with `value`, numeric `range` when available, `unit`, `confidence`, `source`, and `quality`.
- Semantic family status policy:
  - `stable`: existing direct AML families such as gait, jump, turn, terminal still.
  - `candidate`: plausible but not yet frozen semantic families such as `SQUAT_REPETITION`, `ACROBATIC_SEQUENCE_CANDIDATE`, `DANCE_LEG_POSE_CANDIDATE`, and weak ballistic candidates.
  - `proxy`: conservative observable proxies such as raised hand, squat hold, climb-up-over proxy, and leg kick proxy.
  - `unknown`: fallback families `UNKNOWN_EVENT_SEQUENCE` and `UNKNOWN_BIMANUAL_FAMILY`, with source event family/cluster counts retained for later subclustering.
- Kept backwards compatibility:
  - existing flat `slots` remain present;
  - `slots.semantic_family_id`, `slots.semantic_family_status`, and `slots.approx_slots` mirror the new fields for downstream code that only reads `slots`.
- Focused validation:
  - py-compile: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py`
  - preview-only extraction: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 012388,M004684,M010447 --output-dir outputs/aml_regression_testset_v2/semantic_gap3_unknown_slots_preview_v2 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix aml_semantic_slots_v2`
  - synthetic unknown fallback check confirmed `EVENT_SEQUENCE -> UNKNOWN_EVENT_SEQUENCE` with source family/cluster counts and low-confidence approximate slots.
- Preview output:
  - `outputs/aml_regression_testset_v2/semantic_gap3_unknown_slots_preview_v2/summary.json`
  - per-case metadata under `outputs/aml_regression_testset_v2/semantic_gap3_unknown_slots_preview_v2/{case_id}/aml_meta.json`
- Interpretation:
  - this does not promote candidate semantic families into final labels;
  - it makes uncertainty explicit so future AML-conditioned training can weight or filter `stable`, `candidate`, `proxy`, and `unknown` families differently.

### Semantic-family status diagnostics v1

- Added `scripts/analyze_aml_semantic_family_status.py`.
- The diagnostic can:
  - read existing preview/probe `summary.json` files that already contain `canonical_actions`;
  - or re-extract coarse canonical actions from a case list / manifest using the current code path.
- Gap8 current-code re-extraction:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/analyze_aml_semantic_family_status.py --case-ids 012388,014448,M000266,M004684,M009712,M010447,004163,007232 --output-json outputs/aml_regression_testset_v2/semantic_gap8_status_slots_v1/semantic_family_status.json --output-md outputs/aml_regression_testset_v2/semantic_gap8_status_slots_v1/semantic_family_status.md --top-n 30 --example-limit 30 --progress-every 4`
  - result: `58` canonical actions over `8` cases, `stable=29`, `proxy=20`, `candidate=9`, `unknown=0`.
- Fixed 250-case regression-set diagnostic:
  - case list source: `outputs/aml_regression_testset_v2/group_01_case_ids.txt` through `group_05_case_ids.txt`.
  - saved case list: `outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt`.
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/analyze_aml_semantic_family_status.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --output-json outputs/aml_regression_testset_v2/semantic_status_250_v1/semantic_family_status.json --output-md outputs/aml_regression_testset_v2/semantic_status_250_v1/semantic_family_status.md --top-n 40 --example-limit 40 --progress-every 50`
  - result: `1434` canonical actions over `250` cases.
  - status distribution: `stable=853` (`59.5%`), `proxy=418` (`29.1%`), `candidate=126` (`8.8%`), `unknown=37` (`2.6%`).
  - unknown case support: `37/250` cases.
  - top unknown source clusters: `LEFT_ARM_PERIODIC/LA_REPEAT`, `RIGHT_ARM_PERIODIC/RA_REPEAT`, `BIMANUAL_PERIODIC/BI_SPREAD`, `RIGHT_ARM_POSTURE/RA_HAND_HIGH`, `LEFT_ARM_POSTURE/LA_HAND_HIGH`, `RIGHT_ARM_PERIODIC/RA_NEAR_FAR`, `BIMANUAL_PERIODIC/BI_HANDS_CLOSE`.
- Diagnostic outputs:
  - `outputs/aml_regression_testset_v2/semantic_gap8_status_slots_v1/semantic_family_status.md`
  - `outputs/aml_regression_testset_v2/semantic_status_250_v1/semantic_family_status.md`
- Interpretation:
  - the semantic-family layer covers the curated gap8 cases without unknown fallback;
  - the 250-case regression set still exposes a small but useful unknown bucket, mostly object/tool mime, arm-only gesture, bimanual range-of-motion, and subtle low-motion state patterns;
  - next mechanism work should target subfamilies for unilateral/bimanual arm mime and object-like interactions before expanding MoMask probes.

### Conservative arm-mime semantic family v2

- Updated `pseudoedit3d/edit/coarse_signature.py` with conservative non-scene semantic families:
  - `BIMANUAL_ARM_MIME_CANDIDATE`: bimanual/upper-body events dominate without reliable locomotion or a stronger semantic family.
  - `UNILATERAL_ARM_MIME_CANDIDATE`: repeated or held one-arm upper-body motion without a stable action name.
  - `STATIC_OR_SUBTLE_STATE_PROXY`: only static/subtle state evidence is available.
- Updated `pseudoedit3d/edit/coarse_prompt_renderer.py` with conservative probe wording:
  - `makes a bimanual upper-body gesture`
  - `makes repeated left/right arm gestures`
  - `holds a mostly still subtle pose`
- Kept the rule motion-only:
  - no same-case HML3D captions are read;
  - no object/tool/scene words such as dumbbell, drink, dog, discussion, wall, or rail are emitted.
- 250-case status regression after arm-mime v2:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/analyze_aml_semantic_family_status.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --output-json outputs/aml_regression_testset_v2/semantic_status_250_armmime_v2/semantic_family_status.json --output-md outputs/aml_regression_testset_v2/semantic_status_250_armmime_v2/semantic_family_status.md --top-n 50 --example-limit 50 --progress-every 50`
  - result: `1433` canonical actions over `250` cases.
  - status distribution changed from v1 `stable=853`, `proxy=418`, `candidate=126`, `unknown=37` to v2 `stable=853`, `proxy=427`, `candidate=138`, `unknown=15`.
  - unknown action share dropped from `2.6%` to `1.0%`.
- Gap8 regression after arm-mime v2:
  - output: `outputs/aml_regression_testset_v2/semantic_gap8_status_slots_v2/semantic_family_status.md`
  - result unchanged at `58` canonical actions, `stable=29`, `proxy=20`, `candidate=9`, `unknown=0`.
- Preview-only prompt check:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 009941,M008179,000055,002795 --output-dir outputs/aml_regression_testset_v2/semantic_armmime_preview_v2 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix aml_armmime_v2`
  - output: `outputs/aml_regression_testset_v2/semantic_armmime_preview_v2/summary.json`
  - checks: `009941` and `M008179` use bimanual upper-body gesture wording; `000055` uses repeated right-arm gesture wording; no object/tool labels were introduced.
- Remaining unknown bucket:
  - mostly no-event / too-subtle clips, single raised-hand posture, and vertical-only in-place motion not strong enough for the current in-place gait rule.
  - next useful mechanism is a separate no-event/subtle-motion audit and an object-like arm interaction candidate that still avoids scene/object nouns unless external evidence exists.

### Unknown-family closure and required-slot audit v3

- Step 1 audit:
  - 250-case output: `outputs/aml_regression_testset_v2/semantic_unknown_audit_250_v1/semantic_family_status.md`.
  - gap8 output: `outputs/aml_regression_testset_v2/semantic_unknown_audit_gap8_v1/semantic_family_status.md`.
  - 250-case status before closure: `1433` canonical actions, `stable=853`, `proxy=427`, `candidate=138`, `unknown=15`.
  - gap8 remained `unknown=0`.
- Step 2 semantic-family closure:
  - no-event / zero-span fallback actions now become hidden `STATIC_OR_SUBTLE_STATE_PROXY` rather than `UNKNOWN_EVENT_SEQUENCE`.
  - low repeated vertical in-place bounce with bimanual/torso evidence now becomes `IN_PLACE_GAIT_PROXY` rather than `UNKNOWN_BIMANUAL_FAMILY`.
  - fallback `EVENT_SEQUENCE` / `BIMANUAL_ACTION` actions are dropped when all source events are already explained by later semantic/proxy actions.
  - 250-case status after closure: `1427` canonical actions, `stable=853`, `proxy=436`, `candidate=138`, `unknown=0`.
  - output: `outputs/aml_regression_testset_v2/semantic_status_250_after_step2_v1/semantic_family_status.md`.
- Step 3 approximate-slot audit:
  - `scripts/analyze_aml_semantic_family_status.py` now reports required per-family `approx_slots` coverage and examples.
  - `ROTATION_DOMINANT` now copies `angle_deg` and `angle_bin` from the rotation signature into canonical action slots.
  - squat candidate requirements allow `magnitude|vertical_amplitude_m`, because some squat candidates are posture-depth estimates while others are vertical-cycle estimates.
  - required-slot missing count is `0` on the 250-case audit.
  - output: `outputs/aml_regression_testset_v2/semantic_slot_audit_250_v3/semantic_family_status.md`.
- Step 4 renderer policy:
  - `coarse_prompt_renderer` now reads `semantic_family.status`.
  - `unknown` actions are excluded from probe text.
  - `candidate` and `proxy` actions are salience-penalized and rendered with conservative observable wording.
  - `probe_visible=false` actions remain structural only; no-evidence clips render as `moves naturally`.
  - preview output: `outputs/aml_regression_testset_v2/semantic_renderer_step4_preview_v1/summary.json`.
- Representative preview checks:
  - `M000239`: `a person raises the left hand high`.
  - `002795`: `a person makes a small in-place bouncing motion`.
  - `M006210`: `a person moves naturally`.

### AML condition manifest v1

- Added `pseudoedit3d/edit/aml_condition_schema.py` as the shared required-slot and condition-weight schema.
- Updated `scripts/analyze_aml_semantic_family_status.py` to reuse the shared schema rather than owning a duplicate required-slot table.
- Added `scripts/export_aml_condition_manifest.py`.
- Manifest record contract:
  - one JSONL row per case;
  - `conditions` stores one item per canonical action;
  - each condition stores `family_id`, `source_family`, `status`, `condition_weight`, `slot_values`, `slot_confidences`, `slot_qualities`, full `approx_slots`, and `missing_required_slots`;
  - HML3D text is kept only as `selected_hml3d_prompt_for_reference_only`, not as the AML condition.
- Default weights:
  - `stable=1.0`
  - `candidate=0.7`
  - `proxy=0.5`
  - `unknown`, missing required slots, or `probe_visible=false` -> `0.0`
- 250-case export:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --output-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/summary.json --output-md outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/summary.md --top-n 80 --progress-every 50`
  - result: `250` cases, `1427` conditions, `missing_required_condition_count=0`, `zero_weight_condition_count=55`.
  - status distribution: `stable=853`, `proxy=436`, `candidate=138`.
- Gap8 export:
  - output: `outputs/aml_regression_testset_v2/aml_condition_manifest_gap8_v1/summary.md`
  - result: `8` cases, `58` conditions, `missing_required_condition_count=0`, `zero_weight_condition_count=8`.

### AML condition screening v1

- Added `scripts/screen_aml_conditions.py`.
- Scope:
  - replace the complex HTML evolution view with simple Markdown/JSON/JSONL screening records;
  - keep every original condition in `screened_conditions.jsonl`;
  - attach `screen_score`, `screen_decision`, and `screen_reason`;
  - export a compact `selected_conditions.jsonl` for downstream dataset/training wiring.
- Scoring rule:
  - combines `condition_weight`, label confidence, slot confidence, slot quality, and required-slot coverage;
  - `condition_weight=0`, `probe_visible=false`, or missing required slots score `0`;
  - default selected threshold is `0.42`;
  - default deferred threshold is `0.315`.
- Gap8 run:
  - command: `python scripts/screen_aml_conditions.py --manifest outputs/aml_regression_testset_v2/aml_condition_manifest_gap8_v1/conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_condition_screening_gap8_v1 --threshold 0.42 --defer-ratio 0.75 --max-selected-per-case 8 --min-selected-per-case 1 --example-cases 8`
  - result: `8` cases, `58` conditions, final decisions `selected=30`, `deferred=20`, `dropped=8`.
- 250-case run:
  - command: `python scripts/screen_aml_conditions.py --manifest outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_condition_screening_250_v1 --threshold 0.42 --defer-ratio 0.75 --max-selected-per-case 8 --min-selected-per-case 1 --example-cases 25`
  - result: `250` cases, `1427` conditions, final decisions `selected=905`, `deferred=458`, `dropped=64`.
- Outputs:
  - gap8 report: `outputs/aml_regression_testset_v2/aml_condition_screening_gap8_v1/screening_report.md`
  - 250-case report: `outputs/aml_regression_testset_v2/aml_condition_screening_250_v1/screening_report.md`
  - compact selected/deferred records: `selected_conditions.jsonl` under each output directory.
- Verification:
  - `python -m py_compile scripts/screen_aml_conditions.py`
  - `git diff --check`
  - JSONL row/condition counts match `summary.json` for both gap8 and 250-case outputs.

### AML selected-condition dataset contract audit v1

- Added `scripts/audit_aml_condition_contract.py`.
- Scope:
  - audit `selected_conditions.jsonl` without introducing a training dataset adapter yet;
  - verify required record fields and condition fields;
  - verify slot type stability for numeric/string/span slots;
  - split train-ready records from empty-selected audit records.
- Gap8 run:
  - command: `python scripts/audit_aml_condition_contract.py --selected-jsonl outputs/aml_regression_testset_v2/aml_condition_screening_gap8_v1/selected_conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_condition_contract_gap8_v1`
  - result: `8` records, `8` train-ready, `0` empty-selected, `30` selected conditions, `20` deferred conditions, contract status `pass`.
- 250-case run:
  - command: `python scripts/audit_aml_condition_contract.py --selected-jsonl outputs/aml_regression_testset_v2/aml_condition_screening_250_v1/selected_conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_condition_contract_250_v1`
  - result: `250` records, `242` train-ready, `8` empty-selected, `905` selected conditions, `458` deferred conditions, contract status `pass`.
  - empty-selected cases: `M006210`, `006351`, `006357`, `007359`, `008449`, `008740`, `010895`, `M011255`.
- Outputs:
  - gap8 contract report: `outputs/aml_regression_testset_v2/aml_condition_contract_gap8_v1/dataset_contract.md`
  - 250-case contract report: `outputs/aml_regression_testset_v2/aml_condition_contract_250_v1/dataset_contract.md`
  - first downstream input: `outputs/aml_regression_testset_v2/aml_condition_contract_250_v1/train_ready_selected_conditions.jsonl`
  - audit-only empty records: `outputs/aml_regression_testset_v2/aml_condition_contract_250_v1/empty_selected_cases.jsonl`
- Verification:
  - `python -m py_compile scripts/audit_aml_condition_contract.py scripts/screen_aml_conditions.py`
  - `git diff --check`
  - slot type issues are `{}` on both gap8 and 250-case audits.

### AML condition batch schema v1

- Added `scripts/export_aml_condition_batch_schema.py`.
- Scope:
  - convert train-ready selected AML conditions into fixed-shape arrays;
  - do not include source/target motion tensors yet;
  - keep this as the first model-facing condition contract before wiring a training dataset.
- Input:
  - `outputs/aml_regression_testset_v2/aml_condition_contract_250_v1/train_ready_selected_conditions.jsonl`
  - records: `242`
  - selected conditions: `905`
- Command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_batch_schema.py --input-jsonl outputs/aml_regression_testset_v2/aml_condition_contract_250_v1/train_ready_selected_conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1 --max-conditions 8`
- Output:
  - `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1/condition_batch.npz`
  - `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1/condition_batch_schema.json`
  - `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1/condition_batch_index.jsonl`
  - `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1/condition_batch_summary.json`
  - `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1/condition_batch_report.md`
- Fixed array shapes:
  - `condition_mask`: `[242, 8]`
  - `family_id`: `[242, 8]`
  - `status_id`: `[242, 8]`
  - `score`: `[242, 8]`
  - `condition_weight`: `[242, 8]`
  - `span`: `[242, 8, 2]`
  - `span_norm`: `[242, 8, 4]`
  - `numeric_slots`: `[242, 8, 18]`
  - `categorical_slots`: `[242, 8, 4]`
- Smoke result:
  - real selected conditions from `condition_mask`: `905`
  - `num_selected.sum()`: `905`
  - truncated cases: `0`
  - span coverage: `1.0`
  - padding ids and padded spans are consistent.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/export_aml_condition_batch_schema.py scripts/audit_aml_condition_contract.py scripts/screen_aml_conditions.py`
  - `git diff --check`
  - note: the default `python` on this machine lacks `numpy`; use the `h2char` environment for this exporter.

### AML condition + motion batch smoke v1

- Added `scripts/export_aml_condition_motion_batch.py`.
- Scope:
  - align the fixed AML condition batch with HumanML3D `joints3d.pth`;
  - export padded joints and frame masks for the 250-case train-ready subset;
  - do not train a model and do not alter the condition schema.
- Input:
  - condition batch dir: `outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1`
  - HumanML3D joints: `/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/joints3d.pth`
- Command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_motion_batch.py --condition-batch-dir outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1 --output-dir outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1`
- Output:
  - `outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1/condition_motion_batch.npz`
  - `outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1/condition_motion_alignment.json`
  - `outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1/condition_motion_alignment.jsonl`
  - `outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1/condition_motion_report.md`
- Smoke result:
  - cases: `242`
  - joints shape: `[242, 282, 22, 3]`
  - frame mask shape: `[242, 282]`
  - valid frames: `43500`
  - source frame range: min `55`, mean `179.7521`, max `282`
  - condition real count: `905`
  - mismatched frame count: `0`
  - truncated count: `0`
  - alignment status: `pass`.
- Independent check:
  - `case_index` matches `condition_batch.npz`;
  - `source_num_frames` matches condition `num_frames`;
  - padded frames are zero and masked out.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/export_aml_condition_motion_batch.py scripts/export_aml_condition_batch_schema.py scripts/audit_aml_condition_contract.py scripts/screen_aml_conditions.py`
  - `git diff --check`

### AML condition + motion DataLoader smoke v1

- Added `pseudoedit3d/data/aml_condition_motion_dataset.py`.
- Added `scripts/smoke_aml_condition_motion_loader.py`.
- Dataset scope:
  - read `condition_batch.npz` and `condition_motion_batch.npz`;
  - expose padded HumanML3D joints, frame masks, selected-condition tensors, slot tensors, and case metadata;
  - do not instantiate a model and do not train.
- Important contract note:
  - the first default PyTorch collate attempt failed because `selected_families` is variable-length metadata;
  - added `collate_aml_condition_motion_samples` to stack tensors and keep metadata as Python lists.
- Command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/smoke_aml_condition_motion_loader.py --condition-batch-dir outputs/aml_regression_testset_v2/aml_condition_batch_schema_250_v1 --motion-batch-dir outputs/aml_regression_testset_v2/aml_condition_motion_batch_250_v1 --output-dir outputs/aml_regression_testset_v2/aml_condition_motion_loader_smoke_250_v1 --batch-size 16 --num-workers 0 --example-cases 10`
- Output:
  - `outputs/aml_regression_testset_v2/aml_condition_motion_loader_smoke_250_v1/loader_smoke.json`
  - `outputs/aml_regression_testset_v2/aml_condition_motion_loader_smoke_250_v1/loader_smoke.md`
- Smoke result:
  - status: `pass`
  - dataset length: `242`
  - checked batches: `16`
  - checked samples: `242`
  - batch size: `16`
  - condition count from masks: `905`
  - valid frame count from masks: `43500`
  - mask mismatches: `[]`
- First batch tensor shapes:
  - `joints`: `[16, 282, 22, 3]`
  - `frame_mask`: `[16, 282]`
  - `condition_mask`: `[16, 8]`
  - `condition_family_id`: `[16, 8]`
  - `condition_status_id`: `[16, 8]`
  - `condition_score`: `[16, 8]`
  - `condition_span_norm`: `[16, 8, 4]`
  - `condition_numeric_slots`: `[16, 8, 18]`
  - `condition_categorical_slots`: `[16, 8, 4]`
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/data/aml_condition_motion_dataset.py scripts/smoke_aml_condition_motion_loader.py`
  - `git diff --check`

### AML MoMask review-pack orchestration v1

- Decision:
  - pause the AML condition encoder work;
  - before model-facing conditioning, run more `AML -> AutoPrompt -> MoMask` semantic review on HumanML3D;
  - prioritize manual GT-vs-generated GIF inspection for the fixed 250-case regression set.
- Added `scripts/run_aml_momask_review_pack.py`.
- Purpose:
  - orchestrate the existing `scripts/run_momask_aml_autoprompt_probe.py`, `scripts/visualize_momask_auto_gt.py`, and `scripts/analyze_momask_probe_kinematics.py`;
  - run the 250-case set in five groups of 50;
  - support `--reuse-existing` for interrupted runs;
  - generate per-group `index.md` files and a master `review_manifest.json`.
- Dry run:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_aml_momask_review_pack.py --case-list outputs/aml_regression_testset_v2/group_01_case_ids.txt --output-root outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1_dryrun --review-name aml_review250_semantic_v1_dryrun --ext-prefix aml_review250_semantic_v1_dryrun --skip-generation --skip-visualization --skip-kinematic`
  - result: `group_01` probe summary and index generated for `50` cases without MoMask generation.
  - dry-run index: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1_dryrun/group_01/index.md`
  - dry-run master index: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1_dryrun/index.md`
- Full review command to run after confirmation:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_aml_momask_review_pack.py --case-list-glob 'outputs/aml_regression_testset_v2/group_[0-9][0-9]_case_ids.txt' --output-root outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1 --review-name aml_review250_semantic_v1 --ext-prefix aml_review250_semantic_v1 --prompt-mode coarse --max-events 8 --time-steps 10 --cond-scale 4 --gpu-id 0 --reuse-existing --frame-stride 4`
- Notes from dry-run prompt table:
  - the current semantic renderer is improved enough to justify a systematic review bundle;
  - remaining language-coverage watchpoints are still visible, e.g. arm/object-like actions, throw/catch, drink/wave, and some walking-like clips with extra leg-kick clauses.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/run_aml_momask_review_pack.py scripts/run_momask_aml_autoprompt_probe.py scripts/visualize_momask_auto_gt.py scripts/analyze_momask_probe_kinematics.py`

### AML MoMask review-pack group_01 50-case trial

- Scope:
  - ran only `group_01` from the fixed 250-case regression set;
  - generated motion with the current `AML -> AutoPrompt -> MoMask` path;
  - rendered GT-vs-generated GIFs for manual semantic coverage review.
- Command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_aml_momask_review_pack.py --case-list outputs/aml_regression_testset_v2/group_01_case_ids.txt --output-root outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1 --review-name aml_review250_semantic_v1 --ext-prefix aml_review250_semantic_v1 --prompt-mode coarse --max-events 8 --time-steps 10 --cond-scale 4 --gpu-id 0 --reuse-existing --frame-stride 4`
- Output:
  - master index: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/index.md`
  - group index: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/index.md`
  - probe summary: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/probe/summary.json`
  - GIF summary: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/gifs/summary.json`
  - GIF directory: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/gifs/`
  - kinematic sanity: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/kinematic_sanity.md`
- Result:
  - MoMask generations: `50/50`
  - rendered GIFs: `50/50`
  - kinematic rows: `50/50`
  - aggregate sanity flags: `root_path_mismatch: 12`, `vertical_amp_mismatch: 12`, `unexpected_jumpiness: 2`
- Notes:
  - this is not a final metric; the sanity flags are only a triage layer before manual GIF inspection;
  - several flagged examples are expected to distinguish `AutoPrompt semantic loss` from `MoMask realization failure` during manual review.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/run_aml_momask_review_pack.py scripts/run_momask_aml_autoprompt_probe.py scripts/visualize_momask_auto_gt.py scripts/analyze_momask_probe_kinematics.py`
  - `git diff --check`

### AML manual-review seed4 semantic patch v1

- Input manual observations from `group_01` GIF review:
  - `000189`: `walk in place` was over-rendered as left/right kick clauses.
  - `000263`: bended-knee / lunge-like motion was not captured by a semantic family.
  - `004303`: `does a lunge` was rendered as kick plus squat.
  - `000905`: AutoPrompt semantics were plausible, but MoMask generated root translation was visibly larger than GT.
- Code changes:
  - `pseudoedit3d/edit/coarse_signature.py`: hide leg-forward proxy actions when dominated by `IN_PLACE_GAIT`; add `LUNGE_CANDIDATE` from leg-forward extension plus low-body posture; hide low-level jump/gait/squat components under lunge.
  - `pseudoedit3d/edit/coarse_prompt_renderer.py`: render `LUNGE_CANDIDATE` as a lunge phrase and give it higher prompt salience than low-level leg/squat components.
  - `pseudoedit3d/edit/aml_condition_schema.py`: add required slots for `LUNGE_CANDIDATE`.
  - `scripts/analyze_momask_probe_kinematics.py`: add softer manual-review flags `root_path_scale_review` and `vertical_amp_scale_review` in addition to severe mismatch flags.
  - `scripts/run_aml_momask_review_pack.py`: hide non-probe-visible / semantically dominated canonical ids from future review index tables.
- Preview-only command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 000189,000263,004303,000905 --output-dir outputs/aml_regression_testset_v2/manual_review_seed4_prompt_patch_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix manual_review_seed4_prompt_patch_v1`
- Patched prompts:
  - `000189`: `a person walks in place`
  - `000263`: `a person does a right-leg lunge`
  - `004303`: `a person raises the left hand high, then does a right-leg lunge, then comes to a stop and stands still`
  - `000905`: unchanged semantic prompt; now caught by `root_path_scale_review` for manual MoMask-scale review.
- Output:
  - manual seed md: `outputs/aml_regression_testset_v2/manual_review_seed4_prompt_patch_v1/manual_review_seed.md`
  - manual seed json: `outputs/aml_regression_testset_v2/manual_review_seed4_prompt_patch_v1/manual_review_seed.json`
  - prompt preview: `outputs/aml_regression_testset_v2/manual_review_seed4_prompt_patch_v1/summary.json`
  - condition manifest: `outputs/aml_regression_testset_v2/manual_review_seed4_condition_manifest_patch_v1/conditions.jsonl`
  - review-threshold sanity: `outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/kinematic_sanity_review_thresholds_v2.md`
- Checks:
  - `000189` condition manifest keeps leg proxy evidence but marks those conditions `probe_visible=false`, `condition_weight=0.0`; `IN_PLACE_GAIT` stays selected.
  - `000263` and `004303` expose `LUNGE_CANDIDATE` with candidate weight `0.7` and no missing required slots.
  - `000905` has `root_path_scale_review` under the softer review threshold.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py scripts/analyze_momask_probe_kinematics.py scripts/run_aml_momask_review_pack.py scripts/run_momask_aml_autoprompt_probe.py scripts/export_aml_condition_manifest.py`
  - `git diff --check`

### AML language coverage weak-label audit v1

- Motivation:
  - manual GIF review is useful for discovering failure modes, but it should not be the main scaling mechanism;
  - this audit converts manual-review patterns into caption/AML/AutoPrompt weak labels and writes simple markdown/json/jsonl outputs.
- Added:
  - `scripts/audit_aml_language_coverage.py`
- Issue buckets:
  - `missing_composed_family`: caption suggests a geometry-recoverable composed action family missing from AML output.
  - `object_or_intent_ambiguous`: caption names object/intent semantics that skeleton geometry can usually support only as candidate/proxy evidence.
  - `prompt_priority_error`: a reasonable family exists, but lower-level wording dominates AutoPrompt.
  - `momask_realization_or_scale_review`: optional MoMask kinematic sanity flag; should not drive AML taxonomy changes alone.
- Group_01 command:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_aml_language_coverage.py --summary-json outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/probe/summary.json --kinematic-json outputs/aml_regression_testset_v2/aml_momask_review250_semantic_v1/group_01/kinematic_sanity_review_thresholds_v2.json --output-dir outputs/aml_regression_testset_v2/aml_language_coverage_group01_v1 --active-samples-per-type 12`
- Group_01 output:
  - report: `outputs/aml_regression_testset_v2/aml_language_coverage_group01_v1/coverage_report.md`
  - summary: `outputs/aml_regression_testset_v2/aml_language_coverage_group01_v1/summary.json`
  - per-case jsonl: `outputs/aml_regression_testset_v2/aml_language_coverage_group01_v1/coverage_cases.jsonl`
  - active sample list: `outputs/aml_regression_testset_v2/aml_language_coverage_group01_v1/active_sample_case_ids.txt`
- Group_01 result:
  - cases: `50`
  - cases with issues: `46`
  - issue counts: `momask_realization_or_scale_review:47`, `missing_composed_family:23`, `object_or_intent_ambiguous:15`, `prompt_priority_error:13`
  - key labels include `hand_to_head_or_phone`, `cheer_or_dance`, `locomotion_prompt_priority`, `combat_or_martial_arts`, `arm_swing_or_windmill`, and `lunge`.
  - `000189` is now automatically flagged as `locomotion_prompt_priority` in the old group_01 summary, matching the manual observation that kick wording dominated walking.
- Manual seed patch sanity:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_aml_language_coverage.py --summary-json outputs/aml_regression_testset_v2/manual_review_seed4_prompt_patch_v1/summary.json --output-dir outputs/aml_regression_testset_v2/aml_language_coverage_manual_seed4_patch_v1 --active-samples-per-type 12`
  - result: `4/4` cases without weak-label issues; covered labels are `locomotion_prompt_priority:2` and `lunge:2`.
- 250-case patched manifest:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --case-list <all five group case lists> --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v1/summary.json --output-md outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v1/summary.md --progress-every 50`
  - result: `250` valid cases, `1420` conditions, `stable:860`, `proxy:326`, `candidate:234`, `missing_required_condition_count:0`.
  - caution: `LUNGE_CANDIDATE` appears `133` times, which is likely too broad and needs targeted refinement before treating lunge coverage as solved.
- 250-case coverage audit:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_aml_language_coverage.py --condition-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v1/conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v1 --active-samples-per-type 20`
  - report: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v1/coverage_report.md`
  - summary: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v1/summary.json`
  - active sample list: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v1/active_sample_case_ids.txt`
  - result: `250` cases, `121` cases with weak-label issues, active sample count `45`.
  - issue counts: `missing_composed_family:87`, `object_or_intent_ambiguous:52`, `prompt_priority_error:23`.
  - top labels: `sit_or_stand_transition:26`, `hand_to_head_or_phone:25`, `arm_swing_or_windmill:23`, `locomotion_prompt_priority:23`, `tennis_or_ball_strike:18`.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/audit_aml_language_coverage.py`

### AML composed-family refinement after weak-label audit v1

- Motivation:
  - the first 250-case patched manifest exposed that `LUNGE_CANDIDATE` was too broad;
  - the user also raised the larger concern that manual GIF inspection and future CLIP fallback should not become the main AML scaling mechanism.
- Boundary document:
  - `docs/design/aml_clip_boundary.md`
  - rule: `geometry_recoverable` failures must be fixed in AML extraction/composition/renderer priority; future CLIP-like resolvers may only attach optional metadata for `object_or_intent_ambiguous` cases.
- Lunge refinement:
  - `pseudoedit3d/edit/coarse_signature.py`
  - changed lunge composition from broad `leg-forward + low/torso posture` to a conservative `leg-forward + local squat/low-body anchor` rule;
  - suppresses lunge when strong root translation explains the leg event or when the low posture is a long sitting-like state.
- Lunge targeted previews:
  - v1: `outputs/aml_regression_testset_v2/lunge_tighten_preview_v1/summary.json`
  - v2: `outputs/aml_regression_testset_v2/lunge_tighten_preview_v2/summary.json`
  - preserved lunge prompts for `000263` and `004303`;
  - removed obvious lunge over-triggering in `011643`, `006986`, `014448`, `M005037`, and `002950`.
- 250-case lunge check:
  - manifest: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v2/conditions.jsonl`
  - coverage: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v2/coverage_report.md`
  - `LUNGE_CANDIDATE` count reduced from `133` in patch v1 to `7` in patch v2.
  - coverage audit: `119/250` cases with weak-label issues; `missing_composed_family:89`, `object_or_intent_ambiguous:52`, `prompt_priority_error:16`.
- Sit/stand candidate addition:
  - `pseudoedit3d/edit/coarse_signature.py`: added conservative `SIT_DOWN_CANDIDATE`, `STAND_UP_CANDIDATE`, and `SIT_STAND_CYCLE_CANDIDATE` from low-body posture plus vertical transition evidence.
  - `pseudoedit3d/edit/coarse_prompt_renderer.py`: renders the new families as `sits down`, `stands up`, or `sits down, then stands back up`.
  - `pseudoedit3d/edit/aml_condition_schema.py`: adds required approximate slots for the new families.
- Sit/stand preview:
  - v1: `outputs/aml_regression_testset_v2/sitstand_preview_v1/summary.json`
  - v2: `outputs/aml_regression_testset_v2/sitstand_preview_v2/summary.json`
  - v2 restored `000263` to `a person does a right-leg lunge` after v1 over-prioritized stand-up;
  - v2 captures sit/stand evidence in examples including `007581`, `M000886`, and `M004095`, while keeping `004303` as lunge.
- 250-case sit/stand check:
  - manifest: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v3/conditions.jsonl`
  - coverage: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v3/coverage_report.md`
  - summary: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v3/summary.json`
  - condition summary: `250` valid cases, `1427` conditions, `missing_required_condition_count:0`.
  - new family counts: `STAND_UP_CANDIDATE:7`, `SIT_STAND_CYCLE_CANDIDATE:1`, `SIT_DOWN_CANDIDATE:1`.
  - `LUNGE_CANDIDATE` count is now `4`.
  - weak-label issues: `116/250` cases; `missing_composed_family:81`, `object_or_intent_ambiguous:52`, `prompt_priority_error:16`.
  - `sit_or_stand_transition` issue count reduced from `26` to `18`.
- Decision:
  - stop this rule pass here; remaining sit/stand cases are more mixed with kneeling, crawling, prone recovery, and object interactions, and should be handled as separate family design rather than ad hoc expansion.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py`

### AML proto registry and jumping-jack coverage patch

- Motivation:
  - `coarse_signature.py` had multiple hard-coded `prototype_id` sets for candidate/proxy status, cover suppression, prompt dominance, fallback removal, and primary metadata attachment.
  - This made AML extension fragile: every new composed family required editing several unrelated branches.
  - `jumping_jack` remained a weak-label miss after patch v3/v4, and several manual-review cases were still rendered as low-level jump/hand/squat clauses.
- Registry:
  - added `pseudoedit3d/edit/aml_proto_registry.json`;
  - moved high-level proto groups into JSON:
    - `semantic_family_status.unknown/candidate/proxy`;
    - `semantic_cover_suppression.sit_stand_cover/lunge_cover/climb_cover`;
    - `semantic_action_groups.hand_high/leg_forward_pose/leg_kick/emit_and_cover`;
    - `dominance.dominant_prototypes/hideable_targets/dominant_groups/hide_by_dominant`;
    - `fallback_actions.redundant`;
    - `primary_action_metadata.*`.
  - `pseudoedit3d/edit/coarse_signature.py` now loads the registry with a cached helper; algorithmic thresholds remain in code.
- Jumping-jack refinement:
  - added secondary `JUMPING_JACK` composition from repeated `BI_RAISE_SPREAD` plus vertical cycles;
  - the secondary candidate can inspect evidence already covered by primary `VERTICAL_JUMP`/`IN_PLACE_GAIT`, then semantic dominance hides the lower-level prompt clauses;
  - candidate extraction first attempts a global composed family, and falls back to windows split by strong root translation so cases such as `011643` can keep `jumping jacks` before later walking.
- Targeted preview:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,008227,005722,M005904,M001919,M008014,000263,004303 --output-dir outputs/aml_regression_testset_v2/jumpingjack_preview_v4 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix jumpingjack_preview_v4`
  - output: `outputs/aml_regression_testset_v2/jumpingjack_preview_v4/summary.json`
  - representative prompts:
    - `011643`: `a person does jumping jacks 5 times, then walks naturally for about 3.0 meters, then raises both arms`
    - `003082`: `a person does jumping jacks 5 times`
    - `007808`: `a person does jumping jacks 4 times`
    - `M001919`: `a person does jumping jacks 8 times`
    - lunge sanity preserved: `000263` remains `a person does a right-leg lunge`; `004303` remains a left-hand-high plus right-leg-lunge prompt.
- 250-case manifest:
  - manifest: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v4/conditions.jsonl`
  - summary: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v4/summary.json`
  - report: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v4/summary.md`
  - result: `250` valid cases, `1439` conditions, `missing_required_condition_count:0`.
  - key family counts: `JUMPING_JACK:20`, `LUNGE_CANDIDATE:4`, `SIT_DOWN_CANDIDATE:1`, `STAND_UP_CANDIDATE:7`, `SIT_STAND_CYCLE_CANDIDATE:1`.
- 250-case coverage audit:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_aml_language_coverage.py --condition-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_patch_v4/conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v5 --active-samples-per-type 20`
  - report: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v5/coverage_report.md`
  - summary: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v5/summary.json`
  - active samples: `outputs/aml_regression_testset_v2/aml_language_coverage_250_patch_v5/active_sample_case_ids.txt`
  - result: weak-label issue cases reduced from `103/250` in patch v4 to `99/250`; `missing_composed_family` reduced from `64` to `57`; `jumping_jack` issue count reduced from `7` to `0`; `jumping_jack` covered-label count increased from `6` to `13`.
- Residual risk:
  - `JUMPING_JACK` family count increased from `7` to `20`; this is not explosive, but it can absorb geometrically similar arm/vertical motions such as swimming-like flailing, jump-rope-like hopping, or windmill-like arm swings.
  - Treat those as future family splits (`swim_or_prone_motion`, `jump_rope`, `arm_swing_or_windmill`) rather than CLIP-only fixes.

### AML top-down family taxonomy scaffold

- Motivation:
  - the user pointed out that AML family splitting was becoming bottom-up: each manual/audit failure was patched as an isolated family;
  - the next stage should define a top-down family taxonomy first, then map WordNet/InternVid/HML3D terms and geometry detectors into that taxonomy.
- WordNet / InternVid reference:
  - read-only reference: `docs/InternVid.ipynb`
  - relevant section: `WordNet取用词汇`
  - notebook currently sketches:
    - `wn.all_synsets(wn.VERB)` for action verbs;
    - WordNet noun synsets for activity nouns;
    - synonym expansion through WordNet lemmas;
    - human-subject sentence filtering for InternVid captions using spaCy.
  - decision: use WordNet/InternVid as lexical proposal sources, not as runtime AML taxonomy or as direct family promotion.
- Added taxonomy files:
  - `pseudoedit3d/edit/aml_family_taxonomy.json`
  - `pseudoedit3d/edit/aml_family_taxonomy.py`
  - `docs/design/aml_family_taxonomy_v1.md`
- Taxonomy parent groups:
  - `ROOT_LOCOMOTION`
  - `VERTICAL_IMPULSE`
  - `BODY_LEVEL_POSTURE`
  - `GROUND_PRONE_KNEEL`
  - `UPPER_LIMB_GESTURE`
  - `LOWER_LIMB_ACTION`
  - `BILATERAL_RHYTHMIC_EXERCISE`
  - `ROTATION_SPIN`
  - `ACROBATICS_INVERSION`
  - `ACTIVITY_INTENT_PROXY`
  - `UNKNOWN_OR_FALLBACK`
- Code integration:
  - `pseudoedit3d/edit/coarse_signature.py`: semantic family descriptors now include taxonomy metadata:
    - `taxonomy_parent_id`
    - `taxonomy_parent_label`
    - `taxonomy_recoverability`
    - `taxonomy_evidence_axes`
    - `taxonomy_secondary_parent_ids`
    - `ambiguity_boundary`
  - `scripts/export_aml_condition_manifest.py`: condition records now persist the same taxonomy fields and manifest summaries include:
    - `taxonomy_parent_counts`
    - `taxonomy_recoverability_counts`
  - `scripts/audit_aml_language_coverage.py`: coverage summaries and reports now include:
    - `issue_taxonomy_parent_counts`
    - `visible_taxonomy_parent_counts`
- 250-case taxonomy manifest:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/summary.json --output-md outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/summary.md --progress-every 50`
  - result: `250` valid cases, `1439` conditions, `missing_required_condition_count:0`.
  - top taxonomy parents by condition count:
    - `ROTATION_SPIN:492`
    - `ROOT_LOCOMOTION:275`
    - `BODY_LEVEL_POSTURE:205`
    - `UPPER_LIMB_GESTURE:204`
    - `LOWER_LIMB_ACTION:107`
    - `VERTICAL_IMPULSE:101`
    - `BILATERAL_RHYTHMIC_EXERCISE:21`
- 250-case taxonomy audit:
  - command: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_aml_language_coverage.py --condition-jsonl outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/conditions.jsonl --output-dir outputs/aml_regression_testset_v2/aml_language_coverage_250_taxonomy_v1 --active-samples-per-type 20`
  - report: `outputs/aml_regression_testset_v2/aml_language_coverage_250_taxonomy_v1/coverage_report.md`
  - summary: `outputs/aml_regression_testset_v2/aml_language_coverage_250_taxonomy_v1/summary.json`
  - issue totals unchanged from patch v5 by design: `99/250` cases with issues, `151/250` clean.
  - top issue taxonomy parents:
    - `UPPER_LIMB_GESTURE:189`
    - `ROOT_LOCOMOTION:64`
    - `BODY_LEVEL_POSTURE:58`
    - `ACTIVITY_INTENT_PROXY:23`
    - `BILATERAL_RHYTHMIC_EXERCISE:16`
    - `GROUND_PRONE_KNEEL:13`
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/aml_family_taxonomy.py pseudoedit3d/edit/coarse_signature.py scripts/export_aml_condition_manifest.py scripts/audit_aml_language_coverage.py`
  - `python -m json.tool pseudoedit3d/edit/aml_family_taxonomy.json`
  - `python -m json.tool pseudoedit3d/edit/aml_proto_registry.json`
  - `git diff --check`

### WordNet cached action lexicon and Layer3 call-order note

- User requirement:
  - WordNet should not be queried every time AML extraction runs.
  - Download/build once, then read a stable JSON/YAML artifact.
- Added builder:
  - `scripts/build_wordnet_action_lexicon.py`
  - runtime policy in artifact: `offline_builder_only; AML extraction must read this JSON and must not query WordNet`
  - `nltk` is imported only inside the builder's WordNet loading function, not by AML runtime modules.
- Environment setup:
  - installed `nltk` into `h2char` with:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m pip install nltk`
  - used proxy for the WordNet corpus download:
    - `http_proxy=http://10.130.136.133:7890`
    - `https_proxy=http://10.130.136.133:7890`
  - downloaded and validated:
    - `/home/guoruoxi/nltk_data/corpora/wordnet.zip`
- Cached artifact:
  - `outputs/aml_lexicon/wordnet_action_terms_v1.json`
  - size: about `21M`
  - summary:
    - `27986` terms
    - `700` WordNet-mapped terms plus `3` curated multi-word bridge phrases
    - top mapped taxonomy parents:
      - `ROOT_LOCOMOTION:137`
      - `UPPER_LIMB_GESTURE:126`
      - `ACTIVITY_INTENT_PROXY:115`
      - `LOWER_LIMB_ACTION:73`
      - `ROTATION_SPIN:61`
      - `BODY_LEVEL_POSTURE:56`
      - `GROUND_PRONE_KNEEL:55`
      - `VERTICAL_IMPULSE:52`
      - `BILATERAL_RHYTHMIC_EXERCISE:25`
      - `ACROBATICS_INVERSION:13`
  - curated bridge phrases are included because WordNet does not reliably contain current AML multi-word targets such as `jumping jack` and `skip rope`.
- Updated source metadata:
  - `pseudoedit3d/edit/aml_family_taxonomy.json`: WordNet marked as `offline_cached_source`, with builder and cached artifact path.
  - `docs/design/aml_family_taxonomy_v1.md`: WordNet runtime policy and install/build commands.
- Added call-order note:
  - `docs/design/aml_layer3_coarse_signature_call_order.md`
  - records:
    - current artifact paths;
    - Layer3 construction route;
    - `build_coarse_action_program` function order;
    - condition manifest conversion route.
- Coarse signature structure cleanup:
  - renamed the jumping-jack-only secondary selector into parent-level `_bilateral_rhythmic_candidate_actions`;
  - current behavior is unchanged, but future jump-rope / cheer / windmill-like splits should enter through this bilateral rhythmic exercise selector rather than adding one selector per family to the main pipeline.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_wordnet_action_lexicon.py pseudoedit3d/edit/coarse_signature.py scripts/export_aml_condition_manifest.py scripts/audit_aml_language_coverage.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m json.tool outputs/aml_lexicon/wordnet_action_terms_v1.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m json.tool pseudoedit3d/edit/aml_family_taxonomy.json`
  - `git diff --check`

### AML generic motion-pattern prototype refactor

- Motivation:
  - the primary prototype assignment had become bottom-up and case-like:
    `post_jump_recovery`, `squat_events`, `JUMPING_JACK`, and `CELEBRATORY_DANCE_GESTURE` were embedded directly in `assign_seeded_prototype`;
  - the user requested a top-down family structure where action names such as jumping jack / cheer / squat are aliases or descendants, not hard-coded primary family decisions.
- Code structure:
  - `pseudoedit3d/edit/coarse_signature.py`
    - added `_motion_patterns_axis`, which computes reusable evidence patterns from Layer3 axes:
      - `coupled_locomotion`
      - `post_vertical_recovery_step`
      - `low_body_repetition`
      - `bilateral_rhythmic_coordination`
      - `bilateral_rhythmic_gesture`
    - refactored `assign_seeded_prototype` into a short ordered rule chain:
      - `_rule_low_body_repetition`
      - `_rule_bilateral_rhythmic_coordination`
      - `_rule_bilateral_rhythmic_gesture`
      - `_rule_ballistic_translation`
      - `_rule_vertical_jump`
      - `_rule_translating_gait`
      - `_rule_in_place_gait`
      - `_rule_in_place_gait_proxy`
      - `_rule_rotation`
      - `_rule_upper_body_or_subtle`
      - `_rule_bimanual_fallback`
    - new primary families:
      - `LOW_BODY_REPETITION`
      - `LOW_BODY_REPETITION_WITH_ARM_LIFT`
      - `BILATERAL_RHYTHMIC_COORDINATION`
      - `BILATERAL_RHYTHMIC_GESTURE_CANDIDATE`
    - legacy aliases are normalized through `active_family_id` / registry aliasing:
      - `SQUAT_REPETITION -> LOW_BODY_REPETITION`
      - `SQUAT_ARM_LIFT -> LOW_BODY_REPETITION_WITH_ARM_LIFT`
      - `JUMPING_JACK -> BILATERAL_RHYTHMIC_COORDINATION`
      - `CELEBRATORY_DANCE_GESTURE -> BILATERAL_RHYTHMIC_GESTURE_CANDIDATE`
  - `pseudoedit3d/edit/coarse_prompt_renderer.py`
    - renders new families using conservative motion wording:
      - bilateral rhythmic coordination
      - bilateral rhythmic upper-body gesture
      - low-body repetition
    - legacy family ids are normalized before rendering.
  - `pseudoedit3d/edit/aml_condition_schema.py`, `scripts/export_aml_condition_manifest.py`, and `scripts/audit_aml_language_coverage.py`
    - condition and audit paths normalize legacy ids to active families so fixed batch vocab should not keep old bottom-up labels.
  - `pseudoedit3d/edit/aml_proto_registry.json`
    - added centralized `legacy_aliases`;
    - runtime metadata groups now use active family ids.
  - `pseudoedit3d/edit/aml_family_taxonomy.py/json`
    - added `active_family_id`;
    - legacy aliases remain in `family_overrides`, but are removed from active parent `children`.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/generic_rule_refactor_preview_v2 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix generic_rule_refactor_preview_v2`
  - summary:
    - `outputs/aml_regression_testset_v2/generic_rule_refactor_preview_v2/summary.json`
  - condition manifest:
    - `outputs/aml_regression_testset_v2/generic_rule_refactor_preview_v2/conditions.jsonl`
    - `outputs/aml_regression_testset_v2/generic_rule_refactor_preview_v2/conditions_summary.json`
  - observed canonical ids contain no legacy hits among:
    - `JUMPING_JACK`
    - `SQUAT_REPETITION`
    - `SQUAT_ARM_LIFT`
    - `CELEBRATORY_DANCE_GESTURE`
  - representative prompts:
    - `011643`: `a person repeatedly coordinates both arms with vertical body motion 5 times, then walks naturally for about 3.0 meters, then raises both arms`
    - `003082`: `a person repeatedly coordinates both arms with vertical body motion 5 times`
    - `000263`: `a person does a right-leg lunge`
    - `004303`: `a person raises the left hand high, then does a right-leg lunge, then comes to a stop and stands still`
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py pseudoedit3d/edit/aml_family_taxonomy.py scripts/audit_aml_language_coverage.py scripts/export_aml_condition_manifest.py scripts/build_wordnet_action_lexicon.py`
  - `python -m json.tool pseudoedit3d/edit/aml_proto_registry.json`
  - `python -m json.tool pseudoedit3d/edit/aml_family_taxonomy.json`
  - `python -m json.tool outputs/aml_lexicon/wordnet_action_terms_v1.json`
  - `git diff --check`

### AML primary seeded-family spec refactor

- Motivation:
  - after the generic family pass, `assign_seeded_prototype` still depended on a Python `_rule_*` chain;
  - the user pointed out that signature assignment was still bottom-up and difficult to extend;
  - this pass moves the primary seeded-family match order and threshold combinations into a JSON family spec file.
- Added:
  - `pseudoedit3d/edit/aml_family_specs.json`
    - stores `primary_seeded_prototypes` in priority order;
    - supports `all` / `any` / `not`, field-path predicates, literal/path/template outputs, simple confidence expressions, and case-based outputs;
    - primary families remain top-down IDs such as `BILATERAL_RHYTHMIC_COORDINATION`, not legacy action labels such as `JUMPING_JACK`.
- Changed:
  - `pseudoedit3d/edit/coarse_signature.py`
    - added cached `_family_specs` / `_primary_seeded_family_specs`;
    - added generic matcher helpers `_condition_matches`, `_spec_matches`, `_resolve_spec_value`, `_prototype_from_spec`, and `_match_seeded_family_spec`;
    - removed the primary `_rule_low_body_repetition`, `_rule_bilateral_rhythmic_coordination`, `_rule_bilateral_rhythmic_gesture`, `_rule_ballistic_translation`, `_rule_vertical_jump`, `_rule_translating_gait`, `_rule_in_place_gait`, `_rule_in_place_gait_proxy`, and `_rule_rotation` chain;
    - `assign_seeded_prototype` now routes:
      `signature/events -> _prototype_context -> _match_seeded_family_spec -> upper-body/subtle fallback -> bimanual fallback -> EVENT_SEQUENCE`;
    - `_motion_patterns_axis` now keeps reusable evidence fields instead of final `matched` booleans for low-body/bilateral-rhythmic patterns.
- Documentation:
  - `docs/design/aml_layer3_coarse_signature_call_order.md` now records the spec-driven call order.
  - `docs/design/aml_coarse_signature_pipeline.md` lists `aml_family_specs.json` as an active main module.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix spec_primary_refactor_preview_v1`
  - output:
    - `outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1/summary.json`
  - result:
    - all 8 prompts match `generic_rule_refactor_preview_v2`;
    - no visible prompt drift in this smoke set.
- Condition manifest:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --summary-json outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1/summary.json --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1/conditions_summary.json --output-md outputs/aml_regression_testset_v2/spec_primary_refactor_preview_v1/conditions_summary.md --top-n 20`
  - result:
    - cases: `8`
    - conditions: `33`
    - missing required conditions: `0`
    - legacy family hits among `JUMPING_JACK`, `SQUAT_REPETITION`, `SQUAT_ARM_LIFT`, `CELEBRATORY_DANCE_GESTURE`: `0`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_family_specs.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py pseudoedit3d/edit/aml_family_taxonomy.py scripts/export_aml_condition_manifest.py scripts/audit_aml_language_coverage.py`
  - `git diff --check`

### AML pattern tree primary matcher refactor

- Motivation:
  - the flat `aml_family_specs.json` removed hard-coded Python rule chains, but it was still a flat priority list;
  - the user requested a bottom-up pattern-cluster tree with a WordNet-like parent/child structure, rather than case-by-case family rules.
- Added:
  - `pseudoedit3d/edit/aml_pattern_tree.json`
    - root/abstract/primary nodes:
      - `MOTION_PATTERN`
      - `WHOLE_BODY_PATTERN`
      - `LIMB_GESTURE_PATTERN`
      - `LOCOMOTION_PATTERN`
      - `VERTICAL_IMPULSE_PATTERN`
      - `BODY_LEVEL_PATTERN`
      - `ROTATION_PATTERN`
      - `BILATERAL_RHYTHMIC_PATTERN`
    - primary leaves include:
      - `TRANSLATING_GAIT_PATTERN`
      - `IN_PLACE_GAIT_PATTERN`
      - `BALLISTIC_TRANSLATION_PATTERN`
      - `REPEATED_VERTICAL_JUMP_PATTERN`
      - `LOW_BODY_REPETITION_PATTERN`
      - `BILATERAL_ARM_LEG_VERTICAL_COORDINATION_PATTERN`
      - `BILATERAL_RHYTHMIC_GESTURE_PATTERN`
    - lexical aliases such as `jumping_jack`, `squat`, `cheer`, and `dance` are metadata, not primary detector ids.
  - `pseudoedit3d/edit/aml_pattern_tree.py`
    - generic tree loader;
    - `PatternMatch`;
    - condition matcher;
    - output resolver;
    - `match_pattern_tree`;
    - `select_primary_pattern_match`.
- Changed:
  - `pseudoedit3d/edit/coarse_signature.py`
    - `assign_seeded_prototype` now routes:
      `signature/events -> _prototype_context -> _select_seeded_pattern_prototype -> upper-body/subtle fallback -> bimanual fallback -> EVENT_SEQUENCE`;
    - primary prototypes now include:
      - `pattern_node_id`
      - `pattern_path`
      - `pattern_taxonomy_parent_id`
      - `pattern_tree_matches`
  - `pseudoedit3d/edit/legacy/aml_family_specs.json`
    - previous flat spec moved to legacy for migration reference only.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/pattern_tree_primary_preview_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix pattern_tree_primary_preview_v1`
  - output:
    - `outputs/aml_regression_testset_v2/pattern_tree_primary_preview_v1/summary.json`
  - result:
    - all 8 prompts match `spec_primary_refactor_preview_v1`;
    - representative pattern paths:
      - `011643`: `MOTION_PATTERN/WHOLE_BODY_PATTERN/LOCOMOTION_PATTERN/TRANSLATING_GAIT_PATTERN`
      - `003082`: `MOTION_PATTERN/WHOLE_BODY_PATTERN/VERTICAL_IMPULSE_PATTERN/REPEATED_VERTICAL_JUMP_PATTERN`
      - `006986`: `MOTION_PATTERN/WHOLE_BODY_PATTERN/BODY_LEVEL_PATTERN/LOW_BODY_REPETITION_WITH_ARM_LIFT_PATTERN`
  - observed multi-match examples:
    - `011643`: `TRANSLATING_GAIT_PATTERN`, `IN_PLACE_GAIT_PATTERN`
    - `006986`: `SINGLE_VERTICAL_JUMP_PATTERN`, `LOW_BODY_REPETITION_WITH_ARM_LIFT_PATTERN`
    - this is expected; `primary_selection_order` chooses the current primary while preserving match evidence for audit.
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/aml_pattern_tree.py pseudoedit3d/edit/coarse_signature.py`

### AML pattern tree event-proxy metadata refactor

- Motivation:
  - primary prototype matching was already tree-driven, but `_semantic_candidate_actions` still owned a local hard-coded `semantic_map`;
  - this made residual/proxy candidates harder to extend and harder to audit as a tree.
- Changed:
  - `pseudoedit3d/edit/aml_pattern_tree.json`
    - added abstract taxonomy branches:
      - `UPPER_LIMB_GESTURE_PATTERN`
      - `LOWER_LIMB_ACTION_PATTERN`
      - `GROUND_PRONE_KNEEL_PATTERN`
      - `ACROBATICS_PATTERN`
      - `STATE_FALLBACK_PATTERN`
    - added `event_proxy` leaves for Layer-3 event pairs such as:
      - `LEFT_ARM_POSTURE/LA_HAND_HIGH -> LEFT_HAND_RAISED_HIGH_PATTERN`
      - `RIGHT_ARM_POSTURE/RA_HAND_HIGH -> RIGHT_HAND_RAISED_HIGH_PATTERN`
      - `WHOLE_BODY_POSTURE/WB_SQUAT_HOLD -> SQUAT_HOLD_PATTERN`
      - `LEFT_LEG_ACTION/LL_KICK_FORWARD -> LEFT_LEG_KICK_FORWARD_PATTERN`
      - `RIGHT_LEG_ACTION/RL_KICK_FORWARD -> RIGHT_LEG_KICK_FORWARD_PATTERN`
      - `WHOLE_BODY_PATH/ROOT_CIRCULAR_PATH -> CIRCULAR_WALK_PATH_PATTERN`
      - `WHOLE_BODY_CLIMB/CLIMB_UP_OVER_PROXY -> CLIMB_UP_OVER_PROXY_PATTERN`
      - `WHOLE_BODY_ACROBATICS/* -> CARTWHEEL_CANDIDATE_PATTERN` / `INVERTED_ACROBATICS_CANDIDATE_PATTERN`
    - added `composed_candidate` leaves for procedural temporal candidates:
      - `LUNGE_CANDIDATE_PATTERN`
      - `SIT_DOWN_CANDIDATE_PATTERN`
      - `STAND_UP_CANDIDATE_PATTERN`
      - `SIT_STAND_CYCLE_CANDIDATE_PATTERN`
      - `ACROBATIC_SEQUENCE_CANDIDATE_PATTERN`
      - `DANCE_LEG_POSE_CANDIDATE_PATTERN`
      - fallback state leaves such as `TERMINAL_STILL_PATTERN`.
  - `pseudoedit3d/edit/aml_pattern_tree.py`
    - added `event_proxy_map`, `event_proxy_for_event`, `event_proxy_action_fields`;
    - added `family_pattern_nodes`, `pattern_node_for_family`, `action_pattern_metadata_for_family`, and `action_pattern_metadata_for_node`.
  - `pseudoedit3d/edit/coarse_signature.py`
    - removed the local `semantic_map`;
    - `_semantic_candidate_actions` now gets event-proxy fields from `aml_pattern_tree.py`;
    - `_attach_action_metadata` now fills `pattern_node_id`, `pattern_path`, and `pattern_taxonomy_parent_id` for every visible action whose family is represented in the pattern tree;
    - canonical `semantic_family` and `slots` records preserve the same pattern metadata for md/json review.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix pattern_tree_event_proxy_preview_v1`
  - output:
    - `outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/summary.json`
  - result:
    - prompt changes vs `pattern_tree_primary_preview_v1`: `0`;
    - visible actions missing `pattern_node_id`: `0`;
    - examples:
      - `004303`: `LEFT_HAND_RAISED_HIGH_PATTERN`, `LUNGE_CANDIDATE_PATTERN`, `TERMINAL_STILL_PATTERN`
      - `011643`: `BILATERAL_ARM_LEG_VERTICAL_COORDINATION_PATTERN`, `TRANSLATING_GAIT_PATTERN`
- Condition manifest:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --summary-json outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/summary.json --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/conditions_summary.json --output-md outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/conditions_summary.md --top-n 20`
  - result:
    - cases: `8`
    - conditions: `33`
    - missing required conditions: `0`
    - status counts: `stable=14`, `proxy=14`, `candidate=5`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/aml_pattern_tree.py pseudoedit3d/edit/coarse_signature.py`
  - `git diff --check`

### AML pattern tree composed matcher refactor

- Motivation:
  - after the event-proxy refactor, sit/stand and lunge still chose their final family id directly inside `_semantic_candidate_actions`;
  - the temporal evidence still needs procedural pairing, but the family decision should be represented as pattern-tree nodes.
- Changed:
  - `pseudoedit3d/edit/aml_pattern_tree.json`
    - added `match` and `outputs` for:
      - `LUNGE_CANDIDATE_PATTERN`
      - `SIT_DOWN_CANDIDATE_PATTERN`
      - `STAND_UP_CANDIDATE_PATTERN`
      - `SIT_STAND_CYCLE_CANDIDATE_PATTERN`
    - added `composed_selection_order`, with sit/stand cycle before one-way sit/stand and lunge.
  - `pseudoedit3d/edit/aml_pattern_tree.py`
    - added `composed_selection_order`;
    - added `select_composed_pattern_match`.
  - `pseudoedit3d/edit/coarse_signature.py`
    - added local `make_composed_action`;
    - sit/stand now builds a context like `low_body_vertical_transition` and routes it through `select_composed_pattern_match`;
    - lunge now builds a context like `low_body_leg_forward_pair` and routes it through `select_composed_pattern_match`;
    - low-level pairing, overlap, gap, and near-translation checks stay in Python for now because they depend on event spans and temporal assignment.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix pattern_tree_composed_preview_v1`
  - output:
    - `outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/summary.json`
  - result:
    - prompt changes vs `pattern_tree_event_proxy_preview_v1`: `0`;
    - visible actions missing `pattern_node_id`: `0`;
    - examples:
      - `000263`: `LUNGE_CANDIDATE`, `right_leg_lunge_candidate`, `LUNGE_CANDIDATE_PATTERN`
      - `004303`: `LUNGE_CANDIDATE`, `right_leg_lunge_candidate`, `LUNGE_CANDIDATE_PATTERN`
      - `006986`: `STAND_UP_CANDIDATE`, `low_to_up`, `STAND_UP_CANDIDATE_PATTERN`
      - `007581`: `STAND_UP_CANDIDATE`, `low_to_up`, `STAND_UP_CANDIDATE_PATTERN`
- Condition manifest:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --summary-json outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/summary.json --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/conditions_summary.json --output-md outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/conditions_summary.md --top-n 20`
  - result:
    - cases: `8`
    - conditions: `33`
    - missing required conditions: `0`
    - status counts: `stable=14`, `proxy=14`, `candidate=5`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/aml_pattern_tree.py pseudoedit3d/edit/coarse_signature.py`
  - `git diff --check`

### AML registry consistency refactor

- Motivation:
  - `coarse_signature.py`, renderer, condition schema, coverage audit, and WordNet builder still carried several Python-local prototype/pattern lists after the tree matcher landed.
  - This made extension brittle: adding a family required edits across Python files instead of tree/registry JSON.
- Changed:
  - added `pseudoedit3d/edit/aml_proto_registry.py` as the shared runtime registry reader;
  - moved condition required approx slots into `pseudoedit3d/edit/aml_proto_registry.json` under `condition_schema`;
  - moved renderer clause/salience and probe aliases into `pseudoedit3d/edit/aml_proto_registry.json`;
  - moved semantic action emitters, cover/suppression groups, fallback entrypoints, and primary cover modes into `pseudoedit3d/edit/aml_proto_registry.json`;
  - moved language coverage weak-label specs into `pseudoedit3d/edit/aml_language_coverage_specs.json`;
  - moved WordNet builder seed/regex/family-term config into `pseudoedit3d/edit/aml_wordnet_lexicon_config.json`;
  - extended `pseudoedit3d/edit/aml_pattern_tree.json` so upper-body fallback, subtle fallback, bimanual fallback, event sequence, acrobatic sequence, dance-like leg pose, and terminal still are tree/event-proxy nodes rather than local prototype switches.
- Preview:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_momask_aml_autoprompt_probe.py --case-ids 011643,003082,007808,M001919,000263,004303,006986,007581 --output-dir outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1 --skip-generation --prompt-mode coarse --max-events 8 --ext-prefix pattern_tree_consistency_refactor_preview_v1`
  - output:
    - `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/summary.json`
  - regression vs `pattern_tree_composed_preview_v1`:
    - prompt changes: `0`
    - probe alias changes: `0`
    - canonical id changes: `0`
- Condition manifest:
  - `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/conditions.jsonl`
  - `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/conditions_summary.md`
- Language coverage audit:
  - `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/language_coverage_audit/coverage_report.md`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `python -m json.tool pseudoedit3d/edit/aml_proto_registry.json`
  - `python -m json.tool pseudoedit3d/edit/aml_language_coverage_specs.json`
  - `python -m json.tool pseudoedit3d/edit/aml_wordnet_lexicon_config.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/aml_pattern_tree.py pseudoedit3d/edit/aml_proto_registry.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py scripts/audit_aml_language_coverage.py scripts/build_wordnet_action_lexicon.py scripts/export_aml_condition_manifest.py scripts/run_momask_aml_autoprompt_probe.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_wordnet_action_lexicon.py --output /tmp/wordnet_action_terms_config_smoke.json --no-include-nouns --max-sample-synsets 1`

### AML tree cleanup plus geometry sidecar refresh

- Motivation:
  - keep the AML tree as the mainline condition schema;
  - keep geometry clustering as a diagnostic sidecar instead of replacing AML;
  - remove Python-local action lists where the same behavior can be represented by tree/registry nodes.
- Changed:
  - `pseudoedit3d/edit/coarse_pattern_evidence.py`
    - residual gait/turn and residual proxy evidence now routes through tree/registry groups;
    - gait-like leg swing events are emitted as hidden zero-weight proxies so sidecar provenance is preserved without polluting prompts;
    - low-body repetition variables were renamed away from `squat`-specific function names.
  - `pseudoedit3d/edit/geometry_sidecar.py`
    - separates true unmapped events from covered-context geometry;
    - sidecar summary schema is `aml_geometry_sidecar_summary_v2`.
  - `scripts/export_aml_geometry_sidecar.py`
    - exports Markdown/JSON summaries with separate context-only and unable-to-name sections.
  - `pseudoedit3d/edit/aml_pattern_tree.json`
    - added generic event proxy nodes for low-body hold, torso periodic motion, left/right arm periodic gestures.
  - `pseudoedit3d/edit/aml_proto_registry.json`
    - added required approx slots, prompt renderer specs, and probe aliases for the generic proxy families;
    - registry consistency check now has no required family missing alias/prompt config.
- Refreshed 250-case condition manifest:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/tree_cleanup_step5_proxy_review250_v3/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/tree_cleanup_step5_proxy_review250_v3/summary.json --output-md outputs/aml_regression_testset_v2/tree_cleanup_step5_proxy_review250_v3/summary.md --top-n 80 --progress-every 50`
  - result:
    - cases: `250`
    - conditions: `2302`
    - missing required conditions: `0`
    - zero-weight conditions: `587`
    - status counts: `proxy=1287`, `candidate=780`, `stable=235`
- Refreshed 250-case geometry sidecar:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_geometry_sidecar.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --output-jsonl outputs/aml_regression_testset_v2/geometry_sidecar_review250_v4/geometry_sidecar.jsonl --output-summary-json outputs/aml_regression_testset_v2/geometry_sidecar_review250_v4/summary.json --output-md outputs/aml_regression_testset_v2/geometry_sidecar_review250_v4/summary.md --top-n 80 --progress-every 50`
  - result:
    - cases: `250`
    - geometry events: `6473`
    - true unable-to-name clusters: `27`
    - context-only geometry clusters: `48`
    - stable geometry mappings: `1`
    - one-to-many geometry clusters: `55`
  - top remaining unable-to-name clusters:
    - `BIMANUAL_PERIODIC/BI_RAISE`: `56`, share `0.4516`
    - `BIMANUAL_PERIODIC/BI_SPREAD`: `79`, share `0.4293`
    - `WHOLE_BODY_LOCOMOTION/LOCO_MIXED_MEDIUM`: `3`, share `0.4286`
    - `WHOLE_BODY_LOCOMOTION/LOCO_TURN_LEFT_SMALL`: `5`, share `0.3571`
    - `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_SLOW`: `12`, share `0.3333`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_proto_registry.json`
  - `python -m json.tool pseudoedit3d/edit/aml_family_taxonomy.json`
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_pattern_evidence.py pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/geometry_sidecar.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py scripts/export_aml_geometry_sidecar.py scripts/export_aml_condition_manifest.py`
  - residual grep for old Python-local selector names returned no hits.

### AML generic proxy coverage pass

- Motivation:
  - after the first sidecar cleanup, the largest true unable-to-name clusters were generic bimanual arm raise/spread, locomotion-coupled arm periodic events, small turn locomotion fragments, and residual vertical body-height events;
  - these should be represented as conservative proxy families instead of activity names.
- Changed:
  - `pseudoedit3d/edit/aml_family_taxonomy.json`
    - added `BIMANUAL_ARM_RAISE_SPREAD_PROXY` under `UPPER_LIMB_GESTURE`;
    - added `WHOLE_BODY_VERTICAL_MOTION_PROXY` under `VERTICAL_IMPULSE`.
  - `pseudoedit3d/edit/aml_proto_registry.json`
    - registered the new proxy families in `semantic_family_status.proxy`;
    - added required approx slots, probe aliases, and prompt renderer clauses;
    - included the new proxies in aggregate/emit/skip residual groups.
  - `pseudoedit3d/edit/aml_pattern_tree.json`
    - added event proxy nodes for `BI_RAISE`, `BI_SPREAD`, `BI_RAISE_SPREAD`;
    - added event proxy nodes for `LA_REPEAT_LOCO`, `LA_REPEAT_ALT_LOCO`, `RA_REPEAT_LOCO`, `RA_REPEAT_ALT_LOCO`;
    - added event proxy nodes for `WB_VERT_UP`, `WB_VERT_DOWN`, `WB_VERT_CYCLE`, `WB_VERT_REP`, `WB_VERT_REP_ALT`.
  - `pseudoedit3d/edit/coarse_pattern_evidence.py`
    - residual `LOCO_TURN_*` events without a nearby rotation driver now emit neutral `TURN_SEGMENT` sparse actions.
- Refreshed 250-case condition manifest:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_condition_manifest.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --max-residual-events 8 --output-jsonl outputs/aml_regression_testset_v2/tree_cleanup_step9_generic_proxy_review250_v1/conditions.jsonl --output-summary-json outputs/aml_regression_testset_v2/tree_cleanup_step9_generic_proxy_review250_v1/summary.json --output-md outputs/aml_regression_testset_v2/tree_cleanup_step9_generic_proxy_review250_v1/summary.md --top-n 80 --progress-every 50`
  - result:
    - cases: `250`
    - conditions: `2635`
    - missing required conditions: `0`
    - zero-weight conditions: `570`
    - status counts: `proxy=1648`, `candidate=752`, `stable=235`
    - new generic proxy counts:
      - `BIMANUAL_ARM_RAISE_SPREAD_PROXY`: `131`
      - `WHOLE_BODY_VERTICAL_MOTION_PROXY`: `127`
- Refreshed 250-case geometry sidecar:
  - command:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/export_aml_geometry_sidecar.py --case-list outputs/aml_regression_testset_v2/semantic_status_250_v1/case_ids.txt --output-jsonl outputs/aml_regression_testset_v2/geometry_sidecar_step9_generic_proxy_review250_v1/geometry_sidecar.jsonl --output-summary-json outputs/aml_regression_testset_v2/geometry_sidecar_step9_generic_proxy_review250_v1/summary.json --output-md outputs/aml_regression_testset_v2/geometry_sidecar_step9_generic_proxy_review250_v1/summary.md --top-n 80 --progress-every 50`
  - result:
    - cases: `250`
    - geometry events: `6473`
    - true unable-to-name clusters: `15`
    - context-only geometry clusters: `48`
  - true unable-to-name cluster count progression:
    - step6 bimanual proxy: `25`
    - step7 loco/turn/arm proxy: `18`
    - step8 vertical proxy: `16`
    - step9 generic proxy: `15`
  - remaining top unable clusters are dominated by weak/mixed root drift:
    - `WHOLE_BODY_LOCOMOTION/LOCO_MIXED_MEDIUM`: `3`, share `0.4286`
    - `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_SLOW`: `12`, share `0.3333`
    - `WHOLE_BODY_LOCOMOTION/LOCO_MIXED_SLOW`: `3`, share `0.3000`
    - `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_MEDIUM`: `1`, share `0.2500`
- Verification:
  - `python -m json.tool pseudoedit3d/edit/aml_pattern_tree.json`
  - `python -m json.tool pseudoedit3d/edit/aml_proto_registry.json`
  - `python -m json.tool pseudoedit3d/edit/aml_family_taxonomy.json`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile pseudoedit3d/edit/coarse_pattern_evidence.py pseudoedit3d/edit/coarse_signature.py pseudoedit3d/edit/geometry_sidecar.py pseudoedit3d/edit/coarse_prompt_renderer.py pseudoedit3d/edit/aml_condition_schema.py pseudoedit3d/edit/aml_pattern_tree.py scripts/export_aml_geometry_sidecar.py scripts/export_aml_condition_manifest.py`

### Motion-corpus pattern-tree mainline switch

- Decision:
  - stop treating the hand-built AML tree as the source of truth for future
    pattern structure;
  - use HumanML3D as both a motion corpus and a text corpus;
  - induce structure from `motion cluster + Layer3 event-BPE`;
  - use `text-BPE + caption aliases + WordNet` only to name and audit
    motion-derived nodes.
- New docs:
  - `docs/design/motion_corpus_pattern_tree_mainline.md`
  - `docs/design/motion_cluster_bpe_tree_induction.md`
  - `docs/design/text_bpe_wordnet_naming_layer.md`
- Legacy cleanup:
  - moved `docs/design/motion_bpe_baseline.md` to `legacy/motion_bpe_baseline/docs/design/motion_bpe_baseline.md`;
  - moved `scripts/learn_motion_bpe.py` to `legacy/motion_bpe_baseline/scripts/learn_motion_bpe.py`;
  - recorded the move in `legacy/README.md`.
- New offline proposal script:
  - `scripts/propose_motion_pattern_tree_candidates.py`
  - input:
    - `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/bpe_phrase_to_pattern_tree_candidates.json`
    - `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/bpe_motif_audit.json`
  - output:
    - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/summary.json`
    - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidates.json`
    - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidate_report.md`
- First proposal result:
  - input records: `29228`
  - input BPE stable candidates: `8`
  - proposed offline candidate nodes: `7`
  - important candidate families:
    - torso hunch + vertical rise, language name evidence: `sit_down`
    - bimanual raise-spread, language name evidence: `jumping_jack`
    - low-body hold + squat hold, language name evidence: `sit_down`
    - bimanual hands-close / near-far arm cycles, language evidence: `cheer_dance`, kept diagnostic
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/propose_motion_pattern_tree_candidates.py scripts/audit_hml3d_layer3_event_bpe.py`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidates.json`

### Motion-BPE motif tier audit

- Motivation:
  - the first motion-tree proposal had only `7` candidate nodes because it
    intentionally used only `8` stable caption-alias motifs;
  - this could be misread as the full-corpus scan discovering only seven
    structures, so the proposal script now also emits all `256` learned BPE
    motifs by tier.
- Updated:
  - `scripts/propose_motion_pattern_tree_candidates.py`
    - added `motion_bpe_motif_tiers.json`
    - added `motion_bpe_motif_tiers.md`
    - preserved the strict `motion_pattern_tree_candidates.json` output
- Output:
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json`
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.md`
- Tier result:
  - total motifs: `256`
  - `named_motion_candidate`: `8`
  - `motion_stable_unnamed`: `25`
  - `legacy_aligned_diagnostic`: `133`
  - `language_weak_diagnostic`: `0`
  - `generic_or_low_purity`: `90`
- Interpretation:
  - the corpus has many stable motion motifs, but only a small fraction have
    clean caption-alias names under the current sidecar;
  - the next useful work is to inspect the `25` `motion_stable_unnamed` motifs
    and decide whether they become unnamed structural tree nodes, need better
    text-BPE/WordNet names, or are old-tree artifacts such as over-broad
    kick/hand-high patterns.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/propose_motion_pattern_tree_candidates.py`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/summary.json`

### Candidate-node manual review and language naming layer

- User reviewed the first `7` offline candidate nodes from:
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidate_report.md`
- Manual review artifact:
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/manual_review_v1.json`
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/manual_review_v1.md`
- Manual decisions:
  - `motion_node_0001`: reframe as stand-up / rise-from-low-or-seated transition evidence; `sit_down` is implied by starting pose, not the vertical-rise event itself.
  - `motion_node_0002`: downgrade to jumping-jack upper-body component.
  - `motion_node_0003`: split; current low-body hold / squat-hold family mixes sit/squat/kneel-like states.
  - `motion_node_0004`: downgrade to reusable component; can appear in multiple patterns.
  - `motion_node_0005`: keep diagnostic; broad bilateral near-far arm motion.
  - `motion_node_0006`: promote candidate as the closest complete jumping-jack composite because bimanual raise-spread co-occurs with vertical body motion.
  - `motion_node_0007`: downgrade to upper-body hands-high component.
- New language naming script:
  - `scripts/build_text_bpe_wordnet_naming_layer.py`
- Naming layer output:
  - `outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/summary.json`
  - `outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json`
  - `outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.md`
- Naming layer summary:
  - motion nodes: `7`
  - caption cases from BPE sequence rows: `28823`
  - phrase vocab size: `5465`
  - alias clusters: `12`
  - WordNet terms: `27986`
- Important observation:
  - language evidence can strongly name `motion_node_0002` and `motion_node_0007` as `jumping_jack`, but manual motion review correctly downgrades them to components;
  - this confirms the new policy that language names motion-derived structure but does not decide promotion.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_text_bpe_wordnet_naming_layer.py scripts/propose_motion_pattern_tree_candidates.py`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/manual_review_v1.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json`

### Promoted motion-tree draft v1

- Motivation:
  - start iterating from reviewed motion candidates into a draft tree without
    adding case-by-case logic;
  - consume artifacts instead of hard-coding action names or node ids.
- New script:
  - `scripts/build_promoted_motion_tree_draft.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_pattern_tree_candidates.json`
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/manual_review_v1.json`
  - `outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json`
- Outputs:
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/summary.json`
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json`
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.md`
- Draft result:
  - candidate nodes: `7`
  - reviewed nodes: `7`
  - promoted candidates: `1`
  - structural components: `4`
  - split required: `1`
  - diagnostic only: `1`
- Case-by-case guard:
  - builder uses manual `decision` and `recommended_structural_role` as typed
    disposition overlay;
  - builder uses motion evidence, source motifs, and geometry overlap for
    structural links;
  - builder does not use text labels or WordNet terms to create structure;
  - language label conflicts are emitted only in `naming_conflicts`.
- Important fix after risk audit:
  - initial component-link logic allowed shared strong language labels as a
    link signal;
  - it was corrected so component links now require motion-geometry overlap
    only;
  - after the fix, the only component link is geometry-based:
    `motion_node_0006 -> motion_node_0002` with Jaccard `0.5`.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_promoted_motion_tree_draft.py scripts/build_text_bpe_wordnet_naming_layer.py scripts/propose_motion_pattern_tree_candidates.py`
  - `python -m json.tool outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json`
  - linted structural fields for common language/action words; no hits in status, structural role, motion family key, required geometry clusters, component relationship, or split method.

### Motion split planner v1

- Motivation:
  - continue from the reviewed/promoted draft without adding case-by-case action logic;
  - handle nodes marked `split_required` by mining motion context axes from the full HML3D Layer3 event-BPE sequences;
  - keep text labels and WordNet evidence as diagnostics only.
- New script:
  - `scripts/plan_motion_node_splits.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json`
  - `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl`
  - `outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json`
- Outputs:
  - `outputs/aml_regression_testset_v2/motion_split_planner_v1/summary.json`
  - `outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.json`
  - `outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_axis_summary.json`
  - `outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.md`
- Result for current split node:
  - `motion_node_0003` / source motif `<M0111>` has `169` occurrences in `169` unique cases;
  - exact context signatures are too sparse to become stable groups directly;
  - coarse motion-context axis groups are usable and produce `21` candidate axes;
  - top split axes include overlap/arm, after/arm, before/arm, overlap/locomotion, after/leg, overlap/leg, after/vertical, and overlap/other-context groups.
- Current interpretation:
  - `motion_node_0003` should not be promoted as one named action node;
  - it is better treated as a low-body/torso component that needs split by surrounding motion context;
  - the compact JSON intentionally excludes language diagnostics from structural fields.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/plan_motion_node_splits.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/plan_motion_node_splits.py --draft outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json --bpe-sequences outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl --naming-layer outputs/aml_regression_testset_v2/text_bpe_wordnet_naming_layer_v1/text_bpe_wordnet_naming_layer.json --output-dir outputs/aml_regression_testset_v2/motion_split_planner_v1`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_axis_summary.json`

### Motion split proposals v1

- Motivation:
  - convert generic motion-context split axes into child-node candidates for
    human review;
  - keep this as an offline proposal layer, not a runtime AML tree mutation;
  - avoid case-by-case action names by selecting axes from body-channel
    structure (`relation`, `context_bucket`, `cluster_id`, support).
- New script:
  - `scripts/build_motion_split_proposals.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json`
  - `outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.json`
- Outputs:
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/summary.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.md`
- Result:
  - source split plan count: `1`
  - proposal count: `1`
  - split child candidates: `18`
  - promotion queue candidates: `13`
  - readiness counts: `review_for_promotion=6`, `review_as_minor_split=7`, `low_support_diagnostic=5`
  - selected axes for `motion_node_0003`: overlap/vertical, overlap/leg,
    overlap/locomotion, overlap/other, overlap/state, overlap/rotation.
- Current interpretation:
  - `motion_node_0003` remains a parent component;
  - vertical and leg overlap children are the strongest review targets;
  - locomotion/other/state children are useful minor or context splits;
  - low-support rotation and acrobatics-derived children stay diagnostic unless
    future full-corpus evidence strengthens them.
- Guardrails:
  - arm and torso context axes are deferred as modifiers for this low-body
    source node;
  - compact promotion queue excludes caption aliases, text labels, phrases, and
    WordNet fields;
  - full proposal JSON keeps caption alias diagnostics only under diagnostics.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_motion_split_proposals.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_motion_split_proposals.py --draft outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json --split-plan outputs/aml_regression_testset_v2/motion_split_planner_v1/motion_split_plan.json --output-dir outputs/aml_regression_testset_v2/motion_split_proposals_v1`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue.json`
  - linted `motion_split_promotion_queue.json` for caption/label/phrase/WordNet/language keys; no hits.

### Motion split review artifacts v1

- Motivation:
  - make split-child review easier than reading raw JSON;
  - include example HML3D captions in human-facing review tables so manual
    screening can use the original text context;
  - persist the current manual decision that the six `review_for_promotion`
    candidates are acceptable promotion candidates;
  - keep the source clear: these candidates come from the full HML3D
    Layer3/event-BPE audit, not from the 250-case MoMask/GIF review subset.
- New script:
  - `scripts/render_motion_split_review_artifacts.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue.json`
- Outputs:
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue_review.md`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue_review.csv`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.md`
- Result:
  - queue rows: `13`
  - `promote_candidate`: `6`
  - `review_pending`: `7`
  - promoted seed candidates are the six `review_for_promotion` rows:
    vertical up, vertical down, right/left leg-forward pose, left/right kick-forward.
- Human-review convention:
  - review MD/CSV includes up to three raw HML3D captions per example case;
  - captions are read from
    `/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/texts/<case_id>.txt`;
  - if raw captions are unavailable, the renderer can fall back to the caption
    cached in the full event-BPE audit;
  - manual seed JSON remains caption-free.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/render_motion_split_review_artifacts.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/render_motion_split_review_artifacts.py --promotion-queue outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_promotion_queue.json --output-dir outputs/aml_regression_testset_v2/motion_split_proposals_v1`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json`
  - linted `manual_split_review_seed_v1.json` for caption/label/phrase/WordNet/language keys; no hits.

### Motion pattern forest v1

- Motivation:
  - move from isolated queue review to an actual offline forest draft;
  - allow multiple roots because action-pattern variation is broad and should
    not be forced into one global tree;
  - insert currently accepted split children first, then adjust after real
    inspection.
- New script:
  - `scripts/build_motion_pattern_forest.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json`
  - `outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json`
- Main output:
  - `outputs/aml_regression_testset_v2/motion_pattern_forest_v1/summary.json`
  - `outputs/aml_regression_testset_v2/motion_pattern_forest_v1/motion_pattern_forest.json`
  - `outputs/aml_regression_testset_v2/motion_pattern_forest_v1/motion_pattern_forest.md`
  - `outputs/aml_regression_testset_v2/motion_pattern_forest_v1/motion_pattern_forest_tree.txt`
- Main forest size:
  - nodes: `13`
  - edges: `7`
  - roots: `6`
  - max depth: `1`
  - node kinds: `component=4`, `diagnostic=1`, `pattern_candidate=1`,
    `pattern_variation_candidate=6`, `variation_parent=1`
  - edge types: `component=1`, `variation=6`
- Pending-inclusive diagnostic output:
  - `outputs/aml_regression_testset_v2/motion_pattern_forest_with_pending_v1/`
  - nodes: `25`
  - edges: `19`
  - pending variations: `12`
- Current structure:
  - `motion_node_0003` is a low-body variation parent with six promoted
    variation children: vertical up, vertical down, right/left leg-forward
    pose, left/right kick-forward;
  - `motion_node_0006` is a composite pattern candidate with
    `motion_node_0002` as a component;
  - other reviewed components remain separate roots for now.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_motion_pattern_forest.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_motion_pattern_forest.py --draft outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json --split-proposals outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json --manual-split-review outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json --output-dir outputs/aml_regression_testset_v2/motion_pattern_forest_v1`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_motion_pattern_forest.py --draft outputs/aml_regression_testset_v2/promoted_motion_tree_draft_v1/promoted_motion_tree_draft.json --split-proposals outputs/aml_regression_testset_v2/motion_split_proposals_v1/motion_split_proposals.json --manual-split-review outputs/aml_regression_testset_v2/motion_split_proposals_v1/manual_split_review_seed_v1.json --output-dir outputs/aml_regression_testset_v2/motion_pattern_forest_with_pending_v1 --include-pending-children`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_pattern_forest_v1/motion_pattern_forest.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/motion_pattern_forest_with_pending_v1/motion_pattern_forest.json`

### Full candidate motion forest v1

- Motivation:
  - clarify that the earlier `13` / `25` node forest is only a reviewed seed
    forest, not the full HumanML3D Motion-BPE forest;
  - build a broader offline forest directly from all full-HML3D Motion-BPE motif
    tiers, while keeping language aliases and the old AML tree as diagnostics
    only;
  - separate structural candidate families from legacy/low-purity diagnostic
    families before any runtime AML tree update.
- New script:
  - `scripts/build_full_candidate_motion_forest.py`
- Inputs:
  - `outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json`
  - `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl`
- Forest policy:
  - root nodes are geometry-family nodes grouped by required geometry cluster
    sets;
  - leaf nodes are Motion-BPE motifs;
  - caption aliases, caption keywords, and previous tree family IDs are stored
    under diagnostics and are not used to create edges;
  - family status is computed from child motif tiers:
    `candidate_family`, `mixed_family`, or `diagnostic_family`.
- Default structural candidate output:
  - `outputs/aml_regression_testset_v2/full_candidate_motion_forest_v1/`
  - includes tiers: `named_motion_candidate`, `motion_stable_unnamed`
  - included motifs: `33`
  - geometry families: `14`
  - total nodes: `47`
  - edges: `33`
  - total scanned cases: `28823`
  - unique case coverage: `5882`
  - coverage ratio: `0.204073`
  - family status counts: `candidate_family=14`
- Legacy-inclusive diagnostic output:
  - `outputs/aml_regression_testset_v2/full_candidate_motion_forest_with_legacy_v1/`
  - includes tiers: `named_motion_candidate`, `motion_stable_unnamed`,
    `legacy_aligned_diagnostic`
  - included motifs: `166`
  - geometry families: `66`
  - total nodes: `232`
  - unique case coverage: `16403`
  - coverage ratio: `0.569094`
  - family status counts: `candidate_family=14`, `diagnostic_family=52`
- All-tier diagnostic output:
  - `outputs/aml_regression_testset_v2/full_candidate_motion_forest_all_tiers_v1/`
  - includes all `256` Motion-BPE motifs from the tier artifact
  - geometry families: `89`
  - total nodes: `345`
  - edges: `256`
  - unique case coverage: `21181`
  - coverage ratio: `0.734865`
  - tier counts: `named_motion_candidate=8`, `motion_stable_unnamed=25`,
    `legacy_aligned_diagnostic=133`, `generic_or_low_purity=90`
  - family status counts: `candidate_family=11`, `mixed_family=3`,
    `diagnostic_family=75`
- Key interpretation:
  - `25` nodes was the manual seed/pending forest size;
  - the current full-tier forest exposes the real Motion-BPE scale:
    `89` geometry-family roots and `256` motif leaves;
  - the default `47` node output is the cleaner candidate subset for review,
    while the all-tier `345` node output is the full diagnostic map.
- Verification:
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m py_compile scripts/build_full_candidate_motion_forest.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_full_candidate_motion_forest.py --motif-tiers outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json --case-bpe-sequences outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl --output-dir outputs/aml_regression_testset_v2/full_candidate_motion_forest_v1`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_full_candidate_motion_forest.py --motif-tiers outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json --case-bpe-sequences outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl --output-dir outputs/aml_regression_testset_v2/full_candidate_motion_forest_with_legacy_v1 --include-legacy-diagnostic`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_full_candidate_motion_forest.py --motif-tiers outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/motion_bpe_motif_tiers.json --case-bpe-sequences outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/case_bpe_sequences.jsonl --output-dir outputs/aml_regression_testset_v2/full_candidate_motion_forest_all_tiers_v1 --tiers named_motion_candidate,motion_stable_unnamed,legacy_aligned_diagnostic,generic_or_low_purity,language_weak_diagnostic`
  - `python -m json.tool outputs/aml_regression_testset_v2/full_candidate_motion_forest_v1/full_candidate_motion_forest.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/full_candidate_motion_forest_with_legacy_v1/full_candidate_motion_forest.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/full_candidate_motion_forest_all_tiers_v1/full_candidate_motion_forest.json`

### Multi-channel Motion-BPE extraction design

- Motivation:
  - the current full-HML3D Event-BPE audit is useful but still flattens
    concurrent Layer3 events into one sorted sequence;
  - examples such as arm raise while running, jumping-jack arm spread plus
    vertical motion, and dance/karate/swimming-like coordination need explicit
    parallel structure;
  - the current `256` BPE motifs are a configured merge budget, not evidence
    that the motion corpus has only `256` reusable subwords.
- Current single-sequence audit facts:
  - records: `29228`
  - original Layer3 event-token occurrences: `393032`
  - base event-token types: `649`
  - BPE token occurrences after merge: `326070`
  - learned merge motif types: `256`
  - final BPE vocabulary types: `905 = 649 base symbols + 256 merge symbols`
  - compression ratio: `0.829627`
  - token granularity: `geometry`
- New design doc:
  - `docs/design/multi_channel_motion_bpe_extraction.md`
- Updated docs:
  - `docs/design/motion_cluster_bpe_tree_induction.md`
  - `docs/design/motion_corpus_pattern_tree_mainline.md`
  - `docs/design/design_overview.md`
  - `docs/README.md`
- Extraction design summary:
  - Step 0: build a HumanML3D corpus index with case ids, joints, captions, and
    provenance.
  - Step 1: extract dense body-channel observables from joints.
  - Step 2: segment observables into channel events with span, direction,
    magnitude, speed, count, confidence, and source observables.
  - Step 3: assign normalized event tokens per channel.
  - Step 4: build temporal overlap graphs over channel events.
  - Step 5: construct parallel packets from overlapping cross-channel events.
  - Step 6: create three BPE views: per-channel sequences, packet sequences,
    and relation triples.
  - Step 7: learn BPE motifs with sequence, parallel, repetition, and packet
    sequence merge operators.
  - Step 8: score merges by support, compression, channel coverage, relation
    consistency, and numeric consistency; language remains audit-only.
  - Step 9: run merge-budget and support sweeps instead of treating `256` as
    final.
  - Step 10: audit motifs with relation profiles, numeric profiles, examples,
    caption diagnostics, and legacy diagnostics.
  - Step 11: group motifs into motion-derived motif families.
  - Step 12: convert motif families into offline motion pattern forest
    candidates.
- Key schema additions:
  - channel event fields: `channel`, `span`, `direction`, `duration_bin`,
    `magnitude_bin`, `speed_bin`, `count_bin`, `source_observables`;
  - rotation-specific handle: `angular_speed_bin`;
  - locomotion-specific handles: `distance_bin`, `speed_bin`, `path_bin`;
  - packet fields: `members`, `member_channels`, `packet_symbol`,
    `relation_summary`;
  - motif fields: `operator`, `channels`, `relation_profile`,
    `numeric_profile`.
- Proposed implementation artifact:
  - script: `scripts/audit_hml3d_multichannel_motion_bpe.py`
  - output dir:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/`
- Proposed output files:
  - `multi_channel_event_corpus.jsonl`
  - `channel_event_vocab.json`
  - `overlap_packet_corpus.jsonl`
  - `packet_vocab.json`
  - `multichannel_motion_bpe_vocab.json`
  - `case_multichannel_bpe_sequences.jsonl`
  - `motif_audit.json`
  - `motif_family_candidates.json`
  - `motion_pattern_forest_candidates.json`
  - `summary.json`
  - `audit_report.md`
- Validation checklist:
  - compare channel-event counts to the current `393032` Layer3 event baseline;
  - report how many events become parallel packets;
  - verify that jumping-jack-like cases produce bimanual + vertical packets;
  - verify that arm-motion-with-running cases produce root + arm packets;
  - verify that rotation uses angular-speed bins;
  - run `256/512/1024/2048` merge sweeps;
  - compare motif purity, coverage, and compression against the old
    single-sequence Event-BPE baseline.

### Full-HML3D multi-channel Motion-BPE audit v1

- New script:
  - `scripts/audit_hml3d_multichannel_motion_bpe.py`
- Output:
  - `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/`
- Source:
  - `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl`
- Command:
  - `/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' /mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_hml3d_multichannel_motion_bpe.py --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1 --num-merges 96 --min-pair-count 120 --min-pair-support 60 --write-heavy-corpora`
- Runtime:
  - elapsed: `2:11.23`
  - max RSS: `3312552 KB`
- Summary:
  - source records: `29228`
  - channel events: `393032`
  - channel event types: `1019`
  - packets: `87058`
  - packet types: `25339`
  - single-member packets: `46355`
  - parallel packets: `40703`
  - relation count: `1209868`
  - relation type counts: `lead_lag=321989`, `parallel=713807`,
    `same_channel_adjacent=174072`
  - original multi-view token count: `480090`
  - learned motifs: `96`
  - final token count: `348944`
  - final vocab size: `18981`
  - compression ratio: `0.72683`
  - covered cases: `19647 / 29228`
  - case coverage: `0.672198`
  - motif families: `43`
  - forest nodes: `139`
  - forest edges: `96`
  - operator counts: `SEQ_CHANNEL_MERGE=86`, `SEQ_PACKET_MERGE=10`
  - packet motif ratio: `0.104167`
- Output files:
  - `summary.json`
  - `audit_report.md`
  - `channel_event_vocab.json`
  - `packet_vocab.json`
  - `multichannel_motion_bpe_vocab.json`
  - `case_multichannel_bpe_sequences.jsonl`
  - `motif_audit.json`
  - `motif_family_candidates.json`
  - `motion_pattern_forest_candidates.json`
  - `multi_channel_event_corpus.jsonl`
  - `overlap_packet_corpus.jsonl`
- Line counts:
  - `multi_channel_event_corpus.jsonl`: `29228`
  - `overlap_packet_corpus.jsonl`: `29228`
  - `case_multichannel_bpe_sequences.jsonl`: `106507`
- Important implementation note:
  - this symbolic audit is CPU-oriented; GPU is not useful for the current hot
    path because the work is JSON/dict handling, span-overlap graph building,
    Counter statistics, and token replacement;
  - the first heavy relation-view run with `--include-relation-view` and `256`
    merges was stopped because it stayed CPU-bound for several minutes;
  - v1 therefore learns over per-channel and packet sequences by default,
    while still reporting relation counts and writing packet corpora;
  - BPE learning was optimized to use lightweight symbol sequences and then
    reconstruct structured token sequences once from the learned merge table.
- Verification:
  - `python -m py_compile scripts/audit_hml3d_multichannel_motion_bpe.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_hml3d_multichannel_motion_bpe.py --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_smoke_v5 --max-records 200 --num-merges 32 --min-pair-count 4 --min-pair-support 3`
  - `python -m json.tool outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/summary.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/motif_audit.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/motif_family_candidates.json`
  - `python -m json.tool outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/motion_pattern_forest_candidates.json`
  - `head -n 1 outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/multi_channel_event_corpus.jsonl | python -m json.tool`
  - `head -n 1 outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/overlap_packet_corpus.jsonl | python -m json.tool`

### Multi-channel Motion-BPE script cleanup

- Updated:
  - `scripts/audit_hml3d_multichannel_motion_bpe.py`
  - `docs/design/multi_channel_motion_bpe_extraction.md`
- Purpose:
  - make the script easier to tune directly;
  - keep Motion-BPE motion-only;
  - cache the expensive channel-event / packet extraction stage.
- Script behavior:
  - captions and caption aliases are retained only for examples and naming
    diagnostics;
  - hard-coded caption keyword tags were removed from the Motion-BPE audit;
  - BPE merges and motif-family grouping use motion symbols only.
- New commands:
  - self-test:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_hml3d_multichannel_motion_bpe.py --self-test`
  - small cached tuning run:
    - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_hml3d_multichannel_motion_bpe.py --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_clean_v1 --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_clean_cache_v1 --max-records 200 --num-merges 32 --min-pair-count 4 --min-pair-support 3`
- Debug output:
  - `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_clean_v1/summary.json`
  - `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_clean_v1/audit_report.md`
  - cache:
    - `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_clean_cache_v1/`
- Debug summary:
  - records: `200`
  - channel events: `2812`
  - packets: `600`
  - parallel packets: `278`
  - learned motifs: `32`
  - compression ratio: `0.758499`
  - motif families: `20`
  - forest nodes: `52`
  - second run cache status: `hit`
- Verification:
  - `python -m py_compile scripts/audit_hml3d_multichannel_motion_bpe.py`
  - `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/audit_hml3d_multichannel_motion_bpe.py --self-test`
  - searched the script for the removed feedback keyword/action regex terms;
    no hits.
