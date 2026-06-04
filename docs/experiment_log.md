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
