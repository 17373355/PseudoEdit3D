# v0 AML Priority TODO

This is the shared checklist for the current AML design and training iteration.
We mark an item done only after its artifact exists and has been reviewed.

## Status Legend

- `[ ]` not started
- `[~]` implemented, waiting for review
- `[x]` reviewed and accepted
- `[!]` blocked or needs redesign

## Global Progress Board

This board shows the full project process from corpus mining to training.

| Stage | Purpose | Current status | Main artifact | Current blocker / review gate |
| --- | --- | --- | --- | --- |
| G0. Legacy cleanup | keep active scripts readable | `[~] implemented, needs final review` | `docs/design/script_inventory.md`, `legacy/README.md` | confirm no active script imports from `legacy/` |
| G1. HumanML3D Layer3 corpus | convert full HML3D motion into symbolic event records | `[~] implemented` | `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl` | check whether missing targets need new observables |
| G2. Multi-channel Motion-BPE | learn channel motifs and cross-channel coordination motifs | `[~] v4 coord-role 3k sanity complete` | `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_3k/` | review whether composition candidates should be promoted or kept as components |
| G3. Motion candidate forest | group motion-derived motifs into reviewable family nodes | `[~] composition forest v0 built` | `outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v0_structure_groups/` | review structure groups before promotion |
| G4. Manual pseudo-GT audit | test known weak targets without creating rules | `[~] self-reviewed, needs user spot-check` | `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/manual_text_target_self_review.md` | user spot-check `cartwheel`, `sit`, `swim` split decisions |
| G5. Caption/WordNet naming | attach language names to existing motion nodes | `[~] implemented v0` | `outputs/aml_regression_testset_v2/hml3d_caption_wordnet_name_candidates_v0/` | filter low-quality n-grams and classify phrase types |
| G6. Reviewed AML pattern forest | promote accepted motion nodes into AML vocabulary | `[~] v2 recall audit says full nodes need composition mining` | `outputs/aml_regression_testset_v2/aml_pattern_forest_v0/`, `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_all_units/` | do not promote more full nodes until composition candidates are mined |
| G7. AML condition schema | define trainable condition representation | `[~] manifest + audit filter implemented` | `outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean/` | user review before training |
| G8. Condition encoder data | export train/eval condition batches | `[~] clean 250-case batch exported` | `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean/` | connect to encoder smoke config |
| G9. Smoke training | train/evaluate small AML condition encoder | `[!] alignment pass, split warn` | `outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/` | current clean set is too small/sparse for generalization |
| G10. MoMask visual audit | render selected prompts/motions for semantic inspection | `[~] group_01 failure ledger started` | `outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/group_01_failure_audit_v0/` | implement general observables for the top failure families, not case-specific prompt rules |
| G11. Full training decision | decide whether to scale | `[ ] not started` | training plan + acceptance metrics | gated by G2/G4/G5/G9 |

## Current Snapshot

Implemented but not fully reviewed:

- Full HumanML3D Layer3 corpus exists.
- Multi-channel Motion-BPE v1 exists.
- Multi-channel Motion-BPE v2 all-confusions observable refinement exists.
- Multi-channel Motion-BPE v3/v4 adds raw-joint sidecar events for:
  arm orbit/large-arc trajectories, reach/retract, hand-to-head proximity,
  leg lateral spread/repeat, and body-level low/high transition cycles.
- Multi-channel Motion-BPE v2 composition-score coordination selection exists.
- v1/v2 dense candidate forests and promotion self-reviews exist.
- Manual registry audit exists for 14 text targets.
- Caption/WordNet naming sidecar exists.
- v0 pipeline design doc exists.

Not yet started:

- Condition schema v0.
- v0 MoMask visual review.

Current evidence:

```text
Manual target audit:
  targets: 14
  text pseudo-GT cases: 7302
  indexed HML3D cases: 29228
  indexed BPE symbols: 5736

Caption/WordNet naming:
  caption cases: 29228
  captions: 87372
  retained phrases: 12030
  motion nodes: 164
  node-name candidates: 1968
```

Current weak points:

- Many object/environment/intent-heavy targets are not recovered by motion-only
  motifs: `basketball`, `tennis`, `climb`, `duck_under`.
- Some currently recovered targets may be proxy-heavy, especially `cartwheel`
  and `swim`.
- Caption phrase mining still contains low-quality fragments.
- v2 observable refinement reduces false full-pattern candidates, but the merge
  policy still needs to learn cleaner composed motifs from refined components.
- v2 composition-score removes gait/path false coordination candidates, but it
  is conservative and has not recovered new full-action patterns yet.
- Coactivation recall audit shows the missing full-pattern issue is mostly a
  composition-mining gap: full-pattern evidence often exists at the all-channel
  unit level, but the current coordination stage only composes selected
  per-channel `<CHM_*>` motifs and therefore misses many stable combinations.
- The first composable AML pattern program exists and can export a conservative
  train-clean 250-case condition batch.
- Composition pattern forest v0 now mines closed all-channel coactivation
  itemsets and exports a four-level candidate forest:
  `root -> structure_group -> composition_family -> variant`.
- Group_01 native-vs-AML visual audit exposed recurring missing families:
  `unilateral_arm_circle`, `sit_stand_cycle`, `bilateral_spread_jump`,
  `side_sway_or_rock`, `step_up_hop_sequence`, `strike_or_punch_sequence`,
  `prone_swim_or_flail`, and object-activity weak names such as basketball and
  tennis.
- The first general observable fix targets `unilateral_arm_circle` and
  `large_arm_arc` as raw-joint motion evidence, not as caption keywords.
- The v4 coord-role promotion review surface now separates likely named
  compositions from reusable components:
  `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_promotion_review/`.

## Layer Contract

The layers have different precision/recall goals.

- Observable micro-events should favor recall. At this layer, it is better to
  over-emit measurable joint/root/body signals than to miss evidence needed by
  later mining. Extra events are acceptable if they keep span, channel,
  geometry, magnitude, count, and source evidence traceable.
- Merge, proxy, and promotion should favor precision. These layers must prevent
  frequent but low-importance components from dominating a pattern. Promotion
  must use motion scope, structure score, support, caption/WordNet naming
  evidence, and visual/manual review gates; support alone is not sufficient.
- A noisy micro-event is a tuning problem only if it floods BPE or hides stronger
  evidence. A noisy proxy/promotion is a correctness problem because it changes
  the semantic action node exposed to AML and downstream editing.

## Current Critical Path

```text
review high-signal target audits
-> classify failure reasons
-> improve Motion-BPE merge policy / naming filters
-> promote reviewed pattern forest v0
-> define/audit condition schema
-> export clean condition batches
-> smoke train condition encoder
-> MoMask visual audit
-> decide full training
```

The current immediate checkpoint is not training. The immediate checkpoint is
whether the v0 Motion-BPE/naming outputs are clean enough to promote a small
reviewed AML pattern forest.

The current newest checkpoint is tree search: can an input motion's structural
evidence search the composable pattern program and return the right semantic
level, edit scope, and candidate condition node.

## P0: Freeze The v0 Pipeline Contract

Goal: make the current full-HML3D AML pipeline reproducible before adding more
logic.

- [~] Define v0 AML pipeline boundary.
  - Artifact: `docs/design/v0_aml_pipeline.md`
  - Review check: motion builds structure; language names; manual registry only audits.

- [~] Keep manual text target registry audit-only.
  - Artifact: `configs/motion_pattern_text_targets.json`
  - Review check: registry is not imported by Motion-BPE learning or runtime AML rules.

- [~] Run all manual text target audits.
  - Artifact: `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/`
  - Review check: each target has `review.md`, `pattern_pseudo_gt_audit.json`,
    `recall_candidate_symbols.json`, and `pattern_family_proposal.json`.

- [~] Run caption/WordNet name mining.
  - Artifact: `outputs/aml_regression_testset_v2/hml3d_caption_wordnet_name_candidates_v0/`
  - Review check: naming sidecar attaches names to existing motion nodes only.

- [ ] Create a one-command v0 reproduction script.
  - Proposed artifact: `scripts/run_v0_aml_pipeline.py` or `scripts/run_v0_aml_pipeline.sh`
  - Review check: reruns Motion-BPE audit, coordination forest, manual target audit,
    and caption/WordNet naming with fixed output paths or a version suffix.

## P1: Improve Motion-BPE Quality

Goal: make the symbolic motion vocabulary cleaner before using it as a condition
schema.

- [~] Audit high-signal manual targets first.
  - Scope: `jumping_jack`, `cartwheel`, `swim`, `sit`
  - Artifact: `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/manual_text_target_self_review.md`
  - Review check: user confirms `cartwheel`, `sit`, and `swim` split/downgrade decisions.

- [~] Diagnose uncovered manual targets.
  - Scope: `jump_rope`, `stand_up`, `kneel`, `karate_or_martial`, `dance`,
    `ballet`, `basketball`, `tennis`, `climb`, `duck_under`
  - Artifact: `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/manual_text_target_self_review.md`
  - Review check: classify failure as extraction gap, BPE merge gap, naming gap,
    object/environment ambiguity, or not representable by current motion channels.

- [~] Add better channel observables only where the audit proves a gap.
  - Artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_all_confusions_full/`
  - Implemented split categories: leg-forward gait/pose/impulse, weak root
    drift, root path fragments, turn angle/tempo/path role, vertical gait bounce,
    low-body transitions, torso context, arm symmetry/coupling, bimanual context.
  - Review check: v2 promotion self-review shows fewer dense composition rows
    and no full-action promotion from impure components.
  - Layer contract: this step may be recall-heavy. It should expose candidate
    evidence broadly, but every emitted micro-event must remain localized,
    typed by channel/geometry, and removable or downweighted by later merge
    policy.
  - New group_01 failure ledger:
    `outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/group_01_failure_audit_v0/failure_ledger.md`
  - Next observable priorities:
    - `[~]` unilateral arm circle/swing trajectory, direction, and alternation.
      - Implemented in `scripts/audit_hml3d_multichannel_motion_bpe.py` as
        `--observable-refinement v3` with raw-joint sidecar events:
        `LEFT_ARM_TRAJECTORY/*_ARM_ORBIT_CYCLE_*`,
        `RIGHT_ARM_TRAJECTORY/*_ARM_ORBIT_CYCLE_*`, and
        `*_LARGE_ARM_ARC_*`.
      - Debug artifacts:
        `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v3_arm_traj_group01_probe/`
        and
        `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v3_arm_traj_3k/`.
      - Current check: `008692` produces right-arm orbit-cycle tokens;
        `003191`, `003082`, and `009072` produce large-arm-arc tokens;
        `007581` and `000576` no longer trigger after threshold tightening.
    - `[~]` sit/stand state machine with sit-down, sit-up, stand-up,
      sit-back-down temporal order.
      - Implemented as raw-joint body-level sidecar events:
        `WHOLE_BODY_LEVEL/WB_LEVEL_LOW_SUSTAINED`,
        `WB_LEVEL_RISE_FROM_LOW`, `WB_LEVEL_DESCEND_TO_LOW`,
        `WB_LEVEL_LOW_HIGH_LOW_CYCLE`, and
        `WB_LEVEL_HIGH_LOW_HIGH_CYCLE`.
      - Generic `WB_LEVEL_HIGH_STANDLIKE` is disabled by default because it is
        ubiquitous context and dominates BPE if emitted.
      - Debug artifacts:
        `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v3_body_level_no_high_group01_probe/`
        and
        `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v3_arm_body_no_high_3k/`.
      - Current check: `006986` emits low-high-low sit/stand/sit structure;
        `007581` emits high-low-high sit/stand structure; `003020` and
        `003191` no longer emit generic high-state noise.
    - `[~]` hand-to-head/face proximity.
      - Implemented as raw-joint sidecar events:
        `LEFT_ARM_PROXIMITY/*_HAND_APPROACH_HEAD`,
        `*_HAND_NEAR_HEAD_HOLD`, `*_HAND_LEAVE_HEAD`, and
        `*_HAND_NEAR_HEAD_REPEATED`.
      - This is a geometry observable for hand-near-head/face contact-like
        motion, not a semantic label such as drinking, phone, or face-touch.
    - `[~]` leg lateral spread/repeat.
      - Implemented as raw-joint sidecar events:
        `LEFT_LEG_LATERAL/*_LEG_LATERAL_REPEAT`,
        `*_LEG_LATERAL_ABDUCT`, `*_LEG_LATERAL_ADDUCT`, and
        `*_LEG_LATERAL_OUT_*`.
      - The signal is baseline-relative foot displacement along the body
        lateral axis, so normal standing/walking should not dominate BPE.
    - `[~]` bilateral spread-jump evidence for jumping-jack-like motion.
      - Current evidence is not a named jumping-jack node yet. It is a
        composable structure: bilateral large arm arcs/orbits + bilateral leg
        lateral repeats + vertical/body-level rhythm.
    - `[~]` lateral sway/rocking periodicity.
      - Current evidence comes from alternating left/right leg lateral repeats;
        root/torso sway still needs a cleaner direct observable.
    - `[~]` strike/punch reach-and-retract cycles.
      - Current evidence comes from arm reach/retract sidecars. It should be
        composed with torso/root context before promotion.
  - v4 debug artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_micro_sidecar_failure_probe/`
  - v4 3k sanity artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_micro_sidecar_3k/`
  - Current check:
    `003082`, `011643`, `014607`, and `M008014` expose spread-jump evidence;
    `007581` exposes sit/stand plus hand-to-head approach/hold/leave;
    `008692` exposes right-arm orbit cycles; `003020` exposes lateral repeat;
    `000576` no longer emits leg-lateral noise in the failure probe.
  - v4 micro-event 3k sanity metrics:
    `channel_event_count` increases from 43350 to 50113; `channel_event_type_count`
    increases from 1833 to 2062; `forest_node_count` stays controlled
    (73 -> 76). New sidecar counts are: arm reach/retract 1901,
    hand proximity 2389, leg lateral 2473. Compression gets slightly worse
    (`channel_bpe_output_ratio` 0.765 -> 0.796), so the next issue is merge
    policy, not adding more one-off observables.

- [~] Improve multichannel BPE merge policy.
  - Current artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_3k/`
  - Debug artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_failure_probe/`
  - Review check: compare v1 vs v2 on compression, motif purity, target audit recall,
    and false-positive rate.
  - Layer contract: this step must be precision-heavy. Merge and proxy nodes
    should suppress generic frequent components unless they form a stable,
    reviewable composition with clear motion scope.
  - Current result:
    - Coordination mining now uses high-signal raw-joint events and high-signal
      channel motifs as seeds instead of only selected `<CHM_*>` motifs.
    - Coactivation signatures are role-level structures such as
      `left_arm:arm_large_arc + right_arm:arm_large_arc +
      bimanual:bilateral_arm_vertical_cycle + left/right_leg:leg_lateral_repeat`,
      while exact geometry remains as evidence.
    - On the 17-case failure probe, coordination motifs increase from 1 to 12
      and recover interpretable structures for jumping-jack-like arm/leg/vertical
      coordination, right-arm orbit, martial-like low-body/hand/leg coordination,
      and sit/stand body-level transitions.
    - On the 3k sanity run, coordination motifs increase from 3 to 21,
      `forest_node_count` increases from 76 to 106, and `case_coverage` increases
      from 0.534 to 0.592. Families are now tagged by motion scope:
      14 `stable_component_candidate`, 21 `component_candidate`, and
      7 `composition_candidate`.
    - Current limitation: many high-support motifs are valid components
      rather than full semantic actions. Promotion should therefore use
      `motion_scope`, caption/WordNet naming, and visual spot checks instead of
      support alone.
    - Diagnostic note: full jumping-jack-like structure exists in the
      coactivation sequence, but is fragmented across many nearby role
      signatures. The selected v4 merge set currently surfaces the clean
      upper/vertical component as `named_component_review`, not a full pattern.
      Next step is composition closure / family-level grouping over related
      role signatures, not more micro-events.

- [~] Build v4 coord-role promotion/component review.
  - Script:
    `scripts/build_v4_coord_role_promotion_review.py`
  - Artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_promotion_review/`
  - Current result:
    42 families total: 2 `promote_review`, 5 `composition_review`,
    1 `named_component_review`, 12 `component_review`, and
    22 `component_library`.
  - Review check:
    `sit_down` body-level/torso compositions are the only immediate named
    promote-review rows; jumping-jack-like evidence is deliberately kept as a
    named component until upper/vertical/lower closure is promoted.

- [~] Audit missed full-pattern coactivations.
  - Artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_all_units/`
  - Comparison artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit/`
  - Result:
    - `channel_motifs` view: 2096 coactivation symbols, most targets collapse
      to selected upper-body components.
    - `all_units` view: 21935 coactivation symbols; `jumping_jack` has a clean
      unselected upper+vertical+bimanual candidate with 65 / 67 target precision
      and 65 / 368 text-pseudo-GT recall.
  - Review check: passed for diagnosis. The next implementation is a real
    composition-mining stage above channel units, not more proxy labels.

- [~] Add composition-BPE / closed coactivation mining above channel units.
  - Artifact:
    `outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v0_structure_groups/`
  - Script:
    `scripts/build_hml3d_composition_pattern_forest_v0.py`
  - Goal: merge repeated cross-channel sets such as
    `vertical arm-raise coupled + bilateral high arm pose + bimanual raise-spread`
    into stable composition candidates while preserving component provenance.
  - Current result:
    - 29228 caption-indexed cases.
    - 34453 all-channel coactivation transactions.
    - 3752 closed itemsets.
    - 64 structure groups.
    - 379 composition families.
    - 1219 exported variants.
  - Review check: inspect structure groups before promotion. Caption aliases are
    naming diagnostics only; motion structure labels come from channel/geometry
    roles.

- [ ] Separate component motifs from full action motifs.
  - Example: jumping-jack arm raise-spread alone should be a component; arm+vertical
    coordination may be a full pattern candidate.
  - Artifact: motif family schema field such as `motif_scope=component|coordination|sequence`.
  - Review check: forest no longer treats common components as complete action names.

## P3.5: Make The Pattern Forest Searchable

Goal: turn the forest into a program that can be searched by motion evidence and
later used by the AML condition interface.

- [~] Export composable AML pattern program.
  - Artifact: `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v0/`
  - Script: `scripts/export_aml_composable_pattern_program_v0.py`
  - Current result: 1666 program nodes, 443 condition entries, 64 structure
    groups, 379 composition families, 1219 variants.
  - Review check: inspect `aml_composable_pattern_program_tree.txt` and verify
    semantic levels/edit scopes are meaningful.

- [~] Add lightweight AML program loader and tree search API.
  - Module: `pseudoedit3d/edit/aml_composable_pattern_program.py`
  - Package import: `from pseudoedit3d.edit import load_composable_pattern_program, search_program_nodes`
  - Review check: loader works without importing heavy numeric dependencies.

- [~] Add motion-to-tree search debug script.
  - Script: `scripts/search_aml_composable_pattern_program_v0.py`
  - Artifact: `outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0/`
  - Current 250-case result:
    - 250 channel cases.
    - 192 cases with coactivation windows.
    - 340 searched windows.
    - 37 whole-body/full-composition candidates.
    - 26 composed multi-part candidates.
    - 29 transition cases.
    - 32 component-dominant cases.
    - 68 diagnostic/ambiguous cases.
    - 58 unmatched/local-only cases.
  - Review check: inspect `search_report.md` and selected review examples.

- [~] Add final case-level condition manifest exporter.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0/`
    and strict training variant
    `outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/`
  - Script: `scripts/export_aml_program_condition_manifest_v0.py`
  - Requirement: select reviewed `condition_entry_id`s per case/span with
    confidence, semantic level, edit scope, and source evidence.
  - Review check: no diagnostic-only node enters positive training conditions.
  - Current status: `[~] implemented`
  - Current default span2/debug result: 250 cases, 156 train-ready, 377 selected,
    544 deferred, contract pass.
  - Current strict span1/training result: 250 cases, 156 train-ready, 236
    selected, 544 deferred, contract pass, no duplicate selected conditions per
    span.
  - Current audit-filtered train-clean result: 250 cases, 40 train-ready, 50
    selected, 730 deferred, contract pass. The quality audit confirms all
    selected conditions are `train_candidate`.

- [~] Export condition batch schema from the program manifest.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0/`
    and strict training variant
    `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_strict_span1/`
  - Command uses the `h2char` Python environment because base Python currently
    lacks `numpy`.
  - Review check: compatible with existing `AmlConditionMotionDataset` contract
    or documented adapter.
  - Current status: `[~] implemented`
  - Current strict span1 batch: 156 cases, 236 selected conditions, max 8
    conditions, span coverage 1.0, no truncation.
  - Current train-clean batch:
    `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean/`
    with 40 cases, 50 selected conditions, max 8 conditions, span coverage 1.0,
    no truncation, score mean 0.8179.

## P2: Improve Caption / WordNet Naming

Goal: turn motion-derived nodes into readable candidate names without allowing
language to create the tree.

- [ ] Filter low-quality caption n-grams.
  - Problem examples: `several jumping`, `over stands`, incomplete phrase fragments.
  - Artifact: updated `mine_hml3d_caption_wordnet_name_candidates_v0.py`.
  - Review check: top names are readable action phrases, not dangling fragments.

- [ ] Add phrase type labels.
  - Proposed labels: `action`, `object_activity`, `body_part_component`,
    `environment_relation`, `directional_context`, `generic_motion`.
  - Artifact: `name_candidates.json` includes phrase type and ambiguity notes.
  - Review check: object-heavy labels such as basketball/tennis are separated from
    pure motion names.

- [ ] Build name-to-motion and motion-to-name ambiguity tables.
  - Artifact: `name_motion_alignment_report.md`
  - Review check: identifies one-name-to-many and one-motion-to-many cases.

- [ ] Use WordNet as taxonomy hint, not final authority.
  - Artifact: report columns for WordNet parent candidates and HML3D evidence.
  - Review check: a motion node is not promoted just because a WordNet term exists.

## P3: Build The AML Pattern Forest Candidate

Goal: convert reviewed Motion-BPE families into an AML pattern vocabulary.

- [ ] Define reviewed node schema.
  - Proposed fields: `node_id`, `parent_id`, `scope`, `channels`, `geometry`,
    `support_cases`, `name_candidates`, `accepted_name`, `edit_handles`.
  - Artifact: `docs/design/aml_pattern_node_schema.md`
  - Review check: every node keeps channel/geometry/source-symbol provenance.

- [~] Promote reviewed component/full-action nodes into a v0 forest.
  - Artifact: `outputs/aml_regression_testset_v2/aml_pattern_forest_v0/`
  - Review check: forest contains accepted nodes only; rejected/diagnostic nodes stay sidecar.

- [~] Rank dense candidate families for promotion review.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0/`
    and `outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v2_all_confusions/`
  - Review check: no dense node is promoted automatically; `composition_review`
    rows are manually split into accepted/component/pending/diagnostic before
    they enter the reviewed forest.

- [~] Self-review the dense promotion table.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v2_composition_score/promotion_self_review.md`
  - Review check: v2 composition-score has 49 reviewed dense families, 2
    downgraded composition rows, 41 kept components, 6 naming-only rows, and no
    visual-review blockers.

- [ ] Map forest nodes to condition slots.
  - Artifact: `docs/design/aml_condition_schema_v0.md`
  - Review check: condition slots support body part, span, count, direction,
    magnitude, speed, and confidence.

## P4: Condition Encoder And Training Iteration

Goal: connect the reviewed AML vocabulary to trainable condition data.

- [~] Freeze v0 condition batch schema.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean/`
  - Review check: schema can represent composed/transition/full candidates with
    span, slots, score, weight, and family vocab.

- [~] Export train/eval condition batches.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean/`
    and
    `outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean/`
  - Review check: no text registry leakage into training labels except designated
    audit metadata.
  - Current status: self-audited pass for the 250-case train-clean smoke batch.

- [~] Train a small AML condition encoder smoke model.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/`
  - Script: `scripts/train_aml_condition_encoder_smoke.py`
  - Review check: overfits a small subset and preserves condition alignment.
  - Current result: loader smoke passed on 40 cases / 50 conditions / 6046 valid
    frames. Overfit smoke on 50 condition rows reduced normalized MSE from
    0.9868 to 0.0019 on span-level geometry targets.
  - Split audit:
    `outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/smoke_summary.md`
  - Current limitation: row/case split smoke is `warn`. The 40-case / 50-condition
    train-clean set has only 14 shared condition types and 33 fine program-family
    ids; many numeric slots remain placeholder values (`0.0` / `unknown`), so
    held-out geometry is under-specified. This is an alignment/data smoke pass,
    not a generalization pass and not a motion-generation result.

- [ ] Build `train_clean_plus` condition data.
  - Goal: keep the current hard-positive train-clean set, but add reviewed
    high-confidence auxiliary/weak conditions and real numeric residues for
    distance, vertical amplitude, turn angle, count, and direction.
  - Proposed artifact:
    `outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean_plus/`
  - Review check: split smoke should beat global/family mean baselines before
    larger encoder training.

- [~] Run MoMask/autoprompt semantic review on selected cases.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/`
  - Current group_01 result:
    - 50 case summaries.
    - 50 MoMask generations from selected/native HML3D prompts.
    - 50 MoMask generations from AML AutoPrompts.
    - 50 four-panel GIFs with `GT Motion`, `HML3D Captions + AutoPrompt`,
      `MoMask from HML3D`, and `MoMask from AML`.
    - First GIF sanity: `1900x650`, 51 frames.
  - Command:
    ```bash
    /mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/run_aml_momask_review_pack.py \
      --case-list outputs/aml_regression_testset_v2/group_01_case_ids.txt \
      --output-root outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0 \
      --review-name aml_momask_native_vs_aml_review250_v0 \
      --ext-prefix aml_momask_native_vs_aml_review250_v0 \
      --native-compare \
      --gpu-id 1 \
      --time-steps 10 \
      --cond-scale 4 \
      --reuse-existing \
      --frame-stride 4 \
      --skip-kinematic
    ```
  - Review check: compare each case against the previous group_01 failure list,
    but now separate:
    - native MoMask cannot realize the HML3D prompt;
    - AML prompt omits or misnames the semantics;
    - AML prompt is reasonable but MoMask realizes it poorly.
  - Group_01 failure ledger:
    `outputs/aml_regression_testset_v2/aml_momask_native_vs_aml_review250_v0/group_01_failure_audit_v0/failure_ledger.md`
  - Ledger summary: 11 reviewed failures, with top source categories
    `naming_boundary_gap`, `observable_missing`, and `temporal_order_gap`.

- [ ] Decide whether to scale to full training.
  - Gate: only after P1/P2 audits show stable vocabulary and P4 smoke training passes.

## P5: Paper / Report Tracking

Goal: keep the research story reproducible.

- [ ] Maintain an experiment table for each Motion-BPE version.
  - Artifact: `docs/experiment_log.md`
  - Review check: records command, output path, metrics, and interpretation.

- [ ] Maintain a failure taxonomy.
  - Artifact: `docs/design/aml_failure_taxonomy.md`
  - Review check: failures are categorized, not patched case-by-case.

- [ ] Keep legacy scripts out of active path.
  - Artifact: `docs/design/script_inventory.md`, `legacy/README.md`
  - Review check: active scripts do not import from `legacy/`.

## Immediate Next Review Queue

Recommended order:

1. Review `manual_text_target_audits_v0/jumping_jack/review.md`.
2. Review `manual_text_target_audits_v0/cartwheel/review.md`.
3. Review `manual_text_target_audits_v0/swim/review.md`.
4. Review `manual_text_target_audits_v0/sit/review.md`.
5. Improve naming phrase quality.
6. Build a coactivation recall audit to find missed full-pattern compositions.
