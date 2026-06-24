# v0 AML Priority TODO

This is the shared checklist for the current AML design and training iteration.
We mark an item done only after its artifact exists and has been reviewed.

## Current Direction, 2026-06-24

The project structure is being converged under `AML Pattern Mining Explorer v1`.
Motion-BPE is no longer the name of the whole system; it is one optional miner.

Current golden path:

```text
motion evidence extraction
-> candidate pattern mining
-> candidate audit
-> pattern registry
```

Current core artifacts:

- `outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/evidence_cases.jsonl`
- `outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/candidate_patterns.jsonl`
- `outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/pattern_registry.json`
- `outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/audit_report.md`

Current implementation status:

| Priority | Step | Status | Artifact / script | Gate |
| --- | --- | --- | --- | --- |
| P0 | Freeze and document v1 golden path | `[x]` | `docs/design/aml_pattern_mining_explorer_v1.md` | done |
| P1 | Move pre-v1 forest/proposal/program variants to legacy | `[~]` | `legacy/aml_pattern_mining_pre_v1/scripts/` | validate no active imports |
| P2 | Unified axis audit entrypoint | `[x]` | `scripts/run_pattern_axis_audit.py` | smoke compiled |
| P3 | Unified pattern mining bundle exporter | `[x]` | `scripts/export_pattern_mining_explorer_bundle_v1.py` | v1 bundle generated |
| P4 | Keep evidence extraction declarative | `[ ]` | `pseudoedit3d/pattern_mining/evidence_extractors/` | split the frozen v5 extractor later |
| P5 | AML runtime / condition interface from registry | `[ ]` | TBD | after registry review |
| P6 | MoMask native-vs-AML comparison using registry prompts | `[ ]` | TBD | after AML interface |

Older v0/v1/v2/v3/v4/v5 experiment notes below are retained as historical
context, not as the active golden path.

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
| G2. Multi-channel Motion-BPE | learn channel motifs and cross-channel coordination motifs | `[~] v5 stance-width full-HML3D complete` | `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/` | review phase/confound closure before naming full action nodes |
| G3. Motion candidate forest | group motion-derived motifs into reviewable family nodes | `[~] support-state v1 draft forest built` | `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_draft/` | visually review `promote_review`, `review_candidate`, and `split_required` families |
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
- Full-HML3D v4 coord-role Motion-BPE baseline is available:
  `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_full_v0/`.
- Full-HML3D v4 support-state Motion-BPE is the current Motion-BPE mainline:
  `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_full_v0/`.
- Full-HML3D v4 coord-role composition closure review baseline is available:
  `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_full_closure_review_v0/`.
- Current v1 support-state draft pattern forest is available:
  `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_draft/`.
- Current split-axis schema is data-driven and lives at:
  `pseudoedit3d/edit/aml_pattern_split_axes.json`.
  The first axis is `body_level_sit_transition_v0`, but its promoted family is
  deliberately `body_level_low_transition`, not `sit_down`. The caption
  `sit_down` aliases are audit diagnostics only.
- Full-HML3D v5 stance-width Motion-BPE is the current lower-spread recall
  checkpoint:
  `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/`.
  It adds 24256 stance-width sidecar events and raises full-corpus
  bilateral-spread-axis full-rule coverage from 78/362 to 146/362 target cases.
  The comparison report is:
  `outputs/aml_regression_testset_v2/hml3d_v4_v5_stance_width_full_comparison_v0/report.md`.

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
    - `[~]` whole-body support state for floor/prone/inverted separation.
      - Implemented in `scripts/audit_hml3d_multichannel_motion_bpe.py` as
        `--observable-refinement v4` support sidecar events:
        `WHOLE_BODY_SUPPORT/WB_SUPPORT_INVERTED`,
        `WB_SUPPORT_FLOOR_LOW_HORIZONTAL`, and `WB_SUPPORT_HAND_FLOOR_LOW`.
      - This is geometry evidence for support/contact state, not a named
        `cartwheel`, `swim`, `kneel`, or `crawl` rule.
      - Current check: cartwheel-like cases `000452`, `002828`, and `002932`
        emit `WB_SUPPORT_INVERTED`; prone/swim-like `000865` emits
        `WB_SUPPORT_FLOOR_LOW_HORIZONTAL`, not inverted support.
    - v4 debug artifact:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_micro_sidecar_failure_probe/`
    - v4 3k sanity artifact:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_micro_sidecar_3k/`
    - v4 support-state 3k diagnostic artifact:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_3k_diag/`
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
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_full_v0/`
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
    - On the full-HML3D v4 coord-role run, 29228 records produce 486367 channel
      events, 2747 channel-event types, 29142 packet types, 96 learned motifs,
      53 coordination merges, 76 motif families, and 172 forest nodes. Case
      coverage is 15299 / 29228 (`0.5234`). This run is the current full-corpus
      source for closure review.
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
    - Support-state closure note: selection now reserves budget for low-support
      high-specificity scopes so `inversion_acrobatic_candidate` is not
      squeezed out by high-support generic whole-body candidates. In the 3k
      support-state diagnostic closure, 160 selected candidates include
      5 `inversion_acrobatic_candidate` and 8 `floor_prone_or_mime_candidate`
      rows.
    - Support-state 3k draft forest:
      `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_3k_draft/`
      contains 92 nodes, 22 families, and 60 selected source candidates.
      The review pack is:
      `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_3k_review_pack/`.
      Visual spot-check: cartwheel/inversion review examples are now cartwheel
      or flip-like; prone/swim examples remain in floor/prone support families
      and need later split.
    - Support-state full-HML3D run:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_full_v0/`
      covers 18832 / 29228 cases (`0.6443`) with 488601 channel events and
      2234 support sidecar events. Closure review:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_full_v0_closure_review/`
      selects 240 candidates, including 17 `promote_review`,
      8 `inversion_acrobatic_candidate`, and 34
      `floor_prone_or_mime_candidate` rows. Draft forest:
      `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_draft/`
      contains 140 nodes, 26 families, and 104 selected source candidates.
      Review pack:
      `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_review_pack/`.

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

- [~] Build v4 coord-role composition closure review.
  - Script:
    `scripts/build_v4_coord_role_composition_closure.py`
  - Full artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_full_closure_review_v0/`
  - Current result:
    - 79954 coactivation occurrences.
    - 3425 raw closure candidates.
    - 240 exported review candidates.
    - 11 `promote_review` and 229 `composition_review`.
    - Promote-review rows recover full jumping-jack-like coordination, several
      cartwheel/inversion phase candidates, a sit/body-level transition
      candidate, a martial/guard-kick coordination candidate, and one
      cheer/dance coordination candidate.
  - Review check:
    Promotion is still not final. The closure stage reveals that swim/floor-prone
    structures conflict with the current `acrobatics_or_inversion` role and need
    an observable split before they can enter the accepted tree.

- [~] Build v4 closure draft pattern forest.
  - Script:
    `scripts/build_v4_closure_pattern_forest_draft.py`
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_from_v4_closure_draft/`
  - Current result:
    - 57 source closure candidates selected from full closure review.
    - 15 family candidates under 10 roots.
    - 5 family-level `review_candidate` nodes, 3 `split_required` nodes, and
      3 `composition_needs_closure` nodes.
  - Review check:
    Inspect `aml_pattern_forest_v1_draft_tree.txt` first, then
    `aml_pattern_forest_v1_draft_review.md` for examples. Only visual-reviewed
    family nodes should become accepted AML pattern nodes.

- [~] Render v4 closure draft visual review pack.
  - Script:
    `scripts/render_v4_closure_pattern_forest_review_pack.py`
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_from_v4_closure_review_pack/`
  - Current result:
    11 PNG family sheets plus `review_queue.md` and
    `review_decision_template.json`.
  - Review check:
    Fill family-level decisions after visual inspection. The key decisions are
    whether each `review_candidate` is a complete pattern, whether each
    `split_required` family needs a new observable split, and whether each
    `composition_needs_closure` family should merge with an existing candidate
    or be downgraded to a component.

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

- [~] Export reviewed support-state AML program.
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_reviewed_draft/`
  - Script: `scripts/export_aml_composable_pattern_program_v1_support_state.py`
  - Current result: 113 program nodes, 21 unique condition entries, 1 positive
    accepted full-pattern condition, 20 searchable component/split/closure TODO
    conditions with zero positive training weight.
  - Review check: inspect `aml_composable_pattern_program_tree.txt` and confirm
    only reviewed accepted nodes become positive training labels.

- [~] Add lightweight AML program loader and tree search API.
  - Module: `pseudoedit3d/edit/aml_composable_pattern_program.py`
  - Package import: `from pseudoedit3d.edit import load_composable_pattern_program, search_program_nodes`
  - Use `semantic_priority=True` for high-level pattern explanation; leave it
    off for pure local evidence retrieval.
  - Review check: loader works without importing heavy numeric dependencies.

- [~] Add motion-to-tree search debug script.
  - Script: `scripts/search_aml_composable_pattern_program_v0.py`
  - Artifact: `outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0/`
  - Support-state v1 command:
    `python scripts/search_aml_composable_pattern_program_v0.py --support-state-v1 --semantic-priority --max-cases 250`
  - Support-state v1 artifact:
    `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_v0/`
  - Support-state v1 250-case result:
    - 250 channel cases.
    - 208 cases with coactivation windows.
    - 296 searched windows.
    - 2 accepted full-pattern cases.
    - 111 pending-closure candidate cases.
    - 26 pending-split candidate cases.
    - 3 component-hit cases.
    - 108 unmatched/local-only cases.
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

- [~] Add support-state v1 promotion audit.
  - Script: `scripts/audit_v1_support_state_promotion_candidates.py`
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_promotion_audit_v0/`
  - Current result under strict target-alias scoring:
    - 21 audited nodes.
    - 1 `keep_positive` accepted full-pattern node.
    - 2 `split_review` sit/body-level transition nodes.
    - 5 `keep_pending`.
    - 11 `split_or_downgrade`.
    - 2 `keep_component`.
    - No new direct promotions under strict text-alias diagnostics.
  - Review check:
    No new node should be promoted directly from this audit. The next work is
    to add data-driven split/closure axes, starting with sit-down ordered
    body-level transitions and jumping-jack upper/lower/vertical closure.

- [~] Add data-driven split-axis audit for broad transition nodes.
  - Schema: `pseudoedit3d/edit/aml_pattern_split_axes.json`
  - Script: `scripts/audit_v1_support_state_split_axes.py`
  - Artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_split_axis_audit_v0/`
  - Current 250-case support-state search result:
    - 296 searched coactivation windows.
    - 10 accepted `body_level_low_transition` candidate cases.
    - diagnostic target-alias precision: 0.8.
    - diagnostic target-alias recall: 0.5333.
    - 9 `body_level_down_up_cycle_candidate` cases and
      1 `body_level_descend_to_low_candidate` case.
  - Review check:
    This is a split axis, not a direct action name. It can explain why a broad
    tree hit should be routed toward low-body transition evidence, but it should
    not promote `sit_down` until a cleaner seat/contact or support relation is
    available. Torso hunch/recover remains support evidence only.

- [~] Add data-driven closure axis for bilateral spread + vertical rhythm.
  - Schema axis:
    `bilateral_spread_vertical_coordination_v0` in
    `pseudoedit3d/edit/aml_pattern_split_axes.json`
  - Probe search artifact:
    `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v0/`
  - Probe audit artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_split_axis_jumpaxis_probe_audit_v0/`
  - Mixed probe composition:
    100 `jumping_jack` pseudo-GT cases plus cheer/sit/martial/kneel/no-alias
    controls.
  - Current probe result:
    - 298 cases, 440 searched windows.
    - 133 accepted windows, 86 accepted cases for the bilateral spread axis.
    - diagnostic `jumping_jack` precision: 1.0.
    - diagnostic `jumping_jack` recall: 0.86.
    - 117 accepted windows are `bilateral_upper_spread_vertical_component`.
    - 16 accepted windows are
      `bilateral_upper_lower_spread_vertical_coordination`.
  - v5 stance-width sidecar probe:
    - Script update:
      `scripts/audit_hml3d_multichannel_motion_bpe.py`
    - New observable:
      `raw_joint_stance_width` under `observable_refinement=v5`.
    - It emits generic bilateral foot-separation clusters:
      `WB_STANCE_WIDTH_WIDE_BRIEF`, `WB_STANCE_WIDTH_WIDE_HOLD`,
      `WB_STANCE_WIDTH_EXPAND`, `WB_STANCE_WIDTH_CONTRACT`,
      `WB_STANCE_WIDTH_REPEAT`.
    - BPE artifact:
      `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_jumpaxis_probe_v0/`
    - Search artifact:
      `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v5_stance_width_v0/`
    - Split-axis audit:
      `outputs/aml_regression_testset_v2/aml_pattern_split_axis_jumpaxis_probe_v5_stance_width_audit_v0/`
    - Coverage audit:
      `outputs/aml_regression_testset_v2/aml_pattern_split_axis_jumpaxis_probe_v5_stance_width_coverage_v0/`
    - Same 100-target mixed probe:
      - lower-spread case coverage: 23/100 -> 47/100.
      - full-rule case coverage: 17/100 -> 40/100.
      - accepted bilateral axis cases: 86 -> 89.
      - diagnostic precision: 1.0 -> 0.9888.
      - diagnostic recall: 0.86 -> 0.88.
      - full-label accepted windows: 16 -> 39.
      - component-label accepted windows: 117 -> 82.
  - Review check:
    The upper+vertical component is reliable and should enter the component /
    closure library. The full upper+lower+vertical coordination is cleaner but
    still not a final named action node. v5 confirms that a meaningful part of
    the previous recall gap was caused by missing bilateral foot-separation
    observables, not by the coactivation-window assembler. Do not rename the
    upper-only component as a complete jumping-jack action. The remaining
    non-target confound is floor/low-body transition plus leg-strike evidence;
    handle it later with support/contact refinement instead of a case rule.
  - Full-HML3D v5 artifact:
    `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/`
  - Full-HML3D v5 composition forest:
    `outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v5_stance_width_full_v0/`
  - Full-HML3D v5 search/audit artifacts:
    `outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_full_v5_stance_width_v0/`
    and
    `outputs/aml_regression_testset_v2/aml_pattern_split_axis_full_v5_stance_width_coverage_v0/`
  - Full-HML3D v4 -> v5 result:
    - channel events: 488601 -> 512857.
    - stance-width sidecar events: 0 -> 24256.
    - coordination motifs: 64 -> 78.
    - composition structure groups: 60 -> 75.
    - bilateral-spread axis lower-spread coverage on 362 target cases:
      96 -> 168.
    - bilateral-spread full upper+lower+vertical coverage:
      78 -> 146 target cases.
  - Phase-aware closure artifact:
    `outputs/aml_regression_testset_v2/aml_pattern_split_axis_phase_closure_v5_stance_width_full_v0/`
  - Phase-aware closure result with generic arm-quality gate
    `bilateral_high_arm_pose OR large_bilateral_arm_arc`:
    - strict phase-closed cases: 441, including 131 / 362 target cases.
    - strict diagnostic precision: 0.2971.
    - strict diagnostic recall: 0.3619.
    - phase connected-or-closed cases: 782, including 144 / 362 target cases.
    - phase connected-or-closed diagnostic precision: 0.1841.
    - phase connected-or-closed diagnostic recall: 0.3978.
  - Review check:
    v5 should be kept as a generic geometry observable. Phase-aware closure
    confirms that `bilateral_spread_vertical_coordination` is a useful reusable
    structure, but it is still too broad to name directly as `jumping_jack`.
    The next gate is subtype/confound review over the phase-closed set, not a
    caption-specific rule.

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
