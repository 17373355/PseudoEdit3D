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
| G2. Multi-channel Motion-BPE | learn channel motifs and cross-channel coordination motifs | `[~] implemented v1` | `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/` | inspect purity; decide v2 merge/extraction changes |
| G3. Motion candidate forest | group motion-derived motifs into reviewable family nodes | `[~] implemented v1` | `motion_pattern_forest_candidates.json`, `coordination_pattern_forest_loose_v1/` | separate component motifs from full action motifs |
| G4. Manual pseudo-GT audit | test known weak targets without creating rules | `[~] self-reviewed, needs user spot-check` | `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/manual_text_target_self_review.md` | user spot-check `cartwheel`, `sit`, `swim` split decisions |
| G5. Caption/WordNet naming | attach language names to existing motion nodes | `[~] implemented v0` | `outputs/aml_regression_testset_v2/hml3d_caption_wordnet_name_candidates_v0/` | filter low-quality n-grams and classify phrase types |
| G6. Reviewed AML pattern forest | promote accepted motion nodes into AML vocabulary | `[ ] not started` | proposed `outputs/aml_regression_testset_v2/aml_pattern_forest_v0/` | depends on G4/G5 review |
| G7. AML condition schema | define trainable condition representation | `[ ] not started` | proposed `docs/design/aml_condition_schema_v0.md` | depends on reviewed forest node schema |
| G8. Condition encoder data | export train/eval condition batches | `[ ] not started` | proposed `aml_condition_batch_schema_v0/` | depends on G7 |
| G9. Smoke training | train/evaluate small AML condition encoder | `[ ] not started` | proposed config/checkpoint/eval report | depends on condition batch |
| G10. MoMask visual audit | render selected prompts/motions for semantic inspection | `[ ] not started for v0` | proposed GIF review pack | depends on usable v0 prompts/conditions |
| G11. Full training decision | decide whether to scale | `[ ] not started` | training plan + acceptance metrics | gated by G2/G4/G5/G9 |

## Current Snapshot

Implemented but not fully reviewed:

- Full HumanML3D Layer3 corpus exists.
- Multi-channel Motion-BPE v1 exists.
- Coordination candidate forest exists.
- Manual registry audit exists for 14 text targets.
- Caption/WordNet naming sidecar exists.
- v0 pipeline design doc exists.

Not yet started:

- Reviewed AML pattern forest.
- Condition schema v0.
- Condition encoder data export.
- Training smoke test.
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
- We have not yet defined the reviewed pattern-node schema or condition schema.

## Current Critical Path

```text
review high-signal target audits
-> classify failure reasons
-> improve Motion-BPE v2 / naming filters
-> promote reviewed pattern forest v0
-> define condition schema
-> export condition batches
-> smoke train condition encoder
-> MoMask visual audit
-> decide full training
```

The current immediate checkpoint is not training. The immediate checkpoint is
whether the v0 Motion-BPE/naming outputs are clean enough to promote a small
reviewed AML pattern forest.

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

- [ ] Add better channel observables only where the audit proves a gap.
  - Candidate gaps: arm spin direction, sit/stand vertical-pose transition,
    kneel/contact, sport-like unilateral arm swing, object-mime repeated hand path.
  - Artifact: updated Layer3/multichannel record schema, not case-specific rules.
  - Review check: new observable improves multiple cases or a stable family, not one case.

- [ ] Improve multichannel BPE merge policy.
  - Artifact: new versioned output directory, e.g.
    `hml3d_multichannel_motion_bpe_coord_sig_full_v2/`
  - Review check: compare v1 vs v2 on compression, motif purity, target audit recall,
    and false-positive rate.

- [ ] Separate component motifs from full action motifs.
  - Example: jumping-jack arm raise-spread alone should be a component; arm+vertical
    coordination may be a full pattern candidate.
  - Artifact: motif family schema field such as `motif_scope=component|coordination|sequence`.
  - Review check: forest no longer treats common components as complete action names.

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
  - Review check: every node keeps span/channel/numeric provenance.

- [ ] Promote reviewed component/full-action nodes into a v0 forest.
  - Artifact: `outputs/aml_regression_testset_v2/aml_pattern_forest_v0/`
  - Review check: forest contains accepted nodes only; rejected/diagnostic nodes stay sidecar.

- [ ] Map forest nodes to condition slots.
  - Artifact: `docs/design/aml_condition_schema_v0.md`
  - Review check: condition slots support body part, span, count, direction,
    magnitude, speed, and confidence.

## P4: Condition Encoder And Training Iteration

Goal: connect the reviewed AML vocabulary to trainable condition data.

- [ ] Freeze v0 condition batch schema.
  - Artifact: `outputs/.../aml_condition_batch_schema_v0/`
  - Review check: schema can represent both component motifs and full action motifs.

- [ ] Export train/eval condition batches.
  - Artifact: versioned train/eval JSONL or NPZ condition files.
  - Review check: no text registry leakage into training labels except designated
    audit metadata.

- [ ] Train a small AML condition encoder smoke model.
  - Artifact: config, checkpoint path, and smoke metrics.
  - Review check: overfits a small subset and preserves condition alignment.

- [ ] Run MoMask/autoprompt semantic review on selected cases.
  - Artifact: GIF review pack with GT, prompt, generated motion.
  - Review check: compare against previous group_01 failure list.

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
6. Decide Motion-BPE v2 changes.
