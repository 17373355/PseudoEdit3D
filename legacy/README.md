# Legacy Area

This folder stores mechanisms that are useful for historical reference but are no longer the AML mainline.

Migration rule:

- do not delete research artifacts directly
- move only files that are not imported by active AML scripts
- record original path, reason, and replacement

Current target groups:

- `auto_prompt_pattern_batches/`: old HumanML3D batch-wise auto-prompt repair workflow
- `aml_demos/`: layer-specific demo scripts superseded by unified AML extraction/visualization
- `text_phrase_mining/`: earlier hard-coded HumanML3D text phrase mining probes
- `old_stage1_atomic_scaffold/`: earlier training scaffold, only after dependency audit

Current AML mainline is documented in:

- `docs/design/motion_corpus_pattern_tree_mainline.md`
- `docs/design/motion_cluster_bpe_tree_induction.md`
- `docs/design/text_bpe_wordnet_naming_layer.md`
- `docs/design/aml_architecture_coverage.md`
- `docs/design/motion_subword_design.md`
- `docs/design/aml_edit_call_graph.md`
- `docs/experiment_log.md`

No active file should import from this folder.

## Moved Files

### 2026-06-07

- `scripts/run_momask_case_study.py.bak` -> `legacy/auto_prompt_pattern_batches/scripts/run_momask_case_study.py.bak`
  - reason: backup copy of old MoMask auto-prompt probe script
  - replacement: active probe remains `scripts/run_momask_case_study.py`; AML mainline uses `scripts/extract_aml_layers.py` and `scripts/visualize_aml_atomic_program.py`
- `scripts/visualize_momask_case_study.py.bak` -> `legacy/auto_prompt_pattern_batches/scripts/visualize_momask_case_study.py.bak`
  - reason: backup copy of old MoMask case-study visualizer
  - replacement: active probe remains `scripts/visualize_momask_case_study.py`; AML mainline uses `scripts/visualize_aml_atomic_program.py`

### 2026-06-10

- `scripts/dump_micro_events_demo.py` -> `legacy/aml_demos/scripts/dump_micro_events_demo.py`
  - reason: layer-specific demo script superseded by unified AML extraction
  - replacement: `scripts/extract_aml_layers.py`
- `scripts/visualize_micro_events.py` -> `legacy/aml_demos/scripts/visualize_micro_events.py`
  - reason: layer-specific visualizer superseded by unified AML visualization
  - replacement: `scripts/visualize_aml_atomic_program.py`
- `scripts/detect_phase_patterns_demo.py` -> `legacy/aml_demos/scripts/detect_phase_patterns_demo.py`
  - reason: layer-specific phase demo superseded by unified AML extraction
  - replacement: `scripts/extract_aml_layers.py`
- `scripts/visualize_phase_patterns.py` -> `legacy/aml_demos/scripts/visualize_phase_patterns.py`
  - reason: layer-specific visualizer superseded by unified AML visualization
  - replacement: `scripts/visualize_aml_atomic_program.py`
- `scripts/merge_submotion_demo.py` -> `legacy/aml_demos/scripts/merge_submotion_demo.py`
  - reason: layer-specific submotion demo superseded by unified AML extraction
  - replacement: `scripts/extract_aml_layers.py`
- `scripts/visualize_submotions.py` -> `legacy/aml_demos/scripts/visualize_submotions.py`
  - reason: layer-specific visualizer superseded by unified AML visualization
  - replacement: `scripts/visualize_aml_atomic_program.py`
- `scripts/summarize_phase_hierarchy.py` -> `legacy/aml_demos/scripts/summarize_phase_hierarchy.py`
  - reason: layer-specific hierarchy summary superseded by current AML taxonomy/visualization scripts
  - replacement: `scripts/summarize_aml_family_taxonomy.py` and `scripts/extract_aml_layers.py`
- `scripts/summarize_aml_hierarchy.py` -> `legacy/aml_demos/scripts/summarize_aml_hierarchy.py`
  - reason: older hierarchy summary superseded by current AML taxonomy/visualization scripts
  - replacement: `scripts/summarize_aml_family_taxonomy.py` and `scripts/extract_aml_layers.py`
- `pseudoedit3d/edit/iterative.py` -> `legacy/old_stage1_iterative_refinement/pseudoedit3d/edit/iterative.py`
  - reason: old mined-pair iterative refinement loop; not imported by active AML coarse action path
  - replacement: current AML mainline uses `pseudoedit3d/edit/coarse_signature.py` and `scripts/analyze_aml_semantic_family_status.py`
- `scripts/iterate_mined_pairs.py` -> `legacy/old_stage1_iterative_refinement/scripts/iterate_mined_pairs.py`
  - reason: command-line wrapper for the old mined-pair iterative refinement loop
  - replacement: no active replacement; keep for historical Stage 1 experiments only

### 2026-06-15

- `docs/design/motion_bpe_baseline.md` -> `legacy/motion_bpe_baseline/docs/design/motion_bpe_baseline.md`
  - reason: old Layer2 BPE baseline framing is superseded by the full-HML3D Layer3 event-BPE audit and the corpus-derived pattern-tree mainline
  - replacement: `docs/design/motion_corpus_pattern_tree_mainline.md`, `docs/design/motion_cluster_bpe_tree_induction.md`, and `scripts/audit_hml3d_layer3_event_bpe.py`
- `scripts/learn_motion_bpe.py` -> `legacy/motion_bpe_baseline/scripts/learn_motion_bpe.py`
  - reason: old case-list Layer2 submotion BPE script is not imported by active AML scripts and does not produce the new full-corpus tree-induction artifacts
  - replacement: `scripts/audit_hml3d_layer3_event_bpe.py`

### 2026-06-16

- `scripts/build_hml3d_pattern_batch.py` -> `legacy/auto_prompt_pattern_batches/scripts/build_hml3d_pattern_batch.py`
  - reason: old batch-wise HML3D auto-prompt pattern repair workflow; already listed as candidate legacy
  - replacement: current review-set and Motion-BPE workflows use `scripts/build_aml_regression_testset.py`, `scripts/run_aml_momask_review_pack.py`, and `scripts/audit_hml3d_multichannel_motion_bpe.py`
- `scripts/analyze_hml3d_pattern_batch.py` -> `legacy/auto_prompt_pattern_batches/scripts/analyze_hml3d_pattern_batch.py`
  - reason: old analyzer for the batch-wise auto-prompt repair workflow
  - replacement: `scripts/audit_hml3d_multichannel_motion_bpe.py` and generated `audit_report.md`
- `scripts/report_hml3d_batch_sizes.py` -> `legacy/auto_prompt_pattern_batches/scripts/report_hml3d_batch_sizes.py`
  - reason: helper for the old pattern-batch workflow
  - replacement: `scripts/build_aml_regression_testset.py` and `scripts/audit_hml3d_multichannel_motion_bpe.py`
- `scripts/mine_hml3d_missing_patterns.py` -> `legacy/auto_prompt_pattern_batches/scripts/mine_hml3d_missing_patterns.py`
  - reason: caption-pattern gap miner tied to old prompt-pattern batches
  - replacement: text naming now goes through `scripts/build_text_bpe_wordnet_naming_layer.py`
- `scripts/mine_submotion_pairs.py` -> `legacy/early_submotion_mining/scripts/mine_submotion_pairs.py`
  - reason: early micro-event pair/triple mining draft superseded by full-HML3D event-BPE and multi-channel Motion-BPE audits
  - replacement: `scripts/audit_hml3d_layer3_event_bpe.py` and `scripts/audit_hml3d_multichannel_motion_bpe.py`
- `scripts/mine_submotion_candidates.py` -> `legacy/early_submotion_mining/scripts/mine_submotion_candidates.py`
  - reason: early case-list submotion candidate miner superseded by corpus-level BPE audits
  - replacement: `scripts/audit_hml3d_layer3_event_bpe.py` and `scripts/audit_hml3d_multichannel_motion_bpe.py`
- `scripts/mine_projected_submotion_candidates.py` -> `legacy/early_submotion_mining/scripts/mine_projected_submotion_candidates.py`
  - reason: early projected-stream pair/triple mining draft superseded by multi-channel event/packet representation
  - replacement: `scripts/audit_hml3d_multichannel_motion_bpe.py`

### 2026-06-23

- `scripts/mine_hml3d_upperbody_phrases.py` -> `legacy/text_phrase_mining/scripts/mine_hml3d_upperbody_phrases.py`
  - reason: earlier hard-coded upper-body text phrase scanner is superseded by the target-registry based text pseudo-GT audit path
  - replacement: `configs/motion_pattern_text_targets.json`, `scripts/audit_motion_pattern_pseudo_gt.py`, and `scripts/audit_motion_pattern_recall_candidates.py`

### 2026-06-24

Moved pre-v1 AML Pattern Mining Explorer scripts to `legacy/aml_pattern_mining_pre_v1/scripts/`.

Reason: the active mainline has been renamed and narrowed to `AML Pattern Mining Explorer v1`:
`motion evidence extraction -> candidate pattern mining -> candidate audit -> pattern registry`.
The moved scripts are historical BPE-first forests, promotion/proposal drafts, and older parallel
forest/program formats. They are kept for reproducibility but should not be extended as the main path.

Replacement: see `docs/design/aml_pattern_mining_explorer_v1.md` and `docs/design/script_inventory.md`.
Active entrypoints are `scripts/run_pattern_axis_audit.py` and
`scripts/export_pattern_mining_explorer_bundle_v1.py`.

Moved files:

- `scripts/promote_coordination_motif_candidates.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/promote_coordination_motif_candidates.py`
- `scripts/build_coordination_pattern_forest.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_coordination_pattern_forest.py`
- `scripts/propose_motion_pattern_tree_candidates.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/propose_motion_pattern_tree_candidates.py`
- `scripts/build_text_bpe_wordnet_naming_layer.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_text_bpe_wordnet_naming_layer.py`
- `scripts/build_promoted_motion_tree_draft.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_promoted_motion_tree_draft.py`
- `scripts/plan_motion_node_splits.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/plan_motion_node_splits.py`
- `scripts/build_motion_split_proposals.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_motion_split_proposals.py`
- `scripts/render_motion_split_review_artifacts.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/render_motion_split_review_artifacts.py`
- `scripts/build_motion_pattern_forest.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_motion_pattern_forest.py`
- `scripts/build_full_candidate_motion_forest.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_full_candidate_motion_forest.py`
- `scripts/render_motion_forest_review_pack.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/render_motion_forest_review_pack.py`
- `scripts/build_aml_pattern_forest_dense_v0.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_aml_pattern_forest_dense_v0.py`
- `scripts/build_aml_pattern_forest_v0.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_aml_pattern_forest_v0.py`
- `scripts/build_aml_pattern_promotion_review_v0.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_aml_pattern_promotion_review_v0.py`
- `scripts/review_aml_pattern_promotion_table_v0.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/review_aml_pattern_promotion_table_v0.py`
- `scripts/build_v4_closure_pattern_forest_draft.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_v4_closure_pattern_forest_draft.py`
- `scripts/build_v4_coord_role_composition_closure.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_v4_coord_role_composition_closure.py`
- `scripts/build_v4_coord_role_promotion_review.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_v4_coord_role_promotion_review.py`
- `scripts/render_v4_closure_pattern_forest_review_pack.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/render_v4_closure_pattern_forest_review_pack.py`
- `scripts/build_v1_support_state_reviewed_forest_draft.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/build_v1_support_state_reviewed_forest_draft.py`
- `scripts/propose_v1_support_state_review_decisions.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/propose_v1_support_state_review_decisions.py`
- `scripts/audit_v1_support_state_promotion_candidates.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/audit_v1_support_state_promotion_candidates.py`
- `scripts/export_aml_composable_pattern_program_v0.py` -> `legacy/aml_pattern_mining_pre_v1/scripts/export_aml_composable_pattern_program_v0.py`
