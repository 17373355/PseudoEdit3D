# Script Inventory

This document keeps the active `scripts/` folder readable. Files moved to
`legacy/` are recorded in `legacy/README.md`.

## Motion-Corpus / BPE Mainline

- `audit_hml3d_layer3_event_bpe.py`
- `audit_hml3d_multichannel_motion_bpe.py`
- `promote_coordination_motif_candidates.py`
- `build_coordination_pattern_forest.py`
- `audit_motion_pattern_pseudo_gt.py`
- `audit_motion_pattern_recall_candidates.py`
- `build_motion_pattern_family_proposals.py`
- `run_motion_pattern_registry_audits.py`
- `mine_hml3d_caption_wordnet_name_candidates_v0.py`
- `propose_motion_pattern_tree_candidates.py`
- `build_text_bpe_wordnet_naming_layer.py`
- `build_promoted_motion_tree_draft.py`
- `plan_motion_node_splits.py`
- `build_motion_split_proposals.py`
- `render_motion_split_review_artifacts.py`
- `build_motion_pattern_forest.py`
- `build_full_candidate_motion_forest.py`
- `render_motion_forest_review_pack.py`

Purpose:

```text
HumanML3D motion corpus
-> event/multichannel Motion-BPE
-> motif family audit
-> motion pattern forest
-> text/WordNet naming diagnostics
```

## AML Condition / Dataset Mainline

- `build_aml_mining_corpus.py`
- `build_aml_regression_testset.py`
- `export_aml_condition_manifest.py`
- `export_aml_condition_batch_schema.py`
- `export_aml_condition_motion_batch.py`
- `export_aml_geometry_sidecar.py`
- `screen_aml_conditions.py`
- `audit_aml_condition_contract.py`
- `smoke_aml_condition_motion_loader.py`
- `extract_aml_layers.py`
- `summarize_aml_family_taxonomy.py`

Purpose:

```text
AML extraction
-> condition manifest/schema
-> train-ready condition data
```

## MoMask / Visual Review

- `run_aml_momask_review_pack.py`
- `run_momask_aml_autoprompt_probe.py`
- `run_momask_aml_prompt_probe.py`
- `run_momask_batch_from_probe_summary.py`
- `run_momask_case_study.py`
- `visualize_momask_auto_gt.py`
- `visualize_momask_case_study.py`
- `analyze_momask_probe_kinematics.py`

Purpose:

```text
AML AutoPrompt / selected prompt
-> MoMask generation
-> GT vs generated visual review
```

## Active Diagnostics

- `analyze_aml_semantic_family_status.py`
- `audit_aml_language_coverage.py`
- `audit_aml_unmapped_geometry.py`
- `scan_aml_full_clusters.py`
- `summarize_aml_cluster_scan.py`
- `select_aml_cluster_representatives.py`
- `select_aml_split_candidate_representatives.py`
- `analyze_aml_prompt_phrases.py`
- `analyze_vertical_salience.py`
- `analyze_coordination_patterns.py`
- `analyze_bimanual_split_candidates.py`
- `visualize_bimanual_cluster_contact_sheet.py`
- `visualize_bimanual_split_report.py`

Purpose:

```text
Focused audits for known weak areas.
Keep while their findings still feed current AML/Motion-BPE decisions.
```

## General Project Utilities / Older Training

- `scan_dataset.py`
- `build_attribute_cache.py`
- `build_artifacts_from_manifest.py`
- `mine_triplets.py`
- `run_full_pipeline.sh`
- `run_subset_pipeline.py`
- `check_autoprompt_regression.py`
- `compare_scratch_vs_momask.py`
- `eval_prefix_completion.py`
- `preview_prompts.py`
- `infer_and_visualize.py`
- `build_group_split.py`
- `build_hml3d_case_index.py`
- `build_humanml3d_action_lexicon.py`
- `build_wordnet_action_lexicon.py`
- `export_embody3d_chunk_manifest.py`
- `export_momask_multi_atomic_bridge.py`
- `audit_embody3d_daylife.py`
- `visualize_aml_atomic_program.py`
- `visualize_aml_report_artifacts.py`

Purpose:

```text
General data prep, older stage-1 training/eval utilities, lexicon builders,
and visualization helpers. Review individually before moving.
```

## Recently Moved To Legacy

- `legacy/auto_prompt_pattern_batches/scripts/`
- `legacy/early_submotion_mining/scripts/`
- `legacy/motion_bpe_baseline/scripts/`
- `legacy/text_phrase_mining/scripts/`

Do not import from `legacy/` in active scripts.
