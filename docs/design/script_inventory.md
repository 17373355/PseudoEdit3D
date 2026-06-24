# Script Inventory

This document keeps the active `scripts/` folder readable. Files moved to
`legacy/` are recorded in `legacy/README.md`.

## AML Pattern Mining Explorer v1 Golden Path

This is the current mainline. New work should extend this path instead of
creating another parallel forest/proposal/program format.

```text
motion evidence extraction
-> candidate pattern mining
-> candidate audit
-> pattern registry / bundle export
```

Active scripts:

- `audit_hml3d_multichannel_motion_bpe.py`
  - frozen v5 evidence extractor / optional Motion-BPE miner
  - keep for reproducibility; do not keep adding v6/v7 logic here
- `build_hml3d_composition_pattern_forest_v0.py`
  - coactivation / composition candidate mining from multichannel evidence
- `export_aml_composable_pattern_program_v1_support_state.py`
  - current reviewed support-state program exporter
- `search_aml_composable_pattern_program_v0.py`
  - temporary program-search bridge used to make support-state search outputs
- `audit_split_axis_case_coverage.py`
  - split-axis coverage audit helper
- `audit_v1_support_state_split_axes.py`
  - split-axis window scoring helper
- `audit_split_axis_phase_closure.py`
  - phase/order closure audit helper
- `run_pattern_axis_audit.py`
  - preferred unified entrypoint for coverage + split + phase axis audits
- `export_pattern_mining_explorer_bundle_v1.py`
  - preferred final exporter for the four v1 artifacts:
    `evidence_cases.jsonl`, `candidate_patterns.jsonl`,
    `pattern_registry.json`, `audit_report.md`

Primary design doc:

- `docs/design/aml_pattern_mining_explorer_v1.md`

Current bundle:

- `outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/`

## Optional Mining / Naming Diagnostics

These are useful probes, but they are not the golden path and should not define
motion structure by themselves. Captions, WordNet, and future TMR belong to the
audit/naming layer, not the mining layer.

- `audit_hml3d_layer3_event_bpe.py`
- `audit_hml3d_coactivation_recall_v0.py`
- `audit_motion_pattern_pseudo_gt.py`
- `audit_motion_pattern_recall_candidates.py`
- `build_motion_pattern_family_proposals.py`
- `run_motion_pattern_registry_audits.py`
- `mine_hml3d_caption_wordnet_name_candidates_v0.py`
- `build_humanml3d_action_lexicon.py`
- `build_wordnet_action_lexicon.py`

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
- `audit_aml_program_condition_manifest_v0.py`
- `export_aml_program_condition_manifest_v0.py`
- `filter_aml_program_condition_manifest_v0.py`
- `train_aml_condition_encoder_smoke.py`

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
- `build_momask_review_failure_ledger_v0.py`

Purpose:

```text
AML AutoPrompt / selected prompt
-> MoMask generation
-> GT vs generated visual review
```

## Focused AML Diagnostics

Keep these while their findings still feed the current extractor / registry
work. Move them to legacy once the issue is folded into v1 evidence groups.

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
- `export_embody3d_chunk_manifest.py`
- `export_momask_multi_atomic_bridge.py`
- `audit_embody3d_daylife.py`
- `visualize_aml_atomic_program.py`
- `visualize_aml_report_artifacts.py`

Purpose:

```text
General data prep, older stage-1 training/eval utilities, and visualization
helpers. Review individually before moving.
```

## Legacy Groups

- `legacy/aml_pattern_mining_pre_v1/scripts/`
  - pre-v1 BPE-first forests, proposal drafts, promotion tables, and old
    parallel tree/program formats
- `legacy/auto_prompt_pattern_batches/scripts/`
- `legacy/early_submotion_mining/scripts/`
- `legacy/motion_bpe_baseline/scripts/`
- `legacy/text_phrase_mining/scripts/`

Do not import from `legacy/` in active scripts.
