# Legacy Area

This folder stores mechanisms that are useful for historical reference but are no longer the AML mainline.

Migration rule:

- do not delete research artifacts directly
- move only files that are not imported by active AML scripts
- record original path, reason, and replacement

Current target groups:

- `auto_prompt_pattern_batches/`: old HumanML3D batch-wise auto-prompt repair workflow
- `aml_demos/`: layer-specific demo scripts superseded by unified AML extraction/visualization
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
