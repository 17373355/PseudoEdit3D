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

- `docs/design/aml_architecture_coverage.md`
- `docs/design/motion_subword_design.md`
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

