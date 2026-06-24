# AML Pattern Mining Explorer v1

This document freezes the current AML motion-structure discovery work as an
explorer, not as a single Motion-BPE method.

## Rename / Positioning

Use this name for the current system:

```text
AML Pattern Mining Explorer
```

Motion-BPE is now only one optional candidate miner inside the explorer. It is
not the whole method and should not own the main narrative.

## Golden Path

```text
motion evidence extraction
-> candidate pattern mining
-> candidate audit
-> pattern registry
```

The four layers are:

1. Evidence Layer
   - Inputs: raw joints, Layer3 events, sidecar events.
   - Goal: high recall, local measurable signals.
   - Examples: support state, stance width, hand proximity, arm trajectory,
     body level, leg lateral movement.

2. Candidate Miner
   - Inputs: evidence cases and event groups.
   - Goal: propose reusable structure candidates.
   - Main miners: coactivation closure and closed itemsets.
   - Optional miner: Motion-BPE for single-channel temporal motifs,
     repetition, local phase, and baseline comparison.

3. Audit Layer
   - Inputs: candidate patterns plus evidence cases.
   - Goal: decide whether a candidate is a full pattern, component,
     split-required node, or blocked label.
   - Audits: split-axis, phase/order, pseudo-GT caption diagnostics,
     naming/WordNet, future TMR.
   - Captions, WordNet, and TMR never create motion evidence.

4. Pattern Registry
   - Inputs: audited candidates.
   - Goal: produce the compact vocabulary for downstream AML condition work.
   - Status values: `accepted`, `component`, `split_required`, `blocked`,
     `review_candidate`.

## Core Artifacts

The v1 golden path should converge to these files:

```text
evidence_cases.jsonl
candidate_patterns.jsonl
pattern_registry.json
audit_report.md
```

### `evidence_cases.jsonl`

One row per motion case. It should contain motion evidence summaries and review
metadata only. It may include captions for auditing, but captions are not
matching rules.

### `candidate_patterns.jsonl`

One row per candidate pattern. Recommended schema:

```json
{
  "pattern_id": "axis:bilateral_spread_vertical_coordination_v0:strict_phase_closed",
  "source": "split_axis_phase_audit",
  "evidence_groups": ["upper_spread", "vertical_rhythm", "lower_spread"],
  "required_groups": ["upper_spread", "vertical_rhythm", "lower_spread"],
  "optional_groups": ["bilateral_high_arm_pose", "large_bilateral_arm_arc"],
  "negative_groups": ["floor_or_inverted_support_confound"],
  "phase_status": "phase_closed_all_pairs",
  "support_cases": 441,
  "support_windows": null,
  "naming_diagnostics": {
    "target_alias": "jumping_jack",
    "precision": 0.2971,
    "recall": 0.3619
  },
  "status": "component",
  "examples": []
}
```

### `pattern_registry.json`

A compact registry for reviewed/usable nodes. It should not include every debug
candidate. It should contain accepted full patterns, reusable components,
split-required structures, and blocked naming labels.

### `audit_report.md`

Human-readable summary of what changed, what is reliable, and what remains
blocked.

## Current Frozen v5 Explorer Inputs

Current evidence/mining/audit inputs:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/
outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v5_stance_width_full_v0/
outputs/aml_regression_testset_v2/aml_pattern_split_axis_full_v5_stance_width_coverage_v0/
outputs/aml_regression_testset_v2/aml_pattern_split_axis_phase_closure_v5_stance_width_full_v0/
outputs/aml_regression_testset_v2/aml_pattern_axis_audit_v1_bilateral_spread_v5/
outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_reviewed_draft/
```

Current consolidated v1 bundle output:

```text
outputs/aml_regression_testset_v2/aml_pattern_mining_explorer_v1/
```

## What Is No Longer The Main Path

The following are historical/exploratory and should not be extended as the main
pipeline:

- pre-v1 motion forest variants,
- promotion-table draft forests,
- multiple parallel `forest/proposal/program/search` formats,
- Motion-BPE as the sole pattern discovery backbone,
- caption keyword matching as a mining rule.

They can remain in `legacy/` for reproducibility.

## Current Research Conclusion

The current system has already shown:

- Skeleton-only structure is not the same as an action name.
- Jumping-jack-like motion requires upper/lower/vertical/phase composition.
- Support state and stance width are real missing evidence dimensions.
- Phase/order audit is necessary because co-presence is not a motion cycle.
- Many labels must stay as component, approximate, split-required, or blocked.

Therefore the v1 main method is:

```text
evidence groups
-> coactivation / closure candidates
-> phase + split audit
-> pattern registry
```

Motion-BPE is kept as an optional local temporal motif miner and baseline, not
as the system skeleton.
