# AML Coarse Signature Pipeline

Current status: `coarse_v2` is the main AutoPrompt rendering path. The old event-stream renderer is retained only for diagnostics and ablations.

## Main Flow

```text
Raw HumanML3D motion
  -> Layer 0 frame observables
  -> Layer 1 micro events
  -> Layer 2 merged submotion units
  -> Layer 3 atomic / phase / state events
  -> Layer 3.5 event-derived coarse signature
  -> Layer 4 coarse action prototypes
  -> global alias evidence lookup
  -> canonical AML actions + probe aliases
  -> MoMask text probe / future AML condition training
```

## Layer Contract

Only Layer 0-3 may inspect raw motion. Layer 3.5 consumes Layer-3 events, spans, metadata, overlap, temporal order, and confidence. If a required cue is missing, add a clearer upstream event instead of adding hidden raw-motion logic inside the signature stage.

Example: terminal stop is detected as a Layer-3 `WHOLE_BODY_STATE/WB_TERMINAL_STILL` event from motion energy. The coarse signature only consumes that event.

## Probe Alias vs Canonical AML Encoding

These are intentionally separated.

- `probe_alias`: a natural-language phrase chosen to test a pretrained text-conditioned MoMask model.
- `canonical_actions`: structured AML condition targets for future training.
- `global alias evidence`: corpus-level HumanML3D wording statistics used to name or audit motion families. It is never same-case caption input.

Example:

```json
{
  "canonical_id": "BILATERAL_RHYTHMIC_COORDINATION",
  "probe_alias": "bilateral rhythmic arm-leg coordination",
  "slots": {
    "span": [1, 185],
    "count": 8,
    "direction": "in_place",
    "vertical_amplitude_m": 0.229
  }
}
```

The final AML-conditioned model should consume the canonical id and slots, not depend on a single surface phrase.

## Condition Manifest

Training-oriented AML conditions are exported by `scripts/export_aml_condition_manifest.py`.

The exporter writes one JSONL record per case:

- `case_id`, `num_frames`, and reference HML3D prompt for audit only;
- `conditions`: one item per canonical action;
- each condition contains `family_id`, `source_family`, `status`, `condition_weight`, `slot_values`, `slot_confidences`, `slot_qualities`, full `approx_slots`, and `missing_required_slots`.

Default condition weights:

- `stable`: `1.0`
- `candidate`: `0.7`
- `proxy`: `0.5`
- `unknown`, missing required slots, or `probe_visible=false`: `0.0`

The manifest deliberately preserves zero-weight structural conditions so audits can explain what was skipped.

### Semantic Family Status

Canonical actions now carry an explicit semantic-family descriptor:

- `stable`: the current event signature is treated as a direct AML family.
- `candidate`: a plausible higher-level family inferred from motion signatures, such as acrobatic sequence or squat repetition.
- `proxy`: a conservative observable proxy, such as raised hand, low squat hold, or climb-over proxy.
- `unknown`: no stable semantic family is assigned; source event family/cluster counts are retained for later subclustering.

This prevents a candidate phrase from silently becoming a hard label. Downstream AML-conditioned training can choose whether to use all families, only stable families, or stable plus candidate/proxy families with lower confidence.

### Approximate Slots

Each canonical action also exposes `approx_slots` alongside the older flat `slots`.

Approximate slots store:

- `value`: the current motion-derived estimate.
- `range`: a conservative numeric tolerance when the slot is numeric.
- `confidence`: confidence after semantic-family status is considered.
- `source`: whether the slot came from Layer-3 event signatures, semantic joint proxies, or span unions.
- `quality`: whether the value is a direct estimate, candidate/proxy estimate, approximate event count, or categorical estimate.

The old flat slot fields are preserved for compatibility. New training code should prefer `semantic_family.status` and `approx_slots` when it needs uncertainty-aware AML conditions.

Required slot coverage is audited by `scripts/analyze_aml_semantic_family_status.py`. The audit defines per-family required slots, including alternative requirements such as `magnitude|vertical_amplitude_m` for squat candidates, and reports missing examples in the markdown output.

### Pattern Tree Metadata

`pseudoedit3d/edit/aml_pattern_tree.json` is the runtime motion-pattern tree. It is WordNet-like in shape, but it does not import WordNet or query external lexical sources at runtime.

The tree now has three active node roles:

- `primary`: chooses the main seeded prototype from signature-level evidence.
- `event_proxy`: maps a Layer-3 `super_family/cluster_id` pair to a conservative observable semantic proxy.
- `composed_candidate`: matches temporal evidence for configured candidates such as lunge and sit/stand, and records the taxonomy path for metadata-only composed candidates such as acrobatic sequence.

Every visible coarse action whose family appears in this tree receives:

- `pattern_node_id`
- `pattern_path`
- `pattern_taxonomy_parent_id`

This makes the md/json audit path independent from Python-local proto-id lists while still keeping temporal composition logic explicit in code.

## Global Alias Evidence

`scripts/mine_hml3d_upperbody_phrases.py` scans all HumanML3D captions as a global wording inventory and associates upper-body Layer-3 motion keys with word families, surface phrases, support, coverage, precision, and lift.

This layer is only for:

- finding candidate family names;
- auditing whether a motion-derived cluster has a stable language correlate;
- constructing probe aliases for pretrained text-conditioned models.

This layer is not allowed to:

- read a case's own HML3D captions when rendering that case's motion-only AutoPrompt;
- replace canonical AML action ids and numeric slots;
- force weak text associations into action labels.

Current readout:

- reliable: `BI_RAISE_SPREAD|nonloco+vertical` strongly maps to `BILATERAL_RHYTHMIC_COORDINATION`; jumping jack is kept as a lexical alias, not the coarse family;
- reliable: `BI_HANDS_CLOSE|nonloco` strongly maps to `clap_or_hands_together`;
- unresolved: `overhead_clap_or_cheer`, `martial_strike`, `push_shove`, `support_contact`, and instrument/tool mime families are still too mixed for direct AutoPrompt naming.

Current integration policy: strong alias evidence may be attached to canonical slots, but the probe phrase remains conservative unless motion evidence directly supports the action name. Example: `BIMANUAL_HANDS_CLOSE` stores `clap_or_hands_together` as evidence but does not render as `clap`.

## Coarse Families

Seeded prototypes in `coarse_v2`:

- `TRANSLATING_GAIT`: walk/run with direction, speed, and distance.
- `IN_PLACE_GAIT`: walk/run in place.
- `IN_PLACE_GAIT_PROXY`: conservative proxy for low repeated in-place bounce with bimanual/torso evidence but insufficient arm-swing evidence for stable gait.
- `BALLISTIC_TRANSLATION`: jump with root translation, such as jump forward/backward.
- `WEAK_BALLISTIC_CANDIDATE`: low-confidence jump-like candidate saved for program inspection, hidden from probe text by default.
- `VERTICAL_JUMP`: jump in place / upward jump.
- `BILATERAL_RHYTHMIC_COORDINATION`: repeated vertical motion plus repeated bimanual raise-spread.
- `ROTATION_DOMINANT` and `TURN_SEGMENT`: turn/spin with angle bins and numeric angle.
- `TERMINAL_STILL`: motion-derived end-state hold.
- `BIMANUAL_HANDS_CLOSE`: conservative upper-body action for hands moving together. It stores `clap_or_hands_together` as global alias evidence but renders as `brings both hands together`.
- `BIMANUAL_ACTION` and `EVENT_SEQUENCE`: fallback families. They are now dropped from canonical actions when all their source events are explained by later semantic/proxy actions.

## Renderer Policy

The renderer outputs coarse action families first, then residual events not covered by the coarse family. Covered local events must not be repeated as long natural-language clauses.

The renderer may hide low-confidence structured actions from MoMask probe text while still preserving them in `canonical_actions`.

Renderer status policy:

- `stable` actions retain their normal salience.
- `candidate` actions are rendered conservatively and receive a small salience penalty.
- `proxy` actions are rendered only with observable, non-object wording and receive a larger salience penalty.
- `unknown` actions are not rendered into probe text.
- actions with `probe_visible=false`, such as no-evidence subtle proxies, are preserved structurally but hidden from probe text.

MoMask probe rendering is intentionally budgeted:

- keep temporal order for selected clauses;
- prefer coarse whole-body actions over raw residual events;
- cap default probe text to about `34` words;
- allow at most `5` coarse clauses and at most `1` residual clause;
- preserve the full `coarse_action_program` regardless of probe truncation.

This means `auto_prompt` is a compatibility probe for a pretrained text-conditioned model, not the full AML label. The full AML label is `canonical_actions` plus numeric slots and residual program evidence.

## Current Regression Observations

From group-01 first-10 preview v5:

- `M008014`: count improved from `twice` to `8 times` using event-derived evidence.
- `M008235`: no longer misclassified as `jumping jacks`; rendered as a backward jump with arm residuals.
- `M010032`: strong first forward jump is rendered; weak later jump-like candidates are preserved in canonical actions but hidden from text probe.
- `002755`: no `stand still` is emitted because motion-only terminal-still evidence is not strong enough, despite HML3D captions mentioning stop/stand still.

## Active Code Paths

Main modules:

- `pseudoedit3d/edit/coarse_signature.py`
- `pseudoedit3d/edit/aml_pattern_tree.json`: WordNet-like parent/child motion-pattern tree consumed by primary matching, event-proxy lookup, and composed-candidate metadata attachment.
- `pseudoedit3d/edit/aml_pattern_tree.py`: generic tree matcher, output resolver, event-proxy lookup, and family-to-pattern metadata helper.
- `pseudoedit3d/edit/aml_proto_registry.json`: runtime registry for semantic status groups, cover/suppression groups, emitter templates, probe aliases, renderer clauses/salience, fallback entrypoints, and condition required slots.
- `pseudoedit3d/edit/aml_proto_registry.py`: shared registry reader used by runtime modules and condition schema.
- `pseudoedit3d/edit/aml_family_taxonomy.json`: taxonomy and lexical-source metadata; WordNet is an offline cached source, not a runtime detector.
- `pseudoedit3d/edit/aml_language_coverage_specs.json`: weak-label HML3D caption coverage audit specs used by `scripts/audit_aml_language_coverage.py`.
- `pseudoedit3d/edit/aml_wordnet_lexicon_config.json`: offline WordNet builder seed/regex config used by `scripts/build_wordnet_action_lexicon.py`.
- `pseudoedit3d/edit/coarse_prompt_renderer.py`
- `scripts/run_momask_aml_autoprompt_probe.py --prompt-mode coarse`

Diagnostic / legacy modules:

- `pseudoedit3d/edit/aml_prompt_renderer.py`: direct event-stream naturalizer.
- `scripts/run_momask_aml_prompt_probe.py`: old selected-HML3D-vs-event-stream comparison probe.

## Latest Outputs

- v5 prompt preview: `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v5/summary.json`
- previous coarse v1 MoMask GIFs: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_coarse_v1_gifs_hml3d_stride4/`
- semantic-family status 250-case diagnostic: `outputs/aml_regression_testset_v2/semantic_status_250_after_step2_v1/semantic_family_status.md`
- required-slot audit: `outputs/aml_regression_testset_v2/semantic_slot_audit_250_v3/semantic_family_status.md`
- renderer policy preview: `outputs/aml_regression_testset_v2/semantic_renderer_step4_preview_v1/summary.json`
- AML condition manifest: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/conditions.jsonl`
- AML condition manifest summary: `outputs/aml_regression_testset_v2/aml_condition_manifest_250_v1/summary.md`
- pattern-tree event-proxy preview: `outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/summary.json`
- pattern-tree event-proxy condition summary: `outputs/aml_regression_testset_v2/pattern_tree_event_proxy_preview_v1/conditions_summary.md`
- pattern-tree composed preview: `outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/summary.json`
- pattern-tree composed condition summary: `outputs/aml_regression_testset_v2/pattern_tree_composed_preview_v1/conditions_summary.md`
- pattern-tree registry consistency refactor preview: `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/summary.json`
- pattern-tree registry consistency condition summary: `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/conditions_summary.md`
- pattern-tree registry consistency language coverage audit: `outputs/aml_regression_testset_v2/pattern_tree_consistency_refactor_preview_v1/language_coverage_audit/coverage_report.md`

## Remaining Issues

- Pure MoMask text conditioning still struggles with ordered numeric programs.
- Long multi-action motions need temporal chunking and program conditioning; the current budgeted probe only prevents MoMask from being overwhelmed by text length.
- Upper-body semantic families such as clapping overhead, cheering arms, martial strikes, pushing/shoving, support contact, instrument/tool mime, and dance gestures need finer event signatures before alias evidence can be safely used.
- `TERMINAL_STILL` is intentionally conservative and should be evaluated on more terminal-pause cases.
- No-evidence clips are intentionally kept as hidden `STATIC_OR_SUBTLE_STATE_PROXY` actions. They should not become object/tool labels without additional motion or external evidence.
