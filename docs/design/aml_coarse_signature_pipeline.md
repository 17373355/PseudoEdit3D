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
  "canonical_id": "JUMPING_JACK",
  "probe_alias": "jumping jacks",
  "slots": {
    "span": [1, 185],
    "count": 8,
    "direction": "in_place",
    "vertical_amplitude_m": 0.229
  }
}
```

The final AML-conditioned model should consume the canonical id and slots, not depend on a single surface phrase.

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

- reliable: `BI_RAISE_SPREAD|nonloco+vertical` strongly maps to `jumping_jack`;
- reliable: `BI_HANDS_CLOSE|nonloco` strongly maps to `clap_or_hands_together`;
- unresolved: `overhead_clap_or_cheer`, `martial_strike`, `push_shove`, `support_contact`, and instrument/tool mime families are still too mixed for direct AutoPrompt naming.

Current integration policy: strong alias evidence may be attached to canonical slots, but the probe phrase remains conservative unless motion evidence directly supports the action name. Example: `BIMANUAL_HANDS_CLOSE` stores `clap_or_hands_together` as evidence but does not render as `clap`.

## Coarse Families

Seeded prototypes in `coarse_v2`:

- `TRANSLATING_GAIT`: walk/run with direction, speed, and distance.
- `IN_PLACE_GAIT`: walk/run in place.
- `BALLISTIC_TRANSLATION`: jump with root translation, such as jump forward/backward.
- `WEAK_BALLISTIC_CANDIDATE`: low-confidence jump-like candidate saved for program inspection, hidden from probe text by default.
- `VERTICAL_JUMP`: jump in place / upward jump.
- `JUMPING_JACK`: repeated vertical motion plus repeated bimanual raise-spread.
- `ROTATION_DOMINANT` and `TURN_SEGMENT`: turn/spin with angle bins and numeric angle.
- `TERMINAL_STILL`: motion-derived end-state hold.
- `BIMANUAL_HANDS_CLOSE`: conservative upper-body action for hands moving together. It stores `clap_or_hands_together` as global alias evidence but renders as `brings both hands together`.
- `BIMANUAL_ACTION` and `EVENT_SEQUENCE`: fallback families.

## Renderer Policy

The renderer outputs coarse action families first, then residual events not covered by the coarse family. Covered local events must not be repeated as long natural-language clauses.

The renderer may hide low-confidence structured actions from MoMask probe text while still preserving them in `canonical_actions`.

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
- `pseudoedit3d/edit/coarse_prompt_renderer.py`
- `scripts/run_momask_aml_autoprompt_probe.py --prompt-mode coarse`

Diagnostic / legacy modules:

- `pseudoedit3d/edit/aml_prompt_renderer.py`: direct event-stream naturalizer.
- `scripts/run_momask_aml_prompt_probe.py`: old selected-HML3D-vs-event-stream comparison probe.

## Latest Outputs

- v5 prompt preview: `outputs/aml_regression_testset_v2/group_01_coarse_prompt_first10_preview_v5/summary.json`
- previous coarse v1 MoMask GIFs: `outputs/aml_regression_testset_v2/group_01_momask_auto_gt_first10_coarse_v1_gifs_hml3d_stride4/`

## Remaining Issues

- Pure MoMask text conditioning still struggles with ordered numeric programs.
- Long multi-action motions need temporal chunking and program conditioning; the current budgeted probe only prevents MoMask from being overwhelmed by text length.
- Upper-body semantic families such as clapping overhead, cheering arms, martial strikes, pushing/shoving, support contact, instrument/tool mime, and dance gestures need finer event signatures before alias evidence can be safely used.
- `TERMINAL_STILL` is intentionally conservative and should be evaluated on more terminal-pause cases.
