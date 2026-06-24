# AML Pattern Node Schema v0

This document describes the first reviewed motion pattern forest:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_v0/
```

The forest is an offline vocabulary artifact. It is not runtime matching logic
and does not encode case-specific action rules.

## Build Command

```bash
python scripts/build_aml_pattern_forest_v0.py
```

Inputs:

- `configs/aml_pattern_forest_v0_review_policy.json`
- `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/*/pattern_family_proposal.json`
- `outputs/aml_regression_testset_v2/manual_text_target_audits_v0/manual_text_target_self_review.json`

Outputs:

- `aml_pattern_forest.json`: full forest with source examples.
- `aml_pattern_forest_compact.json`: compact program-facing forest.
- `aml_pattern_forest_tree.txt`: human-readable tree.
- `aml_pattern_forest_review.md`: review report.
- `summary.json`: counts.

## Node Fields

- `node_id`: stable node identifier.
- `node_kind`: `root` or `pattern_node`.
- `status`: review state.
- `scope`: how the node may be used.
- `accepted_name`: current human-readable name.
- `language_aliases`: caption/WordNet naming hints only.
- `description`: short review rationale.
- `source_targets`: manual audit targets that support this node.
- `evidence`: support and precision/recall summary from review artifacts.
- `motion_summary`: channels and geometry clusters observed in source variants.
- `source_symbols`: compact source symbols and metrics.

## Status Semantics

- `accepted`: can enter the first condition vocabulary as a reviewed pattern.
- `review_candidate`: can be exposed for manual review and ablation, but should
  not be treated as a final full-action label.
- `component`: reusable motion component; can be part of a composed condition,
  but must not be named as a complete action by itself.
- `pending_composition`: needs a composed sequence/transition rule in Motion-BPE
  v2 before promotion.
- `blocked_by_observable_gap`: current motion observables cannot support this
  language label as a full motion node.

## Scope Semantics

- `full_pattern`: accepted full pattern.
- `full_pattern_candidate`: accepted structural candidate with limited support.
- `transition_pattern`: partial transition pattern, not necessarily a full
  semantic action cycle.
- `transition_pattern_candidate`: transition that still needs composition.
- `floor_or_prone_pattern_candidate`: likely needs split by support state.
- `upper_body_component`, `posture_component`, `torso_component`,
  `root_rotation_component`: reusable components only.
- `language_label_group`: names blocked by missing observables.

## Current v0 Tree

```text
coordination_patterns
  jumping_jack_full_coordination                 accepted full_pattern

acrobatics_inversion
  cartwheel_inverted_acrobatics_candidate        accepted full_pattern_candidate

posture_transitions
  sit_down_transition_candidate                  review_candidate transition_pattern
  stand_up_transition_pending                    pending_composition transition_pattern_candidate

floor_prone_or_mime
  swim_like_prone_or_floor_candidate             review_candidate floor_or_prone_pattern_candidate
  bimanual_hands_close_component                 component upper_body_component

component_library
  bimanual_raise_spread_component                component upper_body_component
  low_body_hold_component                        component posture_component
  squat_or_low_posture_component                 component posture_component
  torso_hunched_forward_component                component torso_component
  fast_small_turn_component                      component root_rotation_component

pending_observables
  object_environment_style_labels_pending        blocked_by_observable_gap language_label_group
```

## Condition Schema Use

The next condition-schema pass should use:

- `accepted` nodes as positive vocabulary entries.
- `review_candidate` nodes as optional review/ablation entries.
- `component` nodes as building blocks inside composed conditions.
- `pending_composition` nodes as Motion-BPE v2 requirements.
- `blocked_by_observable_gap` nodes only as a TODO list for new observables.

Do not train full-action labels directly from `component` or
`blocked_by_observable_gap` nodes.

## v1 Closure Draft Forest

The current Motion-BPE branch also exports a newer draft forest from full-HML3D
v4 coord-role closure candidates:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_from_v4_closure_draft/
```

Build command:

```bash
python scripts/build_v4_closure_pattern_forest_draft.py
```

Inputs:

- `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_coord_role_full_closure_review_v0/composition_closure_candidates.json`
- `pseudoedit3d/edit/aml_semantic_alias_sidecar.json`

Outputs:

- `aml_pattern_forest_v1_draft.json`: full review forest with examples.
- `aml_pattern_forest_v1_draft_compact.json`: compact review forest.
- `aml_pattern_forest_v1_draft_tree.txt`: quick tree view.
- `aml_pattern_forest_v1_draft_review.md`: family-level review table.
- `summary.json`: counts.

This forest is a review draft, not an accepted runtime AML tree. It has three
levels:

```text
root group -> pattern family candidate -> source closure candidate
```

Current full-HML3D v1 draft size:

```text
selected source closure candidates: 57
roots: 10
families: 15
nodes: 82
edges: 72

family statuses:
  review_candidate: 5
  split_required: 3
  composition_needs_closure: 3
  composition_review: remaining source-candidate nodes
```

Interpretation:

- `review_candidate`: structurally stable enough to inspect visually for
  promotion. This currently includes a full jumping-jack-like coordination
  family, a cartwheel/inversion family, a sit/body-level transition family, a
  martial/guard coordination family, and a cheer/dance coordination family.
- `split_required`: naming/scope conflict. The clearest current case is
  swim/floor-prone evidence being mixed with the coarse
  `acrobatics_or_inversion` role; this should trigger observable refinement, not
  direct promotion.
- `composition_needs_closure`: a frequent near-pattern is missing one or more
  discriminative roles from the current itemset. Treat it as evidence for
  closure refinement, not as an action label.

The language alias sidecar groups and names candidates only after motion
closure has produced the candidate. It must not create motion evidence or
runtime matching rules.

## v1 Support-State Closure Probe

The newest diagnostic branch adds a whole-body support observable before
composition closure. The 3k probe artifacts are:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_3k_diag/
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_3k_diag_closure_review_balanced/
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_3k_draft/
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_3k_review_pack/
```

New support-state geometry evidence:

```text
WHOLE_BODY_SUPPORT/WB_SUPPORT_INVERTED
WHOLE_BODY_SUPPORT/WB_SUPPORT_FLOOR_LOW_HORIZONTAL
WHOLE_BODY_SUPPORT/WB_SUPPORT_HAND_FLOOR_LOW
```

This evidence is emitted from body support/contact geometry. It is not a named
action detector. The naming layer may later attach aliases such as cartwheel,
swim, kneel, crawl, or fall-to-knees after the motion structure exists.

Current 3k result:

```text
support sidecar events: 238
selected closure candidates: 160
  inversion_acrobatic_candidate: 5
  floor_prone_or_mime_candidate: 8

draft forest:
  selected source closure candidates: 60
  roots: 10
  families: 22
  nodes: 92
  edges: 82
```

Visual spot-check:

- `family_cartwheel_inversion_acrobatic_candidate_needs_closure.png` contains
  cartwheel/flip-like examples and no longer mixes the prone swimming example
  that appeared before support-state separation.
- `family_kneel_or_fall_to_knees_floor_prone_or_mime_candidate_needs_closure.png`
  correctly groups floor/prone support examples, but it is still broad and
  should later split kneel, crawl, prone-swim, and lie/get-up transitions.

Full-HML3D support-state artifacts:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_full_v0/
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v4_support_state_full_v0_closure_review/
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_draft/
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_review_pack/
```

Current full-HML3D result:

```text
source records: 29228
channel events: 488601
support sidecar events: 2234
covered cases: 18832 / 29228 (0.6443)

closure review:
  selected candidates: 240
  promote_review: 17
  inversion_acrobatic_candidate: 8
  floor_prone_or_mime_candidate: 34

draft forest:
  selected source closure candidates: 104
  roots: 10
  families: 26
  nodes: 140
  edges: 130
```

Full visual spot-check:

- `family_cartwheel_inversion_acrobatic_candidate_promote_review.png` is now a
  clean cartwheel/flip-like review family.
- `family_swim_like_motion_floor_prone_or_mime_candidate_promote_review.png`
  stays in floor/prone support space. It is useful but still broad; it mixes
  swim-like prone motion with lie/get-up and some sit/kneel transitions.
- The former swim/inversion conflict was caused by treating
  `acrobatics_or_inversion:vertical_rhythm` as an inversion-zone item. Closure
  mining now derives zone from channel plus role, so vertical rhythm remains
  vertical unless explicit inversion support/acrobatic evidence is present.

The editable decision draft for these 21 review families is:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_review_decisions_draft/
```

Build commands:

```bash
python scripts/propose_v1_support_state_review_decisions.py
python scripts/build_v1_support_state_reviewed_forest_draft.py
```

The current decision draft is deliberately conservative:

```text
promote: 1
downgrade_to_component: 6
split: 1
split_axis_confirmed: 10
needs_closure: 3
```

The reviewed draft forest is:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_full_v0_reviewed_draft/
```

Current reviewed draft tree:

```text
accepted_full_patterns: 1 family
  family_cartwheel_inversion_acrobatic_candidate_promote_review
reusable_components: 6 families
split_required: 11 families
closure_required: 3 families
```

This is still not final runtime matching logic. It is the first clean promotion
surface: accepted nodes can become AML vocabulary candidates after visual
confirmation, while split/closure/component nodes remain mining TODOs or
building blocks.

The reviewed draft can now be materialized as a searchable AML program:

```text
outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_reviewed_draft/
```

Build command:

```bash
python scripts/export_aml_composable_pattern_program_v1_support_state.py
```

Current v1 support-state program:

```text
program nodes: 113
condition entries: 21
positive condition entries: 1
accepted full-pattern condition:
  AMLV1_FAMILY_CARTWHEEL_INVERSION_ACROBATIC_CANDIDATE_PROMOTE_REVIEW
```

Condition ids are unique family-node ids. The human-readable action name remains
in `motion_structure_label`. Only `status=accepted` conditions receive
`condition_weight_default=1.0`; component, split-required, and closure-required
nodes remain searchable but have zero positive training weight.

Loading:

```python
from pseudoedit3d.edit import (
    SUPPORT_STATE_V1_PROGRAM_PATH,
    load_composable_pattern_program,
    search_program_nodes,
)

program = load_composable_pattern_program(SUPPORT_STATE_V1_PROGRAM_PATH)
hits = search_program_nodes(
    program,
    channels=["whole_body_support", "acrobatics_or_inversion", "left_leg", "right_leg"],
    zones=["inversion", "lower"],
    cluster_ids=[
        "WB_SUPPORT_INVERTED",
        "WB_CARTWHEEL_CANDIDATE",
        "LL_LEG_LATERAL_REPEAT",
        "RL_LEG_LATERAL_REPEAT",
    ],
    event_families=[
        "WHOLE_BODY_SUPPORT",
        "WHOLE_BODY_ACROBATICS",
        "LEFT_LEG_LATERAL",
        "RIGHT_LEG_LATERAL",
    ],
    node_kinds=["pattern_family"],
    semantic_priority=True,
)
```

`semantic_priority=True` should be used when the goal is an interpretable
high-level pattern explanation. It lets an accepted whole-body pattern outrank a
perfectly matching local component. Leave it off for pure local-evidence
retrieval/debugging.

Search debug command:

```bash
python scripts/search_aml_composable_pattern_program_v0.py \
  --support-state-v1 \
  --semantic-priority \
  --max-cases 250
```

Promotion audit:

```bash
python scripts/audit_v1_support_state_promotion_candidates.py
```

Output:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_v1_support_state_promotion_audit_v0/
```

The audit uses strict target aliases only as diagnostics. It does not use text
to match motion. Current 250-case result has no new direct promotions: accepted
cartwheel stays positive, while two sit-down candidates need split review and
most jumping-jack/swim/martial candidates are too broad.

Split-axis audit:

```bash
python scripts/audit_v1_support_state_split_axes.py
```

Schema:

```text
pseudoedit3d/edit/aml_pattern_split_axes.json
```

Output:

```text
outputs/aml_regression_testset_v2/aml_pattern_split_axis_audit_v0/
```

Split axes are data-defined motion-evidence schemas. They sit between raw tree
search and pattern promotion:

```text
tree-search hit
  -> split-axis evidence audit
  -> route to a cleaner structural family or keep pending
```

The first axis, `body_level_sit_transition_v0`, is intentionally named as the
motion structure `body_level_low_transition`. It is not a direct `sit_down`
detector. On the support-state 250-case search output, it accepts 10 candidate
cases with diagnostic `sit_down` alias precision 0.8 and recall 0.5333. This is
good enough to separate a low-body transition branch from broad martial/kick or
generic posture candidates, but not good enough to promote a semantic sit-down
node.

Rules for using split axes:

- The schema may contain required positive evidence groups, optional support
  evidence, negative evidence groups, and label-routing rules.
- Captions and caption aliases are diagnostics only.
- Torso hunch/recover may support a body-level low transition, but cannot name
  `sit_down` by itself.
- A split-axis candidate can become a program-side routing condition or a
  training-ablation condition only after it is reviewed as a structural family.
- A named semantic action such as `sit_down` needs an additional naming or
  support/contact distinction before promotion.

The second axis, `bilateral_spread_vertical_coordination_v0`, audits a
jumping-jack-like structure without using the `jumping_jack` name as a rule. Its
evidence groups are mined motion dimensions:

- `upper_spread`: bimanual raise/spread clusters.
- `bilateral_high_arm_pose`: both hands/arms high.
- `large_bilateral_arm_arc`: large arm arcs or orbit cycles.
- `vertical_rhythm`: whole-body vertical up/down or arm-raise-coupled vertical
  rhythm.
- `lower_spread`: left/right leg lateral abduct/adduct/out/repeat clusters,
  plus v5 bilateral stance-width expansion/contraction/repeat clusters.

The default label is `bilateral_upper_spread_vertical_component`. It is a
component because many cases have clean upper-body plus vertical rhythm but no
explicit lower-leg lateral evidence. The stricter
`bilateral_upper_lower_spread_vertical_coordination` label requires
`upper_spread + vertical_rhythm + lower_spread`.

Mixed-probe audit:

```bash
python scripts/search_aml_composable_pattern_program_v0.py \
  --support-state-v1 \
  --semantic-priority \
  --case-ids <100 jumping-jack pseudo-GT ids + controls> \
  --output-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v0

python scripts/audit_v1_support_state_split_axes.py \
  --search-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v0 \
  --output-dir outputs/aml_regression_testset_v2/aml_pattern_split_axis_jumpaxis_probe_audit_v0
```

Current mixed-probe result:

```text
cases: 298
windows: 440
bilateral spread axis accepted cases: 86
diagnostic jumping-jack precision: 1.0
diagnostic jumping-jack recall: 0.86
accepted windows:
  bilateral_upper_spread_vertical_component: 117
  bilateral_upper_lower_spread_vertical_coordination: 16
```

Interpretation: the upper+vertical component is reliable; the full
upper+lower+vertical coordination is cleaner but lower-recall. This supports a
tree branch where the complete action is built from a high-confidence
upper/vertical component plus optional lower-spread closure, rather than naming
every upper/vertical hit as `jumping_jack`.

### v5 Stance-Width Sidecar

The v4 lower-spread bottleneck was not a coactivation-window problem. A
case/window coverage audit showed:

```text
target jumping-jack cases: 100
upper_spread: 94
vertical_rhythm: 95
lower_spread: 23
case_has_full_rule: 17
```

The missing evidence was mostly a micro-event recall issue: single-leg lateral
events only observe each foot relative to the pelvis, so many cases with clear
bilateral foot separation did not emit a lower-spread token. v5 adds a
motion-only `raw_joint_stance_width` sidecar:

```text
signal: baseline-relative left/right foot separation projected onto the
        body-lateral axis
clusters:
  WHOLE_BODY_STATE/WB_STANCE_WIDTH_WIDE_BRIEF
  WHOLE_BODY_STATE/WB_STANCE_WIDTH_WIDE_HOLD
  WHOLE_BODY_STATE/WB_STANCE_WIDTH_EXPAND
  WHOLE_BODY_STATE/WB_STANCE_WIDTH_CONTRACT
  WHOLE_BODY_STATE/WB_STANCE_WIDTH_REPEAT
```

This is not a `jumping_jack` detector. It is a lower-body geometric observable
that can participate in many future patterns.

Probe command:

```bash
python scripts/audit_hml3d_multichannel_motion_bpe.py \
  --observable-refinement v5 \
  --case-ids <same 298 mixed-probe cases> \
  --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_jumpaxis_probe_v0 \
  --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_jumpaxis_probe_v0_cache \
  --rebuild-cache

python scripts/search_aml_composable_pattern_program_v0.py \
  --support-state-v1 \
  --semantic-priority \
  --case-ids <same 298 mixed-probe cases> \
  --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_jumpaxis_probe_v0/case_multichannel_bpe_sequences.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_jumpaxis_probe_v5_stance_width_v0
```

v4 -> v5 comparison on the same mixed probe:

```text
lower_spread case coverage: 23/100 -> 47/100
case_has_full_rule: 17/100 -> 40/100
accepted bilateral axis cases: 86 -> 89
diagnostic precision: 1.0000 -> 0.9888
diagnostic recall: 0.8600 -> 0.8800
full-label accepted windows: 16 -> 39
component-label accepted windows: 117 -> 82
```

Review interpretation:

- v5 fixes a real lower-body observable gap and should be kept as a geometry
  sidecar.
- The full coordination label is still a closure label, not a final named
  action node.
- The remaining bottleneck is no longer only lower-spread; some target cases
  lack upper-spread or vertical-rhythm evidence in the same accepted window.
- One reviewed non-target candidate remains: a floor/low-body transition with
  leg-strike and arm-spread evidence. It should be treated as a confound type
  for future support/contact refinement, not patched with a case rule.

Full-HML3D v5 stance-width artifacts:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v5_stance_width_full_v0/
outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v5_stance_width_full_v0/
outputs/aml_regression_testset_v2/aml_composable_pattern_program_v1_support_state_search_full_v5_stance_width_v0/
outputs/aml_regression_testset_v2/aml_pattern_split_axis_full_v5_stance_width_coverage_v0/
outputs/aml_regression_testset_v2/hml3d_v4_v5_stance_width_full_comparison_v0/
```

Full-HML3D v4 -> v5 comparison:

```text
channel events: 488601 -> 512857
channel event types: 2779 -> 2874
stance-width sidecar events: 0 -> 24256
coordination motifs: 64 -> 78
motif families: 92 -> 106
composition structure groups: 60 -> 75

bilateral-spread axis on 362 jumping-jack text-target cases:
  lower_spread case coverage: 96 -> 168
  full upper+lower+vertical case coverage: 78 -> 146
  full upper+lower+vertical window coverage: 77 -> 145
```

Interpretation: the full-corpus run confirms the probe conclusion. A large
part of the missing full jumping-jack-like closure was caused by absent
bilateral stance-width evidence. v5 increases useful lower-body coverage, but
the full upper+lower+vertical label remains a reviewed structural closure
candidate. It should not be promoted directly to the named `jumping_jack`
action until phase/order and confound audits are reviewed.

## Dense Candidate Forest

The reviewed v0 tree is intentionally conservative. The broader full-HML3D
candidate pool is exported separately:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_candidates_v0_dense/
```

Build command:

```bash
python scripts/build_aml_pattern_forest_dense_v0.py
```

This dense forest currently contains:

```text
87 motif families
231 motif leaves
323 total nodes
```

Dense statuses mean:

- `coordination_candidate`: multi-channel candidate; inspect first for future
  full-pattern promotion.
- `component_candidate`: single-channel sequence candidate; usually enters the
  component library, not the full action vocabulary.
- `named_candidate`: has caption-name evidence, but the name is only diagnostic.
- `diagnostic_candidate`: low-support or uncertain candidate; keep as evidence.
- `reviewed_*`: leaf already linked to the conservative reviewed v0 tree.

The dense forest is the promotion pool. The reviewed tree is the current
condition-vocabulary seed.

## Composition Pattern Forest v0

The current full-HML3D motion-derived candidate forest is:

```text
outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v0_structure_groups/
```

Build command:

```bash
python scripts/build_hml3d_composition_pattern_forest_v0.py \
  --output-dir outputs/aml_regression_testset_v2/hml3d_composition_pattern_forest_v0_structure_groups
```

Inputs:

- `outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl`
- `outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full/case_multichannel_bpe_sequences.jsonl`

Outputs:

- `composition_pattern_forest.json`: full four-level forest with examples.
- `composition_pattern_forest_compact.json`: compact forest for inspection.
- `composition_pattern_forest_tree.txt`: readable hierarchy.
- `composition_pattern_forest_report.md`: ranked family report.
- `summary.json`: counts.

Current result:

```text
29228 caption-indexed cases
34453 all-channel coactivation transactions
3752 closed itemsets
64 structure groups
379 composition families
1219 exported variants
```

Hierarchy:

```text
composition_root
  structure_group
    composition_family
      composition_variant
```

The structure group is the main review unit. It is generated from normalized
motion roles such as `bimanual_raise_spread_vertical_coordination` or
`low_body_torso_transition`. Caption aliases are stored as
`caption_name_candidates` only; they do not create the tree and should not be
used as runtime matching rules.

Recommended review order:

1. Open `composition_pattern_forest_tree.txt`.
2. Review the `full-body composition candidates` root first.
3. For each `structure_group`, decide whether the motion structure is a
   potential full pattern, transition, component, or diagnostic context.
4. Use `composition_pattern_forest_report.md` only after a structure group looks
   promising; it lists examples and variants.
5. Promote only reviewed structure groups/families into the AML condition
   vocabulary. Do not promote raw variants directly.

## Promotion Review Table

Dense candidates are not promoted directly. The current promotion review table
is exported here:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0/
```

Build command:

```bash
python scripts/build_aml_pattern_promotion_review_v0.py
```

Outputs:

- `promotion_review_table.json`: complete review rows with examples.
- `promotion_review_table.md`: readable ranked audit report.
- `promotion_review_table.csv`: compact spreadsheet-style checklist.
- `summary.json`: recommendation counts.

Current result:

```text
87 dense families
1 already_reviewed
15 composition_review
44 component_review
5 name_only_review
22 diagnostic_keep
```

Promotion labels mean:

- `already_reviewed`: already linked to the conservative v0 forest.
- `composition_review`: frequent multi-channel coordination, but examples and
  names are too diffuse for direct promotion.
- `component_review`: reusable local component candidate, not a complete action
  name.
- `name_only_review`: caption aliases are informative, but the motion structure
  alone is insufficient.
- `diagnostic_keep`: keep as mining evidence only.

The important design point is that `composition_review` is not a promotion. It
marks a candidate that may become useful after purity checks, split/merge, or
better observables. Direct promotion still requires reviewed motion evidence,
not just high support or a caption alias.

## Promotion Self-Review

The current text/structure self-review result is:

```text
outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v0/promotion_self_review.md
```

Build command:

```bash
python scripts/review_aml_pattern_promotion_table_v0.py
```

Current result:

```text
87 reviewed dense families
1 keep_accepted_reference
15 downgrade_to_component
44 keep_component
5 keep_name_alignment_only
22 keep_diagnostic
0 needs_visual_review
```

The key conclusion is conservative: the dense forest does not currently add a
new full-action pattern beyond the already reviewed jumping-jack coordination
reference. The 15 multi-channel `composition_review` rows are useful, but they
should enter the component library or Motion-BPE v2 TODOs, not the full pattern
tree.

The strongest repeated issue is observable conflation:

- `LEFT_LEG_ACTION/LL_KICK_FORWARD` and
  `RIGHT_LEG_ACTION/RL_KICK_FORWARD` often describe normal gait leg swing.
- Whole-body vertical up/down often describes gait bounce, path changes, or
  low-amplitude support changes.
- Arm periodic clusters are too generic without path shape, symmetry,
  object-mime, and coupling evidence.

Motion-BPE v2 should split these components before trying to promote denser
full-action nodes.

## Motion-BPE v2 Observable Refinement

The all-confusions v2 pass implements the split requested by the promotion
self-review. It does not add action names or case-specific rules. It only
refines coarse Layer3 geometry labels into more explicit structural roles before
Motion-BPE runs.

Build command:

```bash
python scripts/audit_hml3d_multichannel_motion_bpe.py \
  --num-merges 256 \
  --min-pair-count 80 \
  --min-pair-support 40 \
  --channel-merge-ratio 0.5 \
  --observable-refinement v2 \
  --write-heavy-corpora \
  --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_all_confusions_full \
  --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_all_confusions_full_cache \
  --rebuild-cache
```

Main outputs:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_all_confusions_full/
outputs/aml_regression_testset_v2/aml_pattern_forest_candidates_v2_all_confusions_dense/
outputs/aml_regression_testset_v2/aml_pattern_forest_promotion_review_v2_all_confusions/
```

The v2 refinement handles the confusion sources identified in the v1 promotion
review:

- leg-forward events are split into gait swing, hold-pose, impulse-like, and
  unresolved forward-leg evidence.
- root locomotion is split into weak root drift, path fragment, gait context,
  and translation context.
- root turns are split by angle, tempo, and path/isolated role.
- vertical events are split into gait bounce, low-body transition, arm-raise
  coordination, jump-up impulse, salient descent, and generic vertical cycles.
- low-body posture is split into descent hold, rise from low, down-up cycle,
  sustained hold, locomotion context, and leg-extension context.
- torso motion is split by low-body, locomotion, vertical, sustained, and
  periodic context.
- arm periodic/posture events are split by bilateral symmetry, vertical
  coupling, locomotion coupling, bimanual context, high-pose hold, and transient
  high-pose roles.
- bimanual events are split by vertical, locomotion, low-body, and hand-high
  context.

Current v1 vs v2 comparison:

```text
v1:
  channel_event_type_count: 1019
  learned_motif_count: 231
  motif_family_count: 87
  coordination_merges: 39
  case_coverage: 0.702238
  promotion self-review: 15 composition rows downgraded to components

v2 all-confusions:
  channel_event_type_count: 2253
  learned_motif_count: 135
  motif_family_count: 53
  coordination_merges: 7
  case_coverage: 0.583105
  promotion self-review: 6 composition rows downgraded to components
```

Interpretation: v2 deliberately reduces dense full-pattern candidates. The
coarser v1 labels allowed frequent but impure combinations to look stable. After
splitting the confusing observables, many candidates become explicit
component/context evidence instead of misleading full-action nodes. This is a
cleanup step before improving the merge policy, not the final pattern forest.

The v2 promotion self-review result is:

```text
53 reviewed dense families
6 downgrade_to_component
41 keep_component
6 keep_name_alignment_only
0 needs_visual_review
```

The next Motion-BPE step should not add action-specific rules. It should improve
the merge policy so clean component/context tokens can compose into higher-level
motifs when their coactivation is actually stable.

## Composition-Score Coordination Selection

After v2 observable refinement, the next audit adds a structure-aware
coordination selection policy:

```bash
python scripts/audit_hml3d_multichannel_motion_bpe.py \
  --num-merges 256 \
  --min-pair-count 80 \
  --min-pair-support 40 \
  --channel-merge-ratio 0.5 \
  --observable-refinement v2 \
  --coordination-selection structure_score \
  --coordination-min-structure-score 5.0 \
  --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_composition_score_full \
  --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_all_confusions_full_cache
```

This does not change the channel-BPE stage. It changes only which overlapping
channel motifs are promoted into `<COM_*>` coordination motifs. The score favors
cross-channel structure such as upper-body plus vertical/low-body/torso context,
and penalizes pure gait/path/drift coactivation.

Current result:

```text
v2 all-confusions support selection:
  coordination_merges: 7
  composition_review rows: 6

v2 composition-score selection:
  coordination_merges: 2
  composition_review rows: 2
  self-review: both downgraded to upper-body components
  needs_visual_review: 0
```

Interpretation: structure-score selection correctly removes the dominant
gait-leg and gait-bounce coactivations from the coordination candidate set. The
remaining coordination motifs are still upper-body components, not full action
patterns. This means the next issue is recall/composition, not false-positive
cleanup: the system must inspect unpromoted coactivation candidates and discover
whether full patterns are missing because the channel motifs do not overlap
cleanly, the threshold is too strict, or the current geometry lacks required
channels.

## Coactivation Recall Audit

The recall audit checks why full patterns are missing. It is diagnostic only:
HumanML3D text targets are pseudo-GT labels for analysis, not rules for
Motion-BPE or runtime AML.

Build command:

```bash
python scripts/audit_hml3d_coactivation_recall_v0.py
```

Main output:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_all_units/
```

Comparison command for the current Motion-BPE coordination stage:

```bash
python scripts/audit_hml3d_coactivation_recall_v0.py \
  --coactivation-source channel_motifs \
  --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit_channel_motifs
```

Existing comparison artifact from the first pass:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v2_coactivation_recall_audit/
```

Current result:

```text
channel_motifs view:
  coactivation symbols: 2096
  selected coordination symbols: 2
  diagnosis: most targets collapse to selected upper-body components

all_units view:
  coactivation symbols: 21935
  selected coordination symbols: 2
  diagnosis counts:
    component_only_best_match: 3
    selected_component_not_full_pattern: 6
    target_fragmented_across_many_coactivations: 3
    text_label_motion_diverse_or_ambiguous: 2
```

The key example is `jumping_jack`:

```text
pseudo-GT cases: 368
target cases with any all-unit coactivation: 357
best unselected candidate:
  geometry:
    WHOLE_BODY_VERTICAL/WB_VERT_ARM_RAISE_COUPLED
    LEFT_ARM_POSTURE/LA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT
    RIGHT_ARM_POSTURE/RA_BILATERAL_HIGH_POSE_VERTICAL_CONTEXT
    BIMANUAL_PERIODIC/BI_RAISE_SPREAD_VERTICAL_CONTEXT
  support: 67 cases
  target hits: 65
  precision: 0.970149
  recall: 0.17663
  structure_score: 10.719508
```

This explains why full pattern nodes did not emerge from the current BPE
setting:

- Full-pattern evidence exists for some targets at the all-channel unit level.
- The current coordination stage only composes learned per-channel `<CHM_*>`
  motifs, so many clean base-event coactivations never become coordination
  candidates.
- Text labels such as `sit`, `stand_up`, `climb`, `tennis`, and `basketball`
  remain motion-diverse or object/environment-heavy; they should not be directly
  promoted as full motion nodes from text pseudo-GT.

The next algorithmic step is composition-BPE or closed coactivation mining above
channel units. It should propose higher-level composition candidates while
keeping their component provenance, rather than adding action-specific proxy
rules.

## Composable AML Pattern Program v0

The program-facing artifact is:

```text
outputs/aml_regression_testset_v2/aml_composable_pattern_program_v0/
```

Build command:

```bash
python scripts/export_aml_composable_pattern_program_v0.py
```

Outputs:

- `aml_composable_pattern_program.json`: full program tree with match
  signatures and condition entries.
- `aml_composable_pattern_program_compact.json`: compact tree for inspection.
- `aml_composable_pattern_program_tree.txt`: readable hierarchy.
- `aml_composable_pattern_search_index.json`: inverted index by channel, zone,
  event family, cluster id, geometry role, and semantic level.
- `aml_condition_vocabulary.json`: draft condition entries for structure groups
  and families.

Current result:

```text
1666 program nodes
443 condition entries
4 roots
64 structure groups
379 composition families
1219 variants
```

The purpose of this tree is not just naming. It is a searchable program for
aligning an input 3D motion to human-interpretable structure:

```text
motion evidence
-> tree search
-> semantic level decision
-> editable condition handle
```

This lets the system decide whether a motion span is closer to a whole-body
pattern, a composed multi-part pattern, a transition, or a local component. The
same hierarchy also defines edit scope: whole-body edits such as jumping farther,
multi-part edits such as coordinating both arms faster, and local edits such as
raising the left hand higher.

Important fields:

- `semantic_level`: `whole_body_pattern_candidate`, `multi_part_coordination`,
  `transition`, `component`, `local_component`, or `diagnostic_context`.
- `edit_scope`: `whole_body`, `multi_part`, `upper_body_or_arm`,
  `lower_body_or_leg`, `root_or_body`, or `local`.
- `composition_policy`: whether the node may bind as a full pattern, composed
  subpattern, transition, reusable component, local edit handle, or diagnostic.
- `match_signature`: structural evidence used for tree search: channels, zones,
  event families, cluster ids, geometry roles, and overlap thresholds.
- `edit_handles`: editable dimensions such as span, count, vertical amplitude,
  body level, leg extension, arm height, arm symmetry, root direction, distance,
  turn angle, and turn direction.

Condition ids:

- `condition_id` is the reusable structural condition type, for example
  `AMLCP_BIMANUAL_RAISE_SPREAD_VERTICAL_COORDINATION`.
- `condition_entry_id` is the unique program-node condition entry and should be
  used when a manifest selects a specific tree node.

Runtime loading:

```python
from pseudoedit3d.edit import (
    condition_vocabulary,
    load_composable_pattern_program,
    search_program_nodes,
)

program = load_composable_pattern_program()
conditions = condition_vocabulary(
    program,
    scopes=["full_composition_candidate"],
    review_statuses=["review_candidate"],
)
hits = search_program_nodes(
    program,
    channels=["bimanual", "whole_body_vertical"],
    zones=["upper", "vertical"],
    cluster_ids=["BI_RAISE_SPREAD", "WB_VERT_UP"],
    event_families=["BIMANUAL_PERIODIC", "WHOLE_BODY_VERTICAL"],
)
```

The `pseudoedit3d.edit` package now lazy-loads public exports, so inspecting the
program does not require importing heavy numeric dependencies.

## Tree Search Debug v0

The current motion-to-tree debug bridge is:

```bash
python scripts/search_aml_composable_pattern_program_v0.py --max-cases 250
```

Output:

```text
outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0/
```

Files:

- `case_tree_search_results.json`: per-case classification and top windows.
- `window_tree_search_results.json`: per-window structural evidence and hits.
- `search_report.md`: readable case summary.
- `summary.json`: aggregate counts.

This script does not use captions as matching rules. It loads multichannel
Motion-BPE channel units, rebuilds all-unit coactivation windows, extracts
motion evidence, and searches the program tree.

Latest 250-case debug result:

```text
250 channel cases
192 cases with coactivation windows
58 cases without coactivation windows
340 searched windows
37 whole-body/full-composition candidate cases
26 composed multi-part cases
29 transition cases
32 component-dominant cases
68 diagnostic/ambiguous cases
58 unmatched/local-only cases
```

Selected review examples can be searched with:

```bash
python scripts/search_aml_composable_pattern_program_v0.py \
  --case-ids 003082,003191,007581,008692,009072,011643,011797,014607,M010447 \
  --output-dir outputs/aml_regression_testset_v2/aml_composable_pattern_program_search_v0_review_examples
```

Current limitation:

- Tree search is recall-oriented. It can place a motion span onto likely tree
  levels, but it is not yet a final condition manifest.
- Broad diagnostic nodes still appear. Case-level classification prioritizes
  editable semantic levels over diagnostic score, but the next matcher pass
  should add temporal coverage, node specificity, and sibling suppression.
- Variant nodes remain debug evidence. Default condition search returns only
  `structure_group` and `composition_family` nodes.

## Program Condition Manifest v0

The tree search output can now be converted into the existing AML
`selected_conditions` contract:

```bash
python scripts/export_aml_program_condition_manifest_v0.py
python scripts/audit_aml_condition_contract.py \
  --selected-jsonl outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0/selected_conditions.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0/contract_audit
```

Default debug/composable output:

```text
outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0/
```

Current result:

```text
250 cases
156 train-ready cases
94 empty-selected cases
377 selected conditions
544 deferred diagnostic/weak conditions
contract status: pass
```

This default uses `--max-selected-per-span 2`, so one span may retain a
whole-body/composed node plus a local component. It is useful for inspecting
compositionality, but it still contains sibling overlap.

Strict training-oriented output:

```bash
python scripts/export_aml_program_condition_manifest_v0.py \
  --max-selected-per-span 1 \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1
```

Current strict result:

```text
250 cases
156 train-ready cases
94 empty-selected cases
236 selected conditions
544 deferred diagnostic/weak conditions
0 duplicate selected conditions per span
contract status: pass
```

The recommended first condition-encoder smoke test should use the strict
`span1` manifest. Keep the default `span2` manifest for compositional audit.

Audit-filtered train-clean output:

```bash
python scripts/audit_aml_program_condition_manifest_v0.py \
  --manifest-jsonl outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/selected_conditions.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/audit

python scripts/filter_aml_program_condition_manifest_v0.py

python scripts/audit_aml_condition_contract.py \
  --selected-jsonl outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean/selected_conditions.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean/contract_audit
```

Current train-clean result:

```text
250 cases
40 train-ready cases
210 empty-selected cases
50 selected train-candidate conditions
730 deferred conditions
contract status: pass
quality audit: all selected conditions are train_candidate
```

The train-clean manifest is intentionally conservative. It removes broad
generic/component-only conditions from the hard-positive training set and keeps
them as deferred diagnostics. Captions are used only as audit hints, never as
runtime matching rules.

Batch schema export:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/export_aml_condition_batch_schema.py \
  --input-jsonl outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_strict_span1/contract_audit/train_ready_selected_conditions.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_strict_span1 \
  --max-conditions 8
```

Current strict batch:

```text
156 cases
236 selected conditions
max conditions: 8
span coverage: 1.0
truncated cases: 0
```

Train-clean batch export:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/export_aml_condition_batch_schema.py \
  --input-jsonl outputs/aml_regression_testset_v2/aml_program_condition_manifest_v0_train_clean/contract_audit/train_ready_selected_conditions.jsonl \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean \
  --max-conditions 8
```

Current train-clean batch:

```text
40 cases
50 selected conditions
max conditions: 8
span coverage: 1.0
truncated cases: 0
score mean: 0.8179
```

The batch now carries three token granularities:

- `family_id`: unique program-family condition entry, fine but sparse.
- `condition_id`: shared structural condition type.
- `motion_structure_id`: shared motion-structure label.

Current train-clean vocabulary sizes:

```text
family_vocab: 35 including pad/unk
condition_vocab: 16 including pad/unk
motion_structure_vocab: 16 including pad/unk
```

## Condition Encoder Smoke v0

The first condition-encoder smoke uses the train-clean batch only. It checks
data alignment and overfitting, not final generation quality.

Motion batch export:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/export_aml_condition_motion_batch.py \
  --condition-batch-dir outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean \
  --output-dir outputs/aml_regression_testset_v2/aml_program_condition_motion_batch_v0_train_clean
```

Current motion batch result:

```text
40 cases
joints shape: [40, 200, 22, 3]
valid frames: 6046
frame mismatches: 0
truncated cases: 0
alignment status: pass
```

Loader smoke:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/smoke_aml_condition_motion_loader.py \
  --condition-batch-dir outputs/aml_regression_testset_v2/aml_program_condition_batch_schema_v0_train_clean \
  --motion-batch-dir outputs/aml_regression_testset_v2/aml_program_condition_motion_batch_v0_train_clean \
  --output-dir outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/loader_smoke \
  --batch-size 8 \
  --num-workers 0
```

Current loader result:

```text
status: pass
dataset: 40
batches: 5
samples: 40
conditions: 50
valid frames: 6046
```

Overfit smoke:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/train_aml_condition_encoder_smoke.py \
  --epochs 500 \
  --batch-size 50 \
  --device auto
```

Current overfit result:

```text
device: cuda
condition rows: 50
initial normalized MSE: 0.986822
final normalized MSE: 0.001878
loss reduction: 0.9981
status: pass
```

The overfit target is a span-level motion geometry summary: root displacement,
root path length, vertical range, mean/max joint displacement, body-part
displacement, and speed. This is a sanity check that condition family, span, and
slot tensors are aligned with the motion batch. It does not show that AML
conditions can generalize or drive a motion generator yet.

Split smoke:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python \
  scripts/train_aml_condition_encoder_smoke.py \
  --mode row_split \
  --token-source family \
  --epochs 400 \
  --batch-size 32 \
  --val-fraction 0.25 \
  --device auto \
  --output-dir outputs/aml_regression_testset_v2/aml_condition_encoder_smoke_v0_train_clean/row_split_family_v2
```

Current split conclusion:

```text
row_split family token: warn, eval MSE 1.3146, global-mean MSE 1.1143
row_split condition token: warn, eval MSE 1.8015, global-mean MSE 1.4312
row_split structure token: warn, eval MSE 1.8015, global-mean MSE 1.4312
case_split: warn, eval MSE 3.0867, 7 validation tokens unseen
```

The split result is the important audit finding. The current train-clean set is
good enough to verify schema alignment, but it is too small and under-specified
for generalization. The next condition-data pass should add more reviewed
conditions and fill real numeric residues instead of only training a larger
encoder on the same 50 rows.
