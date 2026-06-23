# v0 AML Pipeline

This document records the current v0 AML pipeline direction.

## Goal

v0 AML is an offline corpus-mining pipeline for building a motion pattern
forest from HumanML3D. It does not assume the old hand-written AML tree is
correct. The old tree can remain as a compatibility baseline, but v0 treats
HumanML3D motion itself as the primary motion corpus.

The intended flow is:

```text
HumanML3D motion
-> Layer3 event corpus
-> multi-channel Motion-BPE
-> motion motif families / pattern forest
-> manual text-target audits
-> HumanML3D caption + WordNet naming candidates
-> reviewed AML pattern vocabulary
```

Key boundary:

```text
motion builds structure
language proposes names
manual targets test coverage
```

HumanML3D captions, WordNet, and manual target registries do not create motion
nodes directly. They are naming and audit evidence attached after motion motifs
already exist.

## Step 1: Layer3 Event Corpus

Active artifact:

```text
outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl
```

Each record stores:

- `case_id`
- frame count
- HumanML3D captions
- caption alias ids
- Layer3 events
- coarse action diagnostics

Purpose:

```text
dense motion -> symbolic event spans with body part, cluster, direction,
duration, magnitude, count, and provenance
```

This is the input corpus for Motion-BPE.

## Step 2: Multi-Channel Motion-BPE

Active script:

```text
scripts/audit_hml3d_multichannel_motion_bpe.py
```

Current main output:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/
```

Core idea:

1. Split Layer3 events into body channels.
2. Learn within-channel sequential motifs first.
3. Build overlap packets from learned channel motifs.
4. Learn cross-channel coordination motifs from stable overlap packets.
5. Group motifs into motion-derived family candidates.

This separates:

- same-channel temporal composition, such as repeated arm swings;
- cross-channel concurrent composition, such as arms plus vertical body motion;
- language naming, which is delayed to later steps.

Important outputs:

```text
channel_event_vocab.json
packet_vocab.json
multichannel_motion_bpe_vocab.json
case_multichannel_bpe_sequences.jsonl
motif_audit.json
motif_family_candidates.json
motion_pattern_forest_candidates.json
summary.json
audit_report.md
```

## Step 3: Coordination Promotion Forest

Active scripts:

```text
scripts/promote_coordination_motif_candidates.py
scripts/build_coordination_pattern_forest.py
```

Current outputs:

```text
outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/
outputs/aml_regression_testset_v2/coordination_pattern_forest_loose_v1/
```

Purpose:

```text
identify coordination motifs that are structurally stable enough to inspect as
candidate pattern-family nodes
```

This is still review-only. A promoted node is not automatically a runtime AML
rule.

## Step 4: Manual Text-Target Audits

Active registry:

```text
configs/motion_pattern_text_targets.json
```

Active scripts:

```text
scripts/audit_motion_pattern_pseudo_gt.py
scripts/audit_motion_pattern_recall_candidates.py
scripts/build_motion_pattern_family_proposals.py
scripts/run_motion_pattern_registry_audits.py
```

Current batch output:

```text
outputs/aml_regression_testset_v2/manual_text_target_audits_v0/
```

Current v0 snapshot:

```text
target_count:           14
total pseudo-GT cases:  7302
indexed HML3D cases:    29228
indexed BPE symbols:    5736
```

Current target audit summary:

| target | pseudo-GT cases | expanded precision | expanded recall | note |
| --- | ---: | ---: | ---: | --- |
| jumping_jack | 368 | 0.846154 | 0.149457 | has one seeded coordination motif and one precision-preserving expansion |
| jump_rope | 38 | 0.0 | 0.0 | not covered by current motif symbols under default threshold |
| sit | 1530 | 0.697248 | 0.049673 | weak low-body/vertical evidence only |
| stand_up | 1228 | 0.0 | 0.0 | not covered under default threshold |
| kneel | 1284 | 0.0 | 0.0 | not covered under default threshold |
| karate_or_martial | 688 | 0.0 | 0.0 | not covered under default threshold |
| dance | 992 | 0.0 | 0.0 | not covered under default threshold |
| ballet | 78 | 0.0 | 0.0 | not covered under default threshold |
| cartwheel | 148 | 0.670455 | 0.797297 | strong coverage, but mixed with low-body/difficult-pose proxies; needs review |
| basketball | 182 | 0.0 | 0.0 | object/action intent not recovered by current motion-only symbols |
| tennis | 118 | 0.0 | 0.0 | object/action intent not recovered by current motion-only symbols |
| swim | 132 | 0.666667 | 0.227273 | partial coverage through prone/inverted/body-low evidence; needs review |
| climb | 316 | 0.0 | 0.0 | object/environment interaction not recovered |
| duck_under | 200 | 0.0 | 0.0 | environment-relation language not recovered |

Interpretation:

```text
The current Motion-BPE v0 can surface some structurally distinctive targets
such as jumping-jack-like coordination and acrobatics/prone motion. It still
does not cleanly recover many object-, sport-, environment-, or intent-heavy
caption names. These failures are useful audit signals, not reasons to add
case-specific runtime rules.
```

Purpose:

```text
use a small manual target registry as text pseudo-GT to test whether learned
motion symbols recover known weak points such as sit, kneel, dance, tennis,
swim, jump rope, and jumping jack
```

The registry is not a source of motion families. It is an audit set.

The per-target reports expose:

- pseudo-GT case count from HumanML3D captions;
- existing seed motif coverage, if any;
- recall-improving candidate symbols;
- precision-preserving candidate expansion;
- review-only pattern family proposals.

## Step 5: Caption / WordNet Naming Candidates

Target script:

```text
scripts/mine_hml3d_caption_wordnet_name_candidates_v0.py
```

Target output:

```text
outputs/aml_regression_testset_v2/hml3d_caption_wordnet_name_candidates_v0/
```

Current v0 snapshot:

```text
caption cases:                 29228
captions:                      87372
retained caption phrases:      12030
WordNet terms loaded:          27986
phrases with WordNet hints:    10632
motion nodes processed:        164
nodes with name candidates:    164
node-name candidates:          1968
strong candidates:             188
review candidates:             740
diagnostic candidates:         1040
```

Purpose:

```text
mine frequent HumanML3D caption phrases, attach WordNet taxonomy hints, and
align those phrase/name candidates to motion-derived nodes by case overlap
```

This is the path from unnamed motion motifs to human-readable action names.

Expected outputs:

```text
name_candidates.json
name_candidates.md
summary.json
```

The naming layer should answer:

- which text phrases are frequent in the corpus;
- which WordNet/taxonomy parents they suggest;
- which motion nodes they overlap with;
- which names are stable and which are one-to-many ambiguous.

## What v0 Is Not

v0 is not a MoMask generator, not a new autoregressive motion model, and not a
case-by-case prompt patching system.

v0 is also not the final runtime AML tree. It is the offline structure-mining
stage that should produce a cleaner candidate vocabulary for later encoding
and editing.

## Review Criteria

Before promoting v0 outputs into runtime AML rules, inspect:

- compression: sequence length reduction without losing span/channel evidence;
- semantic purity: top examples for a motif/family share real motion structure;
- target audit recall: known text targets are recoverable by learned symbols;
- false positives: recall expansion does not collapse distinct actions;
- naming ambiguity: one motion node may have multiple names, and one name may
  map to multiple motion nodes.

Only after these checks should a candidate enter the AML condition schema.
