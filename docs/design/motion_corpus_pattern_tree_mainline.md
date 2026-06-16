# Motion-Corpus Pattern Tree Mainline

## Status

As of 2026-06-15, the AML pattern-tree mainline is shifted from a hand-built
semantic tree toward a corpus-derived motion tree.

The working rule is:

```text
motion cluster + motion BPE -> motion pattern tree
text BPE + WordNet -> naming layer
tree nodes -> AML condition schema / edit handles / prompt names
```

In short:

```text
motion decides structure
language decides names
WordNet supplies semantic hierarchy
the old hand-built tree is a reference, not the authority
```

## Why This Change

The previous iteration exposed a structural problem: many failures were not
single bugs in `coarse_signature.py`, but symptoms of an overly hand-authored
tree. Examples include `jumping jack`, `sit down`, `karate`, `ballet`,
`tennis`, `dribbling`, `jump rope`, and `cartwheel`.

Those names combine three different signals:

- skeleton-visible motion structure;
- text labels in HumanML3D captions;
- object, scene, sport, or intent context that may not be observable from
  skeleton motion alone.

Therefore the tree should not start from action names. It should start from
recurring motion structure in the corpus, then use language and WordNet to
name the stable structure.

## Current Evidence

The current full-corpus Layer3 event-BPE audit is:

```text
scripts/audit_hml3d_layer3_event_bpe.py
outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/
```

Summary:

- HumanML3D records: `29228`
- original Layer3 event tokens: `393032`
- BPE tokens after merge: `326070`
- compression ratio: `0.829627`
- base vocabulary size: `649`
- learned merges: `256`
- stable tree-candidate motifs: `8`

Important artifacts:

- `layer3_event_bpe_corpus.jsonl`
- `case_bpe_sequences.jsonl`
- `motion_bpe_vocab.json`
- `bpe_motif_audit.json`
- `bpe_phrase_to_pattern_tree_candidates.json`
- `bpe_caption_alias_alignment.json`
- `bpe_motif_audit.md`

The stable candidates are not automatically accepted as runtime nodes. They
are the first inspection set for inducing motion-derived tree nodes.

Current limitation:

- this audit is a single-sequence Event-BPE baseline;
- it flattens concurrent Layer3 events into a sorted token sequence;
- `256` learned merges is the configured merge budget, not the final motion
  vocabulary size;
- the next design step is the concurrency-aware multi-channel Motion-BPE
  extraction defined in `multi_channel_motion_bpe_extraction.md`.

## Document Map

Read these three documents together:

- `motion_corpus_pattern_tree_mainline.md`: this decision record and execution
  order.
- `motion_cluster_bpe_tree_induction.md`: how geometry clusters and motion-BPE
  motifs become motion-derived tree nodes.
- `multi_channel_motion_bpe_extraction.md`: how to extract channel events,
  parallel packets, and multi-channel BPE motifs without flattening concurrent
  motion into accidental adjacency.
- `text_bpe_wordnet_naming_layer.md`: how HumanML3D text-BPE, caption aliases,
  and WordNet name those nodes without creating structure.

Older design documents remain useful for implementation details, but they
should not override this direction.

## Roles Of The Main Components

### Motion Corpus

HumanML3D motion is treated as a corpus. Each clip contributes Layer0/1/2/3
observables and events. This motion corpus is the source of structural
evidence.

Motion evidence can support:

- event clusters;
- repeated local event compositions;
- motif families;
- parent-child relations by inclusion, composition, and shared axes;
- numeric edit handles such as count, span, side, direction, magnitude, and
  duration.

### Motion Cluster

Motion or geometry clusters are the base units. They describe which body
system moved and how. They are allowed to be unnamed.

Examples:

- `WHOLE_BODY_VERTICAL/WB_VERT_UP`
- `BIMANUAL_PERIODIC/BI_RAISE_SPREAD`
- `TORSO_POSTURE/TORSO_HUNCHED_FORWARD`
- `RIGHT_ARM_PERIODIC/RA_NEAR_FAR`

Clusters are used to decide whether a recurring signal is motion-real, noise,
context-only, or currently unmapped.

### Motion BPE

The first Motion-BPE audit runs over Layer3 event-token sequences. The next
mainline Motion-BPE should run over channel events and parallel packets so
that simultaneous upper-body and lower-body motion is represented as
coordination rather than accidental token adjacency.

Motion BPE should answer:

- which event clusters repeatedly occur together;
- which compositions compress the corpus;
- which compositions are stable enough to become motifs;
- whether old AML nodes are too broad, too narrow, or missing.

Motion BPE should not answer:

- the final action name;
- whether a text phrase should create a detector;
- whether an object-dependent activity is visible from skeleton alone.

### Text BPE

HumanML3D text is a second corpus. It is used to mine phrase statistics and
surface names.

Text BPE can discover terms such as:

- `sit down`
- `stand up`
- `jumping jack`
- `jump rope`
- `martial arts`
- `dribble basketball`

These phrases are naming and audit evidence only. They cannot create a motion
tree node unless motion evidence already supports a node or motif.

### WordNet

WordNet supplies broad lexical structure, synonyms, and hypernym paths. It is
useful for naming and grouping labels, especially when many phrases describe
similar activity.

WordNet does not decide skeleton-motion hierarchy. A WordNet parent-child
relation is not automatically a motion-tree parent-child relation.

## Status Of The Old Tree

The old hand-authored AML tree and related runtime material are demoted to:

- bootstrap reference;
- legacy alignment target;
- coverage evaluator;
- source of reusable prompt/materialization machinery;
- temporary runtime dependency until the corpus-derived tree is validated.

It should not be expanded by adding more case-specific action names directly.

The old tree can retain a node only if corpus evidence supports it. Otherwise
the node should be downgraded to a naming alias, a context tag, or legacy-only
coverage reference.

## Execution Plan

### Phase 0: Freeze The Old Tree As An Evaluator

Do not keep growing the old tree by hand. Keep it available to compare against
new motion-derived nodes.

Outputs:

- old-node coverage table;
- old-node retain/downgrade/eliminate decision table;
- list of runtime files that still depend on old node ids.

### Phase 1: Build Motion-Derived Candidate Nodes

Start from the full-HML3D Layer3 event-BPE audit and group stable BPE motifs
with related geometry clusters.

Outputs:

- motif-family candidate JSON;
- candidate markdown report;
- provenance for source motifs, cases, spans, clusters, and old-tree links.

### Phase 2: Mine Language Names

Run text phrase mining and text-BPE over HumanML3D captions, then align phrases
to motion-derived nodes.

Outputs:

- phrase vocabulary;
- caption-alias alignment;
- WordNet naming candidates;
- warnings for object/intent-only labels.

### Phase 3: Propose A New Pattern Tree

Create a non-runtime tree candidate file. It should preserve provenance and
must be inspectable before any runtime replacement.

Outputs:

- `motion_pattern_tree_candidates.json`;
- `motion_pattern_tree_candidate_report.md`;
- split/merge recommendations.

### Phase 4: Compare Against Prompt And MoMask Probes

Use the same 250-case and group-level review protocol, but treat probe results
as downstream validation rather than the tree-construction source.

Outputs:

- coverage improvements;
- unknown/proxy/candidate count changes;
- false-positive audit;
- representative GIF review pack if needed.

### Phase 5: Promote Runtime Schema Changes

Only after a candidate tree passes corpus and probe checks should runtime code
consume it as the authority for action families, prompts, and condition schema.

## Immediate Next Implementation

The next code step is not to edit `coarse_signature.py` again. It is to turn
the full-HML3D event-BPE audit into a compact candidate-tree proposal artifact:

```text
outputs/aml_regression_testset_v2/motion_corpus_tree_candidates_v1/
```

That artifact should group the current stable motifs by motion structure and
language-name evidence, while keeping all decisions offline and reversible.
