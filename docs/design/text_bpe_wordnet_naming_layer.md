# Text-BPE and WordNet Naming Layer

This document defines the language-side naming layer for the new PseudoEdit3D route:

```text
motion cluster + motion BPE
    -> motion-derived pattern tree
    -> text-BPE / caption aliases / WordNet naming candidates
    -> optional semantic labels for tree nodes
```

The core rule is that motion discovers the tree structure. HumanML3D text-BPE and WordNet only help name, audit, and search the resulting motion-derived nodes. The old hand-built AML tree remains useful as legacy, bootstrap, and reference material, but it should not be treated as the source of truth for future tree topology.

## Goals

The naming layer should answer:

- which HumanML3D phrases repeatedly describe a motion-derived motif or cluster;
- which caption aliases group noisy surface phrases into existing semantic buckets;
- which WordNet action hypernym paths make a candidate label interpretable;
- whether a label is pure enough to attach to a tree node as metadata;
- where language evidence disagrees with motion evidence and should stay diagnostic.

The naming layer should not answer:

- whether two motion nodes should be split or merged;
- whether a motion pattern exists;
- whether a weak caption phrase can create a detector;
- whether an activity-intent word such as `combat`, `dance`, or `phone` should override geometry.

## Inputs

The first implementation should read, but not mutate, these sources:

- HumanML3D caption files under the configured HML3D root.
- `pseudoedit3d/edit/aml_semantic_alias_sidecar.json`, which defines caption alias rules and already states that aliases may name detected geometry but must not create motion evidence.
- `outputs/aml_lexicon/wordnet_action_terms_v1.json`, produced by `scripts/build_wordnet_action_lexicon.py`, if available.
- Motion-side audit outputs from the current route, especially:
  - `motion_bpe_vocab.json`
  - `bpe_motif_audit.json`
  - `bpe_phrase_to_pattern_tree_candidates.json`
  - `bpe_caption_alias_alignment.json`

The immediate alignment target is:

```text
outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/bpe_caption_alias_alignment.json
```

Its existing `description` says the file is for corpus audit only and that captions do not create motion evidence. The naming layer should preserve that contract.

## HumanML3D Text-BPE and Phrase Mining

### Corpus Rows

Build a caption corpus where each row preserves traceability:

```json
{
  "case_id": "000000",
  "caption_index": 0,
  "caption_text": "a man kicks something or someone with his left leg.",
  "normalized_text": "a man kicks something or someone with his left leg",
  "source": "humanml3d_text"
}
```

If a HumanML3D text line contains metadata after `#`, the phrase miner should use only the caption text before the first `#`, while keeping any segment metadata as optional provenance if later needed.

### Normalization

Use conservative normalization:

- lowercase text;
- normalize punctuation and whitespace;
- keep body-side words such as `left`, `right`, `both`, `one`;
- keep body-part words such as `arm`, `hand`, `leg`, `foot`, `torso`, `head`;
- keep directional words such as `up`, `down`, `forward`, `backward`, `around`, `clockwise`;
- lemmatize verbs only if the original surface form is also stored;
- avoid dropping object words too early, because they are useful for ambiguity diagnostics even when they are not motion evidence.

The phrase miner should store both `surface_phrase` and `normalized_phrase`. For example, `kicks`, `kicking`, and `kick` may share a normalized form, but the surface inventory is still useful for rendering and audit.

### Phrase Candidates

Use two complementary phrase sources.

First, mine frequent contiguous phrases:

- unigrams, bigrams, trigrams, and short four-grams;
- verb-particle forms such as `sit down`, `stand up`, `turn around`;
- body-part constructions such as `left arm`, `right foot`, `hand to face`;
- repeated activity expressions such as `jumping jack`, `jump rope`, `martial arts`.

Second, train text-BPE over normalized caption tokens:

```text
caption tokens
    -> frequent adjacent token merges
    -> text subword / phrase units
    -> phrase vocabulary with support statistics
```

Text-BPE should be treated as a phrase-discovery baseline, not as a linguistic truth. It is useful because it can discover common corpus phrases without hand-writing every alias. It should be supplemented by phrase scoring:

- `case_support`: number of unique motion cases containing the phrase;
- `caption_support`: number of caption instances containing the phrase;
- `phrase_len`: token length after normalization;
- `pmi` or another association score for multi-word phrases;
- `surface_entropy`: whether the normalized phrase has many inconsistent surface forms;
- `motion_alignment_lift`: whether the phrase is enriched for a motion motif or cluster.

### Phrase Filtering

Do not keep every frequent phrase. Remove or down-rank phrases that are too generic:

- actor-only phrases: `a person`, `the man`, `someone`;
- pure discourse phrases: `appears to`, `seems like`;
- unsupported modifiers: `quickly`, `slowly`, unless the motion side has a stable tempo/control field;
- object-only phrases with no action head, unless they are needed as activity-intent diagnostics.

Filtering should be explicit and auditable. A rejected phrase should be traceable to a rule such as `generic_actor_phrase`, `low_case_support`, `low_motion_lift`, or `object_without_motion_head`.

## Phrase, Caption Alias, and WordNet Relations

The naming layer has three language abstractions, ordered from corpus-specific to lexical-general.

### Text Phrase

A text phrase is a surface or normalized phrase mined from HumanML3D captions.

Examples:

- `kick`
- `left foot`
- `sit down`
- `stand back up`
- `jumping jack`
- `answering the phone`

Text phrases are corpus observations. They can be noisy, annotator-specific, and inconsistent. They should carry statistics and provenance, not authority.

### Caption Alias

A caption alias clusters phrases into a project-specific semantic handle. Existing examples include:

- `jumping_jack`
- `jump_rope`
- `sit_down`
- `sit_down_stand_up`
- `hand_to_face_or_ear`
- `martial_arts`

Caption aliases are useful because they turn many caption patterns into a stable audit key. They are still language-side evidence. They can support naming an already detected motion node, but they cannot create a motion node.

The relationship should be many-to-many:

```text
text phrases <-> caption aliases <-> motion motifs
```

A phrase may map to multiple aliases when ambiguous. An alias may contain many phrases. A motion motif may have several plausible aliases, and an alias may align to several different motifs.

### WordNet Action Hierarchy

WordNet provides lexical hypernym paths and synonym sets for action labels. In this project it should be used only through an offline cached artifact, following the existing `scripts/build_wordnet_action_lexicon.py` policy.

WordNet should contribute:

- canonical lemma candidates;
- synonym and derivational variants;
- verb/noun action evidence;
- lexname metadata;
- hypernym paths for display and grouping;
- broad action-parent candidates.

WordNet should not contribute:

- runtime motion detection;
- tree-node creation;
- split/merge decisions for motion clusters;
- high-confidence labels without corpus and motion support.

For phrases missing from WordNet, especially dataset-specific activity phrases such as `jumping jack`, the layer may keep curated phrase entries with `source=["text_bpe", "caption_alias", "curated_seed"]`.

## Why Language Must Not Decide Motion Tree Structure

HumanML3D captions are valuable but not structurally reliable enough to define the motion tree.

First, captions are noisy. The same motion can be described as `wave`, `raise hand`, `answer phone`, or `scratch head`. These phrases carry different object and intent assumptions even when the observable arm geometry is similar.

Second, captions are uneven in granularity. One caption may say `walks`, another may say `walks forward, turns around, and sits down`. A tree built from text would mix whole-clip activity names with local motion units.

Third, language can describe intent or object context that is absent from the skeleton. For example, `phone`, `rope`, `ladder`, `fight`, and `dance` may be useful labels or probes, but skeleton-only motion may not support them as hard structure.

Fourth, text-BPE optimizes phrase compression over words, while motion-BPE optimizes reusable local structure over motion events. A frequent language phrase is not necessarily a coherent motion unit, and a coherent motion motif may have no stable caption phrase.

Therefore:

- motion clusters and motion-BPE motifs decide candidate nodes;
- full-corpus motion support and motion purity decide splits and merges;
- language evidence can propose names, aliases, and audit warnings;
- high language confidence is a naming signal, not structural evidence;
- low language confidence does not invalidate a motion-derived node.

## Output Schema

The naming layer should produce one primary JSON artifact and optional reports. A suggested filename is:

```text
outputs/.../text_bpe_wordnet_naming_layer.json
```

### Top-Level Artifact

```json
{
  "schema_version": "text_bpe_wordnet_naming_layer_v1",
  "runtime_policy": "language names motion-derived nodes; language does not create or restructure the motion tree",
  "source": {
    "hml_root": "...",
    "caption_alias_sidecar": "pseudoedit3d/edit/aml_semantic_alias_sidecar.json",
    "wordnet_action_lexicon": "outputs/aml_lexicon/wordnet_action_terms_v1.json",
    "motion_bpe_alignment": "outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/bpe_caption_alias_alignment.json"
  },
  "summary": {},
  "phrase_vocab": [],
  "alias_clusters": [],
  "wordnet_paths": [],
  "motion_node_labels": []
}
```

### Phrase Vocab

Each phrase entry should describe corpus evidence and optional alignment to aliases and motion nodes:

```json
{
  "phrase_id": "txt_phrase_000123",
  "surface_phrase": "sits down",
  "normalized_phrase": "sit down",
  "tokens": ["sit", "down"],
  "source": ["text_bpe", "ngram"],
  "case_support": 1530,
  "caption_support": 1842,
  "caption_examples": [
    {"case_id": "002748", "caption_index": 0}
  ],
  "alias_candidates": [
    {
      "alias_id": "sit_down",
      "match_type": "caption_alias_regex",
      "confidence": 0.72
    }
  ],
  "motion_alignment": [
    {
      "motion_node_id": "<M0111>",
      "node_type": "motion_bpe_motif",
      "positive_support": 51,
      "node_total_support": 169,
      "phrase_purity": 0.3018,
      "phrase_recall": 0.0981,
      "top_tree_family": "LOW_BODY_HOLD_PROXY",
      "tree_family_purity": 0.7692
    }
  ],
  "status": "candidate"
}
```

`status` should be one of:

- `candidate`: enough support to keep for audit;
- `stable_name`: high enough confidence and purity to label a motion node;
- `ambiguous`: useful phrase, but maps to multiple incompatible motion signatures;
- `diagnostic_only`: useful for reports, not for naming;
- `rejected`: filtered out with an explicit reason.

### Alias Clusters

Each alias cluster should join text phrases, sidecar aliases, and motion alignment statistics:

```json
{
  "alias_id": "hand_to_face_or_ear",
  "label": "hand near face or ear",
  "source": ["aml_semantic_alias_sidecar", "text_bpe"],
  "phrase_ids": ["txt_phrase_000201", "txt_phrase_000202"],
  "caption_patterns": ["phone", "ear", "face", "near head"],
  "compatible_motion_families": [
    "LEFT_HAND_RAISED_HIGH",
    "RIGHT_HAND_RAISED_HIGH",
    "LEFT_ARM_PERIODIC_GESTURE_PROXY",
    "RIGHT_ARM_PERIODIC_GESTURE_PROXY"
  ],
  "case_support": 1846,
  "motion_motif_alignment": [
    {
      "motif_id": "<M0010>",
      "alias_positive_support": 125,
      "motif_total_support": 963,
      "alias_purity": 0.1298,
      "alias_recall": 0.0677,
      "top_tree_family": "RIGHT_ARM_PERIODIC_GESTURE_PROXY",
      "tree_family_purity": 0.6677
    }
  ],
  "ambiguity_notes": [
    "object_or_intent_may_not_be_visible_in_skeleton"
  ]
}
```

### WordNet Hypernym Path

WordNet entries should be cached, not queried at runtime:

```json
{
  "wordnet_entry_id": "wn_kick_v",
  "normalized_phrase": "kick",
  "lemma": "kick",
  "pos": ["verb", "noun"],
  "source": ["wordnet"],
  "synsets": [
    {
      "name": "kick.v.01",
      "definition": "...",
      "lexname": "verb.contact"
    }
  ],
  "hypernym_paths": [
    ["move", "touch", "strike", "kick"]
  ],
  "taxonomy_parent_candidates": [
    {
      "parent_id": "LOWER_LIMB_ACTION",
      "confidence": 0.9,
      "rule": "seed:kick",
      "ambiguity_boundary": "motion_geometry"
    }
  ],
  "candidate_family_ids": []
}
```

The path is for naming and display. It is not the motion-pattern tree path.

### Candidate Semantic Labels

A motion node can receive multiple candidate labels. The label selector should keep all candidates and mark the chosen one, instead of overwriting alternatives:

```json
{
  "motion_node_id": "<M0111>",
  "node_type": "motion_bpe_motif",
  "motion_tree_path": ["BODY_LEVEL_POSTURE", "LOW_BODY_HOLD_PROXY"],
  "candidate_labels": [
    {
      "label": "kneel or fall to knees",
      "label_type": "caption_alias",
      "alias_id": "kneel_or_fall_to_knees",
      "phrase_ids": ["txt_phrase_000310"],
      "wordnet_path_ids": ["wn_kneel_v"],
      "confidence": 0.62,
      "purity": 0.3018,
      "recall": 0.0981,
      "motion_tree_family_purity": 0.7692,
      "decision": "candidate",
      "reason": "language support is visible but below stable-name threshold"
    }
  ],
  "selected_label": null,
  "naming_status": "unnamed_or_ambiguous"
}
```

Use conservative naming statuses:

- `stable_named`: one label has high support, high purity, and compatible motion family;
- `weak_named`: label is useful for inspection or rendering but not canonical;
- `unnamed_or_ambiguous`: no language label is reliable enough;
- `diagnostic_conflict`: language suggests an incompatible family or activity intent.

### Confidence and Purity

Keep confidence and purity separate.

`purity` should measure empirical concentration:

```text
alias_purity = alias_positive_support / motif_total_support
alias_recall = alias_positive_support / alias_total_cases
tree_family_purity = top_tree_family_count / motif_occurrence_count
```

`confidence` should combine multiple signals:

- phrase case support;
- phrase-BPE stability;
- caption alias sidecar confidence;
- WordNet mapping confidence;
- compatibility with the motion node's family;
- empirical purity and recall;
- ambiguity penalties for object/intent-only phrases.

A label can have high WordNet confidence but low motion purity. That should remain a weak or diagnostic label. A label can have high motion purity but no WordNet entry. That can still be a stable project label if caption evidence is strong and the motion family is compatible.

## Alignment With Existing BPE Caption Alias Output

The file:

```text
outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/bpe_caption_alias_alignment.json
```

currently has:

```json
{
  "summary": {
    "version": "hml3d_layer3_event_bpe_audit_v1",
    "num_records": 29228,
    "token_granularity": "geometry",
    "include_coarse": true,
    "outputs": {
      "caption_alias_alignment": "..."
    }
  },
  "description": "...",
  "alignments": []
}
```

Each alignment row is organized by `caption_alias`:

```json
{
  "caption_alias": "sit_down",
  "alias_case_count": 1530,
  "top_motifs": [
    {
      "motif_id": "<M0010>",
      "alias_positive_support": 138,
      "motif_total_support": 963,
      "alias_total_cases": 1530,
      "alias_purity": 0.1433,
      "alias_recall": 0.0902,
      "top_tree_family": "RIGHT_ARM_PERIODIC_GESTURE_PROXY",
      "tree_family_purity": 0.6677,
      "top_geometry_clusters": [],
      "examples": []
    }
  ]
}
```

The naming layer should load this file as the alias-to-motion reverse view.

Mapping rules:

- `caption_alias` maps to `alias_clusters[].alias_id`.
- `alias_case_count` maps to `alias_clusters[].case_support`.
- each `top_motifs[]` entry maps to `alias_clusters[].motion_motif_alignment[]`.
- `motif_id` maps to `motion_node_labels[].motion_node_id` when that motif is selected for naming.
- `alias_purity`, `alias_recall`, and `tree_family_purity` feed label scoring but do not create a node.
- `top_geometry_clusters` helps explain which motion clusters support the alias.
- `examples` should be copied only as lightweight provenance for audit reports.

The output should also support the opposite lookup:

```text
motion_node_id -> candidate labels -> phrase evidence / alias evidence / WordNet path
```

This requires inverting the existing `alignments` list by `motif_id`. The inversion is a reporting operation, not a tree edit.

## Naming Decision Policy

A candidate label can become `stable_named` only when all of these are true:

- the motion node already exists in the motion-derived tree;
- the label is compatible with the node's motion family or cluster;
- the label has sufficient case support;
- purity is high enough for the intended label granularity;
- there is no stronger incompatible label for the same node;
- object/intent assumptions are either visible in motion or explicitly marked as weak.

Recommended initial thresholds should be conservative and report-only:

- `min_phrase_case_support`: 20
- `min_alias_positive_support`: 20
- `min_stable_alias_purity`: 0.45
- `min_tree_family_purity`: 0.65
- `min_confidence_stable_name`: 0.70

These thresholds are starting points for audit. They should not automatically modify the motion pattern tree.

## Minimal Executable Script Plan

Do not implement these scripts until the schema is accepted. The smallest plan is:

### Step 1: Build Text Phrase Vocabulary

Proposed script:

```text
scripts/build_hml3d_text_phrase_vocab.py
```

Responsibilities:

- read HumanML3D caption text files;
- normalize captions;
- mine n-grams and verb-particle phrases;
- train text-BPE phrase merges;
- export `phrase_vocab.json` and a short markdown audit.

No motion tree writes.

### Step 2: Attach Caption Alias Clusters

Proposed script:

```text
scripts/align_text_phrases_to_caption_aliases.py
```

Responsibilities:

- read `phrase_vocab.json`;
- read `aml_semantic_alias_sidecar.json`;
- map phrases to alias rules;
- report alias coverage and ambiguous phrases;
- export `alias_clusters.json`.

No detector creation.

### Step 3: Attach WordNet Paths

Proposed script:

```text
scripts/attach_wordnet_paths_to_text_phrases.py
```

Responsibilities:

- read `phrase_vocab.json`;
- read cached `wordnet_action_terms_v1.json`;
- attach lemma, synset, hypernym, and taxonomy-parent candidates;
- mark missing WordNet phrases for curated review.

No runtime WordNet dependency.

### Step 4: Score Motion Node Names

Proposed script:

```text
scripts/score_motion_tree_semantic_labels.py
```

Responsibilities:

- read motion-BPE outputs;
- read `bpe_caption_alias_alignment.json`;
- read phrase vocab, alias clusters, and WordNet paths;
- invert alias-to-motif rows into motif-to-label candidates;
- compute confidence, purity, recall, and ambiguity penalties;
- export `text_bpe_wordnet_naming_layer.json`.

No motion-tree structural edits.

### Step 5: Human-Readable Report

Proposed script:

```text
scripts/report_motion_tree_naming_candidates.py
```

Responsibilities:

- summarize stable labels, weak labels, ambiguous nodes, and conflicts;
- show examples per candidate label;
- list high-support unnamed motion nodes;
- list high-support language phrases with no clean motion node.

This report should be the review surface before any future manual promotion.

## Legacy AML Tree Use

The hand-built AML tree should be used in three limited ways:

- bootstrap: provide initial alias sidecar rules and known family names;
- reference: compare new motion-derived nodes against old labels;
- regression: detect when a new naming pass loses important known distinctions.

It should not force the new tree to preserve old topology. When the new motion-BPE tree disagrees with the legacy tree, the naming layer should report the disagreement instead of resolving it structurally.

## Non-Goals

- No code changes in this design step.
- No automatic edits to `aml_pattern_tree.json`.
- No automatic edits to the alias sidecar.
- No runtime WordNet queries.
- No tree split/merge decisions from text alone.
- No claim that every motion node must have a natural-language action name.

