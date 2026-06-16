# Motion Cluster + Motion BPE Tree Induction

## Status

This document defines the new route for inducing a data-driven motion pattern
tree from HumanML3D motions. It is a design document only. It does not propose
editing the runtime `aml_pattern_tree.json` directly.

The core decision is:

- HumanML3D motion is the corpus.
- Layer3 event clusters and motion/geometry clusters provide the primary
  structural evidence.
- Event-BPE motifs provide repeated local composition evidence.
- Text-BPE, caption aliases, and WordNet are naming aids only.
- The hand-built AML tree is a legacy matcher, bootstrap reference, and audit
  target, not the authority for the new structure.

## Current Evidence Snapshot

The full audit at
`outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/` used:

- `29228` HumanML3D records.
- `393032` original Layer3 event tokens.
- `326070` BPE tokens after merging.
- compression ratio `0.829627`.
- `256` learned BPE merges.
- base vocabulary size `649`.
- token granularity `geometry`.
- current coarse AML tree binding included for audit comparison.
- `8` stable pattern-tree candidates under the configured caption-alias
  threshold.

The existing audit already follows the intended separation:

- motion tokens and motifs are learned from Layer3 event order.
- caption aliases are used to measure naming stability.
- current AML tree families are used to measure alignment and coverage.
- a high-purity motif is marked for inspection, not automatic tree editing.

This audit is still a baseline, not the final Motion-BPE design:

- concurrent events are sorted into one sequence before BPE;
- the `256` merges are a configured audit budget, not a natural final
  vocabulary size;
- speed is unevenly represented across channels, especially for rotation where
  angular speed should be explicit;
- the next mainline extraction is the multi-channel design in
  `multi_channel_motion_bpe_extraction.md`.

## Evidence Roles

### Motion / Geometry Clusters

Motion clusters are the base motion evidence. In the current implementation,
each Layer3 event has a `geometry_cluster_id` shaped as:

```text
<super_family>/<cluster_id>
```

Examples:

- `BIMANUAL_PERIODIC/BI_RAISE_SPREAD`
- `WHOLE_BODY_VERTICAL/WB_VERT_UP`
- `TORSO_POSTURE/TORSO_HUNCHED_FORWARD`
- `RIGHT_ARM_PERIODIC/RA_NEAR_FAR`

Cluster-level evidence answers:

- what body part or whole-body system moved.
- what kinematic family the event belongs to.
- which direction, magnitude, duration, count, and numeric fields are attached.
- which frame span the event occupies.
- whether the event is standalone, context-only, or currently unmapped.
- how often the same geometry appears across the corpus.

Motion clusters should decide tree splits when the split is stable in the
motion itself. Text labels should not create a split without a cluster-level
or motif-level reason.

### Motion BPE Motifs

Motion BPE motifs are repeated local sequences over Layer3 event tokens. They
answer a different question from clusters:

- which event clusters repeatedly occur next to each other.
- which local compositions compress the corpus.
- whether a composition is a stable unit rather than an accidental adjacency.
- whether a phrase crosses body parts, for example bimanual arm spread plus
  vertical body rise.
- whether a current AML tree node is too broad, too narrow, or missing a
  recurring composition.

The merge vocabulary records:

- `merge_id`, such as `<M0230>`.
- `parents`, the two symbols merged at that step.
- raw occurrence count.
- support case count.
- example case ids and example occurrences.

Motif audit records add:

- top caption alias and caption-alias purity.
- top current tree family and tree-family purity.
- top geometry clusters.
- top base symbols.
- example captions and linked current AML families.

Motion BPE should be treated as composition evidence, not naming evidence.
For example, a motif may be structurally stable while having no stable caption
alias; such a motif can still be a valid lower-level tree node.

## Induction Pipeline

The new tree should be induced bottom-up:

```text
base event clusters
-> channel events and parallel packets
-> BPE motifs
-> motif families
-> tree nodes
```

The preferred next extraction path is:

```text
dense observables
-> channel-specific events
-> temporal overlap graph
-> parallel packets
-> per-channel and packet-sequence BPE
-> multi-channel motifs
-> motif families
-> motion pattern forest
```

The single-sequence audit remains a useful comparison point, but the
multi-channel route is necessary for motions where upper-body, lower-body,
root, and vertical events happen concurrently.

### 1. Base Event Clusters

Start with Layer3 events sorted by event span and event sort key. For each
event, keep:

- `geometry_cluster_id`.
- `super_family`.
- `cluster_id`.
- `part`.
- `role`.
- `direction`.
- `span`.
- `duration`.
- `magnitude`.
- `unit`.
- `count`.
- `motion_signature`.
- optional current AML links for audit only.

The base token can be coarse or detailed. The current full audit uses geometry
granularity:

```text
<super_family>/<cluster_id>|dir=<direction>|dur=<duration_bin>|mag=<magnitude_bin>|count=<count_bin>
```

This keeps control handles in the token while preserving the ability to roll
them back into schema fields later.

### 2. BPE Motifs

In the baseline audit, BPE runs over flattened event-token sequences. In the
multi-channel design, BPE runs over multiple views:

- per-channel event sequences;
- packet sequences built from temporally overlapping cross-channel events;
- relation triples that preserve parallel, lead-lag, and before-after
  relations.

Each merge creates a motif with:

- ordered parent symbols.
- support case count.
- occurrence count.
- examples.
- base symbols inherited from children.
- event indices and spans for each occurrence.
- participating parts, super-families, and geometry clusters.
- relation profile when the motif spans multiple channels.
- numeric profile for duration, magnitude, speed, and count.

BPE motifs should be accepted as candidate sub-motion units when they have
enough support and are not purely an artifact of sorting concurrent events.

### 3. Motif Families

Motif families group motifs that express the same motion composition with
small variations in magnitude, duration, side, count, or lexical label.

Examples:

- `<M0077>` and `<M0252>` both combine torso hunch with vertical-up motion and
  align with sit-down-like cases. They can form a sit/stand transition family
  candidate, while preserving magnitude as a handle.
- `<M0230>`, `<M0075>`, and `<M0216>` all touch jumping-jack-like evidence but
  represent different components: bimanual spread plus vertical rise, repeated
  bimanual spread, and both hands high. These should not be blindly collapsed;
  the family should record their roles.
- `<M0050>` and `<M0170>` are cheer/dance-like by caption alias but currently
  have weaker tree purity. They are better treated as inspection candidates
  or lower-confidence motif-family members until motion-side separation is
  clearer.

A motif family should have:

- a canonical set of geometry clusters.
- allowed variants for duration, magnitude, count, side, and part.
- a distribution over source motifs.
- corpus support and coverage statistics.
- optional naming hints from captions or WordNet.
- audit links to old AML nodes.

### 4. Tree Nodes

A tree node is promoted from a motif family only when the family is both
motion-stable and useful as a reusable abstraction.

Tree hierarchy should be induced by inclusion and composition:

- abstract parent: shared cluster axes or common body system.
- intermediate node: stable motif family, possibly unnamed.
- leaf node: reusable action-like pattern with handles.
- lexical alias: surface name attached after induction.

The tree should not be a WordNet tree. WordNet can suggest labels such as
`jumping_jack`, `sit_down`, or `cheer_dance`, but the parent-child relation
must come from motion evidence:

- motif-family inclusion.
- cluster overlap.
- event-span composition.
- support and purity.
- one-to-many / many-to-one diagnostics.

## Metrics

All metrics should be computed at the case level by default, with occurrence
level values retained for diagnosis.

Let:

- `C` be all valid corpus cases.
- `O_m` be all occurrences of motif `m`.
- `S_m` be the set of cases containing motif `m`.
- `A_l` be the set of cases with caption alias or text label `l`.
- `T_f` be the set of occurrences or cases linked to old AML family `f`.
- `G_k` be the set of events in geometry cluster `k`.
- `N` be a proposed new tree node.

### Support

Motif support:

```text
support_cases(m) = |S_m|
occurrences(m) = |O_m|
```

Node support:

```text
support_cases(N) = number of cases matched by any accepted source motif or cluster rule for N
support_occurrences(N) = number of matched node occurrences across cases
```

Use case support for promotion decisions. Use occurrence support to diagnose
over-repeated motifs in a small number of cases.

### Purity

Caption-alias purity is a naming diagnostic:

```text
caption_alias_purity(m, l) = |S_m intersect A_l| / |S_m|
```

It should answer: "If this motion motif appears, how consistently do captions
use this name or alias?"

Old-tree family purity is a legacy-alignment diagnostic:

```text
tree_family_purity(m, f) = cases or occurrences of m linked to old family f / support of m
```

It should answer: "Does the motif mostly map to one current AML family?"

Geometry purity is the motion-side stability diagnostic:

```text
geometry_purity(N, k) = events under N that use geometry cluster k / all events under N
```

For a multi-cluster node, use a required-cluster-set version:

```text
cluster_set_purity(N, K) = occurrences of N containing required cluster set K / all occurrences of N
```

Promotion should prioritize geometry and motif purity. Caption purity can name
the node but should not rescue a motion-impure node.

### Coverage

Motif coverage of a caption alias:

```text
alias_recall(m, l) = |S_m intersect A_l| / |A_l|
```

Node coverage of old AML family:

```text
old_family_coverage(N, f) = cases matched by N and old family f / cases matched by old family f
```

Node coverage of a motion cluster:

```text
cluster_coverage(N, k) = events from cluster k covered by N / all events from cluster k
```

Tree coverage:

```text
tree_event_coverage = Layer3 events covered by any induced node / all Layer3 events
tree_case_coverage = cases with at least one induced node / all valid cases
```

Coverage should be reported together with purity. High coverage with low purity
means the node is too broad. High purity with very low coverage means the node
may be a leaf, not a parent.

### One-To-Many And Many-To-One

These diagnostics decide whether text labels, old AML nodes, and new motion
nodes are aligned cleanly.

One text alias to many motifs:

```text
alias -> {motifs}
```

This indicates that a word names multiple motion realizations. Example:
`jumping_jack` may involve bimanual spread, both-hands-high posture, and
vertical rise phases. The tree should preserve subnodes and attach the alias
to the family or composed node.

Many text aliases to one motif:

```text
{aliases} -> motif
```

This indicates a motion primitive is reused by many named activities. The
motif should become a lower-level motion node, not a lexical action leaf.

One old AML family to many motifs:

```text
old_family -> {motif families}
```

This indicates the old node is too broad or is mixing phases. It should be
split, or kept only as an abstract parent.

Many old AML families to one motif:

```text
{old_families} -> motif
```

This indicates duplicate or competing old nodes. The old nodes should be
merged, downgraded to aliases, or marked legacy-only.

One geometry cluster to many motifs:

```text
geometry_cluster -> {motifs}
```

This is normal. It means a primitive participates in multiple compositions.
The geometry cluster should remain a base event node or handle-bearing child.

Many geometry clusters to one motif:

```text
{geometry_clusters} -> motif
```

This is the strongest evidence for a composed tree node.

## Legacy AML Tree Disposition

The old `aml_pattern_tree.json` should be audited node by node against the new
motion-corpus evidence. A node can be retained, downgraded, or removed from the
future data-driven tree.

### Retain

Retain a legacy node as a real motion node when:

- it has high support in the corpus.
- its matched events are motion-pure, not just text-pure.
- its source geometry clusters are stable.
- one-to-many diagnostics show it is a coherent parent rather than a confused
  leaf.
- its children or source motifs form a plausible inclusion/composition
  hierarchy.
- it improves coverage without hiding important variants.

Retained nodes still need provenance fields pointing to source motifs and
geometry clusters. A retained old node is not retained because it already
exists; it is retained because corpus evidence re-discovers it.

### Downgrade

Downgrade a legacy node when:

- it has a useful name but weak motion specificity.
- it is mostly a proxy over one base cluster.
- it mixes several motif families.
- it is text-stable but geometry-unstable.
- it is useful for prompts or inspection but not for structural induction.

Downgrade targets:

- `alias`: lexical name attached to a stronger motion node.
- `abstract_parent`: grouping node with no direct matcher.
- `event_proxy`: base event label, not an action pattern.
- `legacy_audit_only`: kept only for comparing old and new outputs.

### Eliminate

Eliminate a legacy node from the future induced tree when:

- it has low support and low coverage.
- it duplicates another node without adding handles or hierarchy.
- its matches are dominated by unmapped/context/noise events.
- it only exists because WordNet or hand-built taxonomy suggested it.
- it has no stable source motifs and no stable geometry cluster set.

Elimination does not require deleting the current runtime file immediately.
It means the node should not be promoted into the next induced tree snapshot.

## Draft Node Schema

The schema below is for an induced tree snapshot, not for the current runtime
tree. It is intentionally provenance-heavy so that every node can be audited.

```json
{
  "node_id": "MOTION_NODE_JUMPING_JACK_UP_SPREAD_V1",
  "parent_id": "MOTION_FAMILY_BILATERAL_RHYTHMIC_VERTICAL_V1",
  "node_type": "motif_family",
  "status": "candidate",
  "label": {
    "canonical": "jumping_jack_up_spread",
    "display": "jumping jack up-spread phase",
    "source": "caption_alias",
    "aliases": ["jumping_jack"],
    "wordnet_synsets": []
  },
  "motion_definition": {
    "required_geometry_clusters": [
      "BIMANUAL_PERIODIC/BI_RAISE_SPREAD",
      "WHOLE_BODY_VERTICAL/WB_VERT_UP"
    ],
    "optional_geometry_clusters": [],
    "ordered": true,
    "allow_overlap": true,
    "span_relation": "adjacent_or_overlapping",
    "part_handles": ["left_arm", "right_arm", "whole_body"],
    "count_handles": {
      "repeat_count": {
        "source": "event.count",
        "bin_source": "count_bin",
        "default": null
      }
    },
    "numeric_handles": {
      "vertical_magnitude": {
        "unit": "m",
        "source": "WHOLE_BODY_VERTICAL.magnitude",
        "bins": ["m_s", "m_m", "m_l"]
      },
      "arm_spread_magnitude": {
        "unit": "m_or_deg",
        "source": "BIMANUAL_PERIODIC.magnitude",
        "bins": ["m_s", "m_m", "deg_m"]
      },
      "duration": {
        "unit": "frames",
        "source": "span.end - span.start"
      }
    },
    "span_handles": {
      "source_span": "union(source_event_spans)",
      "phase_spans": "per_source_event",
      "normalized_phase_order": ["arms_spread", "body_up"]
    }
  },
  "source_motifs": [
    {
      "motif_id": "<M0230>",
      "role": "core",
      "support_cases": 61,
      "caption_alias": "jumping_jack",
      "caption_alias_purity": 0.918,
      "old_tree_family": "BILATERAL_RHYTHMIC_COORDINATION",
      "old_tree_family_purity": 0.918
    }
  ],
  "source_geometry_clusters": [
    {
      "geometry_cluster_id": "BIMANUAL_PERIODIC/BI_RAISE_SPREAD",
      "role": "arm_spread",
      "required": true
    },
    {
      "geometry_cluster_id": "WHOLE_BODY_VERTICAL/WB_VERT_UP",
      "role": "vertical_impulse",
      "required": true
    }
  ],
  "metrics": {
    "support_cases": 61,
    "support_occurrences": 97,
    "motion_purity": null,
    "caption_alias_purity": 0.918,
    "old_tree_family_purity": 0.918,
    "coverage": {
      "tree_event_coverage": null,
      "alias_recall": null,
      "old_family_coverage": null
    },
    "ambiguity": {
      "one_alias_to_many_motifs": true,
      "many_aliases_to_one_motif": false,
      "one_old_family_to_many_motifs": true,
      "many_old_families_to_one_motif": true
    }
  },
  "legacy_alignment": {
    "old_node_ids": [],
    "old_family_ids": [
      "BILATERAL_RHYTHMIC_COORDINATION",
      "VERTICAL_JUMP"
    ],
    "recommended_disposition_for_old_nodes": "retain_or_split_after_audit"
  },
  "examples": [
    {
      "case_id": "000423",
      "span": [54, 61],
      "event_indices": [14, 15]
    }
  ],
  "provenance": {
    "corpus": "HumanML3D",
    "audit_dir": "outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1",
    "bpe_vocab_version": "motion_bpe_vocab_v1",
    "audit_version": "hml3d_layer3_event_bpe_audit_v1"
  }
}
```

Required schema fields:

- `node_id`: stable id for the induced snapshot.
- `parent_id`: induced parent, not WordNet parent.
- `node_type`: `base_cluster`, `motif`, `motif_family`, `composed_pattern`,
  `abstract_parent`, or `alias_only`.
- `status`: `candidate`, `stable`, `deprecated`, or `legacy_audit_only`.
- `label`: naming metadata only.
- `motion_definition`: the actual matcher definition.
- `source_motifs`: BPE motif provenance.
- `source_geometry_clusters`: geometry evidence provenance.
- `metrics`: support, purity, coverage, ambiguity.
- `legacy_alignment`: comparison against current AML nodes.
- `examples`: auditable case spans.
- `provenance`: exact audit source.

## Initial Stable Candidates

The first manual pass should start from the 8 stable candidates reported by
the full audit. They should be inspected as candidate nodes or candidate motif
families, not applied to runtime matching yet.

| motif | support | caption alias | alias purity | old tree family | old tree purity | motion evidence |
| --- | ---: | --- | ---: | --- | ---: | --- |
| `<M0230>` | 61 | `jumping_jack` | 0.918 | `BILATERAL_RHYTHMIC_COORDINATION` | 0.918 | `BIMANUAL_PERIODIC/BI_RAISE_SPREAD` + `WHOLE_BODY_VERTICAL/WB_VERT_UP` |
| `<M0216>` | 53 | `jumping_jack` | 0.660 | `LEFT_HAND_RAISED_HIGH` | 1.000 | `LEFT_ARM_POSTURE/LA_HAND_HIGH` + `RIGHT_ARM_POSTURE/RA_HAND_HIGH` |
| `<M0077>` | 223 | `sit_down` | 0.601 | `TORSO_HUNCHED_FORWARD` | 0.955 | `TORSO_POSTURE/TORSO_HUNCHED_FORWARD` + `WHOLE_BODY_VERTICAL/WB_VERT_UP` |
| `<M0075>` | 196 | `jumping_jack` | 0.592 | `BILATERAL_RHYTHMIC_COORDINATION` | 0.597 | repeated `BIMANUAL_PERIODIC/BI_RAISE_SPREAD` |
| `<M0252>` | 87 | `sit_down` | 0.540 | `TORSO_HUNCHED_FORWARD` | 0.977 | `TORSO_POSTURE/TORSO_HUNCHED_FORWARD` + larger `WHOLE_BODY_VERTICAL/WB_VERT_UP` |
| `<M0111>` | 169 | `sit_down` | 0.527 | `LOW_BODY_HOLD_PROXY` | 0.769 | `WHOLE_BODY_POSTURE/WB_SQUAT_HOLD` + `WHOLE_BODY_POSTURE/WB_LOW_BODY_HOLD` |
| `<M0050>` | 142 | `cheer_dance` | 0.458 | `BIMANUAL_ARM_MIME_CANDIDATE` | 0.542 | repeated `BIMANUAL_PERIODIC/BI_HANDS_CLOSE` |
| `<M0170>` | 90 | `cheer_dance` | 0.433 | `RIGHT_ARM_PERIODIC_GESTURE_PROXY` | 0.567 | `RIGHT_ARM_PERIODIC/RA_NEAR_FAR` + `LEFT_ARM_PERIODIC/LA_NEAR_FAR` |

Immediate interpretation:

- `<M0230>` is the cleanest promotion candidate because both alias purity and
  old-tree-family purity are high, and the motion composition is clear.
- `<M0077>` and `<M0252>` are strong evidence that sit-like transitions should
  be split into torso-hunch-plus-vertical variants with magnitude handles.
- `<M0216>` is structurally clean but may be a component of jumping-jack rather
  than the whole action.
- `<M0075>` has good alias support but weaker old-tree purity, so it should be
  inspected as a phase or sibling of `<M0230>`.
- `<M0050>` and `<M0170>` are useful for studying cheer/dance arm motifs, but
  their weaker purity argues against immediate promotion as action leaves.

## Minimal Execution Plan

### Phase 0: Freeze Runtime

Do not edit `pseudoedit3d/edit/aml_pattern_tree.json` or runtime matchers in
the first pass. The first output should be an offline induced-tree candidate
artifact and an audit report.

### Phase 1: Candidate Inspection

Use the 8 stable candidates above as the seed set.

For each seed:

- inspect source motif parents in `motion_bpe_vocab.json`.
- inspect examples and top clusters in `bpe_motif_audit.json`.
- inspect reverse caption alignment in `bpe_caption_alias_alignment.json`.
- compute case-level support, occurrence-level support, alias purity, old-tree
  purity, and cluster-set purity.
- record whether the motif is a leaf, phase, component, or abstract parent.

### Phase 2: Family Grouping

Group the 8 seeds into tentative motif families:

- jumping-jack / bilateral rhythmic vertical family:
  `<M0230>`, `<M0216>`, `<M0075>`.
- sit / low-body transition family:
  `<M0077>`, `<M0252>`, `<M0111>`.
- cheer / dance bimanual or alternating arm family:
  `<M0050>`, `<M0170>`.

Keep this grouping provisional. It should be validated by cluster overlap and
case-level co-occurrence, not by caption alias alone.

### Phase 3: Legacy Disposition Table

For each affected old AML family, produce a table:

- old family id.
- matching new motifs.
- support.
- purity.
- coverage.
- one-to-many / many-to-one notes.
- recommendation: retain, downgrade, or eliminate.

Likely first old families to audit:

- `BILATERAL_RHYTHMIC_COORDINATION`
- `LEFT_HAND_RAISED_HIGH`
- `TORSO_HUNCHED_FORWARD`
- `LOW_BODY_HOLD_PROXY`
- `BIMANUAL_ARM_MIME_CANDIDATE`
- `RIGHT_ARM_PERIODIC_GESTURE_PROXY`
- `VERTICAL_JUMP`

### Phase 4: Offline Snapshot

Write a separate induced-tree candidate artifact in a future step, for example:

```text
outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/induced_motion_pattern_tree_candidates.json
```

That artifact should contain:

- node schema records.
- source motif links.
- source geometry cluster links.
- metrics.
- legacy disposition recommendations.
- examples.

This remains offline until validated.

### Phase 5: Validation Before Runtime

Before any runtime tree change, validate:

- whether induced nodes improve event coverage without collapsing distinct
  phases.
- whether old stable runtime behavior is preserved or explicitly replaced.
- whether caption-based names remain labels, not matching requirements.
- whether failure cases are visible in an audit report.
- whether a small held-out subset produces consistent node assignments.

Only after this validation should runtime integration be discussed.

## Non-Goals For The First Pass

- Do not replace `aml_pattern_tree.json`.
- Do not treat WordNet as the source of tree parents.
- Do not add action names directly from captions without motion evidence.
- Do not collapse all motifs with the same caption alias into one node.
- Do not require every high-support motif to become an action leaf.
- Do not hide magnitude, duration, count, part, and span as unstructured text.

## Summary

The new route makes motion evidence primary. Geometry clusters define the base
motion alphabet; motion BPE discovers repeated local phrases; motif families
turn repeated phrases into reusable sub-motion abstractions; and only then are
tree nodes named with help from text aliases or WordNet. The old AML tree
remains useful as a comparison target, but every retained node must be
re-earned from corpus support, purity, coverage, and ambiguity diagnostics.
