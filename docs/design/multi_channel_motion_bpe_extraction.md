# Multi-Channel Motion-BPE Extraction Design

## Status

This document defines the Motion-BPE extraction design. It upgrades the current
single-sequence Layer3 Event-BPE audit into a concurrency-aware, multi-channel
symbolic motion tokenizer.

Current implementation:

```text
scripts/audit_hml3d_multichannel_motion_bpe.py
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/
```

The first implementation is intentionally symbolic and CPU-oriented. It starts
from the existing Layer3 event corpus, builds channel events and overlap
diagnostics, learns channel motifs first, then promotes stable cross-channel
coactivations into coordination motifs. It emits motif family / forest
candidate artifacts and does not replace the old
single-sequence baseline.

The goal is not to create a runtime AML tree immediately. The goal is to
produce a paper-ready and report-ready extraction record that can later support:

- motion pattern forest induction;
- semantic naming by HumanML3D text-BPE and WordNet;
- structured condition schema design;
- edit handles for part, span, count, direction, magnitude, and speed.

## Why The Upgrade Is Needed

The current full-HML3D audit uses a single event sequence:

```text
Layer3 events -> sorted event token sequence -> BPE adjacent merges
```

Current full-audit snapshot:

```text
records:                 29228
original event tokens:    393032
base event-token types:   649
BPE token count:          326070
learned BPE merges:       256
BPE vocabulary types:     905 = 649 base tokens + 256 merge tokens
compression ratio:        0.829627
token granularity:        geometry
```

This was useful as a first audit, but it has two limitations.

First, concurrent motion is flattened. If the upper body raises both arms while
the root runs forward, the current audit emits multiple Layer3 event tokens and
sorts them into one sequence. BPE then sees adjacency, even when the true
relation is parallel overlap.

Second, speed and numeric dynamics are not equally represented across
channels. Rotation has angle bins and duration bins, but angular speed is not a
first-class token field in the default audit. Locomotion has speed-like cluster
names, but distance and speed should become explicit handles.

The upgraded design treats motion as a partially ordered multi-channel event
stream:

```text
dense observables
-> channel events
-> overlap packets
-> multi-channel BPE motifs
-> motif families
-> motion pattern forest
```

## Core Representation

### Event

An event is a time span with a body channel, motion family, numeric attributes,
and provenance.

It should answer:

- which body system moved;
- when it moved;
- in which direction;
- by how much;
- how fast;
- whether it repeats;
- which dense signals and Layer3 events support it.

### Channel

A channel is a body-system stream. The minimum channel set is:

```text
root_locomotion
root_rotation
whole_body_vertical
whole_body_state
torso
left_arm
right_arm
bimanual
left_leg
right_leg
acrobatics_or_inversion
```

Channels are not final semantic families. They are extraction axes. A later
motif can span multiple channels.

### Packet

A packet is a time-local bundle of one or more channel events. It preserves
parallel structure.

Examples:

```text
parallel_packet:
  left_arm:  hand_high
  right_arm: hand_high
  root:      forward_run

parallel_packet:
  bimanual:            raise_spread
  whole_body_vertical: vertical_up
```

Packets are the unit that prevents BPE from confusing concurrency with
sequence.

### Motif

A motif is a reusable unit discovered by BPE-like merging over channel events
and packets.

Motifs can be:

- single-channel sequential motifs;
- cross-channel parallel motifs;
- repeated packets;
- composed motifs that combine parallel packets and sequence.

### Pattern Forest Node

A pattern forest node is a promoted motif family. It is still motion-derived
and may be unnamed. Text-BPE, caption aliases, and WordNet are attached only
after the motion node exists.

## Extraction Pipeline

### Step 0: Corpus Index

Input:

```text
/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/joints3d.pth
/mnt/data/home/guoruoxi/code/momask-codes/dataset/HumanML3D/texts/*.txt
```

Output:

```json
{
  "case_id": "000576",
  "num_frames": 195,
  "motion_source": "HumanML3D",
  "caption_paths": ["texts/000576.txt"],
  "has_joints3d": true
}
```

Design rule:

- captions are loaded for audit and naming only;
- captions must not create motion events;
- every later artifact must preserve `case_id`, frame spans, and provenance.

### Step 1: Dense Motion Observables

Dense observables are frame-level numeric time series extracted from joints.
They are not yet events.

Required observables:

```text
root_xz
root_xz_velocity
root_xz_speed
root_heading_proxy
root_heading_delta
root_angular_speed
root_height
root_vertical_velocity
pelvis_to_foot_height
torso_forward_extent
torso_height_drop
left_hand_height
right_hand_height
left_wrist_to_chest_distance
right_wrist_to_chest_distance
left_arm_speed
right_arm_speed
left_leg_forward_extent
right_leg_forward_extent
left_foot_height
right_foot_height
support_or_contact_proxy
body_inversion_proxy
```

Output schema:

```json
{
  "case_id": "000576",
  "num_frames": 195,
  "observables": {
    "root_xz_speed": {"unit": "m_per_frame", "values": "..."},
    "root_angular_speed": {"unit": "deg_per_frame", "values": "..."},
    "left_hand_height": {"unit": "m", "values": "..."}
  }
}
```

Paper-ready claim:

```text
Layer0 converts each motion into dense, body-channel signals before any
semantic decision is made.
```

### Step 2: Channel Event Extraction

Each observable group is segmented into channel events. This is where dense
signals become symbolic events.

Examples:

```text
root_locomotion:      forward_translate, lateral_translate, active_drift
root_rotation:        left_turn, right_turn, spin, turn_while_walking
whole_body_vertical:  vertical_up, vertical_down, vertical_cycle
torso:                hunch_forward, bend_recover, oscillate
left_arm:             hand_high, repeat_up_down, near_far
right_arm:            hand_high, repeat_up_down, near_far
bimanual:             raise_spread, hands_close, symmetric_near_far
left_leg:             kick_forward, leg_forward_hold, foot_lift
right_leg:            kick_forward, leg_forward_hold, foot_lift
state:                terminal_still, low_body_hold
acrobatics:           inverted_rotation_candidate
```

Channel event schema:

```json
{
  "event_id": "000576:e003",
  "case_id": "000576",
  "channel": "right_arm",
  "super_family": "RIGHT_ARM_PERIODIC",
  "cluster_id": "RA_REPEAT",
  "geometry_cluster_id": "RIGHT_ARM_PERIODIC/RA_REPEAT",
  "span": [42, 91],
  "direction": "up_down",
  "duration": 50,
  "duration_bin": "l",
  "magnitude": 0.31,
  "magnitude_unit": "m",
  "magnitude_bin": "m_m",
  "mean_speed": 0.025,
  "speed_unit": "m_per_frame",
  "speed_bin": "speed_m",
  "count": 4,
  "count_bin": "c4_6",
  "confidence": 0.72,
  "source_observables": ["right_hand_height", "right_arm_speed"],
  "source_layer3_event_indices": [12]
}
```

The key upgrade from the current audit is explicit speed fields:

- rotation: `angular_speed = abs(angle_deg) / duration_frames`;
- locomotion: `root_speed` and `distance_m`;
- limbs: `mean_joint_or_endpoint_speed`;
- repeated events: `cycle_rate = count / duration_frames`.

Rotation token design:

```text
angle_bin:         rot_xs, rot_s, rot_qtr, rot_half, rot_full, rot_multi
duration_bin:      xs, s, m, l, xl
angular_speed_bin: omega_slow, omega_med, omega_fast
```

Locomotion token design:

```text
distance_bin:      dist_xs, dist_s, dist_m, dist_l, dist_xl
speed_bin:         speed_slow, speed_med, speed_fast
path_bin:          straight, curved, turn_path, mixed
```

Limb token design:

```text
amplitude_bin:     amp_xs, amp_s, amp_m, amp_l
speed_bin:         speed_slow, speed_med, speed_fast
repeat_bin:        c1, c2_3, c4_6, c7p
```

### Step 3: Channel Assignment And Normalized Tokens

Each event gets both a rich event record and a compact token symbol.

Token schema:

```json
{
  "token_id": "tok_000576_003",
  "event_id": "000576:e003",
  "case_id": "000576",
  "channel": "right_arm",
  "span": [42, 91],
  "symbol": "right_arm/RA_REPEAT|dir=up_down|dur=l|amp=m_m|speed=speed_m|count=c4_6",
  "base_symbols": [
    "right_arm/RA_REPEAT|dir=up_down|dur=l|amp=m_m|speed=speed_m|count=c4_6"
  ],
  "numeric": {
    "duration": 50,
    "magnitude": 0.31,
    "mean_speed": 0.025,
    "count": 4
  }
}
```

Type count at this stage means:

```text
number of distinct token.symbol strings
```

It is the motion analogue of a text alphabet. It is not the number of action
classes.

### Step 4: Temporal Overlap Graph

For each case, build an overlap graph over channel events.

Two events are related by:

```text
parallel       if overlap_ratio >= threshold
lead_lag       if they overlap weakly or have a small gap
before_after   if one clearly follows the other
same_channel   if they are adjacent within one channel
```

Default thresholds for the first implementation:

```text
parallel_overlap_min = 0.30
lead_lag_gap_max     = 6 frames
same_channel_gap_max = 8 frames
```

Relation schema:

```json
{
  "left_event_id": "000576:e003",
  "right_event_id": "000576:e004",
  "relation": "parallel",
  "overlap_ratio": 0.62,
  "gap": 0,
  "relative_order": "overlap"
}
```

This relation graph is the main fix for the single-sequence baseline.

### Step 5: Parallel Packet Construction

Create packets by grouping temporally overlapping cross-channel events. A
packet is not required to contain all channels; it only contains events that
are active in the same local time region.

Packet schema:

```json
{
  "packet_id": "000576:p004",
  "case_id": "000576",
  "packet_type": "parallel",
  "span": [42, 91],
  "members": [
    {
      "channel": "root_locomotion",
      "event_id": "000576:e001",
      "symbol": "root_locomotion/LOCO_FORWARD|dir=forward|dist=m_l|speed=speed_fast"
    },
    {
      "channel": "right_arm",
      "event_id": "000576:e003",
      "symbol": "right_arm/RA_REPEAT|dir=up_down|dur=l|amp=m_m|speed=speed_m|count=c4_6"
    }
  ],
  "member_channels": ["root_locomotion", "right_arm"],
  "packet_symbol": "PAR[root_locomotion:LOCO_FORWARD + right_arm:RA_REPEAT]",
  "relation_summary": {
    "parallel_edges": 1,
    "lead_lag_edges": 0
  }
}
```

Important rules:

- member ordering inside `PAR[...]` is canonical by channel name, not by frame
  start;
- this makes the packet invariant to small start-time jitter;
- the packet still stores exact event spans and numeric values;
- a packet with one member is allowed and represents a non-parallel event.

### Step 6: Multi-View Sequences

Construct three views for each case.

#### View A: Per-Channel Sequences

One sequence per channel:

```text
left_arm:      LA_HAND_HIGH -> LA_REPEAT -> LA_HAND_LOW
root:          LOCO_FORWARD -> LOCO_TURN_LEFT
vertical:      WB_VERT_UP -> WB_VERT_DOWN -> WB_VERT_UP
```

This view learns within-channel primitives such as arm swing cycles or
turn-and-stop phases.

#### View B: Packet Sequence

A sequence of packets ordered by packet span:

```text
PAR[root_forward + right_arm_repeat]
-> PAR[root_turn + torso_hunch]
-> PAR[terminal_still]
```

This view learns cross-channel compositions without destroying concurrency.

#### View C: Relation Triples

Explicit relation tokens:

```text
REL(left_arm:raise_spread, whole_body_vertical:up, parallel)
REL(root_turn, arm_raise, lead_lag)
REL(kick_forward, terminal_still, before_after)
```

This view is useful for diagnosing whether a motif is truly parallel,
sequential, or a weak adjacency artifact.

### Step 7: Two-Stage Motion-BPE

The active BPE path is intentionally simple and hierarchical:

```text
channel event sequences
-> Channel-BPE temporal motifs          <CHM_0001>
-> timeline projection and overlap
-> raw coactivation units               COACT[channel_a:<CHM_i> + channel_b:<CHM_j>]
-> structural coordination signatures   COORD_SIG[channel_a:geometry + channel_b:geometry]
-> Coordination-BPE motifs              <COM_0001>
```

This order encodes the design choice:

- first learn what a single body channel tends to do over time;
- only then ask which learned channel motifs repeatedly overlap across
  different channels;
- keep text labels out of the merge process.

Implementation note:

```text
v1 learns channel merges on lightweight symbol sequences, reconstructs
structured channel motif sequences, derives coactivation units from overlapping
channel motifs, converts them to geometry-level `COORD_SIG[...]` signatures,
then promotes high-support signatures into coordination motifs.
```

The distinction between `COACT[...]` and `COORD_SIG[...]` matters. `COACT`
preserves the exact member `<CHM_*>` ids for provenance. `COORD_SIG` is the
counting key used for promotion, based on channel and geometry clusters. This
prevents the same coordination pattern from fragmenting across many equivalent
channel-motif ids.

This avoids repeatedly copying event dictionaries during BPE learning. The
symbolic BPE stage is not a good GPU target because its hot path is JSON/dict
processing, span-overlap graph construction, Counter-based pair statistics, and
string/integer token replacement. GPU work is more appropriate for future dense
geometry feature extraction, embedding, CLIP/naming, or learned scoring. For
this audit, useful speedups should come from cached channel-event/packet
corpora, integer token ids, incremental pair statistics, and optional
multi-process relation construction.

#### Stage 1: Same-Channel Sequence Merge

```text
SEQ_CHANNEL_MERGE(channel, token_a, token_b) -> <CHM_k>
```

Example:

```text
left_arm/UP -> left_arm/DOWN
=> left_arm/WAVE_PHASE
```

Use for repeated or cyclic single-limb motion.

#### Stage 2a: Raw Coactivation Unit

```text
COACT[channel_a:<CHM_i> + channel_b:<CHM_j> + ...]
```

Example:

```text
bimanual:<CHM_arm_raise_lower_cycle> overlaps whole_body_vertical:<CHM_up_down_cycle>
=> COACT[bimanual:<CHM_i> + whole_body_vertical:<CHM_j>]
```

This is not yet a named pattern. It is a structural observation that two or
more channel motifs overlap in time.

#### Stage 2b: Coordination Merge

```text
COORDINATION_MERGE(COORD_SIG[...]) -> <COM_k>
```

Example:

```text
COORD_SIG[bimanual:BIMANUAL_PERIODIC/BI_RAISE_SPREAD + whole_body_vertical:WHOLE_BODY_VERTICAL/WB_VERT_DOWN&WHOLE_BODY_VERTICAL/WB_VERT_UP]
=> <COM_bilateral_vertical_coordination_candidate>
```

Use for jumping-jack-like, running-in-place-like, dance-like, or exercise-like
patterns when the overlap is frequent enough across cases.

#### Deferred Extension: Packet Sequence Merge

```text
SEQ_MERGE(packet_a, packet_b)
```

Example:

```text
PAR[low_body_hold] -> PAR[vertical_up]
=> low_to_stand_transition_candidate
```

This is not active in the current script. It is a later extension for
sit/stand, crouch/release, kick/recover, turn/stop, and similar multi-phase
motions after the channel/coordination hierarchy is stable.

### Step 8: Merge Scoring

Each candidate merge should be scored by more than raw frequency.

Recommended score components:

```text
support_cases
occurrence_count
compression_gain
channel_coverage
temporal_consistency
relation_consistency
numeric_consistency
caption_alias_purity_for_audit_only
legacy_family_purity_for_audit_only
```

A first scoring form:

```text
score =
  log(1 + support_cases)
  + 0.25 * log(1 + occurrence_count)
  + 0.50 * relation_consistency
  + 0.30 * numeric_consistency
  + 0.20 * channel_coverage
```

Language terms must not enter the structural merge score. Caption aliases are
reported only after the merge is learned.

### Step 9: Vocabulary Growth And Stopping

Do not assume `256` merges is the correct vocabulary size.

Run a sweep:

```text
num_merges:       256, 512, 1024, 2048
min_support:      40, 80, 120
channel ratio:    0.4, 0.5, 0.7
overlap min:      0.25, 0.30, 0.40
token detail:     geometry, geometry_speed, detailed
```

For each run, report:

```text
base token types
base token occurrences
packet diagnostic token types
packet diagnostic token occurrences
merge motif types
final vocabulary types
channel input token occurrences
channel output token occurrences
channel-BPE output ratio
case coverage
motif purity
average channels per motif
coordination motif ratio
```

Stop conditions should be evidence-based:

- compression gain flattens;
- new motifs fall below support;
- new motifs are mostly low-purity context;
- review burden becomes too high;
- downstream pattern binding no longer improves.

### Step 10: Motif Audit

Every motif should preserve both structure and examples.

Motif schema:

```json
{
  "motif_id": "<COM_0007>",
  "operator": "COORDINATION_MERGE",
  "parents": ["COACT[bimanual:<CHM_0012>+whole_body_vertical:<CHM_0009>]"],
  "support_cases": 184,
  "occurrences": 231,
  "channels": ["bimanual", "whole_body_vertical"],
  "relation_profile": {
    "parallel": 0.82,
    "lead_lag": 0.15,
    "sequential": 0.03
  },
  "numeric_profile": {
    "duration_bins": {"m": 124, "l": 60},
    "magnitude_bins": {"m_s": 103, "m_m": 81},
    "speed_bins": {"speed_m": 142, "speed_fast": 42}
  },
  "top_geometry_clusters": [
    {"id": "BIMANUAL_PERIODIC/BI_RAISE_SPREAD", "count": 184},
    {"id": "WHOLE_BODY_VERTICAL/WB_VERT_UP", "count": 184}
  ],
  "example_occurrences": [
    {
      "case_id": "003082",
      "span": [12, 68],
      "member_event_ids": ["003082:e003", "003082:e004"],
      "caption_reference": "a man jumps up and down ...",
      "caption_policy": "reference only"
    }
  ],
  "naming_diagnostics": {
    "top_caption_alias": "jumping_jack",
    "caption_alias_purity": 0.71
  },
  "top_base_symbols": [
    {"id": "bimanual/BIMANUAL_PERIODIC/BI_RAISE_SPREAD|...", "count": 184}
  ]
}
```

### Step 11: Motif Families

Group motifs into families by motion structure, not by names.

Grouping signals:

```text
required channel set
required geometry cluster set
operator type
relation profile
numeric profile
span pattern
repeat structure
source motif overlap
case overlap
```

Example:

```text
family: bilateral_vertical_coordination
children:
  PAR[bimanual_raise_spread + vertical_up]
  REP[PAR[bimanual_raise_spread + vertical_cycle]]
  SEQ[arms_high -> vertical_down]
labels:
  jumping_jack, exercise, cheer gesture
```

The family can later be named `jumping_jack` only if the motion realization and
language evidence are sufficiently aligned.

### Step 12: Pattern Forest Candidate

The pattern forest should use motif families as candidate nodes.

Forest node fields:

```json
{
  "node_id": "motion_family_0008",
  "node_kind": "motif_family",
  "status": "candidate_family",
  "motion_definition": {
    "channels": ["bimanual", "whole_body_vertical"],
    "required_geometry_clusters": [
      "BIMANUAL_PERIODIC/BI_RAISE_SPREAD",
      "WHOLE_BODY_VERTICAL/WB_VERT_UP"
    ],
    "relation": "parallel_or_repeated_parallel",
    "numeric_handles": ["duration", "vertical_magnitude", "arm_spread_amplitude", "cycle_count"]
  },
  "source_motifs": ["<COM_0007>", "<CHM_0042>"],
  "metrics": {
    "support_cases": 184,
    "case_coverage": 0.0064,
    "parallel_relation_purity": 0.82,
    "motion_purity": 0.77
  },
  "naming_diagnostics": {
    "top_caption_aliases": ["jumping_jack"],
    "policy": "diagnostic only"
  }
}
```

## How This Handles Concurrent Motion

Example:

```text
upper body: both arms raise
lower body/root: running forward
```

Single-sequence BPE view:

```text
BI_RAISE -> LOCO_FORWARD
```

This makes parallel motion look like ordered adjacency.

Multi-channel BPE view:

```text
bimanual channel motif:
  <CHM_arm_raise_lower>
root_locomotion channel motif:
  <CHM_forward_stride>
coactivation:
  COACT[bimanual:<CHM_arm_raise_lower> + root_locomotion:<CHM_forward_stride>]
```

The current script can promote repeated coactivation into a coordination motif:

```text
COACT[...] -> <COM_k>
```

This lets the system separate:

- within-channel temporal words;
- cross-channel coordination words.

## Report-Ready Extraction Summary

When writing the method section, describe the pipeline as:

```text
We first convert each motion sequence into dense body-channel observables.
These observables are segmented into channel-specific motion events with
span, direction, magnitude, speed, and count. Instead of flattening concurrent
events into a single sequence, we build an overlap graph and group temporally
co-active channel events into parallel packets. Motion-BPE then learns
same-channel temporal motifs first. It then projects these motifs back onto
the timeline and promotes high-support cross-channel coactivations into
coordination motifs. The learned motifs induce a motion pattern forest;
HumanML3D text and WordNet are used only after induction to name and audit the
nodes.
```

## Artifact Plan

The next implementation should create a new audit directory rather than
overwrite the single-sequence audit.

Proposed script:

```text
scripts/audit_hml3d_multichannel_motion_bpe.py
```

Proposed output:

```text
outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_v1/
```

Files:

```text
multi_channel_event_corpus.jsonl
channel_event_vocab.json
overlap_packet_corpus.jsonl
packet_vocab.json
multichannel_motion_bpe_vocab.json
case_multichannel_bpe_sequences.jsonl
motif_audit.json
motif_family_candidates.json
motion_pattern_forest_candidates.json
coordination_review.md
summary.json
audit_report.md
review_pack/
```

## Current Script Tuning Guide

The active script is:

```text
scripts/audit_hml3d_multichannel_motion_bpe.py
```

The coordination promotion review script is:

```text
scripts/promote_coordination_motif_candidates.py
```

The coordination forest review script is:

```text
scripts/build_coordination_pattern_forest.py
```

The text-pseudo-GT pattern audit script is:

```text
scripts/audit_motion_pattern_pseudo_gt.py
```

The recall-candidate diagnostic script is:

```text
scripts/audit_motion_pattern_recall_candidates.py
```

The generic family-proposal builder is:

```text
scripts/build_motion_pattern_family_proposals.py
```

Main callable functions:

```text
run_multichannel_motion_bpe(args)
  returns records, merges, sequences, motif rows, family payload, forest payload,
  and summary; it does not write output files.

write_multichannel_motion_bpe_outputs(output_dir, result, args)
  writes JSON/JSONL/Markdown outputs from a result dict.
```

The intended debug loop is:

```bash
python scripts/audit_hml3d_multichannel_motion_bpe.py --self-test

python scripts/audit_hml3d_multichannel_motion_bpe.py \
  --max-records 200 \
  --num-merges 32 --min-pair-count 4 --min-pair-support 3 \
  --output-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug \
  --cache-dir outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_debug_cache
```

For tuning, inspect these first:

```text
summary.json
audit_report.md
motif_audit.json
motif_family_candidates.json
motion_pattern_forest_candidates.json
coordination_review.md
```

The main knobs are:

```text
--num-merges          maximum learned motif count
--channel-merge-ratio fraction of budget reserved for channel motifs before coordination
--min-pair-count     total pair frequency threshold
--min-pair-support   distinct-case support threshold
--parallel-overlap-min
--lead-lag-gap-max
```

The cache stores the expensive intermediate representation:

```text
Layer3 events -> channel events -> relations -> packets -> views
```

Changing only BPE thresholds should reuse the cache. Check:

```text
summary.json -> record_cache.status
```

Expected values:

```text
hit         reused cached channel/packet records
miss_built  built cache because none matched
rebuilt     forced rebuild with --rebuild-cache
disabled    no --cache-dir was provided
```

Motion-BPE remains motion-only. Captions and caption aliases are preserved for
diagnostic naming and human review, but text keywords are not used to create
tokens, select merges, or group motif families.

## Coordination Promotion Queue

`promote_coordination_motif_candidates.py` converts learned
`COORDINATION_MERGE` motifs into an offline review queue. It does not modify
the AML runtime tree.

Inputs:

```text
motif_audit.json
```

Outputs:

```text
coordination_pattern_promotion_candidates.json
coordination_pattern_promotion_review.md
summary.json
```

Default policy:

```text
promote_named_coordination_candidate:
  support >= 30
  caption_alias_purity >= 0.70
  at least 2 channels
  at least 2 geometry clusters

review_structural_coordination_candidate:
  support >= 120
  at least 2 channels

diagnostic_coordination_motif:
  everything else
```

This policy uses caption aliases only as naming diagnostics. The structural
candidate still comes from channels, geometry clusters, relation profile, and
the parent `COORD_SIG[...]` signature.

## Coordination Pattern Forest Review

`build_coordination_pattern_forest.py` groups the promotion queue into a small
offline forest for human inspection. It is still a review artifact, not the
runtime AML tree.

Inputs:

```text
coordination_pattern_promotion_candidates.json
```

Outputs:

```text
coordination_pattern_forest.json
coordination_pattern_forest_tree.txt
coordination_pattern_forest_review.md
summary.json
```

The forest has two node levels:

```text
named_coordination_family / structural_coordination_family
  -> coordination_motif_leaf
```

The grouping policy is deliberately simple:

- named promote candidates group by their top diagnostic caption alias;
- unnamed structural candidates group by required channels plus required
  geometry clusters;
- each leaf keeps the source `<COM_*>` motif, support, structural definition,
  caption diagnostics, and example captions.

Important summary fields:

```text
family_count
leaf_count
node_count
edge_count
status_counts
family_status_counts
leaf_status_counts
```

`status_counts` counts both family and leaf nodes, so use
`family_status_counts` or `leaf_status_counts` when checking how many actual
motif leaves are ready to promote.

## Text Pseudo-GT Audit Points

Some compact action names in HumanML3D can be used as pseudo-GT audit points
for a learned motif, without feeding text into Motion-BPE. `jumping_jack` is
the first such audit point.

Target aliases and regex definitions live in:

```text
configs/motion_pattern_text_targets.json
```

Audit scripts read that registry at runtime. Adding `sit_down`, `karate`,
`ballet`, or `tennis` should be a registry/data change first, not a Python
motion-rule change.

`audit_motion_pattern_pseudo_gt.py` compares:

```text
HumanML3D text pseudo-GT positives
  caption_alias_ids contains target alias
  OR target regex matches caption_texts

learned motif predictions
  case contains selected `<COM_*>` motif in case_multichannel_bpe_sequences
```

The main metric for human review is:

```text
precision_subset_accuracy = true_positive / predicted_case_count
```

This answers: among the subset recognized by the learned motif, how many are
correct according to HumanML3D text pseudo-GT? The audit also reports recall,
false positives, and false negatives so we can see whether the motif is too
narrow.

Example command:

```bash
python scripts/audit_motion_pattern_pseudo_gt.py \
  --target-alias jumping_jack \
  --source-corpus outputs/aml_regression_testset_v2/hml3d_layer3_event_bpe_full_v1/layer3_event_bpe_corpus.jsonl \
  --bpe-sequences outputs/aml_regression_testset_v2/hml3d_multichannel_motion_bpe_coord_sig_full_loose_v1/case_multichannel_bpe_sequences.jsonl \
  --candidates outputs/aml_regression_testset_v2/coordination_pattern_promotion_candidates_loose_v1/coordination_pattern_promotion_candidates.json \
  --output-dir outputs/aml_regression_testset_v2/jumping_jack_pseudo_gt_audit_loose_v1
```

After a seed motif is audited, use
`audit_motion_pattern_recall_candidates.py` to inspect the false negatives. It
finds coordination symbols that occur in missed pseudo-GT cases, then reports:

```text
candidate_precision
union_precision_with_seed
union_recall_with_seed
greedy_precision_preserving_expansion
```

This gives a controlled way to expand a named pattern. For example, current
`jumping_jack` starts from `<COM_0036>`:

```text
seed precision: 0.804878
seed recall:    0.089674
```

With a precision floor of `0.80`, the greedy expansion reaches:

```text
expanded precision: 0.800000
expanded recall:    0.315217
```

With a stricter precision floor of `0.85`, it reaches:

```text
expanded precision: 0.864865
expanded recall:    0.173913
```

The top missed `jumping_jack` variants are not random. They concentrate around
vertical up/down plus arm-high posture signatures, sometimes with
`BIMANUAL_PERIODIC/BI_RAISE_SPREAD`. That means the next improvement should be
a promotion/naming policy for a family of related coordination signatures, not
a one-off case rule.

`build_motion_pattern_family_proposals.py` turns this audit pair into a generic
review-only family proposal:

```text
pseudo-GT audit
recall-candidate diagnostic
optional promotion candidates
-> pattern_family_proposal.json
-> pattern_family_proposal.md
```

The builder is target-agnostic. A target alias such as `jumping_jack` is only an
input label and pseudo-GT definition; the builder does not contain
action-specific motion rules. Variants are assigned generic statuses:

```text
seed_promoted_motif
promote_family_variant_candidate
review_family_variant_candidate
diagnostic_family_variant_candidate
reject_noisy_variant_candidate
```

The status policy is based on candidate precision, incremental true positives,
incremental false positives, and whether the variant is selected by the
precision-preserving expansion.

Multichannel BPE minimum summary fields:

```json
{
  "num_records": 29228,
  "channel_event_count": 0,
  "channel_event_type_count": 0,
  "packet_count": 0,
  "packet_type_count": 0,
  "single_member_packet_count": 0,
  "parallel_packet_count": 0,
  "learned_motif_count": 0,
  "channel_input_token_count": 0,
  "channel_output_token_count": 0,
  "coordination_output_token_count": 0,
  "final_token_count": 0,
  "final_vocab_size": 0,
  "channel_bpe_output_ratio": 0.0,
  "coordination_motif_ratio": 0.0,
  "case_coverage": 0.0
}
```

## Validation Checklist

Before promoting the new representation:

- confirm base channel event counts against the current `393032` Layer3 event
  baseline;
- report how many events become parallel packets;
- check that jumping-jack-like examples produce bimanual + vertical parallel
  packets;
- check that running-with-arm-motion examples produce root + arm parallel
  packets rather than accidental sequence-only motifs;
- check that rotation uses angular speed bins;
- compare 256/512/1024/2048 merge sweeps;
- compare motif purity and coverage against the old single-sequence BPE;
- generate static review packs with GT keyframes and HML3D captions for top
  motif families.

## Non-Goals

This design does not:

- use captions to create motion events;
- make WordNet the tree topology;
- directly replace runtime AML files;
- train an autoregressive motion generator;
- treat the number of BPE merges as the final vocabulary size;
- assume one motion phrase has one action name.
