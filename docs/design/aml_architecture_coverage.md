# AML Architecture Coverage

This document tracks whether the Atomic Motion Language has enough structural coverage before we refine quantity, repetition, and angle reasoning.

## Current Layer Stack

```text
Layer 0: frame observables
  dense motion-native signals

Layer 1: micro-events and state-events
  local changes plus sustained active states

Layer 2: sub-motion units
  local event compositions with retained numeric metadata

Layer 2.5: phase patterns
  repeated or alternating sub-motion sequences

Layer 3: atomic program
  family-first structured motion events

AML-Lang: template language
  deterministic human-readable rendering for inspection
```

## Coverage Categories

### 1. Change Events

Covered by Layer 1 micro-events and Layer 3 event families.

Examples:

- vertical root/body changes: `WB_VERT_UP`, `WB_VERT_DOWN`, `WB_VERT_CYCLE`
- rotation changes: `WB_ROT_LEFT_FULL`, `WB_ROT_RIGHT_HALF`, etc.
- arm changes: `LA_REPEAT`, `RA_REPEAT`, `BI_UP`, `BI_OUT`
- torso changes: `TORSO_BEND_RECOVER`, `TORSO_OSC_FB`

Current evidence:

- `000183` spin is recovered as `WHOLE_BODY_ROTATION/WB_ROT_LEFT_FULL` at frames `8-28`
- exact signed yaw delta is retained: about `383deg`

### 2. Sustained State Events

Initial coverage added for whole-body locomotion.

Examples:

- `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_SLOW`
- `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_MEDIUM`
- `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_FAST`

Current evidence:

- `M004289` now has `WHOLE_BODY_LOCOMOTION/LOCO_ACTIVE_MEDIUM` at frames `18-138`
- this fixes the visualization gap where long stable walking intervals had no active event

### 3. Repetition / Phase Events

Covered by Layer 2.5 phase patterns.

Examples:

- repeated arm cycles
- alternating arm cycles
- repeated vertical hop-like cycles
- torso forward-back oscillation

Current evidence:

- arm repeat clusters are split into isolated and locomotion-coupled variants
- vertical repeated phases are split by alternation when corpus support justifies it

### 4. Numeric Residue

Every event should preserve numeric values instead of only producing a semantic name.

Currently retained fields:

- `start_frame`
- `end_frame`
- `count`
- `magnitude`
- `signed_delta`
- `unit`
- `path_length`
- `mean_speed`
- `active_ratio`
- `supporting_units`

Examples:

- `turn left about a full turn` keeps `signed_delta=383deg`
- `move through space` keeps `path_length=4.05m` and `mean_speed`

### 5. Template Language

Covered by `pseudoedit3d/edit/aml_language.py`.

Example:

```text
frames 8-28 | whole_body.rotation | turn left about a full turn | angle=383deg, signed_delta=383deg | context=body_driver
```

Purpose:

- inspect AML outputs deterministically
- avoid premature free-form naturalization
- keep family, cluster, numeric values, context, and source traceable

## 1000-Case Coverage Snapshot

Latest report:

- `outputs/aml_family_taxonomy_hml3d1000_locomotion_state_v1.json`

Summary:

- `num_cases = 1000`
- `total_layer3_events = 13329`
- `avg_layer3_count = 13.329`
- zero-event cases: `0`
- cases with `layer3_count <= 2`: `2`

Super-family support:

- `WHOLE_BODY_VERTICAL`: 972 cases
- `WHOLE_BODY_LOCOMOTION`: 955 cases
- `WHOLE_BODY_ROTATION`: 814 cases
- `LEFT_ARM_PERIODIC`: 685 cases
- `RIGHT_ARM_PERIODIC`: 675 cases
- `TORSO_PERIODIC`: 501 cases
- `BIMANUAL_PERIODIC`: 433 cases

Interpretation:

- coverage is now broad enough to support architecture-level iteration
- locomotion and rotation are intentionally broad first-pass families
- next work should refine these broad families, not return to one-off caption pattern rules

## Known Gaps

### Locomotion Taxonomy

Current locomotion state is too coarse.

Needed refinements:

- forward / backward / sideways trajectory
- walk / run / step / shuffle
- turn-in-place versus moving while turning
- stop / pause / stationary hold
- path length and step count

### Contact and Support

Current support handling is weak.

Needed observables:

- foot contact on/off
- hand contact or support proxy
- wall / rail support proxy if possible
- stance/swing phases

### Fine Arm and Hand Motion

Current arm families are still coarse.

Needed refinements:

- circling hands
- clapping / tapping / knocking
- hand-to-object support
- elbow-only versus whole-arm motion

### Quantity, Repetition, and Angle Semantics

Need to study LLM/VLM handling of:

- numbers as symbols versus quantities
- counting repeated events
- compositional arithmetic over values
- angle/direction discretization and continuous residuals
- exact numeric control instructions such as `turn 30 degrees more`

Target migration into AML:

- keep coarse symbolic bins for language alignment
- keep continuous numeric residue for control
- expose both in AML-Lang and future conditioning models
