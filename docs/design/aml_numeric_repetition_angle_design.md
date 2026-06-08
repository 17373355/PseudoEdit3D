# AML Numeric, Repetition, and Angle Design

This note translates lessons from LLM/VLM tokenization, numeracy, counting, and spatial reasoning into concrete AML design rules.

## Sources Reviewed

Primary sources:

- Sennrich et al., 2016, `Neural Machine Translation of Rare Words with Subword Units`  
  https://arxiv.org/abs/1508.07909
- Kudo and Richardson, 2018, `SentencePiece`  
  https://arxiv.org/abs/1808.06226
- Thawani et al., 2021, `Representing Numbers in NLP: a Survey and a Vision`  
  https://arxiv.org/abs/2103.13136
- Golkar et al., 2023/2024, `xVal: A Continuous Numerical Tokenization for Scientific Language Models`  
  https://arxiv.org/abs/2310.02989
- Levy and Geva, 2024/2025, `Language Models Encode Numbers Using Digit Representations in Base 10`  
  https://arxiv.org/abs/2410.11781
- Jia et al., 2025/2026, `OmniSpatial`  
  https://arxiv.org/abs/2506.03135
- Zhang et al., 2025/2026, `SpinBench`  
  https://arxiv.org/abs/2509.25390

## Key Lessons for AML

### 1. Subword Units Are Useful, But Numeric Values Need Separate Treatment

BPE-style subword units solve open-vocabulary composition by merging frequent local pieces. SentencePiece further shows that subword modeling can be made language-independent and trained from raw sequences.

AML implication:

- use BPE / unigram-style merging for symbolic motion event sequences
- do not let numeric values disappear inside merged symbols
- every merged unit must retain numeric residue

Bad design:

```text
TURN_FULL_LEFT
```

Better design:

```text
event_type = WHOLE_BODY_ROTATION
cluster = WB_ROT_LEFT_FULL
signed_delta = +383.17
unit = deg
source_span = [8, 28]
supporting_units = [...]
```

The symbolic token provides language alignment; the continuous value supports control.

### 2. LLM Numeracy Suggests Dual Representation: Symbolic Form + Value Residue

Number-representation work argues that numbers differ from ordinary words and require special treatment. xVal shows the value of continuous number encodings for numerically dense data. The digit-representation work shows that LLMs often internally represent numbers digit-wise rather than as continuous scalar values.

AML implication:

Represent every quantity with both:

- symbolic bucket for language/model alignment
- continuous numeric residue for exact control

Recommended fields:

```text
value_bin: small | medium | large | full | multi
value: float
signed_value: float
unit: deg | m | frames | count | m/frame
value_source: measured | inferred | caption_prior
confidence: float
```

For angles:

```text
angle_bin: small | quarter | half | three_quarter | full | multi
signed_delta_deg: float
rotation_direction: left | right
reference_frame: body_heading | world_xz | viewer | object_relative
```

For distances:

```text
path_length_m: float
net_displacement_m: float
mean_speed: float
trajectory_direction: forward | backward | left | right | mixed | unknown
```

### 3. Counting Should Be Evidence-Based, Not Text-Based

VLM spatial benchmarks repeatedly show that counting and spatial reasoning are fragile, especially when the model must infer relations or count repeated/occluded structures. For motion, we should not rely on captions saying `three steps` or `twice`; the count must be derived from detected phase spans.

AML implication:

A repeated event should be represented as a group with evidence, not just a scalar count.

Recommended structure:

```text
super_family = LEFT_ARM_PERIODIC
cluster_id = LA_REPEAT
role = repeated_phase
count = 3
count_confidence = 0.84
period_frames = [12, 11, 13]
phase_spans = [[20, 31], [32, 43], [44, 57]]
phase_template = arm_cycle
alternation = false
supporting_units = [...]
```

This makes `repeat twice`, `three steps`, `jump three times`, and `wave repeatedly` auditable.

### 4. Repetition Needs a Hierarchical Program, Not Flat Text

LLM tokenization compresses frequent local sequences, but motion repetition has explicit temporal structure. A repeated motion should be a nested program node:

```text
Repeat(
  count = 3,
  unit = StepForward,
  spans = [...],
  aggregate = {path_length: 1.8m}
)
```

AML implication:

Keep both views:

- flat events for visualization and simple conditioning
- grouped repeated_phase nodes for counts and control

Current Layer 2.5 already moves in this direction. The next step is to store phase spans and period statistics more explicitly.

### 5. Angle and Spatial Reasoning Require Explicit Reference Frames

VLM spatial reasoning papers show weaknesses in rotation, perspective-taking, and multi-step spatial logic. This maps directly to AML: an angle is meaningless unless the reference frame is specified.

AML implication:

Every angle or direction should record:

```text
reference_frame: body_heading | world_xz | viewer | target_object | support_surface
axis: yaw | pitch | roll | x | y | z
sign_convention: positive_left | positive_ccw | dataset_defined
unwrap_policy: continuous_yaw | shortest_arc | phase_local
```

Current rotation events should therefore be treated as:

```text
whole_body.rotation
axis = yaw
reference_frame = body_heading/world_xz estimate
signed_delta_deg = +383.17
angle_bin = full
```

### 6. Template Language Should Be Deterministic Before Free Naturalization

Because LLMs/VLMs can hallucinate or normalize away numeric detail, AML should first render to a controlled template language.

Current template example:

```text
frames 8-28 | whole_body.rotation | turn left about a full turn | angle=383deg, signed_delta=383deg | context=body_driver
```

Design rule:

- AML-Lang is for debugging, training targets, and round-trip evaluation
- free natural language prompt is a secondary rendering
- never use same-case HumanML3D captions to fill missing AML fields

## Proposed AML Field Schema V2

### Common Fields

```text
part
super_family
cluster_id
role
start_frame
end_frame
confidence
source
source_span
supporting_units
motion_signature
metadata
```

### Numeric Fields

```text
magnitude
signed_delta
unit
value_bin
value_confidence
```

### Quantity Fields

```text
count
count_confidence
phase_spans
period_frames
alternation
repeat_mode
```

### Geometry Fields

```text
axis
reference_frame
sign_convention
unwrap_policy
angle_bin
trajectory_direction
path_length_m
net_displacement_m
mean_speed
```

### Context Fields

```text
context_mode
coupled_with_locomotion
support_mode
contact_mode
body_driver_overlap
```

## Immediate Migration Tasks

### Task A: Explicit Repetition Evidence

Add to repeated_phase events:

- `phase_spans`
- `period_frames`
- `count_confidence`
- `repeat_unit_name`

Why:

- needed for `twice`, `three times`, `three steps`
- enables round-trip count evaluation

### Task B: Rotation Geometry Metadata

Add to rotation events:

- `axis = yaw`
- `reference_frame`
- `sign_convention`
- `unwrap_policy`
- `angle_bin`

Why:

- prevents ambiguous angle claims
- supports edits like `turn 30 degrees more`

### Task C: Locomotion Direction Split

Split broad `LOCO_ACTIVE_*` into trajectory-aware states:

- `LOCO_FORWARD`
- `LOCO_BACKWARD`
- `LOCO_LEFT`
- `LOCO_RIGHT`
- `LOCO_MIXED`
- `LOCO_TURNING`
- `LOCO_STATIONARY`

Why:

- current coverage is broad but semantically coarse
- HumanML3D often distinguishes forward/backward/sideways walking

### Task D: Template Language Expansion

Current AML-Lang is event-level. Add grouped rendering:

```text
frames 20-57 | left_arm.periodic | repeat left arm cycle | count=3 | periods=12,11,13 frames | context=isolated
```

### Task E: Evaluation Protocol

Add targeted checks:

- count accuracy on repeated motion cases
- angle-bin accuracy for turn/spin cases
- signed-angle error in degrees
- locomotion-direction accuracy
- round-trip consistency: motion -> AML -> template -> parse -> AML fields

## Research Position

The main claim should not be that AML copies text tokenization literally.

Better claim:

> AML borrows the compositional discipline of LLM tokenization, but separates symbolic motion units from continuous numeric residues because motion contains measurable geometry that text tokens alone cannot preserve.

Chinese version:

> AML 借鉴 LLM tokenization 的组合式结构，但不会把数值运动信息完全离散化成符号；它同时保留语义 token 和连续数值 residue，因为 motion 的角度、距离、速度、次数是可测量的几何量。
