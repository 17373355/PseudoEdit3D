# Motion Subword Design

## Motivation

The current auto-prompt pipeline has already shown that motion-derived semantics can be more faithful than raw HumanML3D captions. However, patching one high-level pattern at a time does not scale.

The next step is to move from:

- pattern-by-pattern heuristic repair

to:

- a layered motion-semantic representation
- where small time-aligned motion units are discovered first
- and higher-level atomic programs are composed from them

This mirrors the role of subword units in language modeling.

## Analogy to Text Subword Modeling

### Text pipeline

```text
characters -> subwords -> words / phrases -> sentences
```

### Motion pipeline

```text
frame observables -> micro-events -> sub-motion units -> atomic programs
```

In our setting, the analogue of subwords is time-aligned sub-motion units built from smaller motion events.

## Why Not Directly Reuse Human Captions

HumanML3D captions are:

- noisy
- inconsistent across annotators
- uneven in granularity
- often too coarse for structured control

So the motion representation must not be built by directly imitating raw text. Instead:

- motion should define the primary structure
- text should act as a noisy prior and later a probe

## Layered Design

## Layer 0: Frame Observables

This is the lowest level and the next immediate implementation target.

A Layer 0 observable is a time series extracted directly from motion, before any event segmentation.

Examples:

- root heading
- root heading velocity
- root height
- root xz velocity
- pelvis-to-ankle compression
- torso relative height drop
- torso forward extent
- left / right arm raise
- left / right elbow lift
- left / right wrist-to-chest distance
- contact/support proxies

Properties:

- dense over time
- motion-native
- no language assumptions
- usable for both rule-based and learned event induction

## Layer 1: Micro-Events

Micro-events are short, time-aligned primitive transitions extracted from Layer 0 signals.

Examples:

- `ROOT_UP +0.05m @ 20-24`
- `ROOT_DOWN -0.04m @ 25-28`
- `TURN_LEFT +18deg @ 30-35`
- `LEFT_ELBOW_UP +12deg @ 40-43`
- `TORSO_BEND_FWD +0.10 @ 50-58`
- `STOP_LOW_VEL @ 90-102`

Properties:

- short temporal span
- explicit direction / magnitude / duration
- closest motion analogue to text-level subword atoms

## Layer 2: Sub-Motion Units

These are the learned or merged units built from repeated micro-event patterns.

Examples:

- `[LEFT_ELBOW_UP, LEFT_ELBOW_DOWN] -> LEFT_ELBOW_FLAP`
- `[ROOT_UP, ROOT_DOWN] -> HOP`
- `[HOP, HOP, HOP] -> HOP_REPEATED`
- `[WALK_FWD, TURN_180, WALK_BWD] -> TURN_AROUND_AND_WALK_BACK`

This is the level where BPE-like merging or unigram-style segmentation becomes useful.

## Layer 3: Atomic Programs

This is the structured semantic layer exposed to later models.

Each event should ideally contain:

- `type`
- `part`
- `direction`
- `magnitude`
- `unit`
- `count`
- `start_frame`
- `end_frame`
- `confidence`

This is the true target representation.

`auto_prompt` is only a rendered probe or text view of this layer.

## BPE Mapping for Motion

The direct analogue of BPE is not raw motion frames.
It is:

```text
micro-event sequence
-> frequency statistics over adjacent event pairs
-> iterative merge of common local subsequences
-> sub-motion vocabulary
```

Examples:

- `LEFT_ARM_UP + LEFT_ARM_DOWN -> LEFT_ARM_WAVE`
- `ROOT_UP + ROOT_DOWN -> HOP`
- `HOP + HOP -> HOP_TWICE`
- `WALK_FWD + TURN_RIGHT + WALK_BWD -> TURN_AROUND_AND_WALK_BACK`

## Why BPE Alone Is Not Enough

Classic BPE is a strong baseline, but motion has richer ambiguity than text.
A better long-term route is likely:

- BPE as the first baseline
- unigram LM / SentencePiece-style segmentation as the stronger mainline
- optional segmentation regularization for robustness

## Immediate Practical Rule

Do not directly design every high-level pattern from scratch anymore.

Instead:

1. expand and stabilize Layer 0 observables
2. induce micro-events from Layer 0
3. use corpus-level frequency / consistency to build sub-motion units
4. derive atomic programs from sub-motion units
5. render `auto_prompt` only at the end

## What Moves Into Legacy

The current hand-crafted high-level rules are still useful, but should gradually move into a bootstrap / validator role.

That means:

- existing explicit pattern rules remain useful for bootstrapping
- but they should not remain the main long-term representation mechanism

## Immediate Development Plan

### Step 1
Implement a stable Layer 0 observable interface.

### Step 2
Export event-ready observable streams on the current 600-case benchmark.

### Step 3
Define micro-event segmentation rules on top of Layer 0.

### Step 4
Prototype motion-BPE over micro-event sequences.

### Step 5
Compare BPE-style merging against a unigram-style segmentation alternative.

## Near-Term Evaluation

At the Layer 0 / Layer 1 stage, good evaluation targets include:

- event detection precision/recall on representative cases
- reduction of hard-bad categories on the 600-case benchmark
- improved separation of confusing categories:
  - crouch vs stairs
  - hop vs squat
  - bend vs bounce
  - raise vs elbow flap

## Long-Term Payoff

This design is the bridge from:

- heuristic auto-prompt repair

to:

- a real structured motion semantic representation
- and eventually a structured conditioning model

So the next implementation focus should be Layer 0 first, not more ad hoc high-level prompt templates.
