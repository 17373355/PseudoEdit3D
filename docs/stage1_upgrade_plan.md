# Stage 1 Upgrade Plan

## 1. Current diagnosis

Current pipeline is **not** yet learning a true multi-atomic realization task.

What it currently does:
- input:
  - first 20 frames prefix
  - one `EditProgram`
- output:
  - full 60-frame completion
- actual supervision:
  - one clip yields only **one dominant atomic sub-action**
  - selected by the largest proxy-attribute change after the prefix

So the current system is best described as:
- `prefix-conditioned single-atomic realization`

It is **not yet**:
- `prefix-conditioned multi-atomic sequence realization`

## 2. Long-term target

Target task:
- input:
  - prefix motion context
  - an ordered sequence of atomic edits
- output:
  - a full motion that realizes these atomic edits in order

What the model should learn:
- each atomic edit has a distinct body part, attribute, direction, amplitude, and time window
- multiple atomic edits can coexist in one clip
- predicted motion should both:
  - realize each atomic edit locally
  - transition between edits coherently

## 3. Why the current version is insufficient

Main limitations:
- data construction only keeps one dominant atomic event
- `EditProgram` is single-event, not sequence-valued
- model condition is still too close to one global instruction
- evaluation only checks whether one target tendency appears
- visualization still centers around a single program summary

## 4. Upgrade stages

### Stage A. Formal task split

Goal:
- make the distinction between single-atomic and multi-atomic explicit in code and docs

Deliverables:
- rename current task in docs as `single-atomic realization`
- add new task name `multi-atomic sequence realization`
- keep current pipeline runnable while building the upgraded branch

Acceptance:
- docs and configs no longer conflate the two task levels

### Stage B. Multi-atomic data extraction

Goal:
- extract multiple ordered atomic segments from one clip instead of only the best one

Deliverables:
- a clip-level extractor that returns a list of atomic segments
- each segment should include:
  - `part`
  - `attribute`
  - `direction`
  - `delta_value_deg`
  - `start_frame`
  - `end_frame`
  - optional confidence / score
- segment list should be ordered by time
- support pruning and merging rules to avoid noisy duplicates

Acceptance:
- one held-out clip can produce 2+ atomic events when appropriate
- extracted segments are visibly time-ordered and non-overlapping enough to interpret

### Stage C. Sequence EditProgram representation

Goal:
- replace the current single-event program assumption with a sequence program

Deliverables:
- a `MultiEditProgram` or equivalent structured list representation
- serialization to JSON for visualization and debugging
- a model-side tensorized sequence condition

Acceptance:
- one sample can carry multiple atomic edits without flattening them into a single dominant edit

### Stage D. Sequence-conditioned model

Goal:
- let the model consume multiple atomic edits and align them with time windows

Deliverables:
- sequence-conditioned decoder or planner-decoder path
- frame-wise alignment between future frames and atomic edit windows
- support for:
  - per-edit timing
  - per-edit magnitude
  - per-edit part activation

Acceptance:
- ablations show the model reacts differently when edit order or timing changes

### Stage E. Completion pressure and transition quality

Goal:
- ensure each edit is actually completed and transitions are not frozen or over-smoothed

Deliverables:
- per-edit progress loss
- per-edit completion loss
- transition continuity regularization between adjacent edits
- optional outside-span suppression so the model does not move too early everywhere

Acceptance:
- active edit windows no longer behave like slow-motion GT
- later edits do not erase earlier completed edits immediately

### Stage F. Visualization and evaluation

Goal:
- make outputs interpretable for multi-atomic clips

Deliverables:
- visualization panel with structured `key: value` program display
- full sequence program display, not just one prompt sentence
- per-edit success diagnostics:
  - realized / not realized
  - timing lag
  - amplitude ratio
- automatic `train + eval + held-out vis` bundle for every round

Acceptance:
- one GIF makes it obvious which edits were requested and which were realized or missed

## 5. Execution order

Strict order:
1. docs and task split
2. multi-atomic extraction
3. sequence program representation
4. visualization support for sequence programs
5. sequence-conditioned model
6. per-edit completion pressure
7. GPU experiments and iteration

## 6. Short-term milestone before full long-term completion

Before the full sequence model is done, we should reach an intermediate milestone:
- a clip can expose multiple atomic edit targets
- visualization can display the full extracted program sequence
- a baseline can at least realize the first 1-2 edits more explicitly than the current single-dominant-atomic system

## 7. Stop rule

Do not stop at a partial conceptual rewrite.
Only stop once:
- multi-atomic extraction exists
- sequence-conditioned training exists
- held-out visualizations show multiple requested edits clearly
- the model no longer behaves like a single-atomic slow-motion local mover
