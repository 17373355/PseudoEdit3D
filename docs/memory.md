# Memory

This file is the fastest way to re-enter the project.

It is intentionally short. For details, follow the linked docs instead of duplicating them here.

## Current one-line task

Current main task:

- **prefix-conditioned action completion**

Interpretation:

- prefix gives motion context
- `EditProgram` gives structured action intent
- model completes the full clip under that condition

## Current research target

- **Language-Guided Action Regulation**

Important:

- repo name stays `PseudoEdit3D`
- the research framing is no longer just third-person motion editing

See:

- `docs/design_brief.md`

## Read order

If you are returning to this repo:

1. `docs/memory.md`
2. `docs/design_brief.md`
3. `docs/planning.md`
4. `docs/experiment_log.md`
5. `Test.md`

## What each doc means

- `docs/design_brief.md`
  - stable concepts and task framing

- `docs/planning.md`
  - next experiments and open decisions only

- `docs/experiment_log.md`
  - completed runs, configs, losses, output dirs

- `Test.md`
  - concrete commands and local validation notes

## Current important setting

Active setting:

- input:
  - first 20 frames prefix
  - `EditProgram`
- target:
  - full 60-frame completion

Important implementation assumption:

- prefix should be treated as motion context, not as something the model is free to rewrite

## Current main problems

- predicted future often collapses to static pose
- prompt/program and target future are still not always cleanly aligned
- action boundaries inside fixed 60-frame clips are still noisy
- current `EditProgram` injection is still very coarse:
  - one vector
  - projected once
  - broadcast to all frames

## Current naming map

Use these names consistently:

- research target:
  - `Language-Guided Action Regulation`

- current engineering task:
  - `prefix-conditioned action completion`

- avoid overusing:
  - `blind continuation`
  - `plain motion editing`

## Where to look for actual runs

See:

- `docs/experiment_log.md`

That file should answer:

- what was trained
- with which config
- what losses were used
- where the held-out outputs were saved

## Current rule for updates

Update this file only when:

- the main task definition changes
- the naming / framing changes
- the read order or doc roles change

Do not turn this file into a full experiment diary.
