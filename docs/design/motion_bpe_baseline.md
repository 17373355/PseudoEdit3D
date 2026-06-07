# Motion BPE Baseline

## Goal

Build the first true BPE-style merge baseline over motion sub-motion sequences.

## Input Sequence

The BPE baseline does not run on raw poses.
It runs on the sequence:

```text
Layer 0 observables
-> Layer 1 micro-events
-> Layer 2 sub-motion units
```

At this stage, each motion clip is represented as a sequence of symbolic units such as:

- `crouch_descent`
- `hop_ascent`
- `left_arm_lowering`
- `torso_rise_back`
- fallback micro-event symbols when no merge exists

## BPE Analogy

Text BPE:

```text
characters -> repeated pair merge -> subwords
```

Motion BPE:

```text
sub-motion units -> repeated adjacent pair merge -> higher-level motion motifs
```

## Baseline Procedure

1. Encode each case as a sequence of sub-motion unit names.
2. Count adjacent token pairs across the corpus.
3. Select the highest-support pair.
4. Merge that pair into a new token.
5. Rebuild the corpus token sequences.
6. Repeat for a fixed number of merges.

## Constraints

Unlike raw text BPE, motion BPE should preserve:

- temporal locality
- part consistency
- access to original numeric information

So every merged token should store:

- parent tokens
- support case count
- mean span
- component sub-motion sequence

## What This Baseline Can Show

- whether higher-order motion motifs emerge automatically
- whether frequently repeated motion structures can be learned compositionally
- whether the current sub-motion layer is rich enough to support further merge

## What It Cannot Yet Show

- final optimal motion vocabulary
- multi-scale segmentation uncertainty
- robust handling of all long-range and cross-part dependencies

Those belong to later unigram / probabilistic segmentation stages.
