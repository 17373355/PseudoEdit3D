# Paper V2 Proposal

## Title

Program-to-Language Motion Editing: Grounding Natural Edit Instructions onto Fine-Grained Motion Slots

## Main goal

Bridge from structured pseudo edit programs to natural instruction following.

## Core question

Can pseudo edit programs from Paper v1 become a stable latent interface for mapping template text, paraphrases, and later dialogue turns into precise motion edits?

## Main contribution targets

1. a language layer that maps text to structured edit slots
2. automatic text generation from pseudo programs for weak supervision
3. paraphrase-robust edit grounding
4. evaluation on instruction fidelity versus structural edit correctness

## Scope

Included:
- single-turn text first
- template and paraphrased language
- structured slot grounding

Deferred:
- full dialogue memory
- multi-agent task planning
- strong scene interaction constraints

## Deliverables

- text-program aligned training set
- text-conditioned motion editor built on top of Paper v1
- evaluation suite for language grounding error versus motion execution error

## Success criteria

- slot prediction is accurate enough to preserve controllability
- free-form paraphrases map to consistent edits
- language grounding does not destroy locality learned in Paper v1
