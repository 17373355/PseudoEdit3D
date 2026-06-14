# AML and CLIP Boundary

This note records the boundary between motion-derived AML semantics and any future CLIP/text resolver.

## Principle

AML remains motion-first.

```text
motion observables -> AML events -> composed AML families -> condition schema
```

CLIP, if added later, must be a resolver on top of AML evidence, not the source of core AML families.

## What AML Must Own

These categories must be handled by geometry and event rules before any language model is involved:

- locomotion and root path
- in-place gait vs leg-kick priority
- jump, hop, turn, spin, root translation scale
- squat, sit/stand transition, lunge, low-body posture
- arm raise, arm swing/circle, bimanual motion, hand proximity
- acrobatic or inverted-motion candidates

If a failure is labeled `geometry_recoverable` by the coverage audit, the fix belongs in AML extraction, family composition, slot estimation, or renderer priority.

## What CLIP May Resolve

CLIP can be considered only for ambiguous object/intent labels that skeleton geometry cannot prove:

- tennis / ball strike
- drinking
- phone / hand near ear
- swimming-like intent
- martial arts / combat naming
- jump rope
- cheering / dance style

Even in these cases, the condition record should keep the AML family as the primary motion evidence and attach resolver output as optional metadata.

Recommended future shape:

```json
{
  "family_id": "UNILATERAL_ARM_MIME_CANDIDATE",
  "resolver": {
    "type": "clip_or_text",
    "label": "drinking",
    "confidence": 0.62,
    "nullable": true
  }
}
```

## Forbidden Use

Do not use CLIP to:

- replace AML family ids for geometry-recoverable actions
- decide condition weights without AML slot support
- silently turn a proxy family into a stable family
- generate auto-prompt clauses that cannot be traced to AML evidence
- hide MoMask realization failures as language coverage fixes

## Evaluation Contract

Coverage reports should remain bucketed:

- `missing_composed_family`: AML implementation problem
- `prompt_priority_error`: renderer/dominance problem
- `object_or_intent_ambiguous`: optional resolver problem
- `momask_realization_or_scale_review`: generation-model problem

This separation is what makes AML scalable without forcing every ambiguous action into manual GIF review or CLIP dependence.
