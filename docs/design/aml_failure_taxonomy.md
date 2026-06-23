# AML Failure Taxonomy

This document defines how review failures should be classified before changing
the AML tree. The goal is to avoid case-by-case repairs.

## Review Signals

A MoMask review pack has four signals:

- `GT Motion`: the HumanML3D motion.
- `Native MoMask: HML3D Caption`: original MoMask T2M conditioned by the selected
  HumanML3D caption.
- `Native MoMask: AML AutoPrompt`: original MoMask T2M conditioned by a text
  prompt rendered from AML.
- `AML AutoPrompt`: the text view of the extracted AML structure.

Only the AML AutoPrompt is evidence about the current tree. MoMask generation is
a probe, not the final AML model.

## Failure Sources

- `observable_missing`: the necessary kinematic signal is not extracted as a
  channel event.
- `composition_binding_gap`: component events exist, but the system fails to
  bind them into a full motion pattern.
- `temporal_order_gap`: the system detects parts of a sequence but drops their
  order, such as sit -> stand -> sit.
- `naming_boundary_gap`: a motion structure exists but is rendered too generic.
- `renderer_priority_gap`: the right evidence exists but is hidden by prompt
  budget, salience, or cover suppression.
- `object_activity_boundary`: the caption names an object or activity that is
  not directly observable from motion alone.
- `native_momask_limitation`: original MoMask fails even with the HML3D caption.
- `ambiguous_hml3d_caption`: HumanML3D text is underspecified or inconsistent
  with the motion.

## Recoverability

- `geometry_recoverable`: should be recoverable from motion geometry alone.
- `geometry_candidate`: may be recoverable, but needs corpus-level purity audit.
- `weak_name_only`: useful as a caption/name hint, not as a hard geometry node.
- `not_motion_only`: should not enter the motion tree without non-motion context.

## Current Group-01 Failure Families

- `bilateral_spread_jump`: jumping-jack-like coordination.
- `unilateral_arm_circle`: one arm spinning, circling, or windmilling.
- `sit_stand_cycle`: sit down, sit up, stand up, and repeated sit/stand order.
- `side_sway_or_rock`: lateral rocking or swaying while mostly in place.
- `step_up_hop_sequence`: repeated hops with upward/downward level changes.
- `strike_or_punch_sequence`: punching, martial-art strikes, and strike-return
  timing.
- `object_activity_proxy`: basketball, tennis, or similar object-conditioned
  activities that need motion evidence plus weak naming.
- `prone_swim_or_flail`: low/prone body with limb flailing or swimming-like
  cycles.

## Update Rule

For each family, first add or refine general observables, then rerun full-HML3D
mining. Promote a tree node only if the recovered motif is frequent, pure enough,
and has stable edit handles.
