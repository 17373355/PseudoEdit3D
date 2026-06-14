# AML Layer3 and Coarse Signature Call Order

This note records the current source-level route from HumanML3D joints to Layer3 events, coarse signatures, canonical actions, and condition manifests.

## Current Artifacts

- WordNet cached lexicon:
  `outputs/aml_lexicon/wordnet_action_terms_v1.json`
- 250-case selected set summary:
  `outputs/aml_regression_testset_v2/aml_regression_testset_v2.json`
- 250-case fixed condition schema:
  `outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/conditions.jsonl`
- 250-case condition summary:
  `outputs/aml_regression_testset_v2/aml_condition_manifest_250_taxonomy_v1/summary.json`

`aml_regression_testset_v2.json` stores `layer3_count`, feature keys, and rendered prompts, but not full Layer3 event objects. To inspect full Layer3 events for selected cases, use `scripts/extract_aml_layers.py`, whose JSON output contains `layer3_atomic_program`.

## Layer3 Construction

```text
HumanML3D joints
  -> extract_layer0_frame_observables
  -> extract_layer1_micro_events
  -> merge_micro_events
  -> detect_repeated_phases
  -> projected repeated phases by category
  -> dedupe_phase_patterns / dedupe_phase_objects
  -> build_layer3_atomic_program
```

Primary code locations:

- `scripts/extract_aml_layers.py`: explicit multi-layer debug export.
- `scripts/build_aml_regression_testset.py`: selected 250-case set builder.
- `pseudoedit3d/edit/aml_atomic_program.py`: `build_layer3_atomic_program`.

Inside `build_layer3_atomic_program`:

```text
submotions
  -> _submotion_to_event
  -> _build_locomotion_turn_events
  -> _build_terminal_state_events
  -> build_semantic_joint_events
  -> _phase_to_event
  -> abstract_atomic_events
  -> split_bimanual_events
  -> _annotate_context_coupling
  -> sort events
  -> {"events": events}
```

## Coarse Signature Main Entry

The current fixed-condition path enters `coarse_signature.py` from:

```text
scripts/export_aml_condition_manifest.py
  -> src.extract_aml_program(joints)
  -> build_coarse_action_program(aml["layer3"])
  -> _condition_from_action(canonical_action)
  -> conditions.jsonl
```

The user-facing prompt path enters through:

```text
coarse_prompt_renderer.render_coarse_aml_prompt
  -> build_coarse_action_program(layer3)
  -> render clauses from canonical/coarse actions
```

## `build_coarse_action_program` Order

```text
build_coarse_action_program(program_or_events)
  -> coarse_event_utils._indexed_events
  -> coarse_axes.build_event_coarse_signature
       -> _locomotion_axis
       -> _vertical_axis
       -> _rotation_axis
       -> _limb_axis
      -> _support_gait_axis
      -> _state_axis
      -> _temporal_axis
       -> _motion_patterns_axis
           reusable pattern evidence only:
           coupled locomotion, post-vertical recovery fields, low-body event counts,
           bilateral rhythmic counts/source indices
      -> assign_seeded_prototype
       -> _prototype_context
       -> _select_seeded_pattern_prototype
            reads pseudoedit3d/edit/aml_pattern_tree.json through aml_pattern_tree.py
            evaluates WordNet-like parent/child pattern nodes and applies primary_selection_order
            includes upper-body arm mime, subtle proxy, bimanual fallback, and event-sequence fallback nodes
  -> optional bilateral rhythmic count repair
  -> _cover_primary_events
       reads primary cover modes from pseudoedit3d/edit/aml_proto_registry.json
  -> build primary_action
  -> coarse_pattern_evidence._cyclic_bilateral_coordination_actions
       assembles bimanual-periodic + vertical-cycle evidence; emitted family/template comes from aml_pattern_tree.json
  -> coarse_pattern_evidence._semantic_candidate_actions
       -> event_proxy_for_event / event_proxy_action_fields
            reads event_proxy leaves from aml_pattern_tree.json:
            torso hunch, hand high, squat, kick/leg pose, circular path,
            climb proxy, cartwheel/inversion
       -> composed semantic candidates
            low-level temporal evidence for low-body transitions and leg-forward pairs is assembled in code,
            then routed through select_composed_pattern_match against composed_candidate nodes
            acrobatics sequence and dance-like leg pose also route through composed_candidate nodes
  -> coarse_pattern_evidence._post_vertical_translation_recovery_actions
  -> coarse_pattern_evidence._bimanual_contact_actions
  -> coarse_pattern_evidence._secondary_actions
       residual gait and turn segments
  -> coarse_pattern_evidence._vertical_impulse_translation_pair_actions
  -> terminal still append
  -> _apply_semantic_dominance
  -> _drop_redundant_fallback_actions
  -> sort actions by span
      -> coarse_action_materializer._attach_action_metadata
       -> action_pattern_metadata_for_family fallback
            fills pattern_node_id / pattern_path / pattern_taxonomy_parent_id
            for every visible action whose family exists in aml_pattern_tree.json
       -> _probe_alias
            reads surface aliases from aml_proto_registry.json
       -> _semantic_family_descriptor
            -> family_taxonomy_metadata
       -> _approx_slots
       -> _canonical_action
  -> return coarse_action_program_v2
```

The returned object includes:

- `signature`: coarse axes built only from Layer3 events.
- `prototype`: one primary seeded prototype.
- `coarse_actions`: actions after secondary/semantic expansion and metadata attachment.
- `canonical_actions`: compact action records used by condition export.
- `covered_events` and `residual_events`: audit references back to Layer3 event indices.

## Condition Export

`scripts/export_aml_condition_manifest.py` converts each `canonical_action` to fixed condition records:

```text
canonical_action
  -> semantic_family
  -> family_taxonomy_metadata
  -> approx_slots
  -> missing_required_slots
       reads required slot schema from pseudoedit3d/edit/aml_proto_registry.json
  -> action_condition_weight
  -> condition row
```

Each condition row keeps:

- `family_id`
- `status`
- `taxonomy_parent_id`
- `taxonomy_recoverability`
- `ambiguity_boundary`
- `approx_slots`
- `slot_values`
- `slot_confidences`
- `missing_required_slots`
- `condition_weight`
