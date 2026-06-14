# AML Family Taxonomy v1

This note records the top-down structure for AML semantic families. It is meant to stop AML growth from becoming a bottom-up collection of failure-case patches.

## Principle

AML family design starts from motion evidence classes, then maps words onto those classes.

```text
WordNet / InternVid / HML3D words
        -> lexical candidates
        -> AML taxonomy parent
        -> motion evidence contract
        -> detector / composition rule
        -> condition schema and prompt rendering
```

The word list is not the taxonomy. WordNet and InternVid can propose candidate labels and synonyms, but a label becomes an AML family only when it has an explicit motion-evidence contract or is marked as an optional resolver label.

## Files

- `pseudoedit3d/edit/aml_family_taxonomy.json`: top-down parent taxonomy, child family membership, recoverability, evidence axes, and lexical source notes.
- `pseudoedit3d/edit/aml_family_taxonomy.py`: small loader and lookup helpers for condition/audit scripts.
- `pseudoedit3d/edit/aml_proto_registry.json`: operational registry for proto status, suppression, dominance, fallback, and metadata groups.
- `scripts/build_wordnet_action_lexicon.py`: one-shot WordNet builder that exports a cached JSON lexicon.
- `docs/design/aml_clip_boundary.md`: boundary for future CLIP/text resolvers.
- `docs/InternVid.ipynb`: WordNet and InternVid notebook prototype for broad action/activity vocabulary discovery.

## Taxonomy Parents

| parent | role | boundary |
| --- | --- | --- |
| `ROOT_LOCOMOTION` | root translation, in-place gait, circular paths, recovery steps | geometry recoverable |
| `VERTICAL_IMPULSE` | jumps, hops, ballistic translation | geometry recoverable |
| `BODY_LEVEL_POSTURE` | squat, lunge, sit/stand, torso/low-body transitions | geometry recoverable |
| `GROUND_PRONE_KNEEL` | kneel, fall, prone, crawl, ground recovery | geometry recoverable |
| `UPPER_LIMB_GESTURE` | hand high, arm swing/circle, clap, hand-to-face/head proxies | geometry candidate |
| `LOWER_LIMB_ACTION` | kicks and non-gait leg actions | geometry recoverable |
| `BILATERAL_RHYTHMIC_EXERCISE` | bilateral rhythmic coordination and repeated arm/leg gesture patterns; jumping-jack or cheer/dance are lexical aliases | geometry recoverable with exclusions |
| `ROTATION_SPIN` | turns, spins, rotation-dominant motion | geometry recoverable |
| `ACROBATICS_INVERSION` | cartwheel/backflip/handspring-like candidates | geometry candidate |
| `ACTIVITY_INTENT_PROXY` | combat, dance, sport/object activities | geometry candidate plus optional resolver |
| `UNKNOWN_OR_FALLBACK` | fallback or unresolved event signatures | unknown |

## WordNet / InternVid Use

The notebook sketches the WordNet/InternVid discovery logic, but it is not part of runtime extraction:

- extract WordNet verbs with `wn.all_synsets(wn.VERB)`;
- extract WordNet nouns as activity candidates;
- expand candidate words through synonyms;
- use spaCy human-subject filtering for InternVid captions.

WordNet access is now an offline cached step:

```text
scripts/build_wordnet_action_lexicon.py
    -> outputs/aml_lexicon/wordnet_action_terms_v1.json
```

`nltk` is required only for that one-shot builder. If the `h2char` environment does not have it:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python -m pip install nltk
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_wordnet_action_lexicon.py --download-wordnet
```

AML extraction should read the cached JSON if it needs lexical candidates. It should not import `nltk`, call `wn.all_synsets`, or depend on `docs/InternVid.ipynb`.

Each exported term maps to:

- `taxonomy_parent_candidates`: top-down parent candidates;
- `candidate_family_ids`: existing or proposed AML child family candidates;
- `ambiguity_boundary`: `motion_geometry`, `geometry_candidate`, or `object_or_intent_ambiguous`;
- `source`: currently `wordnet`, later optionally `internvid`, `hml3d_caption`, or curated.

## Immediate Design Consequence

Do not keep adding individual families only because a GIF failed. New family work should start from parent-level separation.

For the current failure set:

- `BILATERAL_RHYTHMIC_COORDINATION`, `BILATERAL_RHYTHMIC_GESTURE_CANDIDATE`, `JUMP_ROPE_CANDIDATE`, and related repeated gestures belong under `BILATERAL_RHYTHMIC_EXERCISE`; `JUMPING_JACK` and cheer/dance names should be treated as lexical aliases or resolver labels, not primary coarse-signature families.
- `ARM_SWING_CANDIDATE` and `ARM_CIRCLE_CANDIDATE` belong under `UPPER_LIMB_GESTURE` and should preempt jumping-jack when vertical evidence is weak or arm motion is unilateral/circular.
- `SWIM_LIKE_MOTION_CANDIDATE` and `PRONE_ARM_LEG_MOTION_CANDIDATE` belong under `GROUND_PRONE_KNEEL` or `ACTIVITY_INTENT_PROXY` depending on whether prone/floor evidence is present.
- `COMBAT_SEQUENCE_CANDIDATE` and sport/object labels should keep AML geometry evidence first, then use optional resolver metadata only if needed.

## Evaluation Contract

Condition manifests should include taxonomy metadata per condition:

- `taxonomy_parent_id`
- `taxonomy_parent_label`
- `taxonomy_recoverability`
- `taxonomy_evidence_axes`
- `taxonomy_secondary_parent_ids`
- `ambiguity_boundary`

Coverage audits should report both issue labels and taxonomy-parent counts. Parent-level issue counts tell us which top-down areas are weak before we add another child detector.
