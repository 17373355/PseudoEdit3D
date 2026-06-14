# ICLR / CVPR Follow-Up Strategy Notes

Date: 2026-06-10

This document records the current discussion about how to position the AML paper for ICLR and a later CVPR follow-up, especially relative to recent text-based motion editing work such as:

- `/mnt/data/home/guoruoxi/code/PseudoEdit3D/paper_ref/Cross-Axis Feature Fusion2606.01014v1.pdf`

The purpose is to let parallel sessions synchronize through `docs/` instead of relying on long chat history.

## Current High-Level Plan

The ICLR paper should not be limited to an annotation benchmark only.
It should present AML as a grounded intermediate language between continuous motion and human intent.

ICLR target:

> AML as representation learning, motion annotation language, and controllable editing interface.

CVPR follow-up target:

> AML for embodied motion understanding, instruction execution, and robot / humanoid control.

In this split, AML itself is the main contribution of the ICLR paper.
The CVPR paper should use AML as an already-defined interface and focus on embodied execution, physical grounding, and transfer.

## ICLR Paper Scope

The ICLR paper can include both of the following:

1. `motion -> AML`
   - AML as a motion-derived annotation language.
   - Evaluate whether it is more atomic, consistent, motion-faithful, and lower-noise than raw HumanML3D captions.

2. `source motion + instruction -> AML edit program -> target motion`
   - AML as a controllable editing interface.
   - Evaluate whether structured edit programs support part-level, count-aware, direction-aware, angle-aware, and negative edits.

The unifying claim should be:

> AML is not just a caption rewrite. It is a grounded motion-language interface that makes continuous motion inspectable, editable, compositional, and eventually executable.

### ICLR Should Emphasize

- AML induction from motion.
- AML event / phase / family / program structure.
- AutoPrompt-HumanML3D as a re-annotation benchmark.
- Consistency and denoising relative to HumanML3D captions.
- Structured edit operators over AML programs.
- Comparison against text-only or latent-only conditioning where possible.
- MoMask text probing only as a diagnostic, not as the final AML definition.

### ICLR Should Avoid Overclaiming

- Do not claim that MoMask natural-language probe quality defines AML quality.
- Do not claim every unknown action can already be perfectly named.
- Do not let the editing experiments look like the entire contribution is another text-based editing architecture.
- Do not make embodiment / robot execution the main ICLR claim unless it is fully supported by experiments.

## CVPR Follow-Up Scope

The CVPR follow-up should not re-explain AML as the main contribution.
It should treat AML as a programmatic interface and ask:

> Can AML help embodied agents understand, modify, and execute motion under physical and environmental constraints?

CVPR target topics:

- humanoid / robot execution from AML programs
- human demonstration to robot action program
- language feedback for embodied correction
- closed-loop correction, e.g. "raise your hand higher", "stop clapping"
- scene-aware or object-aware execution
- physical feasibility and embodiment constraints
- human-to-robot or human-to-humanoid transfer

Examples:

- `source motion + "raise the right hand a little higher" -> corrected robot motion`
- `source motion + "stop clapping" -> delete clap cycle from ongoing action`
- `instruction + scene / object constraints -> executable AML program -> robot motion`

## Relation To Cross-Axis Feature Fusion

Cross-Axis Feature Fusion is a text-based 3D human motion editing model.
Its main contribution is an editing architecture:

- source motion + text instruction -> target motion
- joint-anchored transformer
- time-anchored transformer
- cross-axis fusion
- joint-wise Soft-DTW motion difference prediction
- MotionFix evaluation

AML should be differentiated as a representation/interface contribution rather than a stronger fusion architecture.

### Core Difference

Cross-Axis asks:

> How can a neural editor better fuse source motion and natural-language instruction to generate an edited target motion?

AML asks:

> What grounded, atomic, compositional motion language should language instructions operate on before generation or execution?

### Comparison Table

| Dimension | Cross-Axis Feature Fusion | AML |
|---|---|---|
| Primary contribution | Editing model architecture | Motion representation and language interface |
| Main object | Latent conditioning features | Explicit AML events / phases / programs |
| Supervision | MotionFix source-target-instruction triplets | Motion-derived structure, HumanML3D captions as noisy reference corpus, optional edit pairs |
| Joint awareness | Auxiliary Soft-DTW prediction over joint rotations | Explicit part fields and editable joint/body-part slots |
| Interpretability | Indirect, through auxiliary objective | Direct, through readable and executable program fields |
| Editing mechanism | Text and source motion fused into diffusion condition | Instruction parsed into edit operators over source AML program |
| Generalization target | Better MotionFix editing | Reusable motion semantic interface for annotation, editing, transfer, and embodiment |

### Fair Evaluation Against Cross-Axis

When evaluating AML-based editing, Cross-Axis can be a strong baseline on MotionFix-style tasks.

Use common metrics:

- generated-to-target retrieval R@K
- generated-to-source retrieval R@K
- FID against target motions
- qualitative source / target / generated comparisons

Add AML-specific metrics:

- edit localization accuracy
- changed-joint / preserved-joint accuracy
- count error
- direction error
- angle error
- temporal span IoU
- unchanged-region preservation
- long compositional instruction success
- negative edit success, e.g. "stop clapping"

Important: the comparison should show that AML is not merely another conditioning encoder.
It should show that a structured interface improves inspectability, compositionality, and controllability.

## Concurrent Submission / Timing Risk

The desired publication plan may involve an ICLR AML paper and a CVPR follow-up before the ICLR result is known.

Risk:

- If ICLR and CVPR review periods overlap, the CVPR paper must not be substantially similar to the ICLR submission.
- The exact policy must be re-checked for the target year before submission.
- Do not rely on old policy text without checking the current call for papers.

Safe framing:

- ICLR: define and validate AML as a grounded motion language.
- CVPR: use AML for embodied control / execution / physical grounding.

The CVPR paper should have distinct:

- problem setting
- method additions
- experiments
- figures
- claims
- evaluation metrics

It should not be an ICLR experiment appendix turned into a new paper.

## How Much AML To Explain In The CVPR Paper

The CVPR paper can use AML after a concise explanation, but it cannot treat AML as an unexplained black box.

The main text should include a short `AML Interface` section, around half to three quarters of a page.
This section should define the interface contract:

```text
motion -> AML program
instruction -> AML edit operators
source AML program + edit operators -> target executable program
target executable program -> robot / humanoid motion
```

Minimum schema to include in main text:

```text
event = <part, temporal span, direction, magnitude, count, contact/state>
edit = add / delete / modify / stop / reorder
```

Example table:

| Component | Example |
|---|---|
| event | `<right_hand, frames 20-40, upward, small, state>` |
| phase | `repeat(clap_cycle, count=3)` |
| edit operator | `add(clap_cycle, now)` |
| negative edit | `delete(clap_cycle)` |
| numeric edit | `modify(root_step, count=5)` |
| embodied constraint | `project_to_robot_feasible_motion()` |

The CVPR main contribution should then focus on the embodied system built on top of this contract.

## Supplement Strategy For CVPR

It is acceptable to put more AML detail in the supplement, but the main paper must remain self-contained.

Recommended supplement sections:

- `Appendix A: AML Primer`
- `Appendix B: Relation to Concurrent AML Representation Paper`
- `Appendix C: Full AML Schema and Operator Set`
- `Appendix D: Additional Embodied Examples`
- `Appendix E: Failure Cases and Unknown / Approximate Semantic Families`

If the target CVPR policy requires handling concurrent own work, include an anonymized version or anonymized summary of the ICLR AML paper in the supplement and explicitly explain why the CVPR work is non-trivially different.

Do not put the only explanation of AML in the supplement.
Reviewers may not fully read supplement, so the main paper must define the interface and how the embodied method uses it.

## Rebuttal Defense If Reviewers Say AML Is Under-Explained

A rebuttal should not introduce a new method.
It should clarify the interface and point to existing paper sections.

Possible rebuttal logic:

1. AML is used as a programmatic interface in this paper, not reintroduced as the main contribution.
2. The main paper defines the required interface contract in Section X.
3. The supplement gives the full schema and operator set.
4. The embodied contribution is evaluated independently through ablations.
5. The paper includes comparisons showing why AML-program conditioning is needed beyond text-only or latent-only control.

Recommended ablations for this defense:

- text-only embodied controller
- latent motion-token controller
- AML-program controller
- AML-program controller with physical constraints

This ensures reviewers can understand the CVPR contribution even if they do not deeply inspect the concurrent AML representation paper.

## Working Boundary Between The Two Papers

### ICLR Main Claim

> We introduce AML as a grounded, atomic, compositional language for motion representation, annotation, and controllable editing.

### CVPR Main Claim

> We show that AML enables embodied agents to execute and revise motions under physical, environmental, and embodiment-specific constraints.

### One-Sentence Relationship

> ICLR defines and validates the language; CVPR studies what an embodied agent can do with the language.

## Open Decisions

- How much MotionFix editing should be included in the ICLR paper.
- Whether ICLR needs a full AML-conditioned editor, or whether an edit-program prototype plus strong annotation evidence is enough.
- Which embodied benchmark or robot simulator should be used for the CVPR follow-up.
- Whether Cross-Axis code / model will be available and reproducible in time for direct comparison.
- How to anonymize and cite the concurrent ICLR work in the CVPR submission if needed.

## Action Items For Future Sessions

1. Keep ICLR and CVPR docs separate under `docs/paper/`.
2. Do not duplicate the full AML method text into the future CVPR draft.
3. Maintain a short, stable `AML Interface Contract` section that can be reused by downstream papers.
4. Add Cross-Axis to the related-work / baseline list for AML-based editing experiments.
5. Before any CVPR submission, re-check the current CVPR policy for dual submission, supplement, anonymized concurrent work, and arXiv handling.
