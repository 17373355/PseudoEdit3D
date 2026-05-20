# Paper V3 Proposal

## Title

Dialogue-Grounded Embodied Motion Editing with Contact and Task Preservation

## Main goal

Move from single-turn instruction following to multi-turn embodied correction in scene-aware tasks.

## Core question

Can a virtual human executing a scene-grounded task be corrected through natural dialogue while preserving contacts, task intent, and previously accepted edits?

## Main contribution targets

1. dialogue state representation for relative motion corrections
2. slot grounding over history, current motion, and scene/task state
3. contact-aware embodied motion editing
4. evaluation for multi-turn consistency and task preservation

## Scope

Included:
- multi-turn correction
- relative edits such as "a bit higher" or "20 degrees more"
- task and contact preservation

Required foundation from earlier papers:
- structured edit programs from Paper v1
- language-slot alignment from Paper v2

## Deliverables

- embodied multi-turn benchmark design
- contact-aware editor or post-refiner
- dialogue-consistency and task-success metrics

## Success criteria

- later turns compose correctly with earlier accepted edits
- contact/task violations stay controlled
- the model can distinguish local pose correction from global action replacement
