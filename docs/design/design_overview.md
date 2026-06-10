# Design Overview

## Main Framing

The current project is not only a text-to-motion task.
It is a motion-derived semantic annotation project whose long-term target is a structured condition model.

## Core Principle

- motion first
- structure first
- text second

This means:
- `auto_program` is the primary output
- `auto_prompt` is a rendered view for probing and visualization
- HumanML3D captions are noisy semantic priors only

## Current Structured Target

Each event should ideally include:
- `type`
- `part`
- `direction`
- `magnitude`
- `unit`
- `count`
- `start_frame`
- `end_frame`
- `confidence`

## Current Main Issues

- false squat / false crouch
- missing backward-walk semantics
- weak stop detection in some edge cases
- missing limb-level repeated patterns
- missing number / angle awareness
- too-coarse whole-body descriptions

## Current Entry Point

- `../design_brief.md`

## Legacy References

- `../hml3d_experiment_pipeline.md`
- `../stage1_upgrade_plan.md`
