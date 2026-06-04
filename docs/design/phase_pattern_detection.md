# Phase Pattern Detection

## Purpose

Many motion semantics are not single events, but repeated phases:

- repeated hops
- repeated squats
- repeated arm up/down motion
- alternating left-right or up-down cycles

So after Layer 2 `sub-motion` units are formed, we need a Layer 2.5 phase-pattern detector.

## Current V1 Mechanism

The first phase-pattern detector works over the ordered sequence of `SubMotionUnit`.

It currently detects two families:

### 1. Same-unit repetition

```text
A A A
```

Examples:
- `hop_unit` repeated three times
- `crouch_descent` repeated twice

### 2. Alternating repetition

```text
A B A B
```

Examples:
- `left_arm_up` / `left_arm_down` alternating
- `root_up` / `root_down` alternating
- `compress` / `release` alternating

## Why This Matters

This provides a generic way to model:

- count
- cycle
- repeated local phase

without writing a separate high-level rule for every repeated behavior.

## Near-Term Use

The immediate use is to detect:

- repeated hops
- repeated bounce
- repeated crouch
- repeated arm up/down

## Later Extension

Later versions should add:

- phase confidence
- support from duration regularity
- support from amplitude regularity
- part-synchronous multi-stream repeat detection
