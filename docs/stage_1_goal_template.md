# Stage 1 Goal Template

Use this as the starting template for `docs/stage_1_goal.md`.

---

# Stage 1 Goal

## 1. Stage 1 one-line objective

Example:

- Train a stable `prefix-conditioned action completion` baseline that can use a 20-frame motion prefix plus `EditProgram` to complete the remaining future without collapsing into a static pose.

## 2. Current task definition

- input:
  - prefix: first 20 frames of the same clip
  - condition: `EditProgram`
- output:
  - target: complete 60-frame motion

Example:

- input:
  - first 20 frames of the same clip
  - `EditProgram`
- output:
  - complete 60-frame motion

## 3. What Stage 1 must prove

List only 2-4 must-have claims.

Example:

- prefix is respected
- future is dynamically completed rather than frozen
- program-conditioned completion is better than unguided completion
- held-out outputs are visually consistent with the target task

## 4. What Stage 1 does **not** try to prove

Example:

- free-form natural language understanding
- full dialogue interaction
- visual / scene feedback regulation
- long-horizon memory

## 5. Success criteria

### Visual criteria

- [ ] prefix frames remain correct
- [ ] future is not a frozen pose
- [ ] no obvious mixed-action fragment
- [ ] GT / pred comparison is interpretable

### Training criteria

- [ ] training completes reliably
- [ ] checkpoint is saved
- [ ] held-out export works

### Experimental criteria

- [ ] at least one clean baseline run
- [ ] at least one ablation that explains failure/success

## 6. Primary experiment loop

| ID | Question | Config / change | Expected signal | Result |
|---|---|---|---|---|
| E1 |  |  |  |  |
| E2 |  |  |  |  |
| E3 |  |  |  |  |

## 7. Current default baseline

Only fill this section when there is a **stable and trusted** baseline.
If the setup is still changing fast, leave it blank and let the next session decide.

## 8. Current most likely failure modes

- The predicted motion is frozen after the prefix.
- The network isn't able to regress a target motion.
- The training data isn't prepared well enough.


## 9. Next decision points

Example:

- if future-only regression still freezes, inspect network structure
- if future-only works but program-conditioned versions fail, inspect condition injection
- if visual outputs still look mixed, inspect action-boundary construction

## 11. What you should do

- 1. Try to write a minimum baseline training that only contains regression loss and the current network.

- 2. Write Evaluation script so you can tell whether the current result meet the goal? (The frozen motion is easy to detect simply by the output motion file.)

- 3. Train the baseline and evaluate the results.

-4. If the results are bad, try analyze the reason. For the regression only loss, I think you should look at the network and the training data first, decide whether they are able to support a motion regressor. Then update the design.

-5. Repeat the (Training (using cuda available.) + Evaluation + Analyze + Update) loop util we meet the stage 1 Goal.

-6. Additionally, do not forget to update the memory.md (which contains training log), and design brief (which should including the design evolution.)

-7. Be free to add off-the-shelf encoder or new module to ensure the success of the experiment.