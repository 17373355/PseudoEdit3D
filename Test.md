# Test Notes

## Date

2026-05-20

## Scope

This file records the concrete checks, smoke commands, and outcomes used while scaffolding `PseudoEdit3D`.

It covers:
- dataset format inspection
- synthetic training path
- proxy attribute extraction
- mined pseudo triplet generation
- mined training path
- iterative mined-pair refinement
- schema-driven prompt generation
- text-conditioned motion editing smoke training
- inference and GIF visualization
- CMU train/test split workflow

It does not claim that the full dataset pipeline has been executed end to end. The full commands are included below, but the verified runs here were smoke-scale runs.

## Environment

- working tree: `/mnt/data/home/guoruoxi/code/PseudoEdit3D`
- dataset root: `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset`
- python env used for checks: `/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python`
- observed library versions:
  - `torch 2.5.1+cu121`
  - `numpy 2.2.6`

## Dataset schema check

Representative sample:

- `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset/HumanML3D-ACCAD_contact_sequences/Female1Walking_c3d_B10 - walk turn left (45)_poses_origintime_0.0_7.95_frame_0_60_contact.npz`

Observed keys and shapes:

- `poses`: `(60, 156)` `float32`
- `trans`: `(60, 3)` `float32`
- `betas`: `(1, 16)` `float32`
- `incontact`: `(60, 6890)` `bool`
- `incontact_index`: `(60, 6890)` `int64`
- `ground_contact_mask`: `(60, 6890)` `bool`
- `start_frame`: scalar `int64`
- `end_frame`: scalar `int64`
- `original_file`: scalar string

Interpretation:

- `poses` is SMPL-H axis-angle for 52 joints
- the repository currently assumes 60-frame clips with valid `poses` and `trans`
- contact is currently only used coarsely at the clip level in the mining path

## Synthetic path smoke test

Verified objects:

- `pseudoedit3d.data.MotionEditDataset`
- `pseudoedit3d.models.MaskedMotionEditor`
- `pseudoedit3d.training.train_stage1`

Forward-check result:

- dataset length on a 4-sample subset: `4`
- sample tensors:
  - `source_pose (60, 52, 3)`
  - `target_pose (60, 52, 3)`
  - `source_trans (60, 3)`
  - `target_trans (60, 3)`
  - `joint_mask (60, 52)`
  - `time_mask (60,)`
  - `edit_vector (19,)`
- model output shape: `(1, 60, 156)`

Smoke training:

- config: `configs/stage1_upper_body.yaml`
- temporary overrides:
  - `max_clips: 16`
  - `batch_size: 4`
  - `num_workers: 0`
  - `epochs: 1`
- result:
  - `epoch=0 loss=0.585846`

## Attribute cache smoke test

Command used:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_attribute_cache.py \
  --manifest /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_smoke.jsonl \
  --limit 128
```

Outcome:

- command completed successfully
- output file created:
  - `attribute_cache_smoke.jsonl`

What is stored:

- per-clip proxy upper-body attributes
- per-clip summary statistics
- detected active segments for each proxy attribute

Current proxy attributes:

- `left_shoulder_pitch_proxy_deg`
- `right_shoulder_pitch_proxy_deg`
- `both_shoulder_pitch_proxy_deg`
- `left_elbow_flex_proxy_deg`
- `right_elbow_flex_proxy_deg`
- `both_elbow_flex_proxy_deg`
- `torso_pitch_proxy_deg`
- `torso_roll_proxy_deg`

## Mined triplet generation smoke test

Command used:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/mine_triplets.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_smoke.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs_smoke.jsonl \
  --candidate-limit 32 \
  --max-pairs-per-clip 2 \
  --distance-threshold 3.0
```

Outcome:

- input scale: `128` cached clips
- mined output: `235` pairs
- output file created:
  - `mined_pairs_smoke.jsonl`

Mining behavior in this version:

- grouping by `contact_bucket` and dataset family
- candidate ranking by similar root speed
- target pair accepted when:
  - one chosen attribute differs enough
  - other proxy attributes stay relatively close
  - coarse motion statistics stay relatively close
- edit span is estimated from the difference sequence of the chosen attribute

## Mined path smoke test

Verified objects:

- `pseudoedit3d.data.MinedMotionEditDataset`
- `pseudoedit3d.models.MaskedMotionEditor`
- `pseudoedit3d.training.train_stage1` with `data_mode: mined`

Forward-check result:

- dataset length on a 4-pair subset: `4`
- sample tensors:
  - `source_pose (60, 52, 3)`
  - `target_pose (60, 52, 3)`
  - `source_trans (60, 3)`
  - `target_trans (60, 3)`
  - `joint_mask (60, 52)`
  - `time_mask (60,)`
  - `edit_vector (19,)`
- model output shape: `(1, 60, 156)`

Smoke training:

- base config: `configs/stage1_mined_upper_body.yaml`
- temporary overrides:
  - `pair_manifest_path: /mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs_smoke.jsonl`
  - `max_clips: 16`
  - `batch_size: 4`
  - `num_workers: 0`
  - `epochs: 1`
  - `save_dir: /mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs/stage1_mined_upper_body_smoke`
- result:
  - `epoch=0 loss=0.687277`

## Full pipeline commands

These were prepared but not executed in full during the scaffold stage.

### Step 1: scan dataset

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/scan_dataset.py \
  --dataset-root /mnt/data/home/guoruoxi/code/CharRet_multi/dataset \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest_full.jsonl
```

### Step 2: build attribute cache

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_attribute_cache.py \
  --manifest /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest_full.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_full.jsonl
```

### Step 3: mine pseudo triplets

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/mine_triplets.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_full.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs_full.jsonl \
  --candidate-limit 64 \
  --max-pairs-per-clip 4 \
  --distance-threshold 2.5
```

### Step 4: train on mined pairs

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
cp configs/stage1_mined_upper_body.yaml /tmp/pseudoedit_stage1_mined_full.yaml
sed -i 's|/mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs.jsonl|/mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs_full.jsonl|g' /tmp/pseudoedit_stage1_mined_full.yaml
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python train.py \
  --config /tmp/pseudoedit_stage1_mined_full.yaml
```

### Organized script entrypoint

Equivalent organized runner:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
bash scripts/run_full_pipeline.sh full
```

Train as part of the script:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
RUN_TRAIN=1 bash scripts/run_full_pipeline.sh full
```

Smoke mode:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
bash scripts/run_full_pipeline.sh smoke
RUN_TRAIN=1 bash scripts/run_full_pipeline.sh smoke
```

## Inference and GIF visualization

Current visualization output:

- a GIF with four panels:
  - source pose
  - prompt text
  - GT prompt-scoped motion
  - predicted target motion

Verified smoke export:

- output directory:
  - `outputs/HumanML3D-CMU/vis_program_clean_smoke_gt`
- output files:
  - `case_0000.gif`
  - `summary.jsonl`

Smoke command used:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/infer_and_visualize.py \
  --config configs/stage1_mined_cmu_program_clean.yaml \
  --checkpoint outputs/HumanML3D-CMU/stage1_mined_cmu_program_clean/stage1_last.pt \
  --pair-manifest artifacts/jsonl/HumanML3D-CMU/mining/mined_pairs_full.jsonl \
  --output-dir outputs/HumanML3D-CMU/vis_program_clean_smoke_gt \
  --indices 0 \
  --num-cases 1 \
  --frame-limit 20
```

Observed result:

- exported GIF path:
  - `outputs/HumanML3D-CMU/vis_program_clean_smoke_gt/case_0000.gif`
- summary path:
  - `outputs/HumanML3D-CMU/vis_program_clean_smoke_gt/summary.jsonl`
- GIF size:
  - `1600 x 420`
- frame count:
  - `20`

Batch inference command template:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/infer_and_visualize.py \
  --config configs/stage1_mined_cmu_train_program_goal_clean.yaml \
  --checkpoint outputs/HumanML3D-CMU-train/stage1_mined_program_goal_clean/stage1_last.pt \
  --pair-manifest artifacts/jsonl/HumanML3D-CMU-test/mining/mined_pairs_full.jsonl \
  --output-dir outputs/HumanML3D-CMU-test/vis_program_goal_clean_heldout \
  --num-cases 16 \
  --frame-limit 30
```

Start-pose corrected batch inference:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/infer_and_visualize.py \
  --config configs/stage1_mined_cmu_train_startpose_goal_clean.yaml \
  --checkpoint outputs/HumanML3D-CMU-train/stage1_mined_startpose_goal_clean/stage1_last.pt \
  --pair-manifest artifacts/jsonl/HumanML3D-CMU-test/mining/mined_pairs_full.jsonl \
  --output-dir outputs/HumanML3D-CMU-test/vis_program_startpose_goal_clean_heldout \
  --num-cases 16 \
  --frame-limit 30
```

Salient held-out inference:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/infer_and_visualize.py \
  --config configs/stage1_mined_cmu_train_startpose_goal_clean.yaml \
  --checkpoint outputs/HumanML3D-CMU-train/stage1_mined_startpose_goal_clean/stage1_last.pt \
  --pair-manifest artifacts/jsonl/HumanML3D-CMU-test/mining/mined_pairs_full.jsonl \
  --output-dir outputs/HumanML3D-CMU-test/vis_program_startpose_goal_clean_salient \
  --selection salient \
  --num-cases 16 \
  --frame-limit 24 \
  --min-delta-deg 15 \
  --min-duration-frames 8
```

## CMU train/test split workflow

Split rule:

- deterministic split by original clip group
- not by individual 60-frame chunk
- intended to reduce near-duplicate leakage across train and test

Build split manifests:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_group_split.py \
  --subset HumanML3D-CMU \
  --manifest artifacts/jsonl/HumanML3D-CMU/scan/manifest_full.jsonl \
  --seed 42 \
  --test-ratio 0.2
```

Rebuild split-specific artifacts:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_artifacts_from_manifest.py \
  --subset HumanML3D-CMU-train \
  --manifest artifacts/jsonl/HumanML3D-CMU-train/scan/manifest_full.jsonl

/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_artifacts_from_manifest.py \
  --subset HumanML3D-CMU-test \
  --manifest artifacts/jsonl/HumanML3D-CMU-test/scan/manifest_full.jsonl
```

Observed split after fixing `non_contact` bucket parsing:

- train records: `6704`
  - `contact: 2560`
  - `neutral: 1617`
  - `non_contact: 2527`
- test records: `1685`
  - `contact: 617`
  - `neutral: 391`
  - `non_contact: 677`

Observed split-specific mined pairs:

- train mined pairs: `26782`
- test mined pairs: `6708`

Train-split goal-loss config:

- `configs/stage1_mined_cmu_train_program_goal_clean.yaml`

Held-out visualization output already exported:

- `outputs/HumanML3D-CMU-test/vis_program_goal_clean_heldout`
- `outputs/HumanML3D-CMU-test/vis_program_startpose_goal_clean_heldout`
- `outputs/HumanML3D-CMU-test/vis_program_startpose_goal_clean_salient_smoke`

## Known limitations at this stage

- proxy attributes are derived from local axis-angle channels, not full FK joint geometry
- synthetic edits still use direct channel perturbation as a bootstrap mechanism
- mined pairs rely on coarse similarity and may include false-positive edit pairs
- contact is used only as a coarse bucket and policy flag, not a hard motion constraint
- no natural language or dialogue supervision is included in this stage

## Schema-driven prompt layer

Added in this stage:

- configurable label schema:
  - `configs/label_schema.yaml`
- program verbalization:
  - `pseudoedit3d/edit/verbalizer.py`
- character tokenizer:
  - `pseudoedit3d/text/tokenizer.py`
- prompt preview script:
  - `scripts/preview_prompts.py`

Prompt preview smoke command:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/preview_prompts.py \
  --mode synthetic \
  --manifest /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest.jsonl \
  --num-samples 5
```

Observed examples:

- `raise the right arm by 22.5 degrees during the middle of the motion`
- `flex the right arm moderately during the middle of the motion`
- `curl the right arm by 22.5 degrees early in the motion`
- `extend the right arm significantly late in the motion`
- `straighten the left arm by 22.5 degrees during the middle of the motion`

## Text-conditioned motion editing smoke

Current condition modes supported by `train.py`:

- `program`
- `text`
- `hybrid`

Forward-check result for text-only conditioning:

- sample prompt example:
  - `raise the both arms by 15 degrees during the middle of the motion`
- prompt token shape:
  - `(96,)`
- model output shape:
  - `(1, 60, 156)`

Smoke training:

- config: `configs/stage1_text_upper_body.yaml`
- temporary overrides:
  - `max_clips: 16`
  - `batch_size: 4`
  - `num_workers: 0`
  - `epochs: 1`
  - `save_dir: /mnt/data/home/guoruoxi/code/PseudoEdit3D/outputs/stage1_text_upper_body_smoke`
- result:
  - `epoch=0 loss=0.610044`

## Iterative mined-pair refinement

Current implementation adds a simple bootstrapped loop:

1. mine raw pairs from the attribute cache
2. select a seed subset using heuristic mining scores
3. train a mined-pair editor on that subset
4. score all raw pairs with the trained editor
5. filter and refresh the pair set for the next round

Entrypoint:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/iterate_mined_pairs.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_full.jsonl \
  --output-dir /mnt/data/home/guoruoxi/code/PseudoEdit3D/iter_runs/default \
  --num-rounds 2
```

Useful debug knobs:

- `--max-train-pairs`
- `--max-pairs-per-source`
- `--train-epochs`
- `--train-batch-size`
- `--train-num-workers`
- `--mine-candidate-limit`
- `--mine-distance-threshold`

Artifacts per run:

- `iter_raw_pairs.jsonl`
- `iter_seed_pairs.jsonl`
- `round_XX/train_pairs.jsonl`
- `round_XX/scored_pairs.jsonl`
- `round_XX/refined_pairs.jsonl`
- `round_XX/train_outputs/round_XX_last.pt`
- `iteration_summary.jsonl`

Smoke verification performed:

```bash
cd /mnt/data/home/guoruoxi/code/PseudoEdit3D
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/iterate_mined_pairs.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_smoke.jsonl \
  --output-dir /mnt/data/home/guoruoxi/code/PseudoEdit3D/iter_runs/smoke_iter \
  --num-rounds 2 \
  --max-train-pairs 32 \
  --train-epochs 1 \
  --train-batch-size 4 \
  --train-num-workers 0 \
  --mine-candidate-limit 32 \
  --mine-max-pairs-per-clip 2 \
  --mine-distance-threshold 3.0
```

Observed result:

- `raw_pairs=235`
- `round=0 input_pairs=32 refined_pairs=32 last_loss=0.562269315123558`
- `round=1 input_pairs=32 refined_pairs=32 last_loss=0.5716654546558857`
- summary:
  - `/mnt/data/home/guoruoxi/code/PseudoEdit3D/iter_runs/smoke_iter/iteration_summary.jsonl`
