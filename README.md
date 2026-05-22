# PseudoEdit3D

Paper v1 scaffold for fine-grained 3D human motion editing from unlabeled SMPL-H motion.

Current scope:
- consume unlabeled 60-frame SMPL-H motion clips
- bootstrap structured pseudo edit programs from motion geometry
- train a first-stage editor on `(source motion, pseudo edit program) -> target motion`

This repository is intentionally narrow for the first iteration. It does not yet solve natural dialogue grounding or scene-aware contact-preserving editing end to end. The first target is a controllable motion editor trained from self-bootstrapped pseudo supervision.

## Dataset assumption

The initial data source is compatible with the segmented `.npz` files under:

- `/mnt/data/home/guoruoxi/code/CharRet_multi/dataset`

Observed sample schema:

- `poses`: `(60, 156)` float32, SMPL-H axis-angle for 52 joints
- `trans`: `(60, 3)` float32
- `betas`: `(1, 16)` float32
- optional contact fields:
  - `incontact`
  - `incontact_index`
  - `ground_contact_mask`

## Project status

Implemented in this scaffold:
- planning document
- dataset manifest scanner
- minimal SMPL-H clip dataset loader
- first-pass pseudo edit schema
- upper-body proxy attribute extractor and active-span detector
- synthetic pair builder for stage-v1 training
- mined pair pipeline for pseudo triplets from similar real clips
- baseline masked editor model and training entrypoint

Not implemented yet:
- full FK-based attribute extraction
- true contact-aware refinement
- dialogue grounding
- natural language generation/alignment

## Quick start

Use an environment with `torch` and `numpy`, for example:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/scan_dataset.py \
  --dataset-root /mnt/data/home/guoruoxi/code/CharRet_multi/dataset \
  --output data_manifest.jsonl
```

Train the minimal baseline:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python train.py \
  --config configs/stage1_upper_body.yaml
```

Train a prompt-conditioned baseline:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python train.py \
  --config configs/stage1_text_upper_body.yaml
```

Build attribute cache and mined pseudo triplets:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/build_attribute_cache.py \
  --manifest /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache.jsonl

/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/mine_triplets.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache.jsonl \
  --output /mnt/data/home/guoruoxi/code/PseudoEdit3D/mined_pairs.jsonl
```

Iterative mined-pair refinement:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/iterate_mined_pairs.py \
  --attribute-cache /mnt/data/home/guoruoxi/code/PseudoEdit3D/attribute_cache_full.jsonl \
  --output-dir /mnt/data/home/guoruoxi/code/PseudoEdit3D/iter_runs/default \
  --num-rounds 2
```

Preview generated prompt templates:

```bash
/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python scripts/preview_prompts.py \
  --mode synthetic \
  --manifest /mnt/data/home/guoruoxi/code/PseudoEdit3D/data_manifest.jsonl \
  --num-samples 5
```

Organized end-to-end runner:

```bash
bash scripts/run_full_pipeline.sh full
```
