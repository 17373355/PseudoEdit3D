#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/home/guoruoxi/miniconda3/envs/h2char/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data/home/guoruoxi/code/CharRet_multi/dataset}"
MODE="${1:-full}"
RUN_TRAIN="${RUN_TRAIN:-0}"

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "Usage: bash scripts/run_full_pipeline.sh [full|smoke]"
  exit 1
fi

if [[ "${MODE}" == "smoke" ]]; then
  MANIFEST_PATH="${PROJECT_ROOT}/data_manifest_smoke.jsonl"
  ATTRIBUTE_CACHE_PATH="${PROJECT_ROOT}/attribute_cache_smoke.jsonl"
  MINED_PAIRS_PATH="${PROJECT_ROOT}/mined_pairs_smoke.jsonl"
  LIMIT_ARGS=(--limit 128)
  MINE_ARGS=(--candidate-limit 32 --max-pairs-per-clip 2 --distance-threshold 3.0)
  TRAIN_SAVE_DIR="${PROJECT_ROOT}/outputs/stage1_mined_upper_body_smoke"
  TRAIN_MAX_CLIPS="16"
  TRAIN_BATCH_SIZE="4"
  TRAIN_NUM_WORKERS="0"
  TRAIN_EPOCHS="1"
else
  MANIFEST_PATH="${PROJECT_ROOT}/data_manifest_full.jsonl"
  ATTRIBUTE_CACHE_PATH="${PROJECT_ROOT}/attribute_cache_full.jsonl"
  MINED_PAIRS_PATH="${PROJECT_ROOT}/mined_pairs_full.jsonl"
  LIMIT_ARGS=()
  MINE_ARGS=(--candidate-limit 64 --max-pairs-per-clip 4 --distance-threshold 2.5)
  TRAIN_SAVE_DIR="${PROJECT_ROOT}/outputs/stage1_mined_upper_body_full"
  TRAIN_MAX_CLIPS="0"
  TRAIN_BATCH_SIZE="16"
  TRAIN_NUM_WORKERS="4"
  TRAIN_EPOCHS="5"
fi

echo "[1/4] Scan dataset -> ${MANIFEST_PATH}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/scan_dataset.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${MANIFEST_PATH}" \
  "${LIMIT_ARGS[@]}"

echo "[2/4] Build attribute cache -> ${ATTRIBUTE_CACHE_PATH}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_attribute_cache.py" \
  --manifest "${MANIFEST_PATH}" \
  --output "${ATTRIBUTE_CACHE_PATH}" \
  "${LIMIT_ARGS[@]}"

echo "[3/4] Mine pseudo triplets -> ${MINED_PAIRS_PATH}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/mine_triplets.py" \
  --attribute-cache "${ATTRIBUTE_CACHE_PATH}" \
  --output "${MINED_PAIRS_PATH}" \
  "${MINE_ARGS[@]}"

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "[4/4] Skip training because RUN_TRAIN=${RUN_TRAIN}"
  exit 0
fi

TRAIN_CONFIG="$(mktemp)"
cp "${PROJECT_ROOT}/configs/stage1_mined_upper_body.yaml" "${TRAIN_CONFIG}"
sed -i "s|^pair_manifest_path:.*|pair_manifest_path: ${MINED_PAIRS_PATH}|" "${TRAIN_CONFIG}"
sed -i "s|^max_clips:.*|max_clips: ${TRAIN_MAX_CLIPS}|" "${TRAIN_CONFIG}"
sed -i "s|^batch_size:.*|batch_size: ${TRAIN_BATCH_SIZE}|" "${TRAIN_CONFIG}"
sed -i "s|^num_workers:.*|num_workers: ${TRAIN_NUM_WORKERS}|" "${TRAIN_CONFIG}"
sed -i "s|^epochs:.*|epochs: ${TRAIN_EPOCHS}|" "${TRAIN_CONFIG}"
sed -i "s|^save_dir:.*|save_dir: ${TRAIN_SAVE_DIR}|" "${TRAIN_CONFIG}"

echo "[4/4] Train mined editor with config ${TRAIN_CONFIG}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/train.py" --config "${TRAIN_CONFIG}"
