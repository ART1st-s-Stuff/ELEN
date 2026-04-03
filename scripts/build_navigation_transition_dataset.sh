#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/project/peilab/atst"
ELEN_DIR="${ROOT_DIR}/elen"
VAGEN_DIR="${ROOT_DIR}/VAGEN"

export PYTHONPATH="${ELEN_DIR}:${ROOT_DIR}:${VAGEN_DIR}:${PYTHONPATH:-}"

OUTPUT_DIR="${ELEN_OUTPUT_DIR:-${ELEN_DIR}/datasets/navigation_transition}"
IMAGE_DIR="${ELEN_IMAGE_DIR:-${OUTPUT_DIR}/images}"
EPISODES="${ELEN_EPISODES:-100}"
MAX_STEPS="${ELEN_MAX_STEPS:-10}"
FEATURE_MODE="${ELEN_FEATURE_MODE:-both}"
FEATURE_DEVICE="${ELEN_FEATURE_DEVICE:-cpu}"
GPU_DEVICE="${ELEN_GPU_DEVICE:-0}"
TRAIN_RATIO="${ELEN_TRAIN_RATIO:-0.8}"
VAL_RATIO="${ELEN_VAL_RATIO:-0.1}"
TEST_RATIO="${ELEN_TEST_RATIO:-0.1}"

python -m src.build_dataset \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --episodes "${EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --eval_set base \
  --prompt_format latent_plan \
  --feature_mode "${FEATURE_MODE}" \
  --feature_device "${FEATURE_DEVICE}" \
  --gpu_device "${GPU_DEVICE}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  "$@"

