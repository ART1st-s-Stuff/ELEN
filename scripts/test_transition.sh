#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/project/peilab/atst"
ELEN_DIR="${ROOT_DIR}/elen"

export PYTHONPATH="${ELEN_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

DATASET_DIR="${ELEN_DATASET_DIR:-${ELEN_DIR}/datasets/navigation_transition}"
FEATURE_MODE="${ELEN_FEATURE_MODE:-both}"
SAVE_DIR="${ELEN_SAVE_DIR:-${ELEN_DIR}/exps/ffnn_transition_${FEATURE_MODE}}"
CHECKPOINT="${ELEN_CHECKPOINT:-${SAVE_DIR}/best.pt}"
DEVICE="${ELEN_DEVICE:-cpu}"
BATCH_SIZE="${ELEN_BATCH_SIZE:-64}"
ACTION_EMBED_DIM="${ELEN_ACTION_EMBED_DIM:-32}"
HIDDEN_DIM="${ELEN_HIDDEN_DIM:-1024}"
DROPOUT="${ELEN_DROPOUT:-0.1}"

python -m src.test \
  --test_parquet "${DATASET_DIR}/test.parquet" \
  --feature_mode "${FEATURE_MODE}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --action_embed_dim "${ACTION_EMBED_DIM}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  "$@"

