#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/project/peilab/atst"
ELEN_DIR="${ROOT_DIR}/elen"

export PYTHONPATH="${ELEN_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

DATASET_DIR="${ELEN_DATASET_DIR:-${ELEN_DIR}/datasets/navigation_transition}"
FEATURE_MODE="${ELEN_FEATURE_MODE:-both}"
SAVE_DIR="${ELEN_SAVE_DIR:-${ELEN_DIR}/exps/ffnn_transition_${FEATURE_MODE}}"
DEVICE="${ELEN_DEVICE:-cpu}"
BATCH_SIZE="${ELEN_BATCH_SIZE:-64}"
EPOCHS="${ELEN_EPOCHS:-10}"
LR="${ELEN_LR:-1e-3}"
WEIGHT_DECAY="${ELEN_WEIGHT_DECAY:-0.0}"
ACTION_EMBED_DIM="${ELEN_ACTION_EMBED_DIM:-32}"
HIDDEN_DIM="${ELEN_HIDDEN_DIM:-1024}"
DROPOUT="${ELEN_DROPOUT:-0.1}"
COSINE_LOSS_WEIGHT="${ELEN_COSINE_LOSS_WEIGHT:-0.0}"

python -m src.train \
  --train_parquet "${DATASET_DIR}/train.parquet" \
  --val_parquet "${DATASET_DIR}/val.parquet" \
  --feature_mode "${FEATURE_MODE}" \
  --save_dir "${SAVE_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --action_embed_dim "${ACTION_EMBED_DIM}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --cosine_loss_weight "${COSINE_LOSS_WEIGHT}" \
  "$@"

