#!/usr/bin/env bash
# 阶段 2：在 ELEN 根目录运行 — bash scripts/train_navigation_scorer.sh
# 需已训练 LeWM 并得到 *_object.ckpt，以及含 instruction / is_end 的 HDF5（请重新跑 dataset 转换）。
# 环境变量：
#   LEWM_JEPA_CKPT  — 指向 lewm_epoch_*_object.ckpt（必填，除非下面自动探测成功）
#   NAV_H5_PATH     — 默认 ELEN/datasets/navigation/eb_nav_train.h5
#   SCORER_OUT_DIR  — 默认 ELEN/models/navigation_scorer
#   QWEN_EMBED_MODEL — 默认 Qwen/Qwen3-Embedding-0.6B（可与 Qwen3.5 系 embedding 模型互换）
#   SCORER_WANDB_ENABLED/SCORER_WANDB_ENTITY/SCORER_WANDB_PROJECT/
#   SCORER_WANDB_RUN_NAME/SCORER_WANDB_RUN_ID — 阶段2 wandb 配置（可选）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEWM_DIR="${LEWM_DIR:-${ELEN_DIR}/lewm}"
MODEL_SUBDIR="${LEWM_MODEL_SUBDIR:-models/lewm/navigation}"
DATASET_NAV="${NAV_DATASET_DIR:-${ELEN_DIR}/datasets/navigation}"
H5_NAME="${NAV_OUT_H5:-eb_nav_train.h5}"

export STABLEWM_HOME="${STABLEWM_HOME:-${ELEN_DIR}}"
export PYTHONPATH="${ELEN_DIR}:${LEWM_DIR}:${PYTHONPATH:-}"

H5_PATH="${NAV_H5_PATH:-${DATASET_NAV}/${H5_NAME}}"
OUT_DIR="${SCORER_OUT_DIR:-${ELEN_DIR}/models/navigation_scorer}"
RUN_DIR="${STABLEWM_HOME}/${MODEL_SUBDIR}"

if [[ ! -f "${H5_PATH}" ]]; then
  echo "错误: 未找到 HDF5 ${H5_PATH}。请先转换数据集（含 instruction / is_end）。" >&2
  exit 1
fi

JEPA_CKPT="${LEWM_JEPA_CKPT:-}"
if [[ -z "${JEPA_CKPT}" ]]; then
  if compgen -G "${RUN_DIR}"/*_object.ckpt > /dev/null; then
    JEPA_CKPT="$(ls -t "${RUN_DIR}"/*_object.ckpt | head -1)"
    echo "[train_navigation_scorer] 使用探测到的 checkpoint: ${JEPA_CKPT}"
  fi
fi
if [[ -z "${JEPA_CKPT}" || ! -f "${JEPA_CKPT}" ]]; then
  echo "错误: 请设置 LEWM_JEPA_CKPT 为 LeWM 的 *_object.ckpt（例如 ${RUN_DIR}/lewm_epoch_100_object.ckpt）。" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[train_navigation_scorer] H5_PATH=${H5_PATH}"
echo "[train_navigation_scorer] JEPA_CKPT=${JEPA_CKPT}"
echo "[train_navigation_scorer] OUT_DIR=${OUT_DIR}"

cd "${ELEN_DIR}"
DEFAULT_SCORER_EPOCHS="${SCORER_EPOCHS:-20}"

exec python -m src.dev.train_navigation_scorer \
  --h5-path "${H5_PATH}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --epochs "${DEFAULT_SCORER_EPOCHS}" \
  --output-dir "${OUT_DIR}" \
  --wandb-enabled "${SCORER_WANDB_ENABLED:-false}" \
  --wandb-entity "${SCORER_WANDB_ENTITY:-}" \
  --wandb-project "${SCORER_WANDB_PROJECT:-navigation_scorer}" \
  --wandb-run-name "${SCORER_WANDB_RUN_NAME:-}" \
  --wandb-run-id "${SCORER_WANDB_RUN_ID:-}" \
  ${SCORER_EXTRA_ARGS:-} "$@"
