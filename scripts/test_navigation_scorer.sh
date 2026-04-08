#!/usr/bin/env bash
# 在 ELEN 根目录运行 — bash scripts/test_navigation_scorer.sh
# 在验证划分上评估已训练的 navigation_scorer（需与 train 相同的 HDF5、JEPA ckpt、随机种子）。
# 环境变量（与 train_navigation_scorer.sh 对齐）：
#   LEWM_JEPA_CKPT   — LeWM *_object.ckpt（可自动探测）
#   NAV_H5_PATH      — 默认 ELEN/datasets/navigation/eb_nav_train.h5
#   SCORER_CKPT      — 默认 ELEN/models/navigation_scorer/navigation_scorer_best.pt
#   SCORER_OUT_DIR   — 若未设置 SCORER_CKPT，则在此目录下找 navigation_scorer_best.pt
#   QWEN_EMBED_MODEL — 与训练时一致
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
  echo "错误: 未找到 HDF5 ${H5_PATH}。" >&2
  exit 1
fi

JEPA_CKPT="${LEWM_JEPA_CKPT:-}"
if [[ -z "${JEPA_CKPT}" ]]; then
  if compgen -G "${RUN_DIR}"/*_object.ckpt > /dev/null; then
    JEPA_CKPT="$(ls -t "${RUN_DIR}"/*_object.ckpt | head -1)"
    echo "[test_navigation_scorer] 使用探测到的 JEPA checkpoint: ${JEPA_CKPT}"
  fi
fi
if [[ -z "${JEPA_CKPT}" || ! -f "${JEPA_CKPT}" ]]; then
  echo "错误: 请设置 LEWM_JEPA_CKPT 为 LeWM 的 *_object.ckpt。" >&2
  exit 1
fi

SCORER_CKPT="${SCORER_CKPT:-}"
if [[ -z "${SCORER_CKPT}" ]]; then
  if [[ -f "${OUT_DIR}/navigation_scorer_best.pt" ]]; then
    SCORER_CKPT="${OUT_DIR}/navigation_scorer_best.pt"
  elif [[ -f "${OUT_DIR}/navigation_scorer_last.pt" ]]; then
    SCORER_CKPT="${OUT_DIR}/navigation_scorer_last.pt"
  fi
fi
if [[ -z "${SCORER_CKPT}" || ! -f "${SCORER_CKPT}" ]]; then
  echo "错误: 未找到评分器权重。请设置 SCORER_CKPT，或先训练并生成 ${OUT_DIR}/navigation_scorer_best.pt" >&2
  exit 1
fi

echo "[test_navigation_scorer] H5_PATH=${H5_PATH}"
echo "[test_navigation_scorer] JEPA_CKPT=${JEPA_CKPT}"
echo "[test_navigation_scorer] SCORER_CKPT=${SCORER_CKPT}"

cd "${ELEN_DIR}"
exec python -m src.wm.test_navigation_scorer \
  --h5-path "${H5_PATH}" \
  --jepa-ckpt "${JEPA_CKPT}" \
  --scorer-ckpt "${SCORER_CKPT}" \
  ${SCORER_TEST_EXTRA_ARGS:-} "$@"
