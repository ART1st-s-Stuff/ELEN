#!/usr/bin/env bash
# 在 ELEN 目录下使用：bash scripts/test_navigation_lewm.sh
# 对 navigation LeWM 做冒烟 / 快速自检（默认 Lightning fast_dev_run，不跑完整训练）。
# 环境变量与 train_navigation_lewm.sh 一致。
# 默认测试参数见下方 LEWM_TEST_EXTRA_ARGS；勿与训练用的 LEWM_EXTRA_ARGS 混用。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEWM_DIR="${LEWM_DIR:-${ELEN_DIR}/lewm}"
MODEL_SUBDIR="${LEWM_MODEL_SUBDIR:-models/lewm/navigation}"
DATASET_NAV="${NAV_DATASET_DIR:-${ELEN_DIR}/datasets/navigation}"
H5_NAME="${NAV_OUT_H5:-eb_nav_test.h5}"
H5_PATH="${DATASET_NAV}/${H5_NAME}"

export STABLEWM_HOME="${STABLEWM_HOME:-${ELEN_DIR}}"
mkdir -p "${STABLEWM_HOME}/datasets"

if [[ ! -f "${H5_PATH}" ]]; then
  echo "错误: 未找到 ${H5_PATH}。请先运行: bash scripts/download_navigation_dataset.sh" >&2
  exit 1
fi

if [[ ! -e "${STABLEWM_HOME}/${H5_NAME}" ]]; then
  ln -sfn "$(realpath "${H5_PATH}")" "${STABLEWM_HOME}/${H5_NAME}"
fi
if [[ ! -e "${STABLEWM_HOME}/datasets/${H5_NAME}" ]]; then
  ln -sfn "$(realpath "${H5_PATH}")" "${STABLEWM_HOME}/datasets/${H5_NAME}"
fi

if [[ ! -f "${LEWM_DIR}/train.py" ]]; then
  echo "错误: 未找到 ${LEWM_DIR}/train.py。请在 ELEN 下初始化子模块: git submodule update --init lewm" >&2
  exit 1
fi

cd "${LEWM_DIR}"
export PYTHONPATH="${ELEN_DIR}:${PYTHONPATH:-}"

WANDB_SAFE_ID="${WANDB_RUN_ID:-${MODEL_SUBDIR//\//_}}"
# 默认：Lightning fast_dev_run（跑通数据与一步验证）；覆盖示例：LEWM_TEST_EXTRA_ARGS="wandb.enabled=false trainer.max_epochs=1"
DEFAULT_TEST_ARGS="trainer.fast_dev_run=true wandb.enabled=false"
LEWM_TEST_EXTRA_ARGS="${LEWM_TEST_EXTRA_ARGS:-${DEFAULT_TEST_ARGS}}"

echo "[test_navigation_lewm] STABLEWM_HOME=${STABLEWM_HOME}"
echo "[test_navigation_lewm] LEWM_DIR=${LEWM_DIR}"
echo "[test_navigation_lewm] LEWM_TEST_EXTRA_ARGS=${LEWM_TEST_EXTRA_ARGS}"

exec python train.py data=eb_nav "subdir=${MODEL_SUBDIR}" "wandb.config.id=${WANDB_SAFE_ID}" ${LEWM_TEST_EXTRA_ARGS} "$@"
