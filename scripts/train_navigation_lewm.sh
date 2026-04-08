#!/usr/bin/env bash
# 在 ELEN 目录下使用：bash scripts/train_navigation_lewm.sh
# 检查点写入 ELEN/models/lewm/navigation/；需先有 ELEN/datasets/navigation/eb_nav_train.h5。
# STABLEWM_HOME 设为 ELEN。stable_worldmodel 从 ``$STABLEWM_HOME/<name>.h5`` 读 HDF5（与 datasets/ 子目录无关），
# 因此必须在 ``$STABLEWM_HOME/eb_nav_train.h5`` 放置文件或符号链接。
# 若需阶段 2 训练「结束状态」评分器，请用更新后的 dataset 转换脚本生成含 instruction / is_end 的 HDF5。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEWM_DIR="${LEWM_DIR:-${ELEN_DIR}/lewm}"
MODEL_DIR="${LEWM_MODEL_DIR:-${ELEN_DIR}/models/lewm/navigation}"
MODEL_SUBDIR="${LEWM_MODEL_SUBDIR:-models/lewm/navigation}"
DATASET_NAV="${NAV_DATASET_DIR:-${ELEN_DIR}/datasets/navigation}"
H5_NAME="${NAV_OUT_H5:-eb_nav_train.h5}"
H5_PATH="${DATASET_NAV}/${H5_NAME}"

export STABLEWM_HOME="${STABLEWM_HOME:-${ELEN_DIR}}"
mkdir -p "${STABLEWM_HOME}/datasets" "${MODEL_DIR}"

if [[ ! -f "${H5_PATH}" ]]; then
  echo "错误: 未找到 ${H5_PATH}。请先运行: bash scripts/download_navigation_dataset.sh" >&2
  exit 1
fi

# LeWM 实际打开的路径：cache_dir 默认为 get_cache_dir() == STABLEWM_HOME → eb_nav_train.h5
if [[ ! -e "${STABLEWM_HOME}/${H5_NAME}" ]]; then
  ln -sfn "$(realpath "${H5_PATH}")" "${STABLEWM_HOME}/${H5_NAME}"
fi
# 可选：同时在 datasets/ 下保留一份链接，便于浏览
if [[ ! -e "${STABLEWM_HOME}/datasets/${H5_NAME}" ]]; then
  ln -sfn "$(realpath "${H5_PATH}")" "${STABLEWM_HOME}/datasets/${H5_NAME}"
fi

if [[ ! -f "${LEWM_DIR}/train.py" ]]; then
  echo "错误: 未找到 ${LEWM_DIR}/train.py。请在 ELEN 下初始化子模块: git submodule update --init lewm" >&2
  exit 1
fi

if [[ ! -f "${LEWM_DIR}/config/train/data/eb_nav.yaml" ]]; then
  echo "警告: 未找到 ${LEWM_DIR}/config/train/data/eb_nav.yaml；data=eb_nav 会失败。" >&2
  echo "  请在 lewm 中新增该配置（dataset.name: eb_nav_train，仅 pixels/action），参见项目 transcript.md。" >&2
fi

cd "${LEWM_DIR}"
export PYTHONPATH="${ELEN_DIR}:${PYTHONPATH:-}"

echo "[train_navigation_lewm] STABLEWM_HOME=${STABLEWM_HOME}"
echo "[train_navigation_lewm] LEWM_DIR=${LEWM_DIR}"
echo "[train_navigation_lewm] checkpoint/run_dir ≈ ${STABLEWM_HOME}/${MODEL_SUBDIR}"

# lewm 默认 ``wandb.config.id=${subdir}``；subdir 含 ``/`` 时 wandb 会报错（Run ID 不允许 ``/`` 等字符）。
WANDB_SAFE_ID="${WANDB_RUN_ID:-${MODEL_SUBDIR//\//_}}"
echo "[train_navigation_lewm] wandb.config.id=${WANDB_SAFE_ID}" >&2

exec python train.py data=eb_nav "subdir=${MODEL_SUBDIR}" "wandb.config.id=${WANDB_SAFE_ID}" ${LEWM_EXTRA_ARGS:-} "$@"
