#!/usr/bin/env bash
# 在 ELEN 目录下使用：bash scripts/download_navigation_dataset.sh
# 数据写入 ELEN/datasets/navigation/，并生成 eb_nav_train.h5 / eb_nav_test.h5（固定种子下随机 10% episode 为测试）。
# 环境变量：NAV_SPLIT_RATIO（默认 0.1）、NAV_SPLIT_SEED（默认 42）、NAV_OUT_H5 / NAV_OUT_H5_TEST 可改输出文件名。
# 依赖：conda activate elen（或已安装 huggingface_hub、h5py、Pillow）、unzip。
#
# 若转换阶段出现 “Killed”，多为内存不足（OOM）。可追加参数减小占用，例如：
#   bash scripts/download_navigation_dataset.sh --resize 224 224
#   bash scripts/download_navigation_dataset.sh --max-episodes 500
# 额外参数会传给 python -m ... dataset（见 dataset.py --help）。
# 并行加载 episode（默认多线程）：bash scripts/download_navigation_dataset.sh --workers 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 脚本位于 ELEN/scripts/，仓库根即 ELEN
ELEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${NAV_DATASET_DIR:-${ELEN_DIR}/datasets/navigation}"
JSON_NAME="${NAV_JSON_NAME:-eb-nav_dataset_multi_step.json}"
OUT_H5_NAME="${NAV_OUT_H5:-eb_nav_train.h5}"
OUT_H5_TEST_NAME="${NAV_OUT_H5_TEST:-eb_nav_test.h5}"
# 训练/测试划分：固定种子下随机 10% episode 作为测试集（可覆盖 NAV_SPLIT_RATIO / NAV_SPLIT_SEED）
SPLIT_RATIO="${NAV_SPLIT_RATIO:-0.1}"
SPLIT_SEED="${NAV_SPLIT_SEED:-42}"

cd "${ELEN_DIR}"

export PYTHONPATH="${ELEN_DIR}:${PYTHONPATH:-}"

mkdir -p "${DATASET_DIR}"

echo "[download_navigation_dataset] ELEN_DIR=${ELEN_DIR}"
echo "[download_navigation_dataset] DATASET_DIR=${DATASET_DIR}"
echo "[download_navigation_dataset] split: ratio=${SPLIT_RATIO} seed=${SPLIT_SEED} → train=${OUT_H5_NAME} test=${OUT_H5_TEST_NAME}"

python -m src.env.navigation.dataset --download "${DATASET_DIR}"

JSON_PATH="${DATASET_DIR}/${JSON_NAME}"
if [[ ! -f "${JSON_PATH}" ]]; then
  echo "错误: 未找到 ${JSON_PATH}，请检查 HF 仓库结构或设置 NAV_JSON_NAME。" >&2
  exit 1
fi

# 与 HF README 一致：解压到 images/ 子目录，使路径与 JSON 中 images/... 对齐
if [[ -f "${DATASET_DIR}/images.zip" ]] && [[ ! -d "${DATASET_DIR}/images" ]]; then
  echo "[download_navigation_dataset] 解压 images.zip -> images/ ..."
  mkdir -p "${DATASET_DIR}/images"
  unzip -q -o "${DATASET_DIR}/images.zip" -d "${DATASET_DIR}/images" || {
    echo "警告: unzip 失败，请手动: mkdir -p ${DATASET_DIR}/images && unzip ${DATASET_DIR}/images.zip -d ${DATASET_DIR}/images" >&2
  }
fi

echo "[download_navigation_dataset] 转换为 HDF5 ..."
python -m src.env.navigation.dataset \
  --json "${JSON_PATH}" \
  --images-root "${DATASET_DIR}" \
  --output "${DATASET_DIR}/${OUT_H5_NAME}" \
  --split-test-ratio "${SPLIT_RATIO}" \
  --split-seed "${SPLIT_SEED}" \
  --output-test "${DATASET_DIR}/${OUT_H5_TEST_NAME}" \
  "$@"

echo "[download_navigation_dataset] 完成: ${DATASET_DIR}/${OUT_H5_NAME}（训练） ${DATASET_DIR}/${OUT_H5_TEST_NAME}（测试，${SPLIT_RATIO}）"
