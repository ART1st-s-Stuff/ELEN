#!/usr/bin/env bash
# 顺序执行：阶段 1 LeWM → 阶段 2 评分器 MLP。
# 在 ELEN 根目录：bash scripts/train_navigation_two_stage.sh
# 阶段 1 参数可附加在末尾，会传给 train_navigation_lewm.sh。
# 阶段 2 使用 train_navigation_scorer.sh 的环境变量（如 LEWM_JEPA_CKPT、SCORER_OUT_DIR、SCORER_EXTRA_ARGS）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 阶段 1: LeWM (JEPA) ==="
bash "${SCRIPT_DIR}/train_navigation_lewm.sh" "$@"

echo "=== 阶段 2: Navigation 评分器 (MLP) ==="
# 若未显式设置，则尝试用刚训练目录下最新的 *_object.ckpt
export STABLEWM_HOME="${STABLEWM_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_SUBDIR="${LEWM_MODEL_SUBDIR:-models/lewm/navigation}"
RUN_DIR="${STABLEWM_HOME}/${MODEL_SUBDIR}"
if [[ -z "${LEWM_JEPA_CKPT:-}" ]] && compgen -G "${RUN_DIR}"/*_object.ckpt > /dev/null; then
  export LEWM_JEPA_CKPT="$(ls -t "${RUN_DIR}"/*_object.ckpt | head -1)"
  echo "[two_stage] LEWM_JEPA_CKPT=${LEWM_JEPA_CKPT}"
fi

bash "${SCRIPT_DIR}/train_navigation_scorer.sh"
