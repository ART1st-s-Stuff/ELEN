# Architecture And Flow

## 当前主线

- 阶段1（LeWM/JEPA）：基于 EB-Nav HDF5（`pixels` + `action`）训练 world model。
- 阶段2（Terminal Scorer）：冻结 JEPA + 冻结 Qwen embedding，训练终止状态二分类器。

## 关键路径

- `src/env/navigation/dataset.py`：数据转换入口。
- `src/wm/critic.py`：终止状态评分器。
- `src/dev/train_navigation_scorer.py`：评分器训练。
- `src/dev/test_navigation_scorer.py`：评分器验证。
- `lewm/config/train/data/eb_nav.yaml`：LeWM 训练数据配置。

## 与 AI_README 对齐

- 主线集中在 `src/`、`scripts/`、`lewm/config/train/data/eb_nav.yaml`。
- `src_old/` 仅历史代码。
- `lewm/`、`vagen/` 默认非必要不修改。
