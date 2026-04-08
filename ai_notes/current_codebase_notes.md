# AI Notes - Current Codebase (ELEN)

最后更新：2026-04-08（基于当前源码树重读）

## 1) 当前主线功能

项目当前可跑通两阶段 navigation 流程：

1. 阶段1（LeWM/JEPA）：基于 EB-Nav HDF5（`pixels` + `action`）训练 world model。
2. 阶段2（Terminal Scorer）：冻结 JEPA + 冻结 Qwen embedding，训练 `NavigationTerminalScorer` 判断终止状态。

## 2) 关键目录与文件（当前有效）

- `src/env/navigation/dataset.py`
  - EB-Nav JSON -> HDF5 转换入口。
  - 支持下载 HuggingFace 数据集、并行读取 episode、进度条、train/test split。
  - HDF5 输出字段：`ep_len`, `ep_offset`, `pixels`, `action`, `instruction`, `is_end`。
- `src/wm/critic.py`
  - `NavigationTerminalScorer`：MLP 二分类头，输入 `latent + text_emb`，输出 logits。
- `src/dev/train_navigation_scorer.py`
  - 阶段2训练脚本（冻结 JEPA 与 Qwen embedding）。
- `src/dev/test_navigation_scorer.py`
  - 阶段2验证脚本（不训练，仅评估 checkpoint）。
- `lewm/config/train/data/eb_nav.yaml`
  - `dataset.name: eb_nav_train`
  - `keys_to_load: [pixels, action]`
  - 与 LeWM 训练兼容。
- `scripts/`
  - `download_navigation_dataset.sh`：下载+解压+转换，并默认按固定种子划分 train/test。
  - `train_navigation_lewm.sh` / `test_navigation_lewm.sh`：LeWM 训练与 smoke test。
  - `train_navigation_scorer.sh` / `test_navigation_scorer.sh`：评分器训练与验证。
  - `train_navigation_two_stage.sh`：顺序执行 LeWM -> scorer。

## 3) dataset 转换脚本要点（`src/env/navigation/dataset.py`）

### 3.1 动作映射

- 优先使用 `action` 的第一个元素（动作 ID，0..7）映射到：
  `moveahead/moveback/moveright/moveleft/rotateright/rotateleft/lookup/lookdown`。
- 兼容字符串动作名及常见别名。

### 3.2 图像与兼容性

- 需要 `images_root` 下存在 JSON 引用的图像路径（通常 `images/...`）。
- Pillow 兼容：优先 `Image.Resampling.BILINEAR`，旧版本回退 `Image.BILINEAR`。

### 3.3 监督标签

- `instruction`：从 episode 级字段（`instruction/task_instruction/...`）优先提取；缺失时尝试 trajectory 第一个条目。
- `is_end`：仅当 episode 成功(`success>=0.5`)且 timestep 为该 episode 最后一步时置 `1.0`，其余为 `0.0`。

### 3.4 性能策略

- 采用“episode 级并行加载 + 主线程写 HDF5”策略：
  - `--workers N` 控制并行线程（默认 `min(8, cpu_count)`）。
  - HDF5 始终单线程写入，避免线程安全问题。
- `tqdm` 进度条可见，`--no-progress` 可关闭。

### 3.5 CLI（常用）

- `--download DIR`：仅下载数据集仓库到 DIR。
- `--max-episodes N`：仅转换前 N 个 episode（快速冒烟）。
- `--split-test-ratio P` + `--split-seed` + `--output-test`：按 episode 随机划分 train/test。
- `--resize W H`：统一分辨率。

## 4) 脚本约定（`scripts/*.sh`）

### 4.1 下载与转换

- `bash scripts/download_navigation_dataset.sh`
- 默认行为：
  - 下载到 `ELEN/datasets/navigation`
  - 解压 `images.zip` 到 `images/`
  - 生成：
    - train: `eb_nav_train.h5`
    - test: `eb_nav_test.h5`（默认 10% episode，`NAV_SPLIT_RATIO=0.1`）

### 4.2 LeWM 训练读取路径约定

- `train_navigation_lewm.sh` 将 `STABLEWM_HOME` 默认设为 `ELEN`。
- stable_worldmodel 实际按 `$STABLEWM_HOME/<name>.h5` 读取；因此脚本会链接：
  - `${STABLEWM_HOME}/eb_nav_train.h5` -> `${ELEN}/datasets/navigation/eb_nav_train.h5`

### 4.3 两阶段训练

- 一键：`bash scripts/train_navigation_two_stage.sh`
- 阶段2依赖：
  - LeWM 的 `*_object.ckpt`
  - 含 `instruction` 与 `is_end` 的 HDF5（需用当前 dataset.py 重新转换）

## 5) 当前已知注意点

1. 若出现 `Killed`，通常是内存不足（OOM）；优先尝试：
   - `--resize 224 224`
   - `--max-episodes N`
   - 调小 `--workers`
2. 若出现 `No valid episodes after conversion`，优先检查：
   - 图像是否存在（`images/` 目录）
   - JSON 与图片路径是否匹配
3. 若只想快速验证训练链路，先转小集：
   - `--max-episodes 20`

## 6) 快速命令索引（ELEN 根目录）

```bash
# 1) 下载+转换（含默认 train/test 划分）
bash scripts/download_navigation_dataset.sh --resize 224 224

# 2) 仅小规模冒烟（20 episodes）
python -m src.env.navigation.dataset \
  --json datasets/navigation/eb-nav_dataset_multi_step.json \
  --images-root datasets/navigation \
  --output datasets/navigation/eb_nav_train_20ep.h5 \
  --max-episodes 20 --resize 224 224

# 3) LeWM 训练
LEWM_EXTRA_ARGS="wandb.enabled=false" bash scripts/train_navigation_lewm.sh

# 4) 评分器训练
bash scripts/train_navigation_scorer.sh
```

## 7) 与 AI_README 的一致性备注

- 代码主线已集中在 `src/`, `scripts/`, `lewm/config/train/data/eb_nav.yaml`。
- `src_old/` 属历史代码，默认不作为当前实现依据。
- `lewm/` 与 `vagen/` 保持“非必要不修改”的策略。
