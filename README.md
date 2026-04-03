# FFNN State Transition Test Project

本项目在 `elen/` 下实现一个用于预测状态转移的 FFNN 测试流程，数据格式为 `(Img, Action, Img_next)`，并使用离线图像特征。

## 1. 数据格式

构建脚本输出 `train.parquet`、`val.parquet`、`test.parquet`，每行样本包含：

- `image_path`: 当前图像路径
- `action_name`: 动作名称
- `action_id`: 动作离散 ID（用于 embedding）
- `next_image_path`: 下一时刻图像路径
- `img_feat`: `Img` 特征向量（CLIP / MAE / 拼接）
- `next_img_feat`: `Img_next` 特征向量
- `reward`, `done`, `success`, `episode_id`, `step_id`, `feature_mode`

## 2. 特征模式

通过 `--feature_mode` 控制：

- `clip`: 仅 CLIP 特征
- `mae`: 仅 MAE 特征
- `both`: CLIP + MAE 拼接特征

## 3. 模型定义

- 输入：`state_feat` + `action_embedding(action_id)`
- 输出：预测 `next_state_feat`
- 损失：`MSELoss`（可选余弦项加权）

## 4. 运行流程

先安装依赖：

```bash
cd /project/peilab/atst/elen
pip install -r requirements.txt
```

### 4.1 构建数据集（随机策略，来源 VAGEN navigation）

```bash
bash /project/peilab/atst/elen/scripts/build_navigation_transition_dataset.sh
```

示例（小规模 + CLIP）：

```bash
ELEN_EPISODES=10 ELEN_FEATURE_MODE=clip bash /project/peilab/atst/elen/scripts/build_navigation_transition_dataset.sh
```

### 4.2 训练

```bash
bash /project/peilab/atst/elen/scripts/train_transition.sh
```

示例（MAE + 2 epoch）：

```bash
ELEN_FEATURE_MODE=mae ELEN_EPOCHS=2 bash /project/peilab/atst/elen/scripts/train_transition.sh
```

### 4.3 测试

```bash
bash /project/peilab/atst/elen/scripts/test_transition.sh
```

示例（both + 指定 checkpoint）：

```bash
ELEN_FEATURE_MODE=both ELEN_CHECKPOINT=/project/peilab/atst/elen/exps/ffnn_transition_both/best.pt \
  bash /project/peilab/atst/elen/scripts/test_transition.sh
```

## 5. 小规模 smoke test 建议

构建时将 episode 降到 10，训练设置 1-2 个 epoch，用于快速验证全流程和三种特征模式切换。

## 6. 常用环境变量

- `ELEN_FEATURE_MODE`: `clip` / `mae` / `both`
- `ELEN_FEATURE_DEVICE`: 特征提取设备（如 `cpu`、`cuda:0`）
- `ELEN_DEVICE`: 训练/测试设备（如 `cpu`、`cuda:0`）
- `ELEN_EPISODES`, `ELEN_MAX_STEPS`: 数据规模
- `ELEN_EPOCHS`, `ELEN_BATCH_SIZE`, `ELEN_LR`: 训练超参
- 三个脚本都支持命令行透传参数，例如：`bash ... --feature_mode clip`

