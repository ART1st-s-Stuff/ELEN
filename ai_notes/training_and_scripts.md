# Training And Scripts

## 脚本清单

- `scripts/download_navigation_dataset.sh`
- `scripts/train_navigation_lewm.sh`
- `scripts/test_navigation_lewm.sh`
- `scripts/train_navigation_scorer.sh`
- `scripts/test_navigation_scorer.sh`
- `scripts/train_navigation_two_stage.sh`

## 路径约定

- `train_navigation_lewm.sh` 默认将 `STABLEWM_HOME` 设为 `ELEN`。
- 训练读取遵循 `$STABLEWM_HOME/<name>.h5`，脚本会处理到 `eb_nav_train.h5` 的链接。

## 两阶段依赖

- 阶段1：LeWM 生成 `*_object.ckpt`。
- 阶段2：需要带 `instruction` 与 `is_end` 的 HDF5（由当前 `dataset.py` 转换）。

## 快速命令索引

```bash
bash scripts/download_navigation_dataset.sh --resize 224 224
LEWM_EXTRA_ARGS="wandb.enabled=false" bash scripts/train_navigation_lewm.sh
bash scripts/train_navigation_scorer.sh
```
