## Quickstart
```bash
# 1) 下载并转换 navigation 数据
bash scripts/download_navigation_dataset.sh --resize 224 224

# 2) 训练 LeWM（阶段1）
SCORER_WANDB_ENABLED=true \
SCORER_WANDB_ENTITY=<你的entity> \
SCORER_WANDB_PROJECT=navigation \
SCORER_WANDB_RUN_NAME=encoder_exp1 \
bash scripts/train_navigation_lewm.sh

# 3) 训练 terminal scorer（阶段2）
SCORER_WANDB_ENABLED=true \
SCORER_WANDB_ENTITY=<你的entity> \
SCORER_WANDB_PROJECT=navigation \
SCORER_WANDB_RUN_NAME=scorer_exp1 \
bash scripts/train_navigation_scorer.sh
```