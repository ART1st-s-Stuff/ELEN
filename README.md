bash scripts/download_navigation_dataset.sh --resize 224 224 --max-episodes 3000
LEWM_EXTRA_ARGS="wandb.enabled=false" bash scripts/train_navigation_lewm.sh
LEWM_EXTRA_ARGS="wandb.enabled=false" bash scripts/train_navigation_scorer.sh