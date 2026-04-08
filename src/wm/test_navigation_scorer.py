"""在 HDF5 验证划分上评估已训练的 NavigationTerminalScorer（不训练）。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

_ELEN_ROOT = Path(__file__).resolve().parents[2]
_LEWM_ROOT = _ELEN_ROOT / "lewm"
for _p in (_ELEN_ROOT, _LEWM_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from src.wm.train_navigation_scorer import (  # noqa: E402
    NavigationTerminalScorer,
    _H5RowDataset,
    _QwenTextEmbedder,
    _build_pixel_transform,
    _collate_scorer_batch,
    _compute_pos_weight_for_indices,
    _infer_embed_dim,
    _load_jepa,
    evaluate,
)

_LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NavigationTerminalScorer checkpoint.")
    p.add_argument("--h5-path", type=Path, required=True)
    p.add_argument("--jepa-ckpt", type=Path, required=True)
    p.add_argument(
        "--scorer-ckpt",
        type=Path,
        required=True,
        help="navigation_scorer_best.pt 或 navigation_scorer_last.pt",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="LeWM Hydra config.yaml（默认：jepa ckpt 同目录）",
    )
    p.add_argument(
        "--qwen-model",
        type=str,
        default=os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="默认：按训练划分自动 neg/pos，与 train_navigation_scorer 一致",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info("device=%s", device)

    cfg_path = args.config or (args.jepa_ckpt.parent / "config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"未找到 LeWM 配置: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    img_size = int(cfg.img_size)

    jepa = _load_jepa(args.jepa_ckpt, device)
    pixel_transform = _build_pixel_transform(img_size)
    embed_dim = _infer_embed_dim(jepa, img_size, device)

    text_embedder = _QwenTextEmbedder(args.qwen_model, device)

    scorer = NavigationTerminalScorer(
        latent_dim=embed_dim,
        text_dim=text_embedder.text_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.mlp_layers,
        dropout=args.dropout,
    ).to(device)

    payload = torch.load(args.scorer_ckpt, map_location=device, weights_only=False)
    if "scorer" not in payload:
        raise KeyError(f"{args.scorer_ckpt} 缺少键 'scorer'")
    scorer.load_state_dict(payload["scorer"], strict=True)

    full = _H5RowDataset(args.h5_path)
    n_train = int(len(full) * args.train_split)
    n_val = len(full) - n_train
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.pos_weight is not None:
        pw = torch.tensor([args.pos_weight], device=device)
    else:
        tr_idx = list(train_ds.indices)  # type: ignore[attr-defined]
        ratio = _compute_pos_weight_for_indices(full._path, tr_idx)
        _LOGGER.info("BCE pos_weight (auto)=%.4f", ratio)
        pw = torch.tensor([ratio], device=device)

    val_loss, val_acc = evaluate(
        jepa,
        text_embedder,
        scorer,
        val_loader,
        pixel_transform,
        device,
        pw,
    )
    _LOGGER.info("val_loss=%.6f val_acc=%.6f", val_loss, val_acc)
    print(f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}")

    full.close()


if __name__ == "__main__":
    main()
