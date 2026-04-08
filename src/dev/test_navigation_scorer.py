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

from src.dev.train_navigation_scorer import (  # noqa: E402
    NavigationTerminalScorer,
    _H5RowDataset,
    _QwenTextEmbedder,
    _build_pixel_transform,
    _collate_scorer_batch,
    _compute_pos_weight_for_indices,
    _infer_embed_dim,
    _load_jepa,
)

_LOGGER = logging.getLogger(__name__)


def _binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.to(dtype=torch.float32).flatten().cpu()
    scores = scores.to(dtype=torch.float32).flatten().cpu()
    n = int(labels.numel())
    if n == 0:
        return 0.0
    n_pos = int((labels >= 0.5).sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = torch.zeros(n, dtype=torch.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    sum_pos_ranks = float(ranks[sorted_labels >= 0.5].sum().item())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate_with_metrics(
    jepa,
    text_embedder: _QwenTextEmbedder,
    scorer: NavigationTerminalScorer,
    loader: DataLoader,
    pixel_transform,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    scorer.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss, n = 0.0, 0
    tp = fp = tn = fn = 0
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    use_amp = device.type == "cuda"

    for batch in loader:
        pixels_b1chw, texts, y = _collate_scorer_batch(batch, pixel_transform, device)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                enc = jepa.encode({"pixels": pixels_b1chw})
        else:
            enc = jepa.encode({"pixels": pixels_b1chw})
        latent = enc["emb"][:, 0, :].float()
        t_emb = text_embedder(texts).float()
        logits = scorer(latent, t_emb)
        loss = loss_fn(logits.float(), y)
        total_loss += float(loss) * y.size(0)
        n += y.size(0)

        pred = (torch.sigmoid(logits) >= 0.5).float()
        all_scores.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(y.detach().cpu())
        tp += int(((pred == 1) & (y == 1)).sum().item())
        fp += int(((pred == 1) & (y == 0)).sum().item())
        tn += int(((pred == 0) & (y == 0)).sum().item())
        fn += int(((pred == 0) & (y == 1)).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(n, 1)
    val_loss = total_loss / max(n, 1)
    auroc = _binary_auroc(torch.cat(all_labels), torch.cat(all_scores))
    return {
        "val_loss": val_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


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

    metrics = evaluate_with_metrics(
        jepa,
        text_embedder,
        scorer,
        val_loader,
        pixel_transform,
        device,
        pw,
    )
    _LOGGER.info(
        "val_loss=%.6f accuracy=%.6f precision=%.6f recall=%.6f f1=%.6f auroc=%.6f",
        metrics["val_loss"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auroc"],
    )
    print(
        "val_loss={:.6f} accuracy={:.6f} precision={:.6f} recall={:.6f} f1={:.6f} auroc={:.6f}".format(
            metrics["val_loss"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auroc"],
        )
    )

    full.close()


if __name__ == "__main__":
    main()
