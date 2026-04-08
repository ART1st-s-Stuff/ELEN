"""Train frozen LeWM latent + frozen Qwen embedding → MLP terminal scorer (phase 2)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)

# ELEN repo root (parent of ``src/``)
_ELEN_ROOT = Path(__file__).resolve().parents[2]
_LEWM_ROOT = _ELEN_ROOT / "lewm"
for _p in (_ELEN_ROOT, _LEWM_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import stable_pretraining as spt  # noqa: E402

from lewm.utils import get_img_preprocessor  # noqa: E402

from src.wm.critic import NavigationTerminalScorer  # noqa: E402


def _str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"非法布尔值: {v}")


def _none_if_empty(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_instruction(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    return str(raw)


class _H5RowDataset(Dataset):
    """Flat rows: ``pixels``, ``instruction``, ``is_end``."""

    def __init__(self, h5_path: str | Path) -> None:
        self._path = Path(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(h5_path, "r") as f:
            for k in ("pixels", "instruction", "is_end"):
                if k not in f:
                    raise KeyError(
                        f"{h5_path} 缺少数据集 '{k}'。请使用更新后的 "
                        "`python -m src.env.navigation.dataset ...` 重新转换 HDF5。"
                    )
            self._n = f["pixels"].shape[0]

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self._path, "r")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        assert self._h5 is not None
        pix = np.asarray(self._h5["pixels"][idx])
        instr = _decode_instruction(self._h5["instruction"][idx])
        y = float(np.asarray(self._h5["is_end"][idx]))
        return {"pixels": pix, "instruction": instr, "is_end": y}

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def _build_pixel_transform(img_size: int):
    # Match ``lewm/train.py``: one Compose wrapping ``get_img_preprocessor`` output.
    return spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    )


def _collate_scorer_batch(
    batch,
    pixel_transform,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Return ``pixels_b1chw`` (B,1,C,H,W), instructions, labels (B,).

    PyTorch 默认 ``collate_fn`` 会把 ``__getitem__`` 返回的 ``dict`` 合并成
    单个 ``dict``（``pixels`` 为 ``(B,H,W,C)`` 等），不能当作 ``list[dict]`` 遍历。
    此处同时支持合并后的 ``dict`` 与 ``list[dict]``。
    """
    if isinstance(batch, dict):
        px = batch["pixels"]
        if px.dim() == 3:
            px = px.unsqueeze(0)
        tensors: list[torch.Tensor] = []
        for i in range(px.shape[0]):
            arr = np.asarray(px[i].detach().cpu().numpy())
            d = pixel_transform({"pixels": arr})
            tensors.append(d["pixels"])
        raw_instr = batch["instruction"]
        if isinstance(raw_instr, (list, tuple)):
            texts = [str(t) for t in raw_instr]
        else:
            texts = [str(raw_instr)]
        y = batch["is_end"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        else:
            y = y.float().to(device)
        stacked = torch.stack(tensors, dim=0).unsqueeze(1).to(device)
        return stacked, texts, y

    tensors = []
    texts = []
    labels: list[float] = []
    for s in batch:
        d = pixel_transform({"pixels": s["pixels"]})
        tensors.append(d["pixels"])
        texts.append(str(s["instruction"]))
        labels.append(float(s["is_end"]))
    stacked = torch.stack(tensors, dim=0).unsqueeze(1).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return stacked, texts, y


class _QwenTextEmbedder:
    """Frozen Qwen / Qwen3.* embedding via transformers (mean pool over last hidden)."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 512,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        self._model.requires_grad_(False)
        self._device = device
        self._max_length = max_length
        # Infer hidden size once
        with torch.no_grad():
            t = self._tok("ping", return_tensors="pt")
            t = {k: v.to(device) for k, v in t.items()}
            out = self._model(**t)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                self._hidden = int(out.pooler_output.shape[-1])
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                self._hidden = int(out.last_hidden_state.shape[-1])
            else:
                raise RuntimeError(
                    "Qwen 模型前向未返回 pooler_output / last_hidden_state；请检查模型与 transformers 版本。"
                )

    @property
    def text_dim(self) -> int:
        return self._hidden

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        batch = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self._model(**batch)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        h = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand_as(h).float()
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (h * mask).sum(dim=1) / denom


def _load_jepa(ckpt_path: Path, device: torch.device):
    jepa = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    jepa = jepa.to(device)
    jepa.eval()
    jepa.requires_grad_(False)
    return jepa


def _infer_embed_dim(jepa, img_size: int, device: torch.device) -> int:
    """Run a dummy encode to read projector output size (``emb`` last dim)."""
    tf = _build_pixel_transform(img_size)
    arr = (torch.ones(img_size, img_size, 3, dtype=torch.uint8) * 255).cpu().numpy()
    d = tf({"pixels": arr})
    x = d["pixels"].unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        o = jepa.encode({"pixels": x})
    emb = o["emb"]
    return int(emb.shape[-1])


def train_one_epoch(
    jepa,
    text_embedder: _QwenTextEmbedder,
    scorer: NavigationTerminalScorer,
    loader: DataLoader,
    pixel_transform,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    pos_weight: torch.Tensor | None,
) -> float:
    scorer.train()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total, n = 0.0, 0
    use_amp = scaler is not None and device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        pixels_b1chw, texts, y = _collate_scorer_batch(batch, pixel_transform, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    enc = jepa.encode({"pixels": pixels_b1chw})
                    latent = enc["emb"][:, 0, :].float()
                    t_emb = text_embedder(texts).float()
            else:
                enc = jepa.encode({"pixels": pixels_b1chw})
                latent = enc["emb"][:, 0, :].float()
                t_emb = text_embedder(texts).float()
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = scorer(latent, t_emb)
                loss = loss_fn(logits.float(), y)
        else:
            logits = scorer(latent, t_emb)
            loss = loss_fn(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


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
def evaluate(
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
    for batch in tqdm(loader, desc="val", leave=False):
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


def _compute_pos_weight_for_indices(h5_path: Path, indices: list[int] | range) -> float:
    n_pos = 0
    with h5py.File(h5_path, "r") as f:
        ie = f["is_end"]
        for i in indices:
            if float(ie[i]) >= 0.5:
                n_pos += 1
    n = len(indices)
    n_neg = n - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NavigationTerminalScorer (phase 2).")
    p.add_argument("--h5-path", type=Path, required=True, help="HDF5 with pixels, instruction, is_end")
    p.add_argument(
        "--jepa-ckpt",
        type=Path,
        required=True,
        help="LeWM ``*_object.ckpt`` from ModelObjectCallBack (JEPA pickle).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Hydra ``config.yaml`` next to LeWM run (default: ckpt parent / config.yaml).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (h5py 建议 0 除非自行处理多进程打开).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--qwen-model",
        type=str,
        default=os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        help="Hugging Face id for Qwen3.* embedding (env QWEN_EMBED_MODEL overrides default).",
    )
    p.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="BCE positive class weight; default: auto from training split (neg/pos).",
    )
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument(
        "--wandb-enabled",
        type=_str2bool,
        default=_str2bool(os.environ.get("SCORER_WANDB_ENABLED", "false")),
        help="是否启用 Weights & Biases（默认 false，可由环境变量 SCORER_WANDB_ENABLED 覆盖）。",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("SCORER_WANDB_ENTITY"),
        help="W&B entity（可选）。",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("SCORER_WANDB_PROJECT", "navigation_scorer"),
        help="W&B project 名称。",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("SCORER_WANDB_RUN_NAME"),
        help="W&B run 名称（可选）。",
    )
    p.add_argument(
        "--wandb-run-id",
        type=str,
        default=os.environ.get("SCORER_WANDB_RUN_ID"),
        help="W&B run id（可选，断点续训可用）。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    _set_seed(args.seed)

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
    _LOGGER.info("JEPA embed_dim=%d (from dummy encode)", embed_dim)

    text_embedder = _QwenTextEmbedder(args.qwen_model, device)
    _LOGGER.info("Qwen text_dim=%d", text_embedder.text_dim)

    scorer = NavigationTerminalScorer(
        latent_dim=embed_dim,
        text_dim=text_embedder.text_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.mlp_layers,
        dropout=args.dropout,
    ).to(device)

    full = _H5RowDataset(args.h5_path)
    n_train = int(len(full) * args.train_split)
    n_val = len(full) - n_train
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
        _LOGGER.info("BCE pos_weight (auto neg/pos on train split)=%.4f", ratio)
        pw = torch.tensor([ratio], device=device)

    opt = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    meta = {
        "jepa_ckpt": str(args.jepa_ckpt),
        "config": str(cfg_path),
        "h5_path": str(args.h5_path),
        "qwen_model": args.qwen_model,
        "img_size": img_size,
        "embed_dim": embed_dim,
        "text_dim": text_embedder.text_dim,
    }
    (args.output_dir / "train_navigation_scorer_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    run = None
    if args.wandb_enabled:
        wandb_entity = _none_if_empty(args.wandb_entity)
        wandb_run_name = _none_if_empty(args.wandb_run_name)
        wandb_run_id = _none_if_empty(args.wandb_run_id)
        try:
            import wandb
        except Exception as e:
            raise RuntimeError("已启用 wandb，但导入失败。请先安装 wandb 并完成 wandb login。") from e
        run = wandb.init(
            project=args.wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            config={
                "h5_path": str(args.h5_path),
                "jepa_ckpt": str(args.jepa_ckpt),
                "qwen_model": args.qwen_model,
                "img_size": img_size,
                "embed_dim": embed_dim,
                "text_dim": text_embedder.text_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "mlp_layers": args.mlp_layers,
                "dropout": args.dropout,
                "train_split": args.train_split,
                "pos_weight": float(pw.item()),
                "output_dir": str(args.output_dir),
            },
        )
        _LOGGER.info("W&B enabled: project=%s run=%s", args.wandb_project, run.name)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            jepa,
            text_embedder,
            scorer,
            train_loader,
            pixel_transform,
            opt,
            device,
            scaler,
            pw,
        )
        val_metrics = evaluate(
            jepa,
            text_embedder,
            scorer,
            val_loader,
            pixel_transform,
            device,
            pw,
        )
        _LOGGER.info(
            (
                "epoch %d | train_loss=%.4f | val_loss=%.4f | "
                "accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | auroc=%.4f"
            ),
            epoch,
            tr_loss,
            val_metrics["val_loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["auroc"],
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "val/loss": val_metrics["val_loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/f1": val_metrics["f1"],
                    "val/auroc": val_metrics["auroc"],
                },
                step=epoch,
            )
        torch.save(
            {
                "scorer": scorer.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "meta": meta,
            },
            args.output_dir / "navigation_scorer_last.pt",
        )
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(
                {
                    "scorer": scorer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["val_loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_auroc": val_metrics["auroc"],
                    "meta": meta,
                },
                args.output_dir / "navigation_scorer_best.pt",
            )

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.finish()

    full.close()
    _LOGGER.info("Done. Checkpoints under %s", args.output_dir)


if __name__ == "__main__":
    main()
