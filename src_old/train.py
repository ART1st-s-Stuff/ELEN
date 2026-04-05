from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import VALID_ACTIONS, get_feature_dim
from src.dataset import TransitionFeatureDataset
from src.model import TransitionFFNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FFNN transition model")
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--val_parquet", type=str, required=True)
    parser.add_argument("--feature_mode", type=str, choices=["clip", "mae", "both"], required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--action_embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine_loss_weight", type=float, default=0.0)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> TransitionFFNN:
    state_dim = get_feature_dim(args.feature_mode)
    return TransitionFFNN(
        state_dim=state_dim,
        num_actions=len(VALID_ACTIONS),
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_loss_weight: float,
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    if cosine_loss_weight <= 0:
        return mse
    cosine_term = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    return mse + cosine_loss_weight * cosine_term


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cosine_loss_weight: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            state_feat = batch["state_feat"].to(device)
            action_id = batch["action_id"].to(device)
            next_state_feat = batch["next_state_feat"].to(device)
            pred = model(state_feat, action_id)
            loss = compute_loss(prediction=pred, target=next_state_feat, cosine_loss_weight=cosine_loss_weight)
            cosine = F.cosine_similarity(pred, next_state_feat, dim=-1).mean().item()

            bsz = state_feat.size(0)
            total_loss += loss.item() * bsz
            total_cosine += cosine * bsz
            total_count += bsz
    return total_loss / max(total_count, 1), total_cosine / max(total_count, 1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = TransitionFeatureDataset(args.train_parquet)
    val_dataset = TransitionFeatureDataset(args.val_parquet)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            state_feat = batch["state_feat"].to(device)
            action_id = batch["action_id"].to(device)
            next_state_feat = batch["next_state_feat"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(state_feat, action_id)
            loss = compute_loss(prediction=pred, target=next_state_feat, cosine_loss_weight=args.cosine_loss_weight)
            loss.backward()
            optimizer.step()

            bsz = state_feat.size(0)
            running_loss += loss.item() * bsz
            running_count += bsz
            pbar.set_postfix(train_loss=running_loss / max(running_count, 1))

        train_loss = running_loss / max(running_count, 1)
        val_loss, val_cosine = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            cosine_loss_weight=args.cosine_loss_weight,
        )

        latest_path = os.path.join(args.save_dir, "latest.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, latest_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_cosine={val_cosine:.6f}"
        )


if __name__ == "__main__":
    main()

