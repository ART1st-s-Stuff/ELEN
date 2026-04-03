from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import VALID_ACTIONS, get_feature_dim
from src.dataset import TransitionFeatureDataset
from src.model import TransitionFFNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FFNN transition model")
    parser.add_argument("--test_parquet", type=str, required=True)
    parser.add_argument("--feature_mode", type=str, choices=["clip", "mae", "both"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--action_embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> TransitionFFNN:
    return TransitionFFNN(
        state_dim=get_feature_dim(args.feature_mode),
        num_actions=len(VALID_ACTIONS),
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = TransitionFeatureDataset(args.test_parquet)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_mse = 0.0
    total_cosine = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            state_feat = batch["state_feat"].to(device)
            action_id = batch["action_id"].to(device)
            next_state_feat = batch["next_state_feat"].to(device)
            pred = model(state_feat, action_id)

            mse = F.mse_loss(pred, next_state_feat, reduction="none").mean(dim=-1)
            cosine = F.cosine_similarity(pred, next_state_feat, dim=-1)
            bsz = state_feat.size(0)
            total_mse += mse.sum().item()
            total_cosine += cosine.sum().item()
            total_count += bsz

    print(f"test_samples={total_count}")
    print(f"test_mse={total_mse / max(total_count, 1):.6f}")
    print(f"test_cosine={total_cosine / max(total_count, 1):.6f}")


if __name__ == "__main__":
    main()

