from __future__ import annotations

import torch
from torch import nn


class TransitionFFNN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        action_embed_dim: int = 32,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        input_dim = state_dim + action_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state_feat: torch.Tensor, action_id: torch.Tensor) -> torch.Tensor:
        action_feat = self.action_embed(action_id)
        x = torch.cat([state_feat, action_feat], dim=-1)
        return self.mlp(x)

