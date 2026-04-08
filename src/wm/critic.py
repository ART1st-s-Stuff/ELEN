"""Navigation terminal scorer: MLP on LeWM latent + frozen instruction embedding (no LeWM ARPredictor)."""

from __future__ import annotations

import torch
from torch import nn


class NavigationTerminalScorer(nn.Module):
    """
    Binary classifier for "terminal / end state" from concatenated features.

    Args:
        latent: ``(B, D)`` — per-frame embedding from ``JEPA.encode`` → ``emb`` (e.g. ``emb[:, t]``).
        text_emb: ``(B, E)`` — frozen text encoder output (e.g. Qwen embedding).

    Returns:
        Logits ``(B,)`` for ``BCEWithLogitsLoss`` (apply ``sigmoid`` for probabilities in ``[0, 1]``).
    """

    def __init__(
        self,
        latent_dim: int,
        text_dim: int,
        *,
        hidden_dim: int = 512,
        num_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + text_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden):
            layers.extend(
                [
                    nn.Linear(d, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2 or text_emb.dim() != 2:
            raise ValueError(
                f"latent 与 text_emb 期望 (B, D) / (B, E)，得到 {tuple(latent.shape)} / {tuple(text_emb.shape)}"
            )
        if latent.shape[0] != text_emb.shape[0]:
            raise ValueError("batch 维不一致")
        x = torch.cat([latent, text_emb], dim=-1)
        return self.net(x).squeeze(-1)
