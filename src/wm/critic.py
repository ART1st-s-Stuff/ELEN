"""Navigation terminal scorer with LeWM-style AdaLN-Zero conditioning."""

from __future__ import annotations

import torch
from torch import nn

from lewm.module import modulate


class _AdaLNZeroMLPBlock(nn.Module):
    """LeWM-style AdaLN-Zero block for conditioning latent with text embedding."""

    def __init__(self, latent_dim: int, text_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(text_dim, 3 * latent_dim, bias=True),
        )
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout),
        )
        # AdaLN-Zero: zero-init keeps initial behavior close to identity.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = self.ff(modulate(self.norm(x), shift, scale))
        return x + gate * h


class NavigationTerminalScorer(nn.Module):
    """
    Binary classifier for "terminal / end state" from AdaLN-Zero conditioned features.

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
        self.latent_proj = (
            nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [_AdaLNZeroMLPBlock(hidden_dim, text_dim, dropout) for _ in range(num_hidden)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2 or text_emb.dim() != 2:
            raise ValueError(
                f"latent 与 text_emb 期望 (B, D) / (B, E)，得到 {tuple(latent.shape)} / {tuple(text_emb.shape)}"
            )
        if latent.shape[0] != text_emb.shape[0]:
            raise ValueError("batch 维不一致")
        x = self.latent_proj(latent)
        for block in self.blocks:
            x = block(x, text_emb)
        x = self.final_norm(x)
        return self.head(x).squeeze(-1)
