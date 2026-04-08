"""Navigation terminal scorer：复用 LeWM ``ARPredictor``，条件向量由 prompt 嵌入代替原 action 嵌入，输出二分类 logits。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# 与 train_navigation_scorer 等脚本一致：保证可导入 lewm
_ELEN_ROOT = Path(__file__).resolve().parents[2]
_LEWM_ROOT = _ELEN_ROOT / "lewm"
for _p in (_ELEN_ROOT, _LEWM_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from lewm.module import ARPredictor  # noqa: E402


class NavigationTerminalScorer(nn.Module):
    """
    使用与 LeWM 相同的 ``ARPredictor``（AdaLN-Zero + Transformer），
    其中原 predictor 的 ``c``（动作嵌入）替换为 **prompt 嵌入**（经线性层映射到与 latent 相同的 ``input_dim``）。

    Args:
        latent: ``(B, D)`` — 单帧嵌入，例如 ``JEPA.encode`` 的 ``emb[:, t]``。
        text_emb: ``(B, E)`` — 冻结文本编码器输出的 prompt 嵌入（如 Qwen）。

    Returns:
        Logits ``(B,)``，用于 ``BCEWithLogitsLoss``；``sigmoid`` 后为 ``[0, 1]`` 概率（对应类别 1）。
    """

    def __init__(
        self,
        latent_dim: int,
        text_dim: int,
        *,
        hidden_dim: int = 512,
        num_hidden: int = 2,
        dropout: float = 0.1,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = 1
        self.prompt_proj = nn.Linear(text_dim, latent_dim, bias=True)
        self.predictor = ARPredictor(
            num_frames=self.num_frames,
            depth=num_hidden,
            heads=heads,
            mlp_dim=mlp_dim,
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 2 or text_emb.dim() != 2:
            raise ValueError(
                f"latent 与 text_emb 期望 (B, D) / (B, E)，得到 {tuple(latent.shape)} / {tuple(text_emb.shape)}"
            )
        if latent.shape[0] != text_emb.shape[0]:
            raise ValueError("batch 维不一致")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent 最后一维应为 {self.latent_dim}，得到 {latent.shape[-1]}")

        x = latent.unsqueeze(1)
        c = self.prompt_proj(text_emb).unsqueeze(1)
        h = self.predictor(x, c)
        return self.head(h[:, -1, :]).squeeze(-1)
