"""基于 LEWM 的 ARPredictor 的状态价值头：对序列上每个时间步输出 [0,1] 标量分数。"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
from torch import nn

# 以 ELEN 为根目录时可导入 lewm.module
_ELEN_ROOT = Path(__file__).resolve().parents[2]
if str(_ELEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ELEN_ROOT))

from lewm.module import ARPredictor


def _arpredictor_out_features(predictor: ARPredictor) -> int:
    """ARPredictor 前向输出最后一维大小。"""
    out = predictor.transformer.output_proj
    if isinstance(out, nn.Linear):
        return out.out_features
    # Identity：各层工作在 hidden_dim
    return predictor.transformer.layers[0].attn.norm.normalized_shape[0]


class Critic(nn.Module):
    """
    输入为 LEWM 的 state 嵌入 ``emb``，形状 ``(B, T, D)``，与训练时 predictor 的 ``x`` 一致。
    ARPredictor 需要条件 ``c``：此处用与 ``D`` 维一致的全零张量占位（与 ``act_emb`` 同形状）。
    输出 ``(B, T)``，每个时间步一个分数，值域 (0,1)。
    """

    def __init__(self, predictor: ARPredictor):
        super().__init__()
        self.predictor = predictor
        d_out = _arpredictor_out_features(predictor)
        self.value_head = nn.Linear(d_out, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, T, D)，LEWM 的 state / emb。

        Returns:
            scores: (B, T)，sigmoid 后的标量分数。
        """
        if state.dim() != 3:
            raise ValueError(f"state 期望 (B, T, D)，得到 {tuple(state.shape)}")
        b, t, d = state.shape
        act_emb = state.new_zeros(b, t, d)
        h = self.predictor(state, act_emb)
        return torch.sigmoid(self.value_head(h).squeeze(-1))


class CriticWithEmbedding(nn.Module):
    """
    输入为 LEWM 的 state ``(B, T, D)`` 以及其它来源的 embedding ``(B, T, E)``。
    将二者拼接后经线性层映射到 ARPredictor 的 ``input_dim``，条件 ``c`` 由外部 embedding 单独线性映射到同维度。
    输出 ``(B, T)``，每步一个 [0,1] 分数。
    """

    def __init__(
        self,
        predictor: ARPredictor,
        state_dim: int,
        ext_emb_dim: int,
    ):
        super().__init__()
        self.predictor = predictor
        input_dim = predictor.pos_embedding.shape[-1]
        self.merge = nn.Linear(state_dim + ext_emb_dim, input_dim)
        self.cond_from_ext = nn.Linear(ext_emb_dim, input_dim)
        d_out = _arpredictor_out_features(predictor)
        self.value_head = nn.Linear(d_out, 1)

    def forward(self, state: torch.Tensor, ext_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, T, state_dim)
            ext_emb: (B, T, ext_emb_dim)

        Returns:
            scores: (B, T)
        """
        if state.dim() != 3 or ext_emb.dim() != 3:
            raise ValueError("state 与 ext_emb 均应为 (B, T, *)")
        if state.shape[:2] != ext_emb.shape[:2]:
            raise ValueError("state 与 ext_emb 的 B、T 必须一致")
        x = self.merge(torch.cat([state, ext_emb], dim=-1))
        c = self.cond_from_ext(ext_emb)
        h = self.predictor(x, c)
        return torch.sigmoid(self.value_head(h).squeeze(-1))
