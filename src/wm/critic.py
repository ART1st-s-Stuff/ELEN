"""LEWM 状态序列打分：复用 ARPredictor 骨干，对每个时间步输出 (0, 1) 标量。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_elen_root = Path(__file__).resolve().parents[2]
_lewm_dir = _elen_root / "lewm"
if _lewm_dir.is_dir():
    _p = str(_lewm_dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from module import ARPredictor  # noqa: E402


def _predictor_input_dim(predictor: ARPredictor) -> int:
    return int(predictor.pos_embedding.shape[-1])


def _predictor_token_output_dim(predictor: ARPredictor) -> int:
    t = predictor.transformer
    op = t.output_proj
    if isinstance(op, nn.Linear):
        return int(op.out_features)
    return int(t.norm.normalized_shape[0])


class Critic(nn.Module):
    """仅使用 LEWM 状态序列 `emb`（与 JEPA 中 predictor 输入同形状），逐步打分。"""

    def __init__(self, predictor: ARPredictor):
        super().__init__()
        self.predictor = predictor
        d_in = _predictor_input_dim(predictor)
        self._null_cond = nn.Parameter(torch.zeros(1, 1, d_in))
        d_out = _predictor_token_output_dim(predictor)
        self.score_head = nn.Linear(d_out, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, T, D)，与 LEWM `emb` / ARPredictor 的 x 一致，D == predictor input_dim。
        返回: (B, T)，每步一个 [0, 1] 标量。
        """
        b, t, _ = state.shape
        c = self._null_cond.expand(b, t, -1).to(dtype=state.dtype, device=state.device)
        h = self.predictor(state, c)
        return torch.sigmoid(self.score_head(h)).squeeze(-1)


class CriticWithEmbedding(nn.Module):
    """LEWM 状态 + 外部来源的逐帧 embedding，经线性对齐为 ARPredictor 的条件 c。"""

    def __init__(self, predictor: ARPredictor, extra_emb_dim: int):
        super().__init__()
        self.predictor = predictor
        d_in = _predictor_input_dim(predictor)
        self.ext_to_cond = nn.Linear(extra_emb_dim, d_in)
        d_out = _predictor_token_output_dim(predictor)
        self.score_head = nn.Linear(d_out, 1)

    def forward(self, state: torch.Tensor, extra_emb: torch.Tensor) -> torch.Tensor:
        """
        state: (B, T, D_state)，LEWM 状态，D_state == predictor input_dim。
        extra_emb: (B, T, D_extra)，外部 embedding，D_extra == 构造时的 extra_emb_dim。
        返回: (B, T)，每步一个 [0, 1] 标量。
        """
        c = self.ext_to_cond(extra_emb)
        h = self.predictor(state, c)
        return torch.sigmoid(self.score_head(h)).squeeze(-1)
