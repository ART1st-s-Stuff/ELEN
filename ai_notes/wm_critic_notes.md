# WM Critic Notes

## 文件

- `src/wm/critic.py`

## 当前设计

- 模型：`NavigationTerminalScorer`
- 输入：`latent (B, D)`、`text_emb (B, E)`
- 输出：`logits (B,)`

## 融合策略（已更新）

- 已从 `concat(latent, text_emb)` 改为 AdaLN-Zero 条件调制。
- 直接复用 LeWM 代码：`from lewm.module import modulate`。
- 条件块核心：
  - `LayerNorm(elementwise_affine=False)`
  - `SiLU + Linear(text_dim, 3 * hidden_dim)` 生成 `shift/scale/gate`
  - 残差更新：`x + gate * ff(modulate(norm(x), shift, scale))`
- 初始化：调制层最后线性层权重/偏置 zero-init。

## 兼容性

- 构造参数保持不变：`latent_dim/text_dim/hidden_dim/num_hidden/dropout`
- `forward(latent, text_emb)` 签名保持不变
- `src/dev/train_navigation_scorer.py` 与 `src/dev/test_navigation_scorer.py` 调用不需改动
