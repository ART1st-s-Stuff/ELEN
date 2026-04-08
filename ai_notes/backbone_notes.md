# Backbone Notes

## 目录

- `src/backbone/base.py`
- `src/backbone/qwen35.py`
- `src/backbone/factory.py`
- `src/backbone/__init__.py`
- 示例：`src/dev/demo_backbone_interface.py`

## 抽象接口状态

- `base.py` 提供统一抽象：
  - `BackboneConfig`
  - `GenerationRequest/Response`
  - `EmbeddingRequest/Response`
  - `PostTrainingSpec/Handle`
- 标准方法契约：
  - `build_inputs(...)`
  - `generate(...)`
  - `embed_text(...)`
  - `save_adapter(...)` / `load_adapter(...)`
  - `prepare_post_training(...)`

## Qwen35 状态

- `Qwen35Backbone` 为占位实现。
- 支持后端类型声明：`transformers` 与 `openai_compatible`。
- 当前 `generate/embed_text` 仍为 `NotImplementedError`。
- `prepare_post_training` 已返回标准化 handle，便于上层流程对接。
