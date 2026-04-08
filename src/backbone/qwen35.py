"""Qwen3.5 backbone stub implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Backbone,
    BackboneConfig,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
    PostTrainingHandle,
    PostTrainingSpec,
)

_SUPPORTED_BACKENDS = {"transformers", "openai_compatible"}


class Qwen35Backbone(Backbone):
    """Step 1 stub for Qwen3.5 backbone."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        if config.model_family != "qwen3_5":
            raise ValueError(f"Qwen35Backbone 仅支持 model_family='qwen3_5'，收到: {config.model_family}")
        if config.backend_type not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "Qwen35Backbone backend_type 必须是 'transformers' 或 'openai_compatible'，"
                f"收到: {config.backend_type}"
            )

    @property
    def supports_generation(self) -> bool:
        return True

    @property
    def supports_embedding(self) -> bool:
        return True

    @property
    def supports_post_training(self) -> bool:
        return True

    def build_inputs(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return {
            "backend_type": self.config.backend_type,
            "model_id": self.config.model_id,
            "messages": messages,
            "extra": extra or {},
        }

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError(
            "Qwen35Backbone.generate 尚未实现（Step 2 将先接入 transformers 的最小单轮推理）。"
        )

    def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError(
            "Qwen35Backbone.embed_text 尚未实现（Step 2 将先接入 transformers 的最小 embedding）。"
        )

    def save_adapter(self, output_dir: str | Path) -> str:
        # Step 1: return normalized target path for upper-layer pipeline wiring.
        return str(Path(output_dir))

    def load_adapter(self, adapter_path: str | Path) -> None:
        # Step 1: keep as no-op placeholder.
        _ = Path(adapter_path)

    def prepare_post_training(self, spec: PostTrainingSpec) -> PostTrainingHandle:
        payload = {
            "backend_type": self.config.backend_type,
            "model_id": self.config.model_id,
            "output_dir": str(spec.output_dir),
            "train_data": None if spec.train_data is None else str(spec.train_data),
            "eval_data": None if spec.eval_data is None else str(spec.eval_data),
            "resume_from": None if spec.resume_from is None else str(spec.resume_from),
            "extra": dict(spec.extra),
        }
        return PostTrainingHandle(training_type=spec.training_type, payload=payload)
