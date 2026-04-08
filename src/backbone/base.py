"""Unified LLM backbone interface for ELEN."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


BackendType = Literal["transformers", "openai_compatible"]
ModelFamily = Literal["qwen3_5"]
PostTrainingType = Literal["sft", "lora", "adapter", "custom"]


@dataclass(slots=True)
class BackboneConfig:
    """Shared runtime configuration for all backbone implementations."""

    model_family: ModelFamily
    model_id: str
    backend_type: BackendType
    device: str = "cpu"
    dtype: str = "auto"
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRequest:
    """Standardized generation request."""

    inputs: Any
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResponse:
    """Standardized generation response."""

    text: str
    raw: Any | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingRequest:
    """Standardized embedding request."""

    texts: list[str]
    normalize: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResponse:
    """Standardized embedding response."""

    embeddings: Any
    raw: Any | None = None


@dataclass(slots=True)
class PostTrainingSpec:
    """Post-training setup contract (SFT/LoRA/adapter/custom)."""

    training_type: PostTrainingType
    output_dir: str | Path
    train_data: str | Path | None = None
    eval_data: str | Path | None = None
    resume_from: str | Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostTrainingHandle:
    """Opaque handle returned by prepare_post_training."""

    training_type: PostTrainingType
    payload: dict[str, Any] = field(default_factory=dict)


class Backbone(ABC):
    """Base class for all ELEN backbone implementations."""

    def __init__(self, config: BackboneConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def supports_generation(self) -> bool:
        """Whether this backbone supports text generation."""

    @property
    @abstractmethod
    def supports_embedding(self) -> bool:
        """Whether this backbone supports text embedding."""

    @property
    @abstractmethod
    def supports_post_training(self) -> bool:
        """Whether this backbone supports post-training workflows."""

    @abstractmethod
    def build_inputs(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Build backend-specific generation inputs from common message format."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Run one generation call."""

    @abstractmethod
    def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Run one embedding call."""

    @abstractmethod
    def save_adapter(self, output_dir: str | Path) -> str:
        """Save adapter (or related lightweight finetune artifacts)."""

    @abstractmethod
    def load_adapter(self, adapter_path: str | Path) -> None:
        """Load adapter from local path."""

    @abstractmethod
    def prepare_post_training(self, spec: PostTrainingSpec) -> PostTrainingHandle:
        """Prepare post-training resources and return execution handle."""
