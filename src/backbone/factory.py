"""Backbone factory."""

from __future__ import annotations

from .base import Backbone, BackboneConfig
from .qwen35 import Qwen35Backbone


def build_backbone(config: BackboneConfig) -> Backbone:
    """Build a backbone instance from unified config."""
    if config.model_family == "qwen3_5":
        return Qwen35Backbone(config)
    raise ValueError(f"不支持的 model_family: {config.model_family}")
