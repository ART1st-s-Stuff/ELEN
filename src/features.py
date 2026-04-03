from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTMAEModel


@dataclass
class FeatureExtractor:
    feature_mode: str
    clip_processor: CLIPProcessor | None
    clip_model: CLIPModel | None
    mae_processor: ViTImageProcessor | None
    mae_model: ViTMAEModel | None
    device: torch.device

    @classmethod
    def build(
        cls,
        feature_mode: str,
        clip_model_name: str,
        mae_model_name: str,
        device: str,
    ) -> "FeatureExtractor":
        torch_device = torch.device(device)
        clip_processor = clip_model = mae_processor = mae_model = None

        if feature_mode in ("clip", "both"):
            clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            clip_model = CLIPModel.from_pretrained(clip_model_name).to(torch_device).eval()

        if feature_mode in ("mae", "both"):
            mae_processor = ViTImageProcessor.from_pretrained(mae_model_name)
            mae_model = ViTMAEModel.from_pretrained(mae_model_name).to(torch_device).eval()

        return cls(
            feature_mode=feature_mode,
            clip_processor=clip_processor,
            clip_model=clip_model,
            mae_processor=mae_processor,
            mae_model=mae_model,
            device=torch_device,
        )

    @torch.no_grad()
    def extract(self, image: Any) -> list[float]:
        features: list[torch.Tensor] = []
        if self.feature_mode in ("clip", "both"):
            if self.clip_processor is None or self.clip_model is None:
                raise RuntimeError("CLIP extractor is not initialized")
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
            clip_feat = self.clip_model.get_image_features(**clip_inputs).squeeze(0).to(torch.float32)
            clip_feat = F.normalize(clip_feat, dim=-1)
            features.append(clip_feat)

        if self.feature_mode in ("mae", "both"):
            if self.mae_processor is None or self.mae_model is None:
                raise RuntimeError("MAE extractor is not initialized")
            mae_inputs = self.mae_processor(images=image, return_tensors="pt")
            mae_inputs = {k: v.to(self.device) for k, v in mae_inputs.items()}
            mae_out = self.mae_model(**mae_inputs)
            mae_feat = mae_out.last_hidden_state.mean(dim=1).squeeze(0).to(torch.float32)
            mae_feat = F.normalize(mae_feat, dim=-1)
            features.append(mae_feat)

        if not features:
            raise RuntimeError(f"No feature extracted for mode {self.feature_mode}")
        return torch.cat(features, dim=-1).cpu().tolist()

