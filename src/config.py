from __future__ import annotations

from dataclasses import dataclass

VALID_ACTIONS = [
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
]

ACTION_TOKEN_MAP = {
    "moveahead": "<|act_moveahead|>",
    "moveback": "<|act_moveback|>",
    "moveright": "<|act_moveright|>",
    "moveleft": "<|act_moveleft|>",
    "rotateright": "<|act_rotateright|>",
    "rotateleft": "<|act_rotateleft|>",
    "lookup": "<|act_lookup|>",
    "lookdown": "<|act_lookdown|>",
}

FEATURE_MODE_CHOICES = ("clip", "mae", "both")

CLIP_FEATURE_DIM = 512
MAE_FEATURE_DIM = 768


def get_feature_dim(feature_mode: str) -> int:
    if feature_mode == "clip":
        return CLIP_FEATURE_DIM
    if feature_mode == "mae":
        return MAE_FEATURE_DIM
    if feature_mode == "both":
        return CLIP_FEATURE_DIM + MAE_FEATURE_DIM
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


@dataclass(frozen=True)
class SplitRatio:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratio must sum to 1.0, got {total}")

